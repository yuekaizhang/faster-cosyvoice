"""vLLM engine construction, DSpark configuration, and sampling helpers.

Both offline and server engines call :func:`build_llm_kwargs`, so speculative
decoding is configured identically in both paths.  The draft checkpoint's
``config.json`` supplies its method and block size.
"""
import json
import logging
import os

from faster_cosyvoice.config import LLMConfig


def load_draft_config(draft_model: str) -> dict:
    """Load a draft checkpoint config from a local path or Hugging Face."""
    local = os.path.join(draft_model, "config.json")
    if os.path.isfile(local):
        with open(local) as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(draft_model, "config.json")) as f:
        return json.load(f)


def build_speculative_config(cfg: LLMConfig) -> dict | None:
    """Translate :class:`LLMConfig` into vLLM's speculative config.

    DSpark predicts one block that includes the current token, so a draft block
    of size ``N`` contributes ``N - 1`` speculative tokens.  Repetition-penalty
    mirroring requires the patched vLLM pinned in ``pyproject.toml``.
    """
    if not cfg.draft_model:
        return None

    draft_config = load_draft_config(cfg.draft_model)
    method = cfg.method or draft_config.get("speculators_model_type")
    if method is None:
        raise ValueError(
            "draft config.json has no speculators_model_type; set method explicitly"
        )
    block_size = draft_config.get("block_size")
    num_speculative_tokens = cfg.num_spec_tokens or (
        block_size - 1 if block_size else 3
    )
    speculative_config = {
        "model": cfg.draft_model,
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
        "draft_sample_method": cfg.draft_sample_method,
    }
    if cfg.repetition_penalty != 1.0:
        speculative_config["draft_apply_repetition_penalty"] = True
    return speculative_config


def build_llm_kwargs(cfg: LLMConfig) -> dict:
    """Build arguments shared by vLLM's offline and asynchronous engines."""
    kwargs = {
        "model": cfg.target_model,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "max_model_len": cfg.max_model_len,
        "disable_log_stats": False,
    }
    speculative_config = build_speculative_config(cfg)
    if speculative_config is not None:
        kwargs["speculative_config"] = speculative_config
    return kwargs


def create_offline_llm(cfg: LLMConfig):
    from vllm import LLM
    return LLM(**build_llm_kwargs(cfg))


def make_sampling_params(cfg: LLMConfig, seed: int):
    from vllm import SamplingParams
    return SamplingParams(
        temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        max_tokens=cfg.max_tokens, seed=seed)


def read_spec_counters(llm) -> dict:
    counters = {}
    try:
        for m in llm.get_metrics():
            if "spec_decode" in m.name and hasattr(m, "value"):
                counters[m.name] = m.value
    except Exception as e:  # noqa: BLE001
        logging.getLogger(__name__).debug("get_metrics failed: %s", e)
    return counters


def create_async_llm(cfg: LLMConfig):
    """Create the server engine and its EngineCore subprocess.

    Construction blocks while weights load, so the server calls this inside
    the FastAPI lifespan and shuts it down during lifespan teardown.
    """
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(**build_llm_kwargs(cfg), enable_log_requests=False))


def make_stream_sampling_params(cfg: LLMConfig, codec, text_token_len: int,
                                seed: int):
    """Build streaming parameters with dynamic output-length bounds.

    vLLM emits token deltas without detokenizing.  The minimum and maximum are
    respectively 2x and 20x the input text-token count, capped by the model
    limit.  vLLM suppresses EOS until ``min_tokens`` has been reached.
    """
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    max_tok = max(1, min(cfg.max_tokens, 20 * text_token_len))
    min_tok = min(max(1, 2 * text_token_len), max_tok)
    return SamplingParams(
        temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        min_tokens=min_tok,
        max_tokens=max_tok,
        stop_token_ids=[codec.eos_token_id],
        seed=seed, detokenize=False,
        output_kind=RequestOutputKind.DELTA)


async def stream_token_ids(engine, prompt: str, sp, request_id: str):
    """Yield ``(delta_token_ids, finished)`` from an AsyncLLM request.

    Closing this generator after a client disconnect propagates cancellation
    to vLLM, which aborts the engine-side request.
    """
    async for out in engine.generate(prompt, sp, request_id):
        yield list(out.outputs[0].token_ids), out.finished
        if out.finished:
            if out.outputs[0].finish_reason == "length":
                logging.getLogger(__name__).warning(
                    "request %s hit max_tokens (audio likely truncated)",
                    request_id)
            break
