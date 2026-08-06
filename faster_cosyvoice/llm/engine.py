# faster_cosyvoice/llm/engine.py
"""vLLM 引擎参数组装（纯函数，可 CPU 单测）+ 引擎/采样工厂。

speculative_config 组装逻辑与 SpeechSpec benchmark_tts.py 一致：
method ← draft config.json 的 speculators_model_type，
num_speculative_tokens ← block_size - 1，
rp != 1.0 时开 draft_apply_repetition_penalty（vllm PR #48932 mirror）。
"""
import json
import logging
import os

from faster_cosyvoice.config import LLMConfig


def load_draft_config(draft_model: str) -> dict:
    local = os.path.join(draft_model, "config.json")
    if os.path.isfile(local):
        with open(local) as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(draft_model, "config.json")) as f:
        return json.load(f)


def build_llm_kwargs(cfg: LLMConfig) -> dict:
    kwargs = dict(
        model=cfg.target_model,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        disable_log_stats=False,
    )
    if cfg.draft_model:
        dc = load_draft_config(cfg.draft_model)
        method = cfg.method or dc.get("speculators_model_type")
        if method is None:
            raise ValueError("draft config.json 缺 speculators_model_type；"
                             "请显式传 method")
        block = dc.get("block_size")
        num_spec = cfg.num_spec_tokens or ((block - 1) if block else 3)
        sc = {
            "model": cfg.draft_model,
            "method": method,
            "num_speculative_tokens": num_spec,
            "draft_sample_method": cfg.draft_sample_method,
        }
        if cfg.repetition_penalty != 1.0:
            sc["draft_apply_repetition_penalty"] = True
        kwargs["speculative_config"] = sc
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
    """server 用异步引擎（spec §5.1）。构造即拉起 engine-core 子进程
    （阻塞加载）——放 FastAPI lifespan；teardown 调 engine.shutdown()。"""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(**build_llm_kwargs(cfg), enable_log_requests=False))


def make_stream_sampling_params(cfg: LLMConfig, codec, text_token_len: int,
                                seed: int):
    """流式采样参数：DELTA 增量、不 detokenize（stop 字符串会 raise，
    用 stop_token_ids）、动态 min/max（vllm-omni 同款 2×/20× 比例）。
    min_tokens 到达前 vLLM 会抑制一切 stop（含 eos）。"""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    return SamplingParams(
        temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        min_tokens=max(1, 2 * text_token_len),
        max_tokens=min(cfg.max_tokens, 20 * text_token_len),
        stop_token_ids=[codec.eos_token_id],
        seed=seed, detokenize=False,
        output_kind=RequestOutputKind.DELTA)


async def stream_token_ids(engine, prompt: str, sp, request_id: str):
    """AsyncLLM DELTA 流 → 逐次 yield (delta_token_ids, finished)。
    消费方取消（客户端断连）时生成器关闭即自动 abort 引擎侧请求。"""
    async for out in engine.generate(prompt, sp, request_id):
        yield list(out.outputs[0].token_ids), out.finished
        if out.finished:
            break
