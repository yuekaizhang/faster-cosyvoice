# faster_cosyvoice/llm/engine.py
"""vLLM 引擎参数组装（纯函数，可 CPU 单测）+ 引擎/采样工厂。

speculative_config 组装逻辑与 SpeechSpec benchmark_tts.py 一致：
method ← draft config.json 的 speculators_model_type，
num_speculative_tokens ← block_size - 1，
rp != 1.0 时开 draft_apply_repetition_penalty（vllm PR #48932 mirror）。
"""
import json
import os

from faster_cosyvoice.config import LLMConfig


def load_draft_config(draft_model: str) -> dict:
    local = os.path.join(draft_model, "config.json")
    if os.path.isfile(local):
        with open(local) as f:
            return json.load(f)
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(draft_model, "config.json")))


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
    except Exception:  # noqa: BLE001
        pass
    return counters
