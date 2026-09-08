"""Startup environment checks with actionable failure messages."""
import inspect


def _import_problem(mod: str) -> str | None:
    """None = importable；否则返回失败原因（含崩溃型 import）。"""
    # 用内建 __import__ 而非 importlib.import_module，测试才能 monkeypatch builtins.__import__
    try:
        __import__(mod)
        return None
    except ImportError:
        return "不可导入"
    except Exception as e:  # 装了但 import 崩（缺 CUDA 库等）
        return f"导入崩溃: {e!r}"


def _has_draft_mirror() -> bool:
    """探测 patched vllm 的 rep-penalty mirror（vllm PR #48932）。"""
    try:
        from vllm.config import SpeculativeConfig
    except Exception:
        return False
    try:
        src = inspect.getsource(SpeculativeConfig)
    except (OSError, TypeError):
        return False
    return "draft_apply_repetition_penalty" in src


def check_environment(require_flashinfer: bool = True,
                      require_draft_mirror: bool = True) -> list[str]:
    problems: list[str] = []
    for mod, hint in [
        ("vllm", "run `uv sync --frozen`"),
        ("s3tokenizer", "run `uv sync --frozen`"),
        ("onnxruntime", "run `uv sync --frozen`"),
    ]:
        reason = _import_problem(mod)
        if reason is not None:
            problems.append(f"{mod} {reason} — {hint}")
    if require_flashinfer:
        reason = _import_problem("flashinfer")
        if reason is not None:
            problems.append(f"flashinfer {reason} — run `uv sync --frozen`; "
                            "or use the Torch Flow estimator")
        reason = _import_problem("triton")
        if reason is not None:
            problems.append(f"triton {reason}（flashinfer packed 批量必需）")
    if require_draft_mirror and not _has_draft_mirror():
        problems.append(
            "patched vllm 未生效（缺 draft_apply_repetition_penalty）— "
            "run `uv sync --frozen` to install the locked vLLM fork; "
            "or use --draft-model none to disable speculative decoding")
    return problems
