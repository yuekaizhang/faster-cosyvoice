"""Startup environment checks with actionable failure messages."""
import inspect


def _import_problem(mod: str) -> str | None:
    """Return ``None`` when a module imports, otherwise a failure reason."""
    # Using the built-in import keeps this probe easy to monkeypatch in tests.
    try:
        __import__(mod)
        return None
    except ImportError:
        return "is not importable"
    except Exception as e:  # Installed modules can still fail on missing CUDA libraries.
        return f"crashed during import: {e!r}"


def _has_draft_mirror() -> bool:
    """Check whether vLLM supports repetition penalty in the draft model."""
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
            problems.append(
                f"triton {reason} (required for packed FlashInfer batches)"
            )
    if require_draft_mirror and not _has_draft_mirror():
        problems.append(
            "the pinned vLLM patch is missing draft_apply_repetition_penalty — "
            "run `uv sync --frozen` to install the locked vLLM fork; "
            "or use --draft-model none to disable speculative decoding")
    return problems
