"""启动期环境自检：fail fast，报可修复的指引（spec §7）。"""
import inspect


def _importable(mod: str) -> bool:
    # 用内建 __import__（而非 importlib.import_module），使测试可通过
    # monkeypatch builtins.__import__ 模拟缺依赖。
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def _has_draft_mirror() -> bool:
    """探测 patched vllm 的 rep-penalty mirror（vllm PR #48932）。"""
    try:
        from vllm.config import SpeculativeConfig
    except ImportError:
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
        ("vllm", "bash scripts/setup_env.sh 后按其输出 export PYTHONPATH"),
        ("s3tokenizer", "pip install s3tokenizer"),
        ("onnxruntime", "pip install onnxruntime"),
    ]:
        if not _importable(mod):
            problems.append(f"{mod} 不可导入 — {hint}")
    if require_flashinfer:
        if not _importable("flashinfer"):
            problems.append("flashinfer 不可导入 — pip install flashinfer-python；"
                            "或用 --estimator torch 降级")
        if not _importable("triton"):
            problems.append("triton 不可导入（flashinfer packed 批量必需）")
    if require_draft_mirror and not _has_draft_mirror():
        problems.append(
            "patched vllm 未生效（缺 draft_apply_repetition_penalty）— "
            "export PYTHONPATH=third_party/spec-vllm:$PYTHONPATH（见 setup_env.sh 输出）；"
            "或 --draft-model none 关闭投机解码")
    return problems
