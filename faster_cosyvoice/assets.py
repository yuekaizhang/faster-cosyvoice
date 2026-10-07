"""HF 资产下载/校验。缺失时报确切 repo id 与目标路径（spec §7 fail fast）。"""
import os

TOKEN2WAV_REPO = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
TOKEN2WAV_FILES = ["flow.pt", "hift.pt", "campplus.onnx"]


def _snapshot_download(repo_id, local_dir, allow_patterns):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, local_dir=local_dir,
                      allow_patterns=allow_patterns)


def ensure_token2wav_assets(model_dir: str) -> str:
    missing = [f for f in TOKEN2WAV_FILES
               if not os.path.isfile(os.path.join(model_dir, f))]
    if missing:
        os.makedirs(model_dir, exist_ok=True)
        _snapshot_download(TOKEN2WAV_REPO, model_dir, TOKEN2WAV_FILES)
    still = [f for f in TOKEN2WAV_FILES
             if not os.path.isfile(os.path.join(model_dir, f))]
    if still:
        raise FileNotFoundError(
            f"{still} 不在 {model_dir}；请检查到 {TOKEN2WAV_REPO} 的网络/凭证")
    return model_dir
