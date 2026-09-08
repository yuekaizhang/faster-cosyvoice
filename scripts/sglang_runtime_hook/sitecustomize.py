"""Make an explicitly configured local WeText snapshot visible to workers."""

from __future__ import annotations

import os


def _install_wetext_snapshot_override() -> None:
    wetext_model_dir = os.environ.get("WETEXT_MODEL_DIR")
    if not wetext_model_dir:
        return

    import modelscope

    snapshot_download = modelscope.snapshot_download

    def snapshot_download_with_local_wetext(model_id, *args, **kwargs):
        if model_id == "pengzhendong/wetext":
            return wetext_model_dir
        return snapshot_download(model_id, *args, **kwargs)

    modelscope.snapshot_download = snapshot_download_with_local_wetext


_install_wetext_snapshot_override()
