import pytest

import faster_cosyvoice.assets as assets


def test_ensure_skips_download_when_present(tmp_path, monkeypatch):
    for f in assets.TOKEN2WAV_FILES:
        (tmp_path / f).write_bytes(b"x")
    called = []
    monkeypatch.setattr(assets, "_snapshot_download",
                        lambda *a, **k: called.append(1))
    assert assets.ensure_token2wav_assets(str(tmp_path)) == str(tmp_path)
    assert not called


def test_ensure_downloads_when_missing(tmp_path, monkeypatch):
    called = {}

    def fake_download(repo_id, local_dir, allow_patterns):
        called.update(repo=repo_id, patterns=allow_patterns)
        for f in assets.TOKEN2WAV_FILES:
            (tmp_path / f).write_bytes(b"x")

    monkeypatch.setattr(assets, "_snapshot_download", fake_download)
    assets.ensure_token2wav_assets(str(tmp_path))
    assert called["repo"] == "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    assert set(assets.TOKEN2WAV_FILES) <= set(called["patterns"])


def test_ensure_raises_when_still_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "_snapshot_download", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError):
        assets.ensure_token2wav_assets(str(tmp_path))
