import builtins

from faster_cosyvoice.envcheck import check_environment


def test_missing_flashinfer_reported(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "flashinfer":
            raise ImportError("no flashinfer")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    problems = check_environment(require_flashinfer=True, require_draft_mirror=False)
    assert any("flashinfer" in p for p in problems)


def test_crashing_import_reported(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "flashinfer":
            raise OSError("libcudart.so.13: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    problems = check_environment(require_flashinfer=True, require_draft_mirror=False)
    assert any("flashinfer" in p and "libcudart" in p for p in problems)


def test_flashinfer_not_required():
    problems = check_environment(require_flashinfer=False, require_draft_mirror=False)
    assert not any("flashinfer" in p for p in problems)
