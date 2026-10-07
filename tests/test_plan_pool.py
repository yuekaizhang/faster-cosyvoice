# tests/test_plan_pool.py
"""[M3.5] FlashInferDiT pooled plan cache: alternating keys must NOT replan.

Pure-CPU test of the pool logic (FIFO bound, recycle-on-evict, hit == no
plan call): RaggedAttentionRunner is monkeypatched so no CUDA workspace /
flashinfer wrapper is touched.
"""
import pytest
import torch

fdit = pytest.importorskip("faster_cosyvoice.token2wav.flashinfer_dit")


class _FakeRunner:
    """Mirrors RaggedAttentionRunner's plan-key semantics + plan_calls."""

    def __init__(self, num_heads, head_dim, device, workspace_size=0):
        self.plan_calls = 0
        self._planned_key = None

    def plan(self, batch_size, seq_len, dtype, chunk_size=None):
        key = (batch_size, seq_len, dtype, chunk_size)
        if key == self._planned_key:
            return
        self.plan_calls += 1
        self._planned_key = key

    def plan_docs(self, doc_lens, dtype, chunk_size=None):
        key = (tuple(doc_lens), dtype, chunk_size)
        if key == self._planned_key:
            return
        self.plan_calls += 1
        self._planned_key = key


@pytest.fixture
def dit(monkeypatch):
    monkeypatch.setattr(fdit, "RaggedAttentionRunner", _FakeRunner)
    # tiny dims: only the pool is under test; device=cpu (pool is lazy)
    return fdit.FlashInferDiT(dim=64, depth=1, heads=2, dim_head=32,
                              device="cpu", plan_cache_size=4)


def test_alternating_keys_do_not_replan(dit):
    """Streaming regression: chunk-1 keys recur across requests; the old
    single-slot cache replanned them every request (~17ms TTFP tax)."""
    k1 = dict(doc_lens=[100, 100], dtype=torch.float16, chunk_size=50)
    k2 = dict(doc_lens=[200, 200], dtype=torch.float16, chunk_size=50)
    r1 = dit._planned_runner_docs(**k1)
    r2 = dit._planned_runner_docs(**k2)
    assert r1 is not r2 and r1.plan_calls == 1 and r2.plan_calls == 1
    # request 2: same chunk keys -> same ready-planned runners, zero replans
    assert dit._planned_runner_docs(**k1) is r1
    assert dit._planned_runner_docs(**k2) is r2
    assert r1.plan_calls == 1 and r2.plan_calls == 1


def test_pool_bounded_fifo_and_recycled(dit):
    runners = [dit._planned_runner(2, 100 * (i + 1), torch.float16)
               for i in range(4)]
    assert len(dit._runner_pool) == 4
    # 5th key evicts + RECYCLES the oldest runner (workspace reuse)
    r5 = dit._planned_runner(2, 900, torch.float16)
    assert len(dit._runner_pool) == 4
    assert r5 is runners[0] and r5.plan_calls == 2  # replanned for new key
    # evicted key comes back: replans (bounded cache, not unbounded)
    r1b = dit._planned_runner(2, 100, torch.float16)
    assert r1b._planned_key == (2, 100, torch.float16, None)


def test_pool_size_1_matches_old_single_runner_behavior(monkeypatch):
    monkeypatch.setattr(fdit, "RaggedAttentionRunner", _FakeRunner)
    dit = fdit.FlashInferDiT(dim=64, depth=1, heads=2, dim_head=32,
                             device="cpu", plan_cache_size=1)
    r = dit._planned_runner(2, 100, torch.float16)
    assert dit._planned_runner(2, 200, torch.float16) is r  # recycled slot
    assert dit._planned_runner(2, 100, torch.float16) is r
    assert r.plan_calls == 3  # replace-on-change, exactly the old semantics
