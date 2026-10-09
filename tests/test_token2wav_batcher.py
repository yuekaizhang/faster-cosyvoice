import asyncio
from dataclasses import dataclass

import pytest

from faster_cosyvoice.streaming.token2wav_batcher import Token2WavWorker


class FakeT2W:
    def __init__(self):
        self.calls = []

    def stream_step(self, session, plan):
        self.calls.append((session, plan))
        return f"pcm-{session}-{plan}"


@dataclass
class FakeStream:
    name: str
    playback_started_at_s: float | None = None
    emitted_duration_s: float = 0.0

    def __str__(self):
        return self.name


@pytest.mark.asyncio
async def test_submit_returns_result():
    t2w = FakeT2W()
    w = Token2WavWorker(t2w)
    await w.start()
    try:
        out = await w.submit("s1", "p1", chunk_index=0)
        assert out == "pcm-s1-p1"
    finally:
        await w.stop()


@pytest.mark.asyncio
async def test_first_chunks_have_priority():
    """A new stream's first chunk precedes an established stream's next chunk."""
    t2w = FakeT2W()
    w = Token2WavWorker(t2w)
    # Fill the queue before starting the worker so selection order is deterministic.
    f_old = w.submit_nowait("old", "p", chunk_index=3)
    f_new = w.submit_nowait("new", "p", chunk_index=0)
    await w.start()
    try:
        await asyncio.gather(f_old, f_new)
        assert t2w.calls[0][0] == "new" and t2w.calls[1][0] == "old"
    finally:
        await w.stop()


@pytest.mark.asyncio
async def test_urgent_established_stream_preempts_startup():
    """A stream near underrun preempts a new stream's first chunk."""
    t2w = FakeT2W()
    w = Token2WavWorker(t2w, deadline_reserve_s=0.1,
                        clock=lambda: 10.0)
    established = FakeStream("established", playback_started_at_s=9.0,
                             emitted_duration_s=1.05)  # deadline=10.05
    startup = FakeStream("startup")
    f_startup = w.submit_nowait(startup, "first", chunk_index=0)
    f_established = w.submit_nowait(established, "next", chunk_index=1)
    await w.start()
    try:
        await asyncio.gather(f_startup, f_established)
        assert t2w.calls[0][0] is established
    finally:
        await w.stop()


@pytest.mark.asyncio
async def test_startup_preempts_nonurgent_established_stream():
    """Startup wins when established streams have enough buffered audio."""
    t2w = FakeT2W()
    w = Token2WavWorker(t2w, deadline_reserve_s=0.1,
                        clock=lambda: 10.0)
    established = FakeStream("established", playback_started_at_s=9.0,
                             emitted_duration_s=2.0)  # deadline=11.0
    startup = FakeStream("startup")
    f_established = w.submit_nowait(established, "next", chunk_index=1)
    f_startup = w.submit_nowait(startup, "first", chunk_index=0)
    await w.start()
    try:
        await asyncio.gather(f_startup, f_established)
        assert t2w.calls[0][0] is startup
    finally:
        await w.stop()


@pytest.mark.asyncio
async def test_established_streams_use_earliest_playback_deadline():
    t2w = FakeT2W()
    w = Token2WavWorker(t2w, deadline_reserve_s=0.1,
                        clock=lambda: 10.0)
    later = FakeStream("later", playback_started_at_s=9.0,
                       emitted_duration_s=3.0)
    earlier = FakeStream("earlier", playback_started_at_s=9.0,
                         emitted_duration_s=2.0)
    f_later = w.submit_nowait(later, "next", chunk_index=1)
    f_earlier = w.submit_nowait(earlier, "next", chunk_index=1)
    await w.start()
    try:
        await asyncio.gather(f_later, f_earlier)
        assert t2w.calls[0][0] is earlier
    finally:
        await w.stop()


@pytest.mark.parametrize("scheduler_mode", ["deadline", "legacy"])
def test_deadline_reserve_must_be_nonnegative(scheduler_mode):
    with pytest.raises(ValueError):
        Token2WavWorker(
            FakeT2W(),
            deadline_reserve_s=-0.1,
            scheduler_mode=scheduler_mode,
        )


def test_scheduler_mode_must_be_known():
    with pytest.raises(ValueError):
        Token2WavWorker(FakeT2W(), scheduler_mode="unknown")


@pytest.mark.asyncio
async def test_legacy_scheduler_uses_chunk_index_then_arrival():
    t2w = FakeT2W()
    w = Token2WavWorker(t2w, scheduler_mode="legacy")
    f_later_chunk = w.submit_nowait("older", "c3", chunk_index=3)
    f_earlier_chunk = w.submit_nowait("newer", "c1", chunk_index=1)
    await w.start()
    try:
        await asyncio.gather(f_later_chunk, f_earlier_chunk)
        assert t2w.calls == [("newer", "c1"), ("older", "c3")]
    finally:
        await w.stop()


@pytest.mark.asyncio
async def test_submit_after_stop_raises():
    w = Token2WavWorker(FakeT2W())
    await w.start()
    await w.stop()
    with pytest.raises(RuntimeError):
        w.submit_nowait("s", "p", chunk_index=0)


@pytest.mark.asyncio
async def test_stop_does_not_lose_wakeup_after_draining_queue():
    """stop racing with the transition to idle must not hang."""
    w = Token2WavWorker(FakeT2W())
    future = w.submit_nowait("s", "p", chunk_index=0)
    await w.start()
    assert await future == "pcm-s-p"
    await asyncio.wait_for(w.stop(), timeout=1)


@pytest.mark.asyncio
async def test_submit_cancelled_on_generator_exit():
    """GeneratorExit from a disconnected consumer cancels its queued future.

    The coroutine is driven manually because task cancellation would make
    asyncio cancel the awaited future itself and would not test worker logic.
    """
    w = Token2WavWorker(FakeT2W())
    fut_holder = {}
    orig = w.submit_nowait

    def spy(session, plan, chunk_index):
        fut = orig(session, plan, chunk_index)
        fut_holder["fut"] = fut
        return fut

    w.submit_nowait = spy

    coro = w.submit("s", "p", chunk_index=0)
    coro.send(None)                 # Suspend at await fut with no worker running.
    with pytest.raises(GeneratorExit):
        coro.throw(GeneratorExit)   # Simulate Starlette closing the generator.
    assert fut_holder["fut"].cancelled()


@pytest.mark.asyncio
async def test_error_fails_only_that_job():
    class Boom:
        def stream_step(self, session, plan):
            if session == "bad":
                raise RuntimeError("boom")
            return "ok"

    w = Token2WavWorker(Boom())
    await w.start()
    try:
        with pytest.raises(RuntimeError):
            await w.submit("bad", "p", chunk_index=0)
        assert await w.submit("good", "p", chunk_index=0) == "ok"
    finally:
        await w.stop()


# ---------------------------------------------------------------- packed (v2)

class FakeBatchT2W:
    """Record packed calls and return one positional result per request."""

    def __init__(self):
        self.batch_calls = []

    def stream_step_batched(self, sessions, plans):
        self.batch_calls.append((list(sessions), list(plans)))
        return [f"pcm-{s}-{p}" for s, p in zip(sessions, plans, strict=True)]


@pytest.mark.asyncio
async def test_packed_batches_distinct_sessions():
    """Three ready sessions form one packed call with correctly routed results."""
    t2w = FakeBatchT2W()
    w = Token2WavWorker(t2w, mode="packed", max_batch=8)
    f1 = w.submit_nowait("s1", "p1", chunk_index=0)
    f2 = w.submit_nowait("s2", "p2", chunk_index=0)
    f3 = w.submit_nowait("s3", "p3", chunk_index=0)
    await w.start()
    try:
        assert await asyncio.gather(f1, f2, f3) == [
            "pcm-s1-p1", "pcm-s2-p2", "pcm-s3-p3"]
        assert len(t2w.batch_calls) == 1
        assert len(t2w.batch_calls[0][0]) == 3
    finally:
        await w.stop()


@pytest.mark.asyncio
async def test_packed_same_session_split_across_batches():
    """Dependent chunks from one session execute in separate batches."""
    t2w = FakeBatchT2W()
    w = Token2WavWorker(t2w, mode="packed", max_batch=8)
    f0 = w.submit_nowait("s1", "c0", chunk_index=0)
    f1 = w.submit_nowait("s1", "c1", chunk_index=1)
    await w.start()
    try:
        assert await asyncio.gather(f0, f1) == ["pcm-s1-c0", "pcm-s1-c1"]
        assert len(t2w.batch_calls) == 2
        assert t2w.batch_calls[0] == (["s1"], ["c0"])
        assert t2w.batch_calls[1] == (["s1"], ["c1"])
    finally:
        await w.stop()


@pytest.mark.asyncio
async def test_packed_skips_cancelled_future():
    """Cancelled futures are omitted while remaining jobs still form a batch."""
    t2w = FakeBatchT2W()
    w = Token2WavWorker(t2w, mode="packed", max_batch=8)
    f1 = w.submit_nowait("s1", "p1", chunk_index=0)
    f2 = w.submit_nowait("s2", "p2", chunk_index=0)
    f3 = w.submit_nowait("s3", "p3", chunk_index=0)
    f2.cancel()
    await w.start()
    try:
        assert await asyncio.gather(f1, f3) == ["pcm-s1-p1", "pcm-s3-p3"]
        assert len(t2w.batch_calls) == 1
        assert t2w.batch_calls[0][0] == ["s1", "s3"]
    finally:
        await w.stop()


@pytest.mark.asyncio
async def test_packed_batch_error_fails_whole_batch_worker_alive():
    """A packed failure fails that whole batch without killing the worker."""

    class BoomBatch(FakeBatchT2W):
        def __init__(self):
            super().__init__()
            self.fail_next = True

        def stream_step_batched(self, sessions, plans):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("batch boom")
            return super().stream_step_batched(sessions, plans)

    t2w = BoomBatch()
    w = Token2WavWorker(t2w, mode="packed", max_batch=8)
    f1 = w.submit_nowait("s1", "p1", chunk_index=0)
    f2 = w.submit_nowait("s2", "p2", chunk_index=0)
    await w.start()
    try:
        r1, r2 = await asyncio.gather(f1, f2, return_exceptions=True)
        assert isinstance(r1, RuntimeError) and isinstance(r2, RuntimeError)
        # A later submission proves the worker survived the batch failure.
        assert await w.submit("s3", "p3", chunk_index=0) == "pcm-s3-p3"
    finally:
        await w.stop()


@pytest.mark.asyncio
async def test_packed_result_count_mismatch_fails_batch_worker_alive():
    class ShortBatch(FakeBatchT2W):
        def __init__(self):
            super().__init__()
            self.short_next = True

        def stream_step_batched(self, sessions, plans):
            if self.short_next:
                self.short_next = False
                return []
            return super().stream_step_batched(sessions, plans)

    w = Token2WavWorker(ShortBatch(), mode="packed", max_batch=8)
    first = w.submit_nowait("s1", "p1", chunk_index=0)
    await w.start()
    try:
        with pytest.raises(RuntimeError, match="wrong result count"):
            await first
        assert await w.submit("s2", "p2", chunk_index=0) == "pcm-s2-p2"
    finally:
        await w.stop()
