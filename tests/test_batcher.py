# tests/test_batcher.py
import asyncio

import pytest

from faster_cosyvoice.streaming.batcher import Token2WavWorker


class FakeT2W:
    def __init__(self):
        self.calls = []

    def stream_step(self, session, plan):
        self.calls.append((session, plan))
        return f"pcm-{session}-{plan}"


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
    """老请求的后续块排队时，新请求的首块（chunk_index=0）插队。"""
    t2w = FakeT2W()
    w = Token2WavWorker(t2w)
    # 不 start worker——先塞满队列再启动，验证弹出顺序
    f_old = w.submit_nowait("old", "p", chunk_index=3)
    f_new = w.submit_nowait("new", "p", chunk_index=0)
    await w.start()
    try:
        await asyncio.gather(f_old, f_new)
        assert t2w.calls[0][0] == "new" and t2w.calls[1][0] == "old"
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
async def test_submit_cancelled_on_generator_exit():
    """消费方弃等（GeneratorExit 注入，即断连时 Starlette aclose 路径）
    → 排队的 future 被取消。手动驱动协程注入 GeneratorExit：task.cancel()
    会由 asyncio 自带机制取消被 await 的 future，锁不住本修复。"""
    w = Token2WavWorker(FakeT2W())
    fut_holder = {}
    orig = w.submit_nowait

    def spy(session, plan, chunk_index):
        fut = orig(session, plan, chunk_index)
        fut_holder["fut"] = fut
        return fut

    w.submit_nowait = spy

    coro = w.submit("s", "p", chunk_index=0)
    coro.send(None)                 # 推进到 await fut 挂起（worker 未启动）
    with pytest.raises(GeneratorExit):
        coro.throw(GeneratorExit)   # 断连注入
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
        assert await w.submit("good", "p", chunk_index=0) == "ok"  # worker 存活
    finally:
        await w.stop()
