# faster_cosyvoice/streaming/batcher.py
"""token2wav GPU 调度器（spec §5.3/§6.3）。

单消费者 + 单线程 ThreadPoolExecutor：GPU 调用天然串行，事件循环不被阻塞。
选择策略借鉴 Nari 的 deadline-aware policy：即将耗尽 playback credit 的既有
stream 先执行；否则首块优先；最后按最早播放 deadline。这样保留低 TTFA，同时
避免旧的 ``(chunk_index, arrival)`` 排序在开放环流量下形成 chunk wave、饿死
既有 stream。packed 模式以 anchor 为起点，继续选择不同 session 组成一批。

stop() 先排空已排队 job 再退出；stop 后 submit 会 RuntimeError。
"""
import asyncio
import itertools
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class _Job:
    seq: int
    chunk_index: int
    session: Any
    plan: Any
    future: asyncio.Future


class Token2WavWorker:
    def __init__(self, token2wav, mode: str = "serial", max_batch: int = 8,
                 deadline_reserve_s: float = 0.1, clock=time.monotonic):
        assert mode in ("serial", "packed"), f"未知 mode: {mode!r}"
        if not math.isfinite(deadline_reserve_s) or deadline_reserve_s < 0:
            raise ValueError("deadline_reserve_s 必须是有限非负数")
        self._t2w = token2wav
        self._mode = mode          # [M3] packed = 跨 session 批量执行循环
        self._max_batch = max_batch
        self._deadline_reserve_s = deadline_reserve_s
        self._clock = clock
        self._pending: list[_Job] = []
        self._seq = itertools.count()
        self._wakeup: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None
        self._gpu = ThreadPoolExecutor(max_workers=1,
                                       thread_name_prefix="t2w")
        self._stopping = False

    async def start(self) -> None:
        self._wakeup = asyncio.Event()
        self._stopping = False
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        if self._wakeup:
            self._wakeup.set()
        if self._task:
            await self._task
        for job in self._pending:  # stop 竞态期间溜进来的 job：取消勿悬挂
            job.future.cancel()
        self._pending.clear()
        self._gpu.shutdown(wait=False)

    def submit_nowait(self, session, plan, chunk_index: int) -> asyncio.Future:
        if self._stopping:
            raise RuntimeError("Token2WavWorker 已停止")
        fut = asyncio.get_running_loop().create_future()
        self._pending.append(_Job(next(self._seq), chunk_index,
                                  session, plan, fut))
        if self._wakeup:
            self._wakeup.set()
        return fut

    async def submit(self, session, plan, chunk_index: int) -> Any:
        fut = self.submit_nowait(session, plan, chunk_index)
        try:
            return await fut
        except BaseException:
            # 消费方弃等（断连 → GeneratorExit/CancelledError）：取消 job，
            # 避免孤儿 GPU 任务继续推进已丢弃的 session
            fut.cancel()
            raise

    @staticmethod
    def _deadline(job: _Job) -> Optional[float]:
        started = getattr(job.session, "playback_started_at_s", None)
        if started is None:
            return None
        emitted = getattr(job.session, "emitted_duration_s", 0.0)
        return float(started) + float(emitted)

    @classmethod
    def _is_startup(cls, job: _Job) -> bool:
        # StreamSession has the explicit route-boundary state.  The fallback
        # keeps lightweight/fake callers compatible with the public API.
        if hasattr(job.session, "playback_started_at_s"):
            return job.session.playback_started_at_s is None
        return job.chunk_index == 0

    def _pop_next(self, seen_sessions: Optional[set[int]] = None
                  ) -> Optional[_Job]:
        """Choose one ready job using Nari's urgency/startup ordering."""
        seen_sessions = seen_sessions or set()
        candidates = [
            (index, job) for index, job in enumerate(self._pending)
            if not job.future.cancelled() and id(job.session) not in seen_sessions
        ]
        # Cancelled jobs are dead work and must not keep stop()/idle logic busy.
        self._pending[:] = [job for job in self._pending
                            if not job.future.cancelled()]
        if not candidates:
            return None

        # Rebuild indices after cancelled entries were removed.
        candidates = [
            (index, job) for index, job in enumerate(self._pending)
            if id(job.session) not in seen_sessions
        ]
        if not candidates:
            return None
        now = self._clock()
        established = [(index, job, self._deadline(job))
                       for index, job in candidates
                       if not self._is_startup(job)]
        urgent = [(index, job, deadline)
                  for index, job, deadline in established
                  if deadline is not None
                  and now >= deadline - self._deadline_reserve_s]
        if urgent:
            index, selected, _ = min(
                urgent, key=lambda item: (item[2], item[1].seq))
        else:
            startup = [(index, job) for index, job in candidates
                       if self._is_startup(job)]
            if startup:
                index, selected = min(startup, key=lambda item: item[1].seq)
            elif established:
                # One-stage analogue of Nari's round-robin fallback: earliest
                # playback completion deadline, then admission order.
                index, selected, _ = min(
                    established,
                    key=lambda item: (
                        float("inf") if item[2] is None else item[2],
                        item[1].seq))
            else:
                index, selected = min(candidates,
                                      key=lambda item: item[1].seq)
        del self._pending[index]
        return selected

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                while not self._pending:
                    # Clear before checking _stopping so stop() cannot set the
                    # event in the small window between the check and clear.
                    # Otherwise the wakeup is lost and shutdown can hang.
                    self._wakeup.clear()
                    if self._stopping:
                        return
                    await self._wakeup.wait()
                if self._mode == "serial":
                    job = self._pop_next()
                    if job is None:
                        continue
                    session, plan, fut = job.session, job.plan, job.future
                    # 断连后丢弃该 job（spec §7）。GPU 中途取消会留下已推进的
                    # session 状态——安全：断连 session 直接丢弃，绝不复用。
                    if fut.cancelled():
                        continue
                    try:
                        result = await loop.run_in_executor(
                            self._gpu, self._t2w.stream_step, session, plan)
                    except Exception as e:  # noqa: BLE001 —— 只 fail 本 job，worker 存活
                        if not fut.cancelled():
                            fut.set_exception(e)
                        await asyncio.sleep(0)
                        continue
                    if not fut.cancelled():
                        fut.set_result(result)
                    # Let the request coroutine consume the completed chunk
                    # before the worker goes idle or launches another chunk.
                    await asyncio.sleep(0)
                else:
                    await self._run_packed_batch(loop)
        finally:
            # 消费者退出（含被 cancel/异常死亡）：残留 job 取消勿悬挂
            for job in self._pending:
                job.future.cancel()
            self._pending.clear()

    async def _run_packed_batch(self, loop) -> None:
        """[M3] packed 执行循环单轮：贪心组批 → 一次 stream_step_batched。

        组批约束：同 session 后续 chunk 不进同批（mel_cache 有前序 chunk
        依赖，必须串行）——留在 pending queue 下一轮再取。cancelled future
        组批时丢弃。
        """
        batch: list = []      # (session, plan, fut)，anchor/deadline 顺序
        seen_sessions: set = set()
        while self._pending and len(batch) < self._max_batch:
            job = self._pop_next(seen_sessions)
            if job is None:
                break
            session, plan, fut = job.session, job.plan, job.future
            # id() 去重：StreamSession 是 eq=True 的 dataclass（不可哈希）。
            sid = id(session)
            seen_sessions.add(sid)
            batch.append((session, plan, fut))
        if not batch:
            return
        sessions = [b[0] for b in batch]
        plans = [b[1] for b in batch]
        try:
            results = await loop.run_in_executor(
                self._gpu, self._t2w.stream_step_batched, sessions, plans)
        except Exception as e:  # noqa: BLE001 —— worker 存活
            # 整批失败即全批 fail：v1 可接受语义。安全性依据：flow 阶段原子；
            # _finish_chunk 阶段部分 session 可能已推进，但全批 future 均拿到
            # 异常 → 各请求失败 → session 被丢弃且（flush 串行 submit）不会再
            # 收到后续 job，故不会错位。
            for _, _, fut in batch:
                if not fut.cancelled():
                    fut.set_exception(e)
            await asyncio.sleep(0)
            return
        if len(results) != len(batch):
            error = RuntimeError(
                "stream_step_batched 返回数量不匹配: "
                f"expected {len(batch)}, got {len(results)}")
            for _, _, fut in batch:
                if not fut.cancelled():
                    fut.set_exception(error)
            await asyncio.sleep(0)
            return
        for (_, _, fut), result in zip(batch, results, strict=True):
            if not fut.cancelled():
                fut.set_result(result)
        await asyncio.sleep(0)
