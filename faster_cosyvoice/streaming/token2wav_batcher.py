"""Async worker that executes streaming token-to-wave chunks on one GPU thread.

Queue ordering lives in
:mod:`faster_cosyvoice.streaming.token2wav_scheduler`.  This module handles
lifecycle, cancellation, serial execution, and packed batching.
"""
import asyncio
import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from faster_cosyvoice.streaming.token2wav_scheduler import (
    PendingChunk,
    create_scheduling_policy,
)


class Token2WavWorker:
    def __init__(self, token2wav, mode: str = "serial", max_batch: int = 8,
                 deadline_reserve_s: float = 0.1, clock=time.monotonic,
                 scheduler_mode: str = "deadline"):
        if mode not in ("serial", "packed"):
            raise ValueError(f"mode must be 'serial' or 'packed', got {mode!r}")
        self._token2wav = token2wav
        self._mode = mode
        self._max_batch = max_batch
        self._scheduler = create_scheduling_policy(
            scheduler_mode,
            reserve_s=deadline_reserve_s,
            clock=clock,
        )
        self._pending: list[PendingChunk] = []
        self._seq = itertools.count()
        self._wakeup: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None
        self._gpu = ThreadPoolExecutor(max_workers=1,
                                       thread_name_prefix="token2wav")
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
        # Cancel jobs admitted during the small race with stop().
        for job in self._pending:
            job.future.cancel()
        self._pending.clear()
        self._gpu.shutdown(wait=False)

    def submit_nowait(self, session, plan, chunk_index: int) -> asyncio.Future:
        if self._stopping:
            raise RuntimeError("Token2WavWorker has stopped")
        fut = asyncio.get_running_loop().create_future()
        self._pending.append(
            PendingChunk(next(self._seq), chunk_index, session, plan, fut)
        )
        if self._wakeup:
            self._wakeup.set()
        return fut

    async def submit(self, session, plan, chunk_index: int) -> Any:
        fut = self.submit_nowait(session, plan, chunk_index)
        try:
            return await fut
        except BaseException:
            # A disconnected consumer must not leave orphaned GPU work that
            # mutates a session nobody will consume again.
            fut.cancel()
            raise

    def _pop_next_job(
        self, seen_sessions: Optional[set[int]] = None
    ) -> Optional[PendingChunk]:
        """Remove and return the next job selected by the scheduling policy."""
        seen_sessions = seen_sessions or set()
        self._pending[:] = [job for job in self._pending if not job.future.cancelled()]
        candidates = [
            (index, job)
            for index, job in enumerate(self._pending)
            if id(job.session) not in seen_sessions
        ]
        if not candidates:
            return None
        index, selected = self._scheduler.select(candidates)
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
                    job = self._pop_next_job()
                    if job is None:
                        continue
                    session, plan, fut = job.session, job.plan, job.future
                    # A cancelled request discards its entire session, so a
                    # partially advanced session is never reused.
                    if fut.cancelled():
                        continue
                    try:
                        result = await loop.run_in_executor(
                            self._gpu, self._token2wav.stream_step, session, plan)
                    except Exception as e:  # noqa: BLE001 - fail one job, keep worker alive
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
                    await self._run_packed_once(loop)
        finally:
            # Never leave callers waiting if the consumer exits unexpectedly.
            for job in self._pending:
                job.future.cancel()
            self._pending.clear()

    async def _run_packed_once(self, loop) -> None:
        """Greedily form and execute one cross-session packed batch.

        At most one chunk from each session may enter a batch because the next
        chunk depends on the previous chunk's Mel cache.  Additional chunks
        from that session stay queued for the next iteration.
        """
        batch: list = []  # (session, plan, future), in scheduler order
        seen_sessions: set = set()
        while self._pending and len(batch) < self._max_batch:
            job = self._pop_next_job(seen_sessions)
            if job is None:
                break
            session, plan, fut = job.session, job.plan, job.future
            # StreamSession is an unhashable dataclass, so identity is the key.
            sid = id(session)
            seen_sessions.add(sid)
            batch.append((session, plan, fut))
        if not batch:
            return
        sessions = [b[0] for b in batch]
        plans = [b[1] for b in batch]
        try:
            results = await loop.run_in_executor(
                self._gpu, self._token2wav.stream_step_batched, sessions, plans)
        except Exception as e:  # noqa: BLE001 - fail batch, keep worker alive
            # Every request in a failed batch discards its session.  This is
            # safe even if vocoding advanced some sessions before the error.
            for _, _, fut in batch:
                if not fut.cancelled():
                    fut.set_exception(e)
            await asyncio.sleep(0)
            return
        if len(results) != len(batch):
            error = RuntimeError(
                "stream_step_batched returned the wrong result count: "
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
