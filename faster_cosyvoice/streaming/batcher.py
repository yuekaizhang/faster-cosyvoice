# faster_cosyvoice/streaming/batcher.py
"""token2wav GPU 调度器（spec §5.3/§6.3）。

单消费者 + 单线程 ThreadPoolExecutor：GPU 调用天然串行（v1），事件循环不被
阻塞；优先队列按 (chunk_index, 到达序) —— 新请求首块插队保 TTFA 公平
（triton priority 方案的 asyncio 版）。v2 只换消费循环为"弹出所有异 session
ready job → flashinfer packed 一次 forward"，submit 接口不变。
"""
import asyncio
import heapq
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional


class Token2WavWorker:
    def __init__(self, token2wav):
        self._t2w = token2wav
        self._heap: list = []
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
        self._gpu.shutdown(wait=False)

    def submit_nowait(self, session, plan, chunk_index: int) -> asyncio.Future:
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (chunk_index, next(self._seq),
                                    session, plan, fut))
        if self._wakeup:
            self._wakeup.set()
        return fut

    async def submit(self, session, plan, chunk_index: int) -> Any:
        return await self.submit_nowait(session, plan, chunk_index)

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            while not self._heap:
                if self._stopping:
                    return
                self._wakeup.clear()
                await self._wakeup.wait()
            _, _, session, plan, fut = heapq.heappop(self._heap)
            if fut.cancelled():        # 断连后丢弃该 job（spec §7）
                continue
            try:
                result = await loop.run_in_executor(
                    self._gpu, self._t2w.stream_step, session, plan)
            except Exception as e:  # noqa: BLE001 —— 只 fail 本 job，worker 存活
                if not fut.cancelled():
                    fut.set_exception(e)
                continue
            if not fut.cancelled():
                fut.set_result(result)
