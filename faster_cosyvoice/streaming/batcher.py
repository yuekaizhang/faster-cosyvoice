# faster_cosyvoice/streaming/batcher.py
"""token2wav GPU 调度器（spec §5.3/§6.3）。

单消费者 + 单线程 ThreadPoolExecutor：GPU 调用天然串行（v1），事件循环不被
阻塞；优先队列按 (chunk_index, 到达序) —— 新请求首块插队保 TTFA 公平
（triton priority 方案的 asyncio 版）。v2 只换消费循环为"弹出所有异 session
ready job → flashinfer packed 一次 forward"，submit 接口不变。

stop() 先排空已排队 job 再退出；stop 后 submit 会 RuntimeError。
"""
import asyncio
import heapq
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional


class Token2WavWorker:
    def __init__(self, token2wav, mode: str = "serial", max_batch: int = 8):
        assert mode in ("serial", "packed"), f"未知 mode: {mode!r}"
        self._t2w = token2wav
        self._mode = mode          # [M3] packed = 跨 session 批量执行循环
        self._max_batch = max_batch
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
        for entry in self._heap:   # stop 竞态期间溜进来的 job：取消勿悬挂
            entry[-1].cancel()
        self._heap.clear()
        self._gpu.shutdown(wait=False)

    def submit_nowait(self, session, plan, chunk_index: int) -> asyncio.Future:
        if self._stopping:
            raise RuntimeError("Token2WavWorker 已停止")
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (chunk_index, next(self._seq),
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

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                while not self._heap:
                    if self._stopping:
                        return
                    self._wakeup.clear()
                    await self._wakeup.wait()
                if self._mode == "serial":
                    _, _, session, plan, fut = heapq.heappop(self._heap)
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
                        continue
                    if not fut.cancelled():
                        fut.set_result(result)
                else:
                    await self._run_packed_batch(loop)
        finally:
            # 消费者退出（含被 cancel/异常死亡）：残留 job 取消勿悬挂
            for entry in self._heap:
                entry[-1].cancel()
            self._heap.clear()

    async def _run_packed_batch(self, loop) -> None:
        """[M3] packed 执行循环单轮：贪心组批 → 一次 stream_step_batched。

        组批约束：同 session 后续 chunk 不进同批（mel_cache 有前序 chunk
        依赖，必须串行）——留在堆里下一轮再取。cancelled future 组批时丢弃。
        """
        batch: list = []      # (session, plan, fut)，按堆序 = 结果顺序
        seen_sessions: set = set()
        deferred: list = []   # 同 session 撞批的 job：原 entry 回堆
        while self._heap and len(batch) < self._max_batch:
            entry = heapq.heappop(self._heap)
            _, _, session, plan, fut = entry
            if fut.cancelled():   # 断连丢弃（同 serial 语义）
                continue
            # id() 去重：StreamSession 是 eq=True 的 dataclass（不可哈希），
            # 且 batch/deferred 全程持强引用，id 不会因 GC 复用——勿"简化"成 in。
            sid = id(session)
            if sid in seen_sessions:
                deferred.append(entry)
                continue
            seen_sessions.add(sid)
            batch.append((session, plan, fut))
        for entry in deferred:
            heapq.heappush(self._heap, entry)
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
            return
        for (_, _, fut), result in zip(batch, results):
            if not fut.cancelled():
                fut.set_result(result)
