# M2 Streaming Server（faster-cosyvoice）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** spec M2 里程碑——OpenAI `/v1/audio/speech` 流式 server：AsyncLLM(DSpark) token 流 → ChunkPlanner → torch 流式 token2wav（chunk-causal + 确定性重算）→ 增量 WAV/PCM；voice 注册、并发交错、TTFA 与参考实现同量级。

**Architecture:** 单进程 FastAPI。每请求一个 pipeline 协程：AsyncLLM DELTA 流 → codec 提取 → ChunkPlanner → Token2WavWorker（优先队列 + 单线程 GPU executor，v1 串行、v2 换 packed batch）→ audio_encode 增量输出。HiFT 已确认无跨调用状态（左上下文全部来自传入 mel），单实例多 session 交错安全；session 状态 = (tokens, mel_cache, speech_offset)。

**Tech Stack:** 同 M1 环境 + fastapi/uvicorn/httpx（新增 pip extras）。

**执行环境约定（同 M1）：** pytest 与 GPU 步骤在容器内；`source venv/bin/activate && export PYTHONPATH=$PWD/third_party/spec-vllm:$PYTHONPATH && export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface`。提交 `git commit -s --no-verify`。分支：`m2-streaming`（从 main 切）。

**调研确认的关键事实（计划代码依据，来源 spec-vllm/ vllm-omni/ duplex 源码，file:line 见调研记录）：**
- AsyncLLM：`AsyncLLM.from_engine_args(AsyncEngineArgs(**同 LLM kwargs, enable_log_requests=False))`；`generate(prompt, sp, request_id)` 异步生成器，`output_kind=DELTA` 时 `outputs[0].token_ids` 是**增量**（consumer 滞后时合并 1..N 个，不重不漏）；`detokenize=False` 下 `stop_token_ids` 可用（stop 字符串会 raise）；`min_tokens` 到达前抑制一切 stop；**取消消费协程/关闭生成器即自动 abort**；engine 构造在 lifespan（阻塞加载），teardown 调 `engine.shutdown()`；spec decode 与 async 路径无任何不兼容。
- ChunkPlanner 语义（vllm-omni `talker2code2wav_async_chunk`）：hop 初始 15（yaml 值），首块加 pad=ceil(prompt_len/15)×15−prompt_len；就绪门 `available ≥ this_hop + lookahead(3)`；**lookahead 只重发不消费**（emitted 只加 this_hop）；hop 逐块 ×2 封顶 4×15=60；LLM 结束后余量（哪怕 1 个 token）一次 finalize，prefix=全部、不要求 lookahead。
- 每 chunk t2w（duplex `forward_stream`）：flow.inference(累计 tokens[:prefix_len], streaming=True, finalize=is_final) → mel 去 prompt 后按 `token_offset×2` 切新段 → 拼进 session mel_cache → hift 对**全量** mel_cache 重跑（finalize 传递）→ 按 speech_offset 切新音频。CausalConditionalCFM 内置固定 `rand_noise`（set_all_random_seed(0)）→ 重算确定性天然满足（spec D3）。
- WAV 流式头：RIFF/data 长度填 0xFFFFFFFF 的 44 字节头发一次，后续全是 PCM_16 chunk；`struct.pack("<4sI4s4sIHHIIHH4sI", ...)`。
- 动态 token 上下限（vllm-omni）：`min_tokens=max(1, 2×text_token_len)`，`max_tokens=min(2048, 20×text_token_len)`。

---

### Task 1: `streaming/chunker.py` — ChunkPlanner

**Files:**
- Create: `faster_cosyvoice/streaming/__init__.py`、`faster_cosyvoice/streaming/chunker.py`
- Test: `tests/test_chunker.py`

- [ ] **Step 1: 失败测试（黄金序列）**

```python
# tests/test_chunker.py
from faster_cosyvoice.streaming.chunker import ChunkPlanner


def test_no_pad_sequence():
    """prompt 75（15 的倍数 → pad=0）：首块需 15+3=18 可用。"""
    p = ChunkPlanner(prompt_token_len=75)
    assert p.next_chunk(17, finished=False) is None
    c1 = p.next_chunk(18, finished=False)
    assert (c1.prefix_len, c1.token_offset, c1.finalize) == (18, 0, False)
    # hop 翻倍→30：下一块需 15+30+3=48 可用
    assert p.next_chunk(47, finished=False) is None
    c2 = p.next_chunk(48, finished=False)
    assert (c2.prefix_len, c2.token_offset, c2.finalize) == (48, 15, False)
    # hop→60（封顶）：需 45+60+3=108
    c3 = p.next_chunk(108, finished=False)
    assert (c3.prefix_len, c3.token_offset, c3.finalize) == (108, 45, False)
    # hop 保持 60
    c4 = p.next_chunk(168, finished=False)
    assert (c4.prefix_len, c4.token_offset, c4.finalize) == (168, 105, False)


def test_pad_applies_to_first_chunk_only():
    """prompt 71 → pad=4：首块需 15+4+3=22。"""
    p = ChunkPlanner(prompt_token_len=71)
    assert p.next_chunk(21, finished=False) is None
    c1 = p.next_chunk(22, finished=False)
    assert (c1.prefix_len, c1.token_offset, c1.finalize) == (22, 0, False)
    # 消费 19（含 pad），第二块 hop=30 不再加 pad：需 19+30+3=52
    c2 = p.next_chunk(52, finished=False)
    assert (c2.prefix_len, c2.token_offset, c2.finalize) == (52, 19, False)


def test_finalize_flushes_remainder():
    p = ChunkPlanner(prompt_token_len=75)
    p.next_chunk(18, finished=False)          # 消费 15
    c = p.next_chunk(20, finished=True)       # 余 5 个，不足 hop 也 flush
    assert (c.prefix_len, c.token_offset, c.finalize) == (20, 15, True)
    assert p.next_chunk(20, finished=True) is None  # 无余量


def test_finished_with_nothing_left():
    p = ChunkPlanner(prompt_token_len=0)
    assert p.next_chunk(0, finished=True) is None


def test_zero_prompt_no_pad():
    p = ChunkPlanner(prompt_token_len=0)
    c = p.next_chunk(18, finished=False)
    assert (c.prefix_len, c.token_offset, c.finalize) == (18, 0, False)
```

- [ ] **Step 2:** `pytest tests/test_chunker.py -v` → FAIL（模块缺失）

- [ ] **Step 3: 实现**

```python
# faster_cosyvoice/streaming/chunker.py
"""token→chunk 纯数学状态机（vllm-omni talker2code2wav_async_chunk 语义，spec §5.3）。

lookahead 只重发不消费；pad 仅首块；hop 逐块 ×2 封顶 4×chunk_size；
LLM 结束后余量一次 finalize（不要求 lookahead，哪怕不足一个 hop）。
常数 15/3/×2/60 出处：vllm-omni deploy/cosyvoice3.yaml codec_chunk_frames=15。
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkPlan:
    prefix_len: int    # flow 输入 = tokens[:prefix_len]（非 final 含 lookahead）
    token_offset: int  # 本块之前已消费 token 数 = mel 切片起点（×token_mel_ratio）
    finalize: bool


class ChunkPlanner:
    def __init__(self, prompt_token_len: int, chunk_size: int = 15,
                 pre_lookahead: int = 3, scale: int = 2,
                 max_hop: Optional[int] = None):
        self.chunk_size = chunk_size
        self.lookahead = pre_lookahead
        self.scale = scale
        self.max_hop = max_hop if max_hop is not None else 4 * chunk_size
        self.hop = chunk_size
        self.pad = ((math.ceil(prompt_token_len / chunk_size) * chunk_size
                     - prompt_token_len) if prompt_token_len > 0 else 0)
        self.emitted = 0  # 已消费 token 数

    def next_chunk(self, available_total: int,
                   finished: bool) -> Optional[ChunkPlan]:
        available = available_total - self.emitted
        this_hop = self.hop + (self.pad if self.emitted == 0 else 0)
        if not finished:
            if available < this_hop + self.lookahead:
                return None
            plan = ChunkPlan(self.emitted + this_hop + self.lookahead,
                             self.emitted, False)
            self.emitted += this_hop
            self.hop = min(self.max_hop,
                           max(self.chunk_size, self.hop * self.scale))
            return plan
        if available <= 0:
            return None
        plan = ChunkPlan(available_total, self.emitted, True)
        self.emitted = available_total
        return plan
```

- [ ] **Step 4:** `pytest tests/test_chunker.py -v` → 5 PASS；全套 `pytest` → 27 passed, 2 deselected
- [ ] **Step 5: Commit** — `git add faster_cosyvoice/streaming tests/test_chunker.py && git commit -s --no-verify -m "feat: ChunkPlanner streaming state machine"`

---

### Task 2: `streaming/session.py` + `token2wav.stream_step`（torch 流式路径）

**Files:**
- Create: `faster_cosyvoice/streaming/session.py`
- Modify: `faster_cosyvoice/token2wav/token2wav.py`（追加 stream_step 方法）
- Test: `tests/test_session.py`（CPU）+ GPU 冒烟

- [ ] **Step 1: 失败测试**

```python
# tests/test_session.py
from faster_cosyvoice.streaming.session import StreamSession
from faster_cosyvoice.streaming.chunker import ChunkPlanner


def test_session_defaults():
    s = StreamSession(cond=object(), planner=ChunkPlanner(0))
    assert s.tokens == [] and s.mel_cache is None
    assert s.speech_offset == 0 and s.chunk_index == 0
```

- [ ] **Step 2:** `pytest tests/test_session.py -v` → FAIL

- [ ] **Step 3: 实现 session.py**

```python
# faster_cosyvoice/streaming/session.py
"""每请求流式状态（spec §5.2/§6.2）。token2wav 模块本身无状态，
多 session 交错依赖本对象承载 (mel_cache, speech_offset)。"""
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from faster_cosyvoice.streaming.chunker import ChunkPlanner


@dataclass
class StreamSession:
    cond: Any                     # RefCondition
    planner: ChunkPlanner
    tokens: List[int] = field(default_factory=list)
    mel_cache: Optional[torch.Tensor] = None  # (1,80,T) 累计生成 mel（不含 prompt）
    speech_offset: int = 0
    chunk_index: int = 0
```

- [ ] **Step 4: 给 `CosyVoice3Token2Wav` 追加 stream_step（放在 `_flow_single` 之后）**

```python
    @torch.inference_mode()
    def stream_step(self, session, plan) -> torch.Tensor:
        """v1 torch 流式路径（spec D3/§5.2；语义同 duplex forward_stream 单步）：
        flow 全前缀重算(streaming=True) → mel 按 token_offset×2 切新段 → 拼
        session.mel_cache → hift 对全量 mel 重跑 → 按 speech_offset 切新音频。
        HiFT 无跨调用状态，共享实例可多 session 交错；重算确定性由
        CausalConditionalCFM 的固定 rand_noise 保证。返回 (1, N) cpu fp32。"""
        assert self.estimator_mode == "torch", \
            "flashinfer estimator 仅支持 offline（流式 mask 是 M3）"
        cond = session.cond
        token = torch.tensor([session.tokens[:plan.prefix_len]],
                             device=self.device)
        prompt_token = torch.tensor([cond.prompt_tokens_flow],
                                    device=self.device)
        prompt_feat = cond.prompt_feat.to(self.device)
        embedding = cond.spk_embedding.unsqueeze(0).to(self.device)
        mel, _ = self.flow.inference(
            token=token,
            token_len=torch.tensor([token.shape[1]], device=self.device),
            prompt_token=prompt_token,
            prompt_token_len=torch.tensor([prompt_token.shape[1]],
                                          device=self.device),
            prompt_feat=prompt_feat,
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]],
                                         device=self.device),
            embedding=embedding,
            streaming=True, finalize=plan.finalize)
        mel = mel[:, :, plan.token_offset * 2:]
        if session.mel_cache is not None:
            mel = torch.cat([session.mel_cache, mel], dim=2)
        session.mel_cache = mel
        speech, _ = self.hift.inference(speech_feat=mel,
                                        finalize=plan.finalize)
        new = speech[:, session.speech_offset:]
        session.speech_offset += new.shape[1]
        session.chunk_index += 1
        return new.cpu()
```

- [ ] **Step 5:** `pytest tests/test_session.py -v` → 1 PASS；全套 → 28 passed, 2 deselected

- [ ] **Step 6: GPU 冒烟——流式 vs 离线一致性（spec §8 测试 2 的模块级前哨）**

```bash
python - <<'EOF'
import torch
from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession

d = ensure_token2wav_assets("models/Fun-CosyVoice3-0.5B-2512")
m = CosyVoice3Token2Wav(d, estimator_mode="torch")
fe = RefAudioFrontend(f"{d}/campplus.onnx")
cond = fe.process_batch([torch.randn(16000 * 3) * 0.1], [16000])[0]
tokens = [(i * 37) % 6561 for i in range(120)]  # 确定性伪 token，~4.8s

# 流式：逐 chunk 喂
s = StreamSession(cond=cond, planner=ChunkPlanner(len(cond.prompt_tokens_flow)))
s.tokens = tokens
chunks, n_seen = [], 0
while True:
    plan = s.planner.next_chunk(len(s.tokens), finished=(n_seen >= 1))
    if plan is None:
        if n_seen >= 1: break
        n_seen += 1; continue
    chunks.append(m.stream_step(s, plan))
stream_wav = torch.cat(chunks, dim=1)
print("stream chunks:", len(chunks), "samples:", stream_wav.shape[1])
assert stream_wav.shape[1] > 24000 * 3, stream_wav.shape
# 采样数连续性：chunk 拼接即 speech_offset 累计，无缝由构造保证
assert stream_wav.shape[1] == s.speech_offset
print("stream_step smoke OK")
EOF
```
预期：多个 chunk、总样本数 ≈ 120/25×24000≈115200、`stream_step smoke OK`。（与 offline 输出的数值级对比不做——offline 是全注意力图，二者本就不同；质量仲裁在 Task 8 的 ASR 门。）

- [ ] **Step 7: Commit** — `git add faster_cosyvoice/streaming/session.py faster_cosyvoice/token2wav/token2wav.py tests/test_session.py && git commit -s --no-verify -m "feat: StreamSession + torch streaming stream_step"`

---

### Task 3: `llm/engine.py` 异步扩展

**Files:**
- Modify: `faster_cosyvoice/llm/engine.py`（追加三个函数）
- Test: `tests/test_engine.py`（追加）

- [ ] **Step 1: 追加失败测试**

```python
# 追加到 tests/test_engine.py
from faster_cosyvoice.llm.engine import make_stream_sampling_params


class FakeCodec:
    eos_token_id = 158486


def test_stream_sampling_params_dynamic_bounds():
    sp = make_stream_sampling_params(LLMConfig(), FakeCodec(),
                                     text_token_len=10, seed=7)
    assert sp.min_tokens == 20                      # 2×text
    assert sp.max_tokens == 200                     # min(2048, 20×text)
    assert sp.stop_token_ids == [158486]
    assert sp.detokenize is False
    assert sp.seed == 7


def test_stream_sampling_params_caps_at_2048():
    sp = make_stream_sampling_params(LLMConfig(), FakeCodec(),
                                     text_token_len=1000, seed=0)
    assert sp.max_tokens == 2048
```

- [ ] **Step 2:** `pytest tests/test_engine.py -v` → 新增 2 个 FAIL

- [ ] **Step 3: 实现（追加到 engine.py 末尾）**

```python
def create_async_llm(cfg: LLMConfig):
    """server 用异步引擎（spec §5.1）。构造即拉起 engine-core 子进程
    （阻塞加载）——放 FastAPI lifespan；teardown 调 engine.shutdown()。"""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    return AsyncLLM.from_engine_args(
        AsyncEngineArgs(**build_llm_kwargs(cfg), enable_log_requests=False))


def make_stream_sampling_params(cfg: LLMConfig, codec, text_token_len: int,
                                seed: int):
    """流式采样参数：DELTA 增量、不 detokenize（stop 字符串会 raise，
    用 stop_token_ids）、动态 min/max（vllm-omni 同款 2×/20× 比例）。
    min_tokens 到达前 vLLM 会抑制一切 stop（含 eos）。"""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    return SamplingParams(
        temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
        repetition_penalty=cfg.repetition_penalty,
        min_tokens=max(1, 2 * text_token_len),
        max_tokens=min(cfg.max_tokens, 20 * text_token_len),
        stop_token_ids=[codec.eos_token_id],
        seed=seed, detokenize=False,
        output_kind=RequestOutputKind.DELTA)


async def stream_token_ids(engine, prompt: str, sp, request_id: str):
    """AsyncLLM DELTA 流 → 逐次 yield (delta_token_ids, finished)。
    消费方取消（客户端断连）时生成器关闭即自动 abort 引擎侧请求。"""
    async for out in engine.generate(prompt, sp, request_id):
        yield list(out.outputs[0].token_ids), out.finished
        if out.finished:
            break
```

- [ ] **Step 4:** `pytest tests/test_engine.py -v` → 5 PASS；全套 → 30 passed, 2 deselected
- [ ] **Step 5: Commit** — `git add faster_cosyvoice/llm/engine.py tests/test_engine.py && git commit -s --no-verify -m "feat: AsyncLLM factory + streaming sampling params + token stream helper"`

---

### Task 4: `streaming/batcher.py` — Token2WavWorker

**Files:**
- Create: `faster_cosyvoice/streaming/batcher.py`
- Test: `tests/test_batcher.py`

- [ ] **Step 1: 失败测试（fake executor，全 CPU）**

```python
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
```

（需要 `pytest-asyncio`：`uv pip install --python venv/bin/python pytest-asyncio` 并加进 requirements.txt extras，pytest.ini 加 `asyncio_mode = auto`——若装的版本默认 strict，用 auto 免去逐个装饰器。）

- [ ] **Step 2:** `pytest tests/test_batcher.py -v` → FAIL

- [ ] **Step 3: 实现**

```python
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
        fut = asyncio.get_event_loop().create_future()
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
```

- [ ] **Step 4:** `pytest tests/test_batcher.py -v` → 3 PASS；全套 → 33 passed, 2 deselected
- [ ] **Step 5: Commit** — `git add faster_cosyvoice/streaming/batcher.py tests/test_batcher.py requirements.txt pytest.ini && git commit -s --no-verify -m "feat: Token2WavWorker priority scheduler (serial v1 executor)"`

---

### Task 5: `server/audio_encode.py`

**Files:**
- Create: `faster_cosyvoice/server/__init__.py`、`faster_cosyvoice/server/audio_encode.py`
- Test: `tests/test_audio_encode.py`

- [ ] **Step 1: 失败测试**

```python
# tests/test_audio_encode.py
import struct

import torch

from faster_cosyvoice.server.audio_encode import pcm16_bytes, wav_stream_header


def test_wav_header_44_bytes_unknown_length():
    h = wav_stream_header(24000)
    assert len(h) == 44
    assert h[:4] == b"RIFF" and h[8:12] == b"WAVE"
    assert struct.unpack("<I", h[4:8])[0] == 0xFFFFFFFF     # 未知长度
    assert struct.unpack("<I", h[24:28])[0] == 24000        # sample rate
    assert struct.unpack("<I", h[40:44])[0] == 0xFFFFFFFF   # data 长度


def test_pcm16_clip_and_dtype():
    wav = torch.tensor([[0.0, 1.0, -1.0, 2.0]])   # 2.0 会被 clip
    b = pcm16_bytes(wav)
    vals = struct.unpack("<4h", b)
    assert vals[0] == 0 and vals[1] == 32767
    assert vals[2] == -32767 and vals[3] == 32767
```

- [ ] **Step 2:** FAIL → **Step 3: 实现**

```python
# faster_cosyvoice/server/audio_encode.py
"""流式音频编码（spec §5.5）：未知长度 WAV 头一次 + PCM_16 增量。"""
import struct

import numpy as np
import torch


def wav_stream_header(sample_rate: int, num_channels: int = 1,
                      bits_per_sample: int = 16) -> bytes:
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    unknown = 0xFFFFFFFF  # 流式：RIFF/data 长度未知（vllm-omni 同款）
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", unknown, b"WAVE",
        b"fmt ", 16, 1,
        num_channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b"data", unknown)


def pcm16_bytes(wav: torch.Tensor) -> bytes:
    """(1, N) 或 (N,) fp32 [-1,1] → little-endian int16 bytes。"""
    x = wav.detach().cpu().float().numpy().reshape(-1)
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
```

- [ ] **Step 4:** 2 PASS；全套 → 35 passed, 2 deselected
- [ ] **Step 5: Commit** — `git add faster_cosyvoice/server tests/test_audio_encode.py && git commit -s --no-verify -m "feat: streaming WAV header + PCM16 encoding"`

---

### Task 6: `config.py` ServerConfig + `server/protocol.py`（请求模型 + ref 解析）

**Files:**
- Modify: `faster_cosyvoice/config.py`（追加 ServerConfig）
- Create: `faster_cosyvoice/server/protocol.py`
- Test: `tests/test_protocol.py`

- [ ] **Step 1: 失败测试**

```python
# tests/test_protocol.py
import base64
import io

import pytest
import soundfile as sf
import numpy as np

from faster_cosyvoice.config import ServerConfig
from faster_cosyvoice.server.protocol import (SpeechRequest, VoiceRequest,
                                              decode_ref_audio)


def _wav_data_url(seconds=1.0, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), dtype=np.float32), sr,
             format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def test_server_config_defaults():
    c = ServerConfig()
    assert c.gpu_memory_utilization == 0.5   # spec D6 server 档
    assert c.max_ref_seconds == 30
    assert c.port == 8000


def test_speech_request_validation():
    r = SpeechRequest(input="你好", ref_audio=_wav_data_url(), ref_text="嗯")
    assert r.response_format == "wav" and r.stream is False
    with pytest.raises(ValueError):
        SpeechRequest(input="", ref_audio="x", ref_text="y")  # 空 input
    with pytest.raises(ValueError):
        SpeechRequest(input="你好", response_format="mp3",     # 非 wav/pcm
                      ref_audio="x", ref_text="y")


def test_decode_data_url_and_truncate():
    wav, sr = decode_ref_audio(_wav_data_url(seconds=2.0), max_seconds=1.0)
    assert sr == 16000 and abs(len(wav) - 16000) <= 1  # 截断到 1s


def test_voice_or_ref_required():
    with pytest.raises(ValueError):
        SpeechRequest(input="你好")  # 既无 voice 也无 ref_audio+ref_text
```

- [ ] **Step 2:** FAIL → **Step 3: 实现**

`config.py` 追加：

```python
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    gpu_memory_utilization: float = 0.5   # server 档（spec D6；与 token2wav 同卡）
    max_ref_seconds: float = 30.0         # ref 音频超长截断并告警（spec §7）
    request_timeout_s: float = 300.0
    voice_cache_size: int = 256
```

```python
# faster_cosyvoice/server/protocol.py
"""OpenAI /v1/audio/speech 请求模型 + ref 音频解析（spec §5.5/§7）。

ref_audio 支持：data:...;base64 URI、http(s) URL、本地路径。
"""
import base64
import io
import logging
from typing import Optional

import numpy as np
import soundfile as sf
from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)


class SpeechRequest(BaseModel):
    input: str
    model: str = "faster-cosyvoice"
    voice: Optional[str] = None            # 已注册音色名
    ref_audio: Optional[str] = None        # data:/http(s)/path
    ref_text: Optional[str] = None
    response_format: str = "wav"           # wav | pcm
    stream: bool = False
    seed: int = 42

    @field_validator("input")
    @classmethod
    def _non_empty(cls, v):
        if not v.strip():
            raise ValueError("input 不能为空")
        return v

    @field_validator("response_format")
    @classmethod
    def _fmt(cls, v):
        if v not in ("wav", "pcm"):
            raise ValueError("response_format 仅支持 wav|pcm（v1）")
        return v

    @model_validator(mode="after")
    def _voice_or_ref(self):
        if self.voice is None and not (self.ref_audio and self.ref_text):
            raise ValueError("需要 voice 或 (ref_audio + ref_text)")
        return self


class VoiceRequest(BaseModel):
    name: str
    ref_audio: str
    ref_text: str

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        if not v.strip():
            raise ValueError("name 不能为空")
        return v


def decode_ref_audio(ref: str, max_seconds: float = 30.0):
    """→ (1-D float32 numpy, sr)。超长截断并 warning（spec §7）。"""
    if ref.startswith("data:"):
        b64 = ref.split(",", 1)[1]
        data = io.BytesIO(base64.b64decode(b64))
    elif ref.startswith(("http://", "https://")):
        import httpx
        resp = httpx.get(ref, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        data = io.BytesIO(resp.content)
    else:
        data = ref  # 本地路径
    wav, sr = sf.read(data, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    limit = int(max_seconds * sr)
    if len(wav) > limit:
        logger.warning("ref 音频 %.1fs 超过 %.0fs，截断", len(wav) / sr,
                       max_seconds)
        wav = wav[:limit]
    return wav, sr
```

（`fastapi`/`uvicorn`/`httpx`/`pydantic` 加进 requirements.txt extras 并 `uv pip install --python venv/bin/python -r requirements.txt`；pydantic 可能已由层内提供，确认版本 ≥2。）

- [ ] **Step 4:** `pytest tests/test_protocol.py -v` → 5 PASS；全套 → 40 passed, 2 deselected
- [ ] **Step 5: Commit** — `git add faster_cosyvoice/config.py faster_cosyvoice/server/protocol.py tests/test_protocol.py requirements.txt && git commit -s --no-verify -m "feat: ServerConfig + speech/voice request models + ref audio decode"`

---

### Task 7: `server/app.py` + `server/openai_speech.py`（管线接线）

**Files:**
- Create: `faster_cosyvoice/server/openai_speech.py`、`faster_cosyvoice/server/app.py`
- Test: 语法 + GPU 单请求冒烟（完整 e2e 在 Task 8）

- [ ] **Step 1: 实现 `openai_speech.py`**

```python
# faster_cosyvoice/server/openai_speech.py
"""每请求流水线（spec §6.2）：resolve voice → prompt → AsyncLLM DELTA 流
→ ChunkPlanner → batcher → PCM 增量。"""
import json
import logging
import time
import uuid
from typing import AsyncGenerator

import torch

from faster_cosyvoice.llm.engine import (make_stream_sampling_params,
                                         stream_token_ids)
from faster_cosyvoice.llm.prompt import build_prompt
from faster_cosyvoice.server.audio_encode import pcm16_bytes, wav_stream_header
from faster_cosyvoice.server.protocol import SpeechRequest, decode_ref_audio
from faster_cosyvoice.streaming.chunker import ChunkPlanner
from faster_cosyvoice.streaming.session import StreamSession

logger = logging.getLogger(__name__)
SAMPLE_RATE = 24000


class NoSpeechTokens(RuntimeError):
    """LLM 结束且 0 个有效 speech token（spec §7 → 500）。"""


def resolve_condition(state, req: SpeechRequest):
    """→ (RefCondition, ref_text)。voice 命中注册表；否则解析 ref_audio。"""
    if req.voice is not None:
        entry = state.voices.get(req.voice)
        if entry is None:
            raise KeyError(f"voice 未注册: {req.voice}")
        return entry
    wav, sr = decode_ref_audio(req.ref_audio,
                               state.server_cfg.max_ref_seconds)
    cond = state.frontend.process(torch.from_numpy(wav), sr, req.ref_text)
    return cond, req.ref_text


async def synthesize_pcm(state, req: SpeechRequest) -> AsyncGenerator[bytes, None]:
    """yield 原始 PCM_16 chunk（不含 WAV 头）。取消即自动 abort LLM 请求。"""
    t_start = time.perf_counter()
    cond, ref_text = resolve_condition(state, req)
    prompt = build_prompt(state.tokenizer, ref_text, req.input,
                          cond.prompt_tokens_llm)
    text_len = len(state.tokenizer.encode(req.input))
    sp = make_stream_sampling_params(state.llm_cfg, state.codec,
                                     text_len, req.seed)
    session = StreamSession(
        cond=cond, planner=ChunkPlanner(len(cond.prompt_tokens_flow)))
    request_id = str(uuid.uuid4())
    ttfa_ms = None

    async def flush(finished: bool):
        nonlocal ttfa_ms
        while True:
            plan = session.planner.next_chunk(len(session.tokens), finished)
            if plan is None:
                return
            pcm = await state.batcher.submit(session, plan,
                                             chunk_index=session.chunk_index)
            if ttfa_ms is None:
                ttfa_ms = (time.perf_counter() - t_start) * 1000
            yield pcm16_bytes(pcm)
            if plan.finalize:
                return

    async for delta, finished in stream_token_ids(
            state.engine, prompt, sp, request_id):
        session.tokens.extend(state.codec.extract(delta))
        async for chunk in flush(finished=False):
            yield chunk
    async for chunk in flush(finished=True):
        yield chunk

    if not session.tokens:
        raise NoSpeechTokens("LLM 未产出有效 speech token")
    logger.info(json.dumps(dict(
        event="request_done", request_id=request_id,
        ttfa_ms=round(ttfa_ms or -1, 1),
        chunks=session.chunk_index, tokens=len(session.tokens),
        audio_s=round(session.speech_offset / SAMPLE_RATE, 2),
        wall_s=round(time.perf_counter() - t_start, 2)), ensure_ascii=False))


async def synthesize_response_chunks(state, req) -> AsyncGenerator[bytes, None]:
    """流式响应体：wav 先发未知长度头，再全是 PCM。"""
    if req.response_format == "wav":
        yield wav_stream_header(SAMPLE_RATE)
    async for chunk in synthesize_pcm(state, req):
        yield chunk
```

- [ ] **Step 2: 实现 `app.py`**

```python
# faster_cosyvoice/server/app.py
"""FastAPI server（spec §5.4/§5.5）。启动：自检→资产→引擎→warmup→接流量。

用法：python -m faster_cosyvoice.server.app --port 8000 [--draft-model none]
"""
import argparse
import asyncio
import contextlib
import io
import logging
from types import SimpleNamespace

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import LLMConfig, ServerConfig, Token2WavConfig
from faster_cosyvoice.envcheck import check_environment
from faster_cosyvoice.llm.engine import create_async_llm
from faster_cosyvoice.llm.tokens import SpeechTokenCodec
from faster_cosyvoice.server.openai_speech import (NoSpeechTokens, SAMPLE_RATE,
                                                   synthesize_pcm,
                                                   synthesize_response_chunks)
from faster_cosyvoice.server.protocol import (SpeechRequest, VoiceRequest,
                                              decode_ref_audio)
from faster_cosyvoice.streaming.batcher import Token2WavWorker
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
state = SimpleNamespace()


def build_app(llm_cfg: LLMConfig, t2w_cfg: Token2WavConfig,
              server_cfg: ServerConfig) -> FastAPI:

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        problems = check_environment(
            require_flashinfer=False,  # v1 流式 = torch estimator
            require_draft_mirror=(llm_cfg.draft_model is not None
                                  and llm_cfg.repetition_penalty != 1.0))
        if problems:
            raise RuntimeError("环境自检失败：" + "; ".join(problems))
        model_dir = ensure_token2wav_assets(t2w_cfg.model_dir)
        from transformers import AutoTokenizer
        state.llm_cfg = llm_cfg
        state.server_cfg = server_cfg
        state.tokenizer = AutoTokenizer.from_pretrained(llm_cfg.target_model)
        state.codec = SpeechTokenCodec(state.tokenizer)
        state.engine = create_async_llm(llm_cfg)
        state.frontend = RefAudioFrontend(f"{model_dir}/campplus.onnx",
                                          device=t2w_cfg.device,
                                          cache_size=server_cfg.voice_cache_size)
        state.token2wav = CosyVoice3Token2Wav(model_dir, device=t2w_cfg.device,
                                              estimator_mode="torch")
        state.batcher = Token2WavWorker(state.token2wav)
        state.voices = {}
        await state.batcher.start()
        await _warmup()
        logger.info("server ready")
        yield
        await state.batcher.stop()
        state.engine.shutdown()

    async def _warmup():
        """一条 dummy 请求打通全链路（spec §7：warmup 失败即启动失败）。"""
        import base64
        buf = io.BytesIO()
        sf.write(buf, torch.randn(16000).mul(0.05).numpy(), 16000,
                 format="WAV")
        req = SpeechRequest(
            input="你好。", stream=True, response_format="pcm",
            ref_audio="data:audio/wav;base64,"
                      + base64.b64encode(buf.getvalue()).decode(),
            ref_text="测试。")
        n = 0
        async for _ in synthesize_pcm(state, req):
            n += 1
        assert n > 0, "warmup 未产出音频"
        logger.info("warmup OK (%d chunks)", n)

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/audio/speech")
    async def speech(req: SpeechRequest):
        media = "audio/wav" if req.response_format == "wav" else "audio/pcm"
        try:
            if req.stream:
                return StreamingResponse(
                    synthesize_response_chunks(state, req), media_type=media)
            chunks = []
            async with asyncio.timeout(state.server_cfg.request_timeout_s):
                async for c in synthesize_pcm(state, req):
                    chunks.append(c)
            pcm = b"".join(chunks)
            if req.response_format == "pcm":
                return Response(content=pcm, media_type=media)
            import numpy as np
            buf = io.BytesIO()
            sf.write(buf, np.frombuffer(pcm, dtype="<i2"), SAMPLE_RATE,
                     format="WAV")
            return Response(content=buf.getvalue(), media_type=media)
        except KeyError as e:
            raise HTTPException(400, str(e))
        except NoSpeechTokens as e:
            raise HTTPException(500, str(e))
        except TimeoutError:
            raise HTTPException(504, "请求超时")

    @app.post("/v1/audio/voices")
    async def register_voice(req: VoiceRequest):
        wav, sr = decode_ref_audio(req.ref_audio,
                                   state.server_cfg.max_ref_seconds)
        cond = state.frontend.process(torch.from_numpy(wav), sr, req.ref_text)
        state.voices[req.name] = (cond, req.ref_text)
        return {"success": True, "voice": req.name}

    @app.get("/v1/audio/voices")
    async def list_voices():
        return {"voices": sorted(state.voices)}

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--target-model",
                   default="yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF")
    p.add_argument("--draft-model", default="yuekai/cosyvoice3_llm_dspark")
    p.add_argument("--token2wav-dir", default="models/Fun-CosyVoice3-0.5B-2512")
    p.add_argument("--token2wav-device", default="cuda:0")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = p.parse_args()

    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # 同 offline 的 fork segfault 规避
    draft = None if args.draft_model in (None, "none") else args.draft_model
    llm_cfg = LLMConfig(target_model=args.target_model, draft_model=draft,
                        gpu_memory_utilization=args.gpu_memory_utilization)
    t2w_cfg = Token2WavConfig(model_dir=args.token2wav_dir,
                              device=args.token2wav_device,
                              estimator_mode="torch")
    server_cfg = ServerConfig(host=args.host, port=args.port,
                              gpu_memory_utilization=args.gpu_memory_utilization)
    uvicorn.run(build_app(llm_cfg, t2w_cfg, server_cfg),
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 语法 + CPU 导入检查**

```bash
python -m py_compile faster_cosyvoice/server/app.py faster_cosyvoice/server/openai_speech.py && echo OK
python -c "from faster_cosyvoice.server.app import build_app; print('import OK')"
```
（注意 `asyncio.timeout` 需要 py≥3.11——本 venv 是 3.12 ✓。若 pydantic/fastapi 版本导入报错，调整并报告。）

- [ ] **Step 4: GPU 单请求冒烟（前台起 server → curl）**

```bash
python -m faster_cosyvoice.server.app --port 18000 &   # 后台，等 "server ready"
# 就绪后：
curl -s http://localhost:18000/health
python - <<'EOF'
import base64, io, json, time
import numpy as np, soundfile as sf, httpx
buf = io.BytesIO()
sf.write(buf, (np.random.randn(16000*3)*0.05).astype("float32"), 16000, format="WAV")
req = dict(input="今天天气真不错，我们一起去公园散步吧。",
           ref_audio="data:audio/wav;base64,"+base64.b64encode(buf.getvalue()).decode(),
           ref_text="这是一个测试。", stream=True, response_format="wav")
t0 = time.perf_counter(); first = None; total = 0
with httpx.stream("POST", "http://localhost:18000/v1/audio/speech",
                  json=req, timeout=120) as r:
    r.raise_for_status()
    for chunk in r.iter_bytes():
        if first is None: first = time.perf_counter() - t0
        total += len(chunk)
print(json.dumps(dict(ttfb_ms=round(first*1000,1), total_bytes=total)))
EOF
kill %1
```
预期：health ok；ttfb（≈TTFA+网络）在几百 ms 量级（含首次请求的 prompt 处理），total_bytes > 100k。真实 ref 音频（随机噪声 ref）下音质无意义——质量看 Task 8。

- [ ] **Step 5: Commit** — `git add faster_cosyvoice/server && git commit -s --no-verify -m "feat: FastAPI OpenAI speech server (streaming pipeline + voices + warmup)"`

---

### Task 8: 客户端示例 + GPU e2e（并发 + ASR 门 + 流/非流一致性）

**Files:**
- Create: `examples/stream_client.py`、`scripts/run_server.sh`、`tests/gpu/test_server_e2e.py`

- [ ] **Step 1: `examples/stream_client.py`**

```python
# examples/stream_client.py
"""流式客户端：请求 → 保存 wav + 打印 TTFA/时长。

python examples/stream_client.py --url http://localhost:8000 \
    --ref-audio ref.wav --ref-text "参考" --target-text "目标" --out out.wav
"""
import argparse
import base64
import json
import time

import httpx
import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--ref-text", required=True)
    p.add_argument("--target-text", required=True)
    p.add_argument("--out", default="out.wav")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.ref_audio, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode()
    req = dict(input=args.target_text, ref_text=args.ref_text,
               ref_audio="data:audio/wav;base64," + ref_b64,
               stream=True, response_format="wav", seed=args.seed)
    t0 = time.perf_counter()
    ttfa = None
    data = b""
    with httpx.stream("POST", f"{args.url}/v1/audio/speech", json=req,
                      timeout=300) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            if ttfa is None and len(data) > 44:  # 头之后的首个音频块
                ttfa = time.perf_counter() - t0
            data += chunk
    if ttfa is None:
        ttfa = time.perf_counter() - t0
    with open(args.out, "wb") as f:
        f.write(data)
    audio, sr = sf.read(args.out)
    print(json.dumps(dict(ttfa_ms=round(ttfa * 1000, 1),
                          audio_s=round(len(audio) / sr, 2),
                          wall_s=round(time.perf_counter() - t0, 2)),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
```

（注意 0xFFFFFFFF 长度的流式 wav：soundfile 读取时以实际字节为准，若个别版本拒读，用 `np.frombuffer(data[44:], "<i2")` 兜底并在实现时验证。）

- [ ] **Step 2: `scripts/run_server.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export PYTHONPATH=$PWD/third_party/spec-vllm:$PYTHONPATH
export HF_HOME=${HF_HOME:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface}
exec python -m faster_cosyvoice.server.app "$@"
```

- [ ] **Step 3: `tests/gpu/test_server_e2e.py`**

```python
# tests/gpu/test_server_e2e.py
"""server e2e：起 server → 4 并发流式 + 非流对照 → 音频断言 + ASR 门数据落盘。
运行：pytest tests/gpu/test_server_e2e.py -m gpu -v（容器内，~10min）"""
import asyncio
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time

import httpx
import numpy as np
import pytest
import soundfile as sf

PORT = 18100
URL = f"http://localhost:{PORT}"
OUT = "results/pytest_server"

TEXTS = ["今天天气真不错，我们一起去公园散步吧。",
         "人工智能正在改变我们的生活方式。",
         "请记得明天早上八点开会。",
         "这本书的内容非常有趣，推荐大家阅读。"]


def _ref_data_url():
    from datasets import load_dataset, Audio
    ds = load_dataset("yuekai/seed_tts_cosy2", split="wenetspeech4tts")
    ds = ds.cast_column("prompt_audio", Audio(decode=False))
    row = ds[0]
    b64 = base64.b64encode(row["prompt_audio"]["bytes"]).decode()
    return "data:audio/wav;base64," + b64, row["prompt_text"]


async def _stream_one(client, ref_url, ref_text, text, idx):
    req = dict(input=text, ref_audio=ref_url, ref_text=ref_text,
               stream=True, response_format="wav", seed=100 + idx)
    t0 = time.perf_counter()
    ttfa = None
    data = b""
    async with client.stream("POST", f"{URL}/v1/audio/speech", json=req,
                             timeout=300) as r:
        assert r.status_code == 200
        async for chunk in r.aiter_bytes():
            if ttfa is None and len(data) > 44:
                ttfa = time.perf_counter() - t0
            data += chunk
    pcm = np.frombuffer(data[44:], dtype="<i2").astype(np.float32) / 32767
    return ttfa or (time.perf_counter() - t0), pcm


@pytest.mark.gpu
def test_server_streaming_e2e():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "faster_cosyvoice.server.app",
         "--port", str(PORT)])
    try:
        # 等 server ready（warmup 含引擎加载，给足时间）
        deadline = time.time() + 900
        while time.time() < deadline:
            try:
                if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                time.sleep(5)
        else:
            raise TimeoutError("server 未就绪")

        ref_url, ref_text = _ref_data_url()

        async def run_all():
            async with httpx.AsyncClient() as client:
                return await asyncio.gather(*[
                    _stream_one(client, ref_url, ref_text, t, i)
                    for i, t in enumerate(TEXTS)])

        results = asyncio.run(run_all())
        expected = {}
        for i, (ttfa, pcm) in enumerate(results):
            assert len(pcm) > 24000 * 0.5, f"req{i} 音频过短"
            assert np.abs(pcm).mean() > 1e-4, f"req{i} 疑似静音"
            assert ttfa < 5.0, f"req{i} TTFA {ttfa:.1f}s 异常"
            sf.write(os.path.join(OUT, f"stream_{i}.wav"), pcm, 24000)
            expected[f"stream_{i}"] = TEXTS[i]
        with open(os.path.join(OUT, "expected.json"), "w") as f:
            json.dump(expected, f, ensure_ascii=False)
        print("TTFA(ms):", [round(t * 1000) for t, _ in results])

        # 非流式对照（同 seed 同请求 → spec §8 流/非流一致性经 ASR 门）
        req = dict(input=TEXTS[0], ref_audio=ref_url, ref_text=ref_text,
                   stream=False, response_format="wav", seed=100)
        r = httpx.post(f"{URL}/v1/audio/speech", json=req, timeout=300)
        assert r.status_code == 200
        with open(os.path.join(OUT, "nonstream_0.wav"), "wb") as f:
            f.write(r.content)
        expected["nonstream_0"] = TEXTS[0]
        with open(os.path.join(OUT, "expected.json"), "w") as f:
            json.dump(expected, f, ensure_ascii=False)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
```

- [ ] **Step 4: 容器内跑 e2e + ASR 门**

```bash
pytest tests/gpu/test_server_e2e.py -m gpu -v      # 后台跑，含引擎加载 ~10min
python scripts/asr_check.py --wav-dir results/pytest_server \
    --ref-json results/pytest_server/expected.json \
    --paraformer-dir models/sherpa-onnx-paraformer-zh-2023-09-14 \
    --cer-threshold 0.15
```
预期：pytest 1 passed；TTFA 打印（并发 4，热身后应在数百 ms 内）；ASR 门 5 条全过（4 流式 + 1 非流）。任何一条超阈值 → 报告 per-item CER，不要自行放宽阈值。

- [ ] **Step 5: Commit** — `git add examples/stream_client.py scripts/run_server.sh tests/gpu/test_server_e2e.py && git commit -s --no-verify -m "test: server e2e (concurrent streaming + ASR gate) + client example"`

---

### Task 9: README 更新 + 收尾

- [ ] **Step 1:** README 增加 "Streaming server" 一节：`scripts/run_server.sh` 启动、curl/stream_client 用法、voice 注册示例、实测 TTFA 数字（取 Task 8 输出）、`--draft-model none` 与 gpu_memory_utilization 0.5 说明。
- [ ] **Step 2:** 全套 CPU 回归 `pytest` → 40 passed, 2 deselected（按各 task 累计数核对）。
- [ ] **Step 3: Commit** — `git add README.md && git commit -s --no-verify -m "docs: README streaming server section + measured TTFA"`

---

## 与 spec 的对照（自查）

- §1 目标 2（OpenAI 协议流式 + 服务端组 batch v1 口径=LLM continuous batching + t2w 交错）→ Task 3/4/7；D3 v1（torch streaming + 固定 rand_noise）→ Task 2；§5.3 chunker/batcher → Task 1/4；§5.4/§5.5 协议/voices/warmup → Task 6/7；§6.2/§6.3 数据流与并发 → Task 7/8；§7 请求期错误（400/500/504、断连 abort、超时旋钮、>30s 截断）→ Task 4/6/7；§8 测试 2（流/非流同过 ASR 门）+ Server 冒烟（并发 TTFA）→ Task 8。
- 显式不在本计划：M3（flashinfer chunk-causal mask、跨请求 packed batch）；campplus TRT（M2 后视性能定）；voice 持久化（内存注册表，重启即失，README 注明）。
- 已知风险：`asyncio.timeout` 仅限非流式路径（流式超时靠客户端）；流式 wav 的 0xFFFFFFFF 头对个别解码器兼容性需在 Task 8 实测；hift 全量重跑 O(T²) 长输出尾部延迟（spec 风险 2，M2 记录实测值即可）。
- 有意偏离：spec §7 观测项里的"每请求接受率"不实现——AsyncLLM 只暴露全局 `vllm:spec_decode_*` 计数器，无 per-request 归因；请求日志记 TTFA/chunks/tokens/audio_s/wall_s，接受率看引擎全局指标。pydantic 校验失败返回 FastAPI 惯例的 422（spec 写 400，语义等价）。
