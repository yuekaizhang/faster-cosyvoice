# M3 FlashInfer Streaming（faster-cosyvoice）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** spec M3——给 FlashInferDiT 加 chunk-causal custom mask（流式也走 flashinfer），实现跨请求 packed 流式批量（batcher v2 执行器），以 M2 torch 流式为参照 + ASR 门通过后切为流式默认。

**Architecture:** 三层递进：① estimator 层（RaggedAttentionRunner 支持 chunk mask、forward 接受 streaming）；② 单请求层（stream_step 允许 flashinfer 模式，server 加 `--stream-estimator` 旋钮）；③ 批量层（流式版 packed flow + `stream_step_batched` + batcher v2 执行器）。每层有独立 parity/ASR 门，上层 gate 不过不切默认。

**执行环境约定（同 M1/M2）：** 容器内跑 pytest/GPU；`source venv/bin/activate && export PYTHONPATH=$PWD/third_party/spec-vllm:$PYTHONPATH && export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface`。分支 `m3-flashinfer-streaming`。基线：44 CPU passed + 4 GPU tests。

**调研钉死的事实（两份 M3 调研报告，行号均已核实）：**
- **mask 谓词**：`allowed(q,k) = floor(k/50) <= floor(q/50)`，位置从序列 0（= prompt 首 mel 帧）绝对计数，无左上下文限制、prompt 不特殊化、CFG 两行同 mask、10 步 Euler 恒定。offline 等价于只留 padding mask。（torch 参照：`dit.py:163-166` + `mask.py:127-158,223-231`；`num_left_chunks` 参数在 vendored 版本里是 no-op。）
- **前缀稳定性**：chunk mask 下输出前缀稳定到 `floor(T_old/50)*50`；尾部残 chunk 随序列增长而变（CPU 探针：稳定区 diff 9.5e-07，残 chunk 7.9e-03，offline 全破坏 1.1e-02）。管线靠全前缀重算+mel cache 容忍此性质。
- **flashinfer 0.6.13**：`BatchPrefillWithRaggedKVCacheWrapper.plan(custom_mask=<flat bool>)` 可用——per-doc 行主序 `T_i×T_i` 展平按 doc 顺序拼接，True=keep；**`packed_custom_mask` 在此版本对多 doc 是坏的（byte/element 单位错配），禁用**。带 mask 强制 fa2 后端（正确性无损）。GPU 探针 vs SDPA：docs=[250,993,512,1000] chunk=50 → rel 6.3e-4；[3000,3000] → 7.5e-4。plan+mask ≈1.5ms（可按 key 缓存）；mask 内存 packed ≈2MB @T=3000×2。
- **集成点（flashinfer_dit.py）**：① `RaggedAttentionRunner.plan`(223)/`plan_docs`(236) 加 `chunk_size=None` 参数，内部建 mask 传 `custom_mask`，cache key 加 chunk_size；② `forward`(352) 删 353 的 assert，线程化 streaming；③ CUDA graph 分支(361) 加 `and not streaming`；④ `_forward_packed`(374) 线程化 streaming，397 处 plan_docs 传 chunk_size，flat mask 可存进 `_pack_cache` meta；⑤ `_solve_euler_batched`(674-675) 目前硬编码 `False`——批量流式需传 flag。
- **streaming 到达路径**：`forward_estimator(x,mask,mu,t,spks,cond,streaming)` nn.Module 分支已以 kwarg 传给 estimator（`flow_matching.py:126-128`）；TRT 分支不传（与我们无关）。
- **finalize 语义（批量流式必须复现）**：`CausalMaskedDiffWithDiT.inference`（vendored `flow.py:364-410`）在 finalize=False 时把最后 `pre_lookahead_len` 个 token 作为 PreLookaheadLayer 的 context 传入（conv 右上下文），返回 mel 恰好覆盖已消费 token；finalize=True 时全量。批量版必须逐行镜像该逻辑。

**对 vendored 文件的修改原则**：M3 必须改 `flashinfer_dit.py`（spec 本就如此），每处修改用 `# [M3]` 注释标记；其余 vendored cosyvoice/ 文件仍不动。

---

### Task 1: FlashInferDiT chunk-causal mask 支持 + estimator 级 parity 门

**Files:**
- Modify: `faster_cosyvoice/token2wav/flashinfer_dit.py`（集成点 ①②③④）
- Test: `tests/gpu/test_flashinfer_streaming_estimator.py`（新，gpu 标记）

- [ ] **Step 1: 写 mask 构建辅助 + runner 扩展**

在 `RaggedAttentionRunner` 上方加模块级函数：

```python
def _chunk_causal_flat_mask(doc_lens, chunk_size: int, device) -> torch.Tensor:
    """[M3] per-doc 行主序 chunk-causal bool mask，按 doc 顺序展平拼接。
    谓词 allowed(q,k) = k//chunk <= q//chunk（与 vendored subsequent_chunk_mask 等价，
    参照 dit.py:163 streaming 分支）。True=keep（flashinfer custom_mask 语义）。"""
    parts = []
    for n in doc_lens:
        idx = torch.arange(n, device=device)
        parts.append((idx.view(1, -1) // chunk_size
                      <= idx.view(-1, 1) // chunk_size).flatten())
    return torch.cat(parts)
```

`plan()` 与 `plan_docs()` 均加 `chunk_size: int | None = None` 形参：cache key 追加 chunk_size；`chunk_size is not None` 时构建 flat mask 并以 `custom_mask=` 传给 `wrapper.plan(...)`（注意：**用 custom_mask，不用 packed_custom_mask——0.6.13 多 doc 时 packed 有单位错配 bug**，加注释说明）。flat mask 以 cache key 缓存（省 ~1.3ms/chunk 构建）。

- [ ] **Step 2: forward 线程化 streaming**

`FlashInferDiT.forward`(352)：删除 `assert not streaming`；CUDA graph 分支条件(361) 加 `and not streaming`；`_forward_packed` 加 `streaming: bool = False` 形参并在 397 处 `plan_docs(..., chunk_size=self._chunk_size if streaming else None)`；非 triton eager 路径(371) 同理 `plan(..., chunk_size=...)`。`self._chunk_size = 50` 常量来自 vendored DiT 的 static_chunk_size——在 `__init__` 里加 `static_chunk_size` 形参默认 50 并存下（`apply_flashinfer` 处从 `ref` 读 `getattr(ref, 'static_chunk_size', 50)` 传入，保持与权重构造一致）。所有改动带 `# [M3]` 注释。

- [ ] **Step 3: 写 GPU parity 测试（estimator 级，两道门）**

```python
# tests/gpu/test_flashinfer_streaming_estimator.py 结构（实现者写全）：
# 门 A —— mask 正确性（排除 fp16 噪声的属性测试）：
#   FlashInferDiT(fp16) streaming=True，T=160 vs T=200（前 160 输入相同，固定种子），
#   断言前 floor(160/50)*50=150 帧输出逐位/近似相等（fp16 下 allclose atol=1e-3），
#   而 streaming=False 下同一实验前缀不稳定（diff 显著大）。
# 门 B —— 与 torch streaming 参照对齐：
#   build_flow() 加载真实 flow.pt，torch 版 estimator（fp32）streaming=True 输出 vs
#   apply_flashinfer 后（fp16+mask）同输入输出，rel diff < 0.15（duplex fp16 已知
#   可接受阈值；真正质量门在 Task 2/5 的 ASR）。
# 两门都跑 packed 路径（B=1 即 2 CFG docs）与非 triton eager 路径。
```

- [ ] **Step 4: 验证** — `pytest tests/gpu/test_flashinfer_streaming_estimator.py -m gpu -v` → passed；全 CPU `pytest` → 44 passed 不回归；既有 GPU 测试 `pytest tests/gpu/test_interleave.py -m gpu -v` 仍 1 passed（offline 路径未受影响）。
- [ ] **Step 5: Commit** — `feat: FlashInferDiT chunk-causal custom mask (streaming support)`

---

### Task 2: 单请求 flashinfer 流式（stream_step + server 旋钮）

**Files:**
- Modify: `faster_cosyvoice/token2wav/token2wav.py`（stream_step 放开 flashinfer）、`faster_cosyvoice/server/app.py`（--stream-estimator）
- Test: `tests/gpu/test_stream_flashinfer.py`（新）

- [ ] **Step 1: stream_step 支持 flashinfer 模式**

改 stream_step 开头的 assert 为允许两种模式；flow.inference 调用包 `with torch.amp.autocast("cuda", enabled=self.fp16):`（flashinfer 模式 flow 是 fp16——与 `_flow_single`/duplex forward_stream 的 autocast 用法对齐；torch 模式 enabled=False 无行为变化）。mel cache/speech_offset 逻辑不变（hift 恒 fp32，flow.inference 返回已 .float()，vendored flow.py:409）。

- [ ] **Step 2: server 旋钮**

`app.py` main()：`p.add_argument("--stream-estimator", default="torch", choices=["torch", "flashinfer"])`（**默认仍 torch，Task 5 门过后才翻**）；`t2w_cfg = Token2WavConfig(..., estimator_mode=args.stream_estimator)`；lifespan 里 envcheck 的 `require_flashinfer=(args.stream_estimator == "flashinfer")`（同步调整 build_app 签名或经 t2w_cfg 判断）。

- [ ] **Step 3: GPU 测试** — `tests/gpu/test_stream_flashinfer.py`：单实例 flashinfer 模式，ChunkPlanner 驱动 120 伪 token 流式合成：(a) 跑通出 4 chunk、样本数 115200、连续性断言（对齐 test_interleave 的套路）；(b) 双 session 交错 vs 单独 bit-exact（flashinfer 模式下复验交错安全——fp16 确定性成立因 rand_noise 固定 + 无 atomics 的 fa2 路径；若非 bit-exact 但 allclose(atol=1e-3)，记录实际值并放宽为 allclose + 注释原因）。
- [ ] **Step 4: server 冒烟** — `python -m faster_cosyvoice.server.app --port 18002 --stream-estimator flashinfer` 起服 → 一条流式请求 200 + 音频字节 >100k → kill。
- [ ] **Step 5: 验证套件** — 全 CPU pytest 不回归；新 GPU 测试 passed。
- [ ] **Step 6: Commit** — `feat: flashinfer streaming path for stream_step + --stream-estimator knob`

---

### Task 3: 批量流式 flow + stream_step_batched

**Files:**
- Modify: `faster_cosyvoice/token2wav/flashinfer_dit.py`（`_solve_euler_batched` 加 streaming 形参→传 forward_estimator；新增 `flow_inference_batched_streaming`）
- Modify: `faster_cosyvoice/token2wav/token2wav.py`（新增 `stream_step_batched`）
- Test: `tests/gpu/test_stream_batched.py`（新）

- [ ] **Step 1: `flow_inference_batched_streaming(flow, token_list, prompt_feat_list, embedding, finalize_list)`**

以 `flow_inference_batched`(687) 为骨架的流式变体，**逐行镜像 vendored `CausalMaskedDiffWithDiT.inference`（flow.py:364-410）的流式语义**，关键差异：
- finalize=False 的 doc：最后 `flow.pre_lookahead_len` 个 token 作为 `pre_lookahead_layer(inputs, context=...)` 的 context 传入（参照 flow.py 的处理，token 本体不进 mu 序列）；finalize=True 的 doc 与 offline 相同。注意 packed batch 中各 doc finalize 可不同——per-doc 分别处理 embedding/lookahead 后再 pack。
- `_solve_euler_batched(..., streaming=True)` → `forward_estimator(..., streaming)`（改 674-675 的硬编码 False 为形参透传）。
- 返回 per-doc 全前缀 mel（fp32），调用方切片。
实现时先读 flow.py:364-410 与 flow_inference_batched 全文，把两者 diff 点列成注释写在函数头。

- [ ] **Step 2: `CosyVoice3Token2Wav.stream_step_batched(sessions, plans) -> list[Tensor]`**

flashinfer 模式专用：组 `token_list=[s.tokens[:p.prefix_len]]`、`prompt_feat_list`、stacked embeddings、`finalize_list=[p.finalize]` → `flow_inference_batched_streaming` → 逐 session：mel 按 `p.token_offset*ratio` 切 → mel_cache 拼接 → hift 全量重跑 → speech_offset 切 → 更新 session 状态（与 stream_step 逐 session 部分完全同构——抽出共享私有方法 `_finish_chunk(session, plan, mel_full)` 供两者复用，避免逻辑漂移）。

- [ ] **Step 3: GPU parity 门** — `tests/gpu/test_stream_batched.py`：
  - 门 A：B=1 的 `stream_step_batched` vs 单请求 `stream_step`（同 flashinfer 模式、同输入）逐 chunk allclose(atol=1e-3)（理想 bit-exact，记录实际）。
  - 门 B：B=2 混合 finalize 状态（一个 session 在中段、一个在末段 finalize）跑通且每 session 输出与其单独 B=1 批量运行一致。
- [ ] **Step 4: 验证套件** — CPU 不回归 + 新 GPU 测试 passed + interleave 回归仍 passed。
- [ ] **Step 5: Commit** — `feat: batched streaming flow (packed per-doc chunk masks) + stream_step_batched`

---

### Task 4: batcher v2 packed 执行器

**Files:**
- Modify: `faster_cosyvoice/streaming/batcher.py`（v2 执行循环，`mode="serial"|"packed"` 构造参数）
- Modify: `faster_cosyvoice/server/app.py`（`--t2w-batch-mode`，默认 serial，Task 5 门过后翻）
- Test: `tests/test_batcher.py`（追加 CPU 测试）

- [ ] **Step 1: packed 执行循环** — `_run` 在 mode="packed" 时：弹出堆顶后**继续弹出所有 ready 且 session 互异**的 job（同 session 后续 chunk 留堆——有 mel cache 依赖；用 set 去重），按 `max_batch`（构造参数，默认 8）截断，`run_in_executor(self._gpu, self._t2w.stream_step_batched, sessions, plans)` 一次执行，按序分发结果/异常到各 future（单 job 异常只 fail 对应 future——stream_step_batched 整批抛错时全批 fail，可接受并注释）。cancelled future 在组批时跳过。serial 模式行为不变。
- [ ] **Step 2: CPU 测试**（fake `stream_step_batched` 返回 list）：(a) 3 个异 session ready → 一次批量调用收 3 个；(b) 同 session 两个 job 不同批（第二个在下一轮）；(c) 批中一个 future 已取消 → 不进批、其余正常。
- [ ] **Step 3: server 旋钮** — `--t2w-batch-mode {serial,packed}` 默认 serial；packed 要求 `--stream-estimator flashinfer`（app.py 校验，否则启动报错）。
- [ ] **Step 4: 验证** — CPU pytest 全绿（新增 3 个）；GPU：起服 `--stream-estimator flashinfer --t2w-batch-mode packed`，4 并发流式冒烟 200 × 4。
- [ ] **Step 5: Commit** — `feat: batcher v2 packed executor (cross-session batched chunks)`

---

### Task 5: e2e 验收 + 切默认 + 基准对比

- [ ] **Step 1: 复用 `tests/gpu/test_server_e2e.py` 的流程对 flashinfer+packed 配置跑完整 e2e**：临时以环境变量/参数化让 e2e 起服带 `--stream-estimator flashinfer --t2w-batch-mode packed`（给 e2e 测试加 `FCV_E2E_SERVER_ARGS` 环境变量透传，默认空 = 现行为），4 并发 + 非流对照 → ASR 门 5/5（阈值 0.15 不放宽；失败即 DONE_WITH_CONCERNS 报 per-item CER 停下）。
- [ ] **Step 2: 基准对比**（同 e2e 输出的 server 侧 ttfa_ms + wall）：torch+serial vs flashinfer+packed 各跑一轮 4 并发，记录 TTFA/每请求 wall/总 wall。预期：flashinfer 估计器更快（估计器级 ~2-4x）+ packed 批量减少排队，TTFA 与吞吐应不劣于 torch v1；若更差，报数字并停（不切默认）。
- [ ] **Step 3: 门全过 → 切默认**：`--stream-estimator` 默认 flashinfer、`--t2w-batch-mode` 默认 packed；README streaming 节更新（新默认 + 对比数字 + `--stream-estimator torch` 作为回退开关）；spec 若有出入处加实施注记。
- [ ] **Step 4: 全套回归**（CPU 全绿 + 全部 GPU 测试 + 现默认配置的 server e2e + ASR 门）。
- [ ] **Step 5: Commit ×2** — `feat: default streaming to flashinfer + packed batching (gated by ASR parity)` / `docs: README M3 defaults + benchmark`

---

## 非目标（明确不做，防蔓延）
campplus TRT（除非 Task 5 profiling 显示瓶颈）；流式 server 侧超时；voice 注册持久化；`state` 单例重构；CUDA graphs for streaming；hift 增量化（O(T²) 保持，M3 只测不改）。

## 风险
- fp16 chunk-mask 路径质量：由三层门（estimator rel-diff → 交错 bit-exact/allclose → ASR 5/5）兜底，任一门不过则不切默认、M3 以"可选模式"落地。
- `flow_inference_batched_streaming` 的 finalize/lookahead 镜像是最易错点：Task 3 门 A/B 专门约束；实现者必须先读 flow.py:364-410。
- packed 模式下混合长度 doc 的 mask 内存（sum T_i²/8）在 B=8×长输出时 ~10MB 级——可接受，超限由 max_batch 截断兜底。
