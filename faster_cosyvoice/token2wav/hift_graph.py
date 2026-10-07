# faster_cosyvoice/token2wav/hift_graph.py
"""[M3.5-r4] 流式 hift bucketed CUDA graphs（opt-in，镜像 flashinfer_dit 的
sbucket 思路）：finalize=False 中间 chunk 的 hift.inference 整段捕成 per-bucket
CUDA graph（mel pad-to-bucket，静态 in/out buffer，共享 graph pool），把几十次
kernel launch（同卡 vLLM EngineCore 时间片抢占下每个 launch 都是调度点）压成
1 次 replay。

正确性依据：CausalHiFTGenerator 的 finalize=False 路径按设计只返回不受
lookahead 影响的前缀音频——f0_predictor.condnet[0]（look-right=3）与
conv_pre（look-right=4）的右视区、istft 的边界归一化尾部（n_fft-hop=12 样本，
被末尾 480 样本 drop 覆盖）全部被裁掉，其余算子全部左因果。因此把 mel 右侧
zero-pad 到桶长 B、按"真实长度 T 的 eager 调用会返回的长度 L(T)" 切片，
逐样本与 unpadded eager 数学等价（cudnn/cuFFT 按形状选算法 → 非逐位一致，
init 时做一次 max-abs-diff 自检）。L(T) 对 T 线性（斜率 480 = 24kHz/50Hz），
init 用两个探针长度拟合并用第三个长度断言。

finalize=True（最终 chunk）不走 graph：final chunk 不在 TTFP 关键路径上，
且其 istft 边界/f0 全长路径与 pad 语义不同——eager（或 hift_compile 桶化
compile 路径）足够。

capture 细节：
- SineGen2/SourceModuleHnNSF 的固定噪声 buffer（rand_ini/sine_waves/uv）与
  stft_window 原始在 CPU、forward 内 `.to(device)`——capture 期间 pageable H2D
  拷贝非法，install 时先搬到 GPU（之后 `.to()` 变 no-op，eager 路径也省一次
  每调用 H2D 拷贝）。
- capture 前在 side stream 以"精确桶形状"跑两遍 eager：cudnn v8 plan、
  cuFFT plan、fp64 f0 conv 的 workspace 全部预热，capture 区域内不再有
  首遇分配/plan-build。
- capture 用 init 时 stash 的原始 eager decode（hift_compile 的 inductor
  wrapper 不进 graph）：graph 数值 == eager，与 hift_compile 是否开启解耦。
- 返回的是静态输出 buffer 的切片视图（与 flow graph `entry["out"][:, :n]`
  同契约）：调用方须在下一次 replay 前消费（_finish_chunk 立即切片 .cpu()，
  单 token2wav 线程下安全）。
"""
import logging
from typing import List

import torch

logger = logging.getLogger(__name__)

_PROBE = (64, 96, 160)  # L(T) 线性拟合探针（两点拟合 + 第三点断言）


def _istft_capture_safe(hift, magnitude, phase):
    """torch.istft 的 capture-safe 手写等价（irfft + window + fold 重叠相加 +
    窗包络归一 + center crop）：ATen istft 在窗包络检查处做 `.item()` 式
    device sync，会 invalidate CUDA graph capture。参数与 generator._istft
    一致（n_fft=16, hop=4, center=True, onesided），数值差 ~1e-6（fft 实现
    路径不同）。仅 capture 期间替换 hift._istft；eager 路径不变。"""
    n_fft = hift.istft_params["n_fft"]
    hop = hift.istft_params["hop_len"]
    window = hift.stft_window  # 已由 _move_static_buffers 搬 GPU
    magnitude = torch.clip(magnitude, max=1e2)
    spec = torch.complex(magnitude * torch.cos(phase),
                         magnitude * torch.sin(phase))  # (B, F, T)
    frames = torch.fft.irfft(spec, n=n_fft, dim=1)      # (B, n_fft, T)
    frames = frames * window.view(1, -1, 1)
    b, _, t = frames.shape
    length = n_fft + hop * (t - 1)
    y = torch.nn.functional.fold(
        frames, output_size=(1, length), kernel_size=(1, n_fft),
        stride=(1, hop)).reshape(b, length)
    env = torch.nn.functional.fold(
        (window * window).view(1, -1, 1).expand(1, n_fft, t),
        output_size=(1, length), kernel_size=(1, n_fft),
        stride=(1, hop)).reshape(length)
    half = n_fft // 2  # center=True crop（istft 先 crop 再除包络）
    return y[:, half:length - half] / env[half:length - half]


def _move_static_buffers(hift, device) -> None:
    """SineGen2/SourceModuleHnNSF 固定噪声 buffer 与 stft_window 搬 GPU
    （capture 合法化 + eager 免每调用 H2D）。幂等。"""
    hift.stft_window = hift.stft_window.to(device)
    ms = hift.m_source
    if getattr(ms, "uv", None) is not None:
        ms.uv = ms.uv.to(device)
    sg = ms.l_sin_gen
    if getattr(sg, "rand_ini", None) is not None:
        sg.rand_ini = sg.rand_ini.to(device)
    if getattr(sg, "sine_waves", None) is not None:
        sg.sine_waves = sg.sine_waves.to(device)


class HiftStreamGraph:
    """finalize=False 的 hift.inference bucketed graph 执行器。

    install() 后 hift.inference 被 wrapper 替换：finalize=False 且
    T<=max(buckets) 走 graph（lazy capture，首遇一次 ~百 ms 级），
    其余（finalize=True / 超长）走原 inference。"""

    def __init__(self, hift, device: str, buckets: List[int],
                 eager_decode=None):
        self.hift = hift
        self.device = torch.device(device)
        self.buckets = sorted(int(b) for b in buckets)
        self.eager_decode = eager_decode  # hift_compile 前的原始 decode
        self.pool = torch.cuda.graph_pool_handle()
        self.entries = {}
        self.orig_inference = hift.inference
        _move_static_buffers(hift, self.device)
        with torch.cuda.device(self.device):
            self._fit_out_len()

    # ---- L(T)：finalize=False 输出样本数，对 T 线性 ----
    def _eager_infer(self, t: int) -> torch.Tensor:
        mel = torch.zeros(1, 80, t, device=self.device) - 6.0
        with self._eager_decode_ctx():
            return self.orig_inference(speech_feat=mel, finalize=False)[0]

    def _fit_out_len(self) -> None:
        t0, t1, t2 = _PROBE
        l0 = self._eager_infer(t0).shape[1]
        l1 = self._eager_infer(t1).shape[1]
        slope = (l1 - l0) // (t1 - t0)
        self._slope, self._icept = slope, l0 - slope * t0
        l2 = self._eager_infer(t2).shape[1]
        assert l2 == self.out_len(t2), (
            f"hift finalize=False 输出长度非线性: L({t2})={l2} "
            f"!= {self.out_len(t2)}")

    def out_len(self, t: int) -> int:
        return self._slope * t + self._icept

    def _eager_decode_ctx(self, capture_safe_istft: bool = False):
        """capture/探针期间临时还原原始 eager decode（若 hift_compile 已包）；
        capture_safe_istft=True 时同时换上手写 istft（torch.istft 在 capture
        中会 device-sync invalidate）。"""
        import contextlib

        @contextlib.contextmanager
        def ctx():
            cur_decode = self.hift.decode
            cur_istft = self.hift._istft
            if self.eager_decode is not None:
                self.hift.decode = self.eager_decode
            if capture_safe_istft:
                self.hift._istft = (
                    lambda m, p: _istft_capture_safe(self.hift, m, p))
            try:
                yield
            finally:
                self.hift.decode = cur_decode
                self.hift._istft = cur_istft
        return ctx()

    # ---- capture / replay ----
    def _capture(self, bucket: int) -> dict:
        with torch.cuda.device(self.device):
            return self._capture_impl(bucket)

    def _capture_impl(self, bucket: int) -> dict:
        static_in = torch.zeros(1, 80, bucket, device=self.device)
        with self._eager_decode_ctx(capture_safe_istft=True):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):  # 精确形状预热：cudnn/cuFFT plan、workspace
                    self.orig_inference(speech_feat=static_in, finalize=False)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out, _ = self.orig_inference(speech_feat=static_in,
                                             finalize=False)
        logger.info("hift stream graph captured: bucket=%d out=%s",
                    bucket, tuple(out.shape))
        return {"graph": graph, "in": static_in, "out": out}

    def run(self, speech_feat: torch.Tensor):
        """graph 路径：pad→replay→按 L(T) 切片（静态 buffer 视图）。"""
        t = speech_feat.shape[2]
        bucket = next(b for b in self.buckets if b >= t)
        entry = self.entries.get(bucket)
        if entry is None:
            entry = self._capture(bucket)
            self.entries[bucket] = entry
        s = entry["in"]
        s[:, :, :t].copy_(speech_feat)
        s[:, :, t:].zero_()
        entry["graph"].replay()
        return entry["out"][:, :self.out_len(t)], None

    def install(self) -> None:
        orig = self.orig_inference
        buckets_max = self.buckets[-1]

        def inference(speech_feat: torch.Tensor, finalize: bool = True):
            if (not finalize and speech_feat.shape[0] == 1
                    and speech_feat.shape[2] <= buckets_max):
                with torch.cuda.device(self.device):
                    return self.run(speech_feat)
            return orig(speech_feat=speech_feat, finalize=finalize)

        self.hift.inference = inference
        self.hift._stream_graph = self  # 便于测试/诊断访问
