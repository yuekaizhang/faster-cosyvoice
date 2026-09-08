"""campplus 说话人 embedding 的 TensorRT 路径（opt-in，默认仍是 ORT-CPU）。

移植自 duplex CosyVoice runtime/triton_trtllm/token2wav_cosyvoice3.py 的
convert_onnx_to_trt / TrtContextWrapper / load_spk_trt / forward_spk_embedding，
按 campplus 单模型简化：fp32-only、单 context（本工程 GPU 调用已串行，
保留 acquire/release 形状以便未来加并发）。

TRT 11 相对原代码（TRT 10）的 API 变化：
- NetworkDefinitionCreationFlag.EXPLICIT_BATCH 已移除（explicit batch 成为
  唯一模式）→ create_network(0)。
- 其余（build_serialized_network / execute_async_v3 / set_tensor_address /
  set_input_shape）不变。
"""
import logging
import os
import queue

import torch

logger = logging.getLogger(__name__)

# campplus.onnx：input "input" (1, T, 80) fp32 动态 T，output (1, 192)
_INPUT_NAME = "input"
_MIN_SHAPE = (1, 4, 80)
_OPT_SHAPE = (1, 500, 80)
_MAX_SHAPE = (1, 3000, 80)
_EMB_DIM = 192


def _import_trt():
    try:
        import tensorrt as trt
        return trt
    except ImportError as e:
        raise RuntimeError(
            "The TensorRT speaker encoder requires the `tensorrt` Python "
            f"package, but importing it failed: {e}. Run `uv sync --frozen`, "
            "or omit --speaker-encoder-tensorrt to use ONNX Runtime CPU.") from e


def convert_onnx_to_trt(plan_path: str, onnx_path: str) -> None:
    """campplus 专用 fp32 build（简化自 duplex convert_onnx_to_trt）。"""
    trt = _import_trt()
    logger.info("Converting %s -> %s ...", onnx_path, plan_path)
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)  # TRT>=10：explicit batch 是默认且唯一
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    # 关 TF32（Ampere+ 默认开）保证纯 fp32。数值基准：TRT 输出与
    # *未优化* ONNX 图（ORT_DISABLE_ALL）逐条 cos=1.000000；ORT 默认
    # ORT_ENABLE_ALL 自己的图优化（BASIC 级即出现）会把结果推离原始图
    # cos 0.9896~0.9998，因此 TRT vs ORT(ENABLE_ALL) 偶见 cos<0.999，
    # 但 TRT 才是对 onnx 原图更忠实的一方（质量门以 ASR CER 对照为准）。
    config.clear_flag(trt.BuilderFlag.TF32)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise ValueError(f"failed to parse {onnx_path}: {errs}")
    profile = builder.create_optimization_profile()
    profile.set_shape(_INPUT_NAME, _MIN_SHAPE, _OPT_SHAPE, _MAX_SHAPE)
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path}")
    with open(plan_path, "wb") as f:
        f.write(engine_bytes)
    logger.info("Successfully converted onnx to trt: %s", plan_path)


class TrtContextWrapper:
    """engine + context 池（同 duplex；本工程串行调用，trt_concurrent=1 足够）。"""

    def __init__(self, trt_engine, trt_concurrent: int = 1,
                 device: str = "cuda:0"):
        self.trt_context_pool = queue.Queue(maxsize=trt_concurrent)
        self.trt_engine = trt_engine
        self.device = device
        for _ in range(trt_concurrent):
            trt_context = trt_engine.create_execution_context()
            assert trt_context is not None, (
                "failed to create trt context (CUDA OOM? try reduce "
                f"trt_concurrent {trt_concurrent})")
            trt_stream = torch.cuda.stream(
                torch.cuda.Stream(torch.device(device)))
            self.trt_context_pool.put([trt_context, trt_stream])

    def acquire_estimator(self):
        return self.trt_context_pool.get(), self.trt_engine

    def release_estimator(self, context, stream):
        self.trt_context_pool.put([context, stream])


def load_campplus_trt(campplus_onnx_path: str, plan_path: str,
                      device: str = "cuda:0") -> TrtContextWrapper:
    """加载（缺失/空文件则先 build）campplus TRT engine → wrapper。"""
    trt = _import_trt()
    if not os.path.exists(plan_path) or os.path.getsize(plan_path) == 0:
        convert_onnx_to_trt(plan_path, campplus_onnx_path)
    with open(plan_path, "rb") as f:
        engine = trt.Runtime(
            trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
    assert engine is not None, f"failed to load trt {plan_path}"
    return TrtContextWrapper(engine, trt_concurrent=1, device=device)


@torch.inference_mode()
def spk_embedding_trt(wrapper: TrtContextWrapper,
                      feat: torch.Tensor) -> torch.Tensor:
    """feat: (T, 80) GPU fp32（CMN 后的 kaldi fbank）→ (192,) cpu fp32。

    execute_async_v3 指针绑定，同 duplex forward_spk_embedding TRT 分支。
    """
    [context, stream], engine = wrapper.acquire_estimator()
    try:
        device = torch.device(wrapper.device)
        with torch.cuda.device(device):
            torch.cuda.current_stream().synchronize()
            feat = feat.unsqueeze(0).to(device).contiguous().float()
            with stream:
                context.set_input_shape(_INPUT_NAME, tuple(feat.shape))
                out = torch.empty((1, _EMB_DIM), dtype=torch.float32,
                                  device=device)
                for i, ptr in enumerate([feat.data_ptr(), out.data_ptr()]):
                    context.set_tensor_address(engine.get_tensor_name(i), ptr)
                assert context.execute_async_v3(
                    torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
    finally:
        wrapper.release_estimator(context, stream)
    return out.cpu().flatten().float()
