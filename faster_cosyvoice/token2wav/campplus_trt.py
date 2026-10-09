"""Optional TensorRT path for the CampPlus speaker encoder.

The implementation is adapted from CosyVoice's Triton runtime and simplified
to one fp32 model and one execution context.  TensorRT 11 always uses explicit
batch mode, so the network is created with ``create_network(0)``.
"""
import logging
import os
import queue

import torch

logger = logging.getLogger(__name__)

# campplus.onnx: dynamic fp32 input "input" (1, T, 80), output (1, 192).
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
    """Build an fp32 TensorRT engine for CampPlus."""
    trt = _import_trt()
    logger.info("Converting %s -> %s ...", onnx_path, plan_path)
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)  # Explicit batch is the only TRT >= 10 mode.
    parser = trt.OnnxParser(network, trt_logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4GB
    # Disable TF32 to match the original fp32 ONNX graph.  ORT's default graph
    # optimizations can produce a larger difference than TensorRT itself.
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
    """Own a TensorRT engine and a small execution-context pool."""

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
    """Load a cached CampPlus engine, building it when absent or empty."""
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
    """Map a normalized GPU fbank ``(T, 80)`` to a CPU embedding ``(192,)``."""
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
