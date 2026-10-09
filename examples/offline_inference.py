# examples/offline_inference.py
"""Offline voice-clone inference.

Single request:
  uv run python examples/offline_inference.py \
      --ref-audio ref.wav --ref-text "Reference text" --target-text "Target text" \
      --output-dir results/single

Dataset:
  uv run python examples/offline_inference.py \
      --dataset yuekai/seed_tts_cosy2 --split wenetspeech4tts \
      --batch-size 8 --output-dir results/wenetspeech4tts

Disable speculative decoding with ``--draft-model none`` or use the original
Torch Flow implementation with ``--flow-estimator torch``.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Use soundfile because torchaudio 2.9 delegates I/O to unavailable ffmpeg libraries.
import soundfile as sf
import torch

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import (
    DEFAULT_OFFLINE_FLOW_GRAPH_BUCKET_SECONDS,
    LLMConfig,
    Token2WavConfig,
    graph_buckets,
)
from faster_cosyvoice.envcheck import check_environment
from faster_cosyvoice.llm.engine import (
    create_offline_llm,
    make_sampling_params,
    read_spec_counters,
)
from faster_cosyvoice.llm.prompt import build_prompt
from faster_cosyvoice.llm.tokens import SpeechTokenCodec
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

TOKENS_PER_SECOND = 25  # CosyVoice3 emits 25 speech tokens per audio second.


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run batched CosyVoice3 inference.")
    p.add_argument("--ref-audio")
    p.add_argument("--ref-text")
    p.add_argument("--target-text")
    p.add_argument("--dataset")
    p.add_argument("--subset", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--target-model",
                   default="yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF")
    p.add_argument("--draft-model", default="yuekai/cosyvoice3_llm_dspark",
                   help="Draft model, or 'none' to disable speculative decoding.")
    p.add_argument("--token2wav-dir", default="models/Fun-CosyVoice3-0.5B-2512")
    p.add_argument("--flow-estimator", default="flashinfer",
                   choices=["flashinfer", "torch"])
    p.add_argument("--estimator", dest="flow_estimator",
                   choices=["flashinfer", "torch"], default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    p.add_argument("--token2wav-batch-size", type=int, default=8)
    p.add_argument("--token2wav-device", default="cuda:0",
                   help="Device for token2wav and the reference-audio frontend.")
    p.add_argument(
        "--offline-flow-cuda-graph",
        action="store_true",
        help="Enable offline batch-size-1 Flow CUDA Graph with built-in "
             "bucket presets. Use --offline-flow-graph-bucket-seconds only "
             "for advanced tuning.",
    )
    flow_graph_group = p.add_mutually_exclusive_group()
    flow_graph_group.add_argument(
        "--offline-flow-graph-bucket-seconds",
        dest="offline_flow_graph_duration_buckets",
        type=graph_buckets.parse_seconds,
        metavar="SECONDS",
        default=None,
        help="Comma-separated total-context duration buckets in seconds for "
             "offline batch-size-1 Flow CUDA Graph, for example 8,12,16,20,24.",
    )
    flow_graph_group.add_argument(
        "--token2wav-cuda-graph-buckets",
        "--t2w-cuda-graph-buckets",
        dest="offline_flow_graph_duration_buckets",
        type=graph_buckets.parse_seconds,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--speaker-encoder-tensorrt", action="store_true",
                   help="Run the CampPlus speaker encoder with TensorRT "
                        "instead of ONNX Runtime CPU.")
    p.add_argument("--campplus-trt", dest="speaker_encoder_tensorrt",
                   action="store_true", default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    p.add_argument("--vocoder-compile", action="store_true",
                   help="Compile the HiFT vocoder and pad offline inputs to "
                        "64-Mel-frame buckets (15-20 seconds startup warmup).")
    p.add_argument("--hift-compile", dest="vocoder_compile", action="store_true",
                   default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p.add_argument("--output-dir", default="results/offline")
    p.add_argument("--seed", type=int, default=42)
    return p


def get_args(argv=None):
    return build_argument_parser().parse_args(argv)


def resolve_offline_flow_graph_buckets(
    args: argparse.Namespace,
) -> tuple[float, ...] | None:
    """Use explicit offline buckets when provided, otherwise the preset."""
    if args.offline_flow_graph_duration_buckets is not None:
        return args.offline_flow_graph_duration_buckets
    if args.offline_flow_cuda_graph:
        return DEFAULT_OFFLINE_FLOW_GRAPH_BUCKET_SECONDS
    return None


def load_items(args):
    """→ list of dict(ref_wav: 1-D tensor, ref_sr, ref_text, target_text, uid)"""
    if args.dataset:
        import io

        from datasets import Audio, load_dataset
        ds = load_dataset(args.dataset, args.subset, split=args.split)
        # datasets>=5 delegates Audio decoding to torchcodec/ffmpeg; decode here instead.
        ds = ds.cast_column("prompt_audio", Audio(decode=False))
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        items = []
        for i, row in enumerate(ds):
            audio = row["prompt_audio"]
            audio_bytes = audio["bytes"]
            if audio_bytes is None:
                audio_bytes = Path(audio["path"]).read_bytes()
            array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
            items.append(dict(
                ref_wav=torch.tensor(array, dtype=torch.float32),
                ref_sr=sr,
                ref_text=row["prompt_text"], target_text=row["target_text"],
                uid=row.get("id", f"{i:05d}"), row_index=i,
                prompt_audio_name=audio.get("path"),
                prompt_audio_bytes=len(audio_bytes),
                prompt_audio_sha256=hashlib.sha256(audio_bytes).hexdigest()))
        return items
    assert args.ref_audio and args.ref_text and args.target_text, \
        "单条模式需要 --ref-audio/--ref-text/--target-text"
    audio_bytes = Path(args.ref_audio).read_bytes()
    array, sr = sf.read(args.ref_audio, dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)
    return [dict(ref_wav=torch.tensor(array, dtype=torch.float32),
                 ref_sr=sr, ref_text=args.ref_text,
                 target_text=args.target_text, uid="single", row_index=0,
                 prompt_audio_name=args.ref_audio,
                 prompt_audio_bytes=len(audio_bytes),
                 prompt_audio_sha256=hashlib.sha256(audio_bytes).hexdigest())]


def main():
    # Avoid gomp_team_start crashes when OpenMP initializes after EngineCore forks.
    # setdefault preserves an explicit deployment-level override.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = get_args()
    draft = None if args.draft_model in (None, "none") else args.draft_model
    llm_cfg = LLMConfig(target_model=args.target_model, draft_model=draft)
    flow_graph_buckets = resolve_offline_flow_graph_buckets(args)
    token2wav_cfg = Token2WavConfig(
        model_dir=args.token2wav_dir,
        device=args.token2wav_device,
        estimator_mode=args.flow_estimator,
        batch_size=args.token2wav_batch_size,
        offline_flow_graph_duration_buckets=flow_graph_buckets,
        vocoder_compile=args.vocoder_compile,
    )

    problems = check_environment(
        require_flashinfer=(args.flow_estimator == "flashinfer"),
        require_draft_mirror=(draft is not None
                              and llm_cfg.repetition_penalty != 1.0))
    if problems:
        print("环境自检失败：\n  - " + "\n  - ".join(problems))
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = ensure_token2wav_assets(token2wav_cfg.model_dir)
    items = load_items(args)
    print(f"{len(items)} 条请求")

    # Persist the exact per-row voice-clone protocol before model execution.
    # This proves that every target uses its own paired prompt audio/text rather
    # than a shared benchmark voice.
    input_manifest_path = os.path.join(args.output_dir, "input_manifest.jsonl")
    with open(input_manifest_path, "w") as manifest:
        for i, item in enumerate(items):
            manifest.write(json.dumps({
                "row_index": item["row_index"],
                "uid": item["uid"],
                "prompt_audio_name": item["prompt_audio_name"],
                "prompt_audio_bytes": item["prompt_audio_bytes"],
                "prompt_audio_sha256": item["prompt_audio_sha256"],
                "prompt_sample_rate": item["ref_sr"],
                "prompt_text": item["ref_text"],
                "target_text": item["target_text"],
                "generation_seed": args.seed + i,
            }, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump({
            **vars(args),
            "resolved_draft_model": draft,
            "protocol": "per-row prompt_audio + prompt_text -> target_text",
        }, f, ensure_ascii=False, indent=2)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    codec = SpeechTokenCodec(tokenizer)
    llm = create_offline_llm(llm_cfg)
    frontend = RefAudioFrontend(f"{model_dir}/campplus.onnx",
                                device=token2wav_cfg.device,
                                campplus_trt=args.speaker_encoder_tensorrt)
    token2wav = CosyVoice3Token2Wav(model_dir, device=token2wav_cfg.device,
                                    estimator_mode=token2wav_cfg.estimator_mode,
                                    cuda_graph_buckets=(
                                        token2wav_cfg
                                        .offline_flow_graph_duration_buckets),
                                    hift_compile=token2wav_cfg.vocoder_compile)

    metrics = dict(llm_wall_s=0.0, token2wav_wall_s=0.0,
                   frontend_wall_s=0.0,
                   output_tokens=0, finished_by_stop=0, failed=[])
    expected = {}
    spec_before = read_spec_counters(llm)

    for s in range(0, len(items), args.batch_size):
        batch = items[s:s + args.batch_size]
        t0 = time.perf_counter()
        conds = frontend.process_batch([it["ref_wav"] for it in batch],
                                       [it["ref_sr"] for it in batch])
        metrics["frontend_wall_s"] += time.perf_counter() - t0
        prompts = [build_prompt(tokenizer, it["ref_text"], it["target_text"],
                                c.prompt_tokens_llm)
                   for it, c in zip(batch, conds, strict=True)]
        t0 = time.perf_counter()
        outs = llm.generate(
            prompts,
            [make_sampling_params(llm_cfg, args.seed + s + j)
             for j in range(len(batch))],
            use_tqdm=False)
        metrics["llm_wall_s"] += time.perf_counter() - t0

        tokens_list, keep = [], []
        for j, o in enumerate(outs):
            gen = o.outputs[0]
            speech = codec.extract(gen.token_ids)
            metrics["output_tokens"] += len(gen.token_ids)
            metrics["finished_by_stop"] += (gen.finish_reason == "stop")
            if not speech:  # Fail this item without aborting the rest of the batch.
                metrics["failed"].append(batch[j]["uid"])
                continue
            tokens_list.append(speech)
            keep.append(j)

        t0 = time.perf_counter()
        wavs = token2wav.offline_batch(
            tokens_list, [conds[j] for j in keep],
            max_batch=token2wav_cfg.batch_size)
        metrics["token2wav_wall_s"] += time.perf_counter() - t0

        for j, wav in zip(keep, wavs, strict=True):
            sf.write(
                os.path.join(args.output_dir, f"{batch[j]['uid']}.wav"),
                wav.float().squeeze(0).numpy(), 24000)
            expected[batch[j]["uid"]] = batch[j]["target_text"]
        print(f"[{s + len(batch)}/{len(items)}] done")

    spec_after = read_spec_counters(llm)
    drafts = spec_after.get("vllm:spec_decode_num_drafts", 0) - \
        spec_before.get("vllm:spec_decode_num_drafts", 0)
    accepted = spec_after.get("vllm:spec_decode_num_accepted_tokens", 0) - \
        spec_before.get("vllm:spec_decode_num_accepted_tokens", 0)
    draft_toks = spec_after.get("vllm:spec_decode_num_draft_tokens", 0) - \
        spec_before.get("vllm:spec_decode_num_draft_tokens", 0)

    audio_s = metrics["output_tokens"] / TOKENS_PER_SECOND
    wall = metrics["llm_wall_s"] + metrics["token2wav_wall_s"]
    summary = dict(
        num_items=len(items), failed=metrics["failed"],
        finished_by_stop=metrics["finished_by_stop"],
        llm_wall_s=round(metrics["llm_wall_s"], 2),
        llm_tok_per_s=round(metrics["output_tokens"]
                            / max(metrics["llm_wall_s"], 1e-9), 1),
        token2wav_wall_s=round(metrics["token2wav_wall_s"], 2),
        frontend_wall_s=round(metrics["frontend_wall_s"], 2),
        audio_seconds=round(audio_s, 1),
        # rtf = (LLM + token2wav) / audio; frontend time is reported separately.
        rtf=round(wall / max(audio_s, 1e-9), 4))
    if drafts:
        summary["spec"] = dict(
            acceptance_rate=round(accepted / max(draft_toks, 1), 4),
            mean_acceptance_len=round(1 + accepted / drafts, 3))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "expected.json"), "w") as f:
        json.dump(expected, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
