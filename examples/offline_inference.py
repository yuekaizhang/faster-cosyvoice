# examples/offline_inference.py
"""Offline voice-clone 推理（spec §5.5/§6.1）。

单条：
  python examples/offline_inference.py \
      --ref-audio ref.wav --ref-text "参考文本" --target-text "目标文本" \
      --output-dir results/single

数据集：
  python examples/offline_inference.py \
      --dataset yuekai/seed_tts_cosy2 --split wenetspeech4tts \
      --batch-size 8 --output-dir results/wenetspeech4tts

关闭投机解码：--draft-model none；torch estimator 降级：--estimator torch
"""
import argparse
import json
import os
import sys
import time

# 本环境 torchaudio 2.9 的 load/save 委托 torchcodec（缺 ffmpeg 共享库），改用 soundfile
import soundfile as sf
import torch

from faster_cosyvoice.assets import ensure_token2wav_assets
from faster_cosyvoice.config import LLMConfig, Token2WavConfig
from faster_cosyvoice.envcheck import check_environment
from faster_cosyvoice.llm.engine import (create_offline_llm, make_sampling_params,
                                         read_spec_counters)
from faster_cosyvoice.llm.prompt import build_prompt
from faster_cosyvoice.llm.tokens import SpeechTokenCodec
from faster_cosyvoice.token2wav.frontend import RefAudioFrontend
from faster_cosyvoice.token2wav.token2wav import CosyVoice3Token2Wav

TOKENS_PER_SECOND = 25  # 25 speech token = 1s 音频


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ref-audio"); p.add_argument("--ref-text")
    p.add_argument("--target-text")
    p.add_argument("--dataset"); p.add_argument("--subset", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--target-model",
                   default="yuekai/Fun-CosyVoice3-0.5B-2512-LLM-HF")
    p.add_argument("--draft-model", default="yuekai/cosyvoice3_llm_dspark",
                   help="'none' 关闭投机解码")
    p.add_argument("--token2wav-dir", default="models/Fun-CosyVoice3-0.5B-2512")
    p.add_argument("--estimator", default="flashinfer",
                   choices=["flashinfer", "torch"])
    p.add_argument("--token2wav-batch-size", type=int, default=8)
    p.add_argument("--output-dir", default="results/offline")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_items(args):
    """→ list of dict(ref_wav: 1-D tensor, ref_sr, ref_text, target_text, uid)"""
    if args.dataset:
        import io

        from datasets import Audio, load_dataset
        ds = load_dataset(args.dataset, args.subset, split=args.split)
        # datasets>=5 的 Audio 解码依赖 torchcodec/ffmpeg，这里用 soundfile 自行解码
        ds = ds.cast_column("prompt_audio", Audio(decode=False))
        if args.limit:
            ds = ds.select(range(min(args.limit, len(ds))))
        items = []
        for i, row in enumerate(ds):
            audio = row["prompt_audio"]
            array, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
            items.append(dict(
                ref_wav=torch.tensor(array, dtype=torch.float32),
                ref_sr=sr,
                ref_text=row["prompt_text"], target_text=row["target_text"],
                uid=row.get("id", f"{i:05d}")))
        return items
    assert args.ref_audio and args.ref_text and args.target_text, \
        "单条模式需要 --ref-audio/--ref-text/--target-text"
    array, sr = sf.read(args.ref_audio, dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)
    return [dict(ref_wav=torch.tensor(array, dtype=torch.float32),
                 ref_sr=sr, ref_text=args.ref_text,
                 target_text=args.target_text, uid="single")]


def main():
    args = get_args()
    draft = None if args.draft_model in (None, "none") else args.draft_model
    llm_cfg = LLMConfig(target_model=args.target_model, draft_model=draft)
    t2w_cfg = Token2WavConfig(model_dir=args.token2wav_dir,
                              estimator_mode=args.estimator,
                              batch_size=args.token2wav_batch_size)

    problems = check_environment(
        require_flashinfer=(args.estimator == "flashinfer"),
        require_draft_mirror=(draft is not None
                              and llm_cfg.repetition_penalty != 1.0))
    if problems:
        print("环境自检失败：\n  - " + "\n  - ".join(problems)); sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = ensure_token2wav_assets(t2w_cfg.model_dir)
    items = load_items(args)
    print(f"{len(items)} 条请求")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    codec = SpeechTokenCodec(tokenizer)
    llm = create_offline_llm(llm_cfg)
    frontend = RefAudioFrontend(f"{model_dir}/campplus.onnx",
                                device=t2w_cfg.device)
    token2wav = CosyVoice3Token2Wav(model_dir, device=t2w_cfg.device,
                                    estimator_mode=t2w_cfg.estimator_mode)

    metrics = dict(llm_wall_s=0.0, t2w_wall_s=0.0, output_tokens=0,
                   finished_by_stop=0, failed=[])
    expected = {}
    spec_before = read_spec_counters(llm)

    for s in range(0, len(items), args.batch_size):
        batch = items[s:s + args.batch_size]
        conds = frontend.process_batch([it["ref_wav"] for it in batch],
                                       [it["ref_sr"] for it in batch])
        prompts = [build_prompt(tokenizer, it["ref_text"], it["target_text"],
                                c.prompt_tokens_llm)
                   for it, c in zip(batch, conds)]
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
            if not speech:  # spec §7：0 个有效 token 记失败不中断整批
                metrics["failed"].append(batch[j]["uid"]); continue
            tokens_list.append(speech); keep.append(j)

        t0 = time.perf_counter()
        wavs = token2wav.offline_batch(
            tokens_list, [conds[j] for j in keep],
            max_batch=t2w_cfg.batch_size)
        metrics["t2w_wall_s"] += time.perf_counter() - t0

        for j, wav in zip(keep, wavs):
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
    wall = metrics["llm_wall_s"] + metrics["t2w_wall_s"]
    summary = dict(
        num_items=len(items), failed=metrics["failed"],
        finished_by_stop=metrics["finished_by_stop"],
        llm_wall_s=round(metrics["llm_wall_s"], 2),
        llm_tok_per_s=round(metrics["output_tokens"]
                            / max(metrics["llm_wall_s"], 1e-9), 1),
        t2w_wall_s=round(metrics["t2w_wall_s"], 2),
        audio_seconds=round(audio_s, 1),
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
