#!/usr/bin/env python3
"""Batch CPU Paraformer ASR and corpus CER for generated Chinese WAV files.

Example:
  python scripts/asr_check.py --wav-dir results/seedtts-zh/draft \
      --ref-json results/seedtts-zh/draft/expected.json \
      --paraformer-dir models/sherpa-onnx-paraformer-zh-2023-09-14 \
      --batch-size 32 --num-threads 16 --no-fail

``expected.json`` maps ``uid`` to ``target_text`` and is written by
``examples/offline_inference.py``. ASR runs on CPU explicitly. Text is NFKC
normalized, lowercased, and stripped of Unicode punctuation and whitespace
before character-level Levenshtein scoring. The primary number is corpus CER:
sum(edit distance) / sum(reference characters).
"""

import argparse
import json
import time
import unicodedata
from pathlib import Path


def normalize_zh(text: str | None) -> str:
    """Normalize Mandarin text for character-level Seed-TTS scoring."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(
        ch for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def edit_distance(ref: str, hyp: str) -> int:
    """Levenshtein distance using O(len(hyp)) memory."""
    dp = list(range(len(hyp) + 1))
    for i, ref_ch in enumerate(ref, 1):
        prev, dp[0] = dp[0], i
        for j, hyp_ch in enumerate(hyp, 1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (ref_ch != hyp_ch),
            )
            prev = old
    return dp[-1]


def cer(ref: str, hyp: str) -> float:
    ref_norm, hyp_norm = normalize_zh(ref), normalize_zh(hyp)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    return edit_distance(ref_norm, hyp_norm) / len(ref_norm)


def nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, (len(ordered) * int(percentile) + 99) // 100)
    return ordered[min(rank, len(ordered)) - 1]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", required=True)
    parser.add_argument("--ref-json", required=True)
    parser.add_argument(
        "--paraformer-dir",
        required=True,
        help="Directory containing model.int8.onnx and tokens.txt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cer-threshold", type=float, default=0.15)
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Report threshold failures without returning a non-zero status",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Default: <wav-dir>/asr_results.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Default: <wav-dir>/asr_summary.json",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_threads < 1:
        parser.error("--batch-size and --num-threads must be positive")
    return args


def main() -> None:
    args = get_args()
    wav_dir = Path(args.wav_dir)
    output_jsonl = Path(args.output_jsonl or wav_dir / "asr_results.jsonl")
    summary_json = Path(args.summary_json or wav_dir / "asr_summary.json")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    refs: dict[str, str] = json.loads(Path(args.ref_json).read_text())
    ref_items = list(refs.items())
    if args.limit is not None:
        ref_items = ref_items[: args.limit]
    missing = [uid for uid, _ in ref_items if not (wav_dir / f"{uid}.wav").is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} WAV files; first: {missing[:5]}")

    import sherpa_onnx
    import soundfile as sf

    model_start = time.perf_counter()
    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=str(Path(args.paraformer_dir) / "model.int8.onnx"),
        tokens=str(Path(args.paraformer_dir) / "tokens.txt"),
        num_threads=args.num_threads,
        provider="cpu",
    )
    model_load_s = time.perf_counter() - model_start

    rows: list[dict] = []
    total_audio_s = 0.0
    decode_wall_s = 0.0
    eval_start = time.perf_counter()
    for start in range(0, len(ref_items), args.batch_size):
        batch = ref_items[start:start + args.batch_size]
        streams = []
        metadata = []
        for uid, ref_text in batch:
            wav_path = wav_dir / f"{uid}.wav"
            audio, sample_rate = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            stream = recognizer.create_stream()
            # sherpa-onnx resamples 24 kHz input to Paraformer's 16 kHz internally.
            stream.accept_waveform(sample_rate, audio)
            streams.append(stream)
            audio_s = len(audio) / sample_rate
            total_audio_s += audio_s
            metadata.append((uid, ref_text, wav_path, audio_s))

        decode_start = time.perf_counter()
        recognizer.decode_streams(streams)
        decode_wall_s += time.perf_counter() - decode_start

        for stream, (uid, ref_text, wav_path, audio_s) in zip(
            streams, metadata, strict=True
        ):
            hyp_text = stream.result.text
            ref_norm = normalize_zh(ref_text)
            hyp_norm = normalize_zh(hyp_text)
            errors = edit_distance(ref_norm, hyp_norm)
            item_cer = errors / max(len(ref_norm), 1)
            rows.append({
                "uid": uid,
                "wav_path": str(wav_path),
                "audio_seconds": round(audio_s, 6),
                "reference": ref_text,
                "hypothesis": hyp_text,
                "reference_normalized": ref_norm,
                "hypothesis_normalized": hyp_norm,
                "reference_characters": len(ref_norm),
                "errors": errors,
                "cer": round(item_cer, 8),
            })
        print(f"[{min(start + len(batch), len(ref_items))}/{len(ref_items)}] ASR done")

    asr_wall_s = time.perf_counter() - eval_start
    total_ref_chars = sum(row["reference_characters"] for row in rows)
    total_errors = sum(row["errors"] for row in rows)
    item_cers = [row["cer"] for row in rows]
    corpus_cer = total_errors / max(total_ref_chars, 1)
    macro_cer = sum(item_cers) / max(len(item_cers), 1)
    above_threshold = sum(row["cer"] > args.cer_threshold for row in rows)
    exact_matches = sum(row["errors"] == 0 for row in rows)

    summary = {
        "schema_version": 1,
        "num_items": len(rows),
        "asr": {
            "library": "sherpa-onnx",
            "model": str(Path(args.paraformer_dir) / "model.int8.onnx"),
            "provider": "cpu",
            "batch_size": args.batch_size,
            "num_threads": args.num_threads,
            "model_load_seconds": round(model_load_s, 4),
            "wall_seconds": round(asr_wall_s, 4),
            "decode_wall_seconds": round(decode_wall_s, 4),
            "audio_seconds": round(total_audio_s, 4),
            "rtf": round(asr_wall_s / max(total_audio_s, 1e-9), 6),
        },
        "normalization": "NFKC + lowercase + remove Unicode punctuation/whitespace",
        "total_reference_characters": total_ref_chars,
        "total_character_errors": total_errors,
        "corpus_cer": round(corpus_cer, 8),
        # Official Chinese Seed-TTS scripts space-separate characters then report WER.
        # Under that protocol this value is identical to corpus CER.
        "seedtts_zh_character_wer": round(corpus_cer, 8),
        "mean_utterance_cer": round(macro_cer, 8),
        "utterance_cer_p50": round(nearest_rank(item_cers, 50) or 0.0, 8),
        "utterance_cer_p90": round(nearest_rank(item_cers, 90) or 0.0, 8),
        "utterance_cer_p95": round(nearest_rank(item_cers, 95) or 0.0, 8),
        "utterance_cer_p99": round(nearest_rank(item_cers, 99) or 0.0, 8),
        "exact_matches": exact_matches,
        "cer_threshold": args.cer_threshold,
        "above_threshold": above_threshold,
    }

    with output_jsonl.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if above_threshold and not args.no_fail:
        raise SystemExit(
            f"ASR gate failed: {above_threshold}/{len(rows)} utterances "
            f"have CER > {args.cer_threshold}"
        )


if __name__ == "__main__":
    main()
