"""Register a reusable voice for the HTTP server.

Local file:
  uv run python examples/register_voice.py --name demo --ref-audio ref.wav \
      --ref-text "Transcript of the reference audio."

Cached Hugging Face dataset row:
  uv run python examples/register_voice.py --name benchmark \
      --dataset yuekai/seed_tts_cosy2 --split test_en --index 0
"""
import argparse
import base64
import json
from pathlib import Path

import httpx


def _load_reference(args) -> tuple[bytes, str]:
    if args.dataset:
        from datasets import Audio, load_dataset

        dataset = load_dataset(args.dataset, args.subset, split=args.split)
        dataset = dataset.cast_column("prompt_audio", Audio(decode=False))
        row = dataset[args.index]
        audio = row["prompt_audio"]
        data = audio["bytes"]
        if data is None:
            data = Path(audio["path"]).read_bytes()
        return data, row["prompt_text"]

    if not args.ref_audio or not args.ref_text:
        raise SystemExit("local mode requires --ref-audio and --ref-text")
    return Path(args.ref_audio).read_bytes(), args.ref_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--name", required=True)
    parser.add_argument("--ref-audio")
    parser.add_argument("--ref-text")
    parser.add_argument("--dataset")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--split", default="test_en")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    if args.dataset and (args.ref_audio or args.ref_text):
        parser.error("choose either --dataset or --ref-audio/--ref-text")
    data, text = _load_reference(args)
    payload = {
        "name": args.name,
        "ref_audio": "data:audio/wav;base64," + base64.b64encode(data).decode(),
        "ref_text": text,
    }
    response = httpx.post(
        f"{args.url.rstrip('/')}/v1/audio/voices", json=payload, timeout=300
    )
    response.raise_for_status()
    print(json.dumps({**response.json(), "ref_text": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
