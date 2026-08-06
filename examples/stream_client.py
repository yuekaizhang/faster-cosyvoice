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
            if ttfa is None and len(data) + len(chunk) > 44:  # 首个音频字节
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
