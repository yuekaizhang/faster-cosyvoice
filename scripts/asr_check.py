# scripts/asr_check.py
"""ASR 质量门（spec D7/§8）：对 results 目录逐 wav 做中文 ASR，与期望文本算 CER。

用法：
  python scripts/asr_check.py --wav-dir results/smoke \
      --ref-json results/smoke/expected.json \
      --paraformer-dir $PARAFORMER_DIR --cer-threshold 0.15

expected.json 格式：{"<uid>": "<期望文本>", ...}（offline_inference.py 自动落盘）。
依赖：pip install sherpa-onnx；paraformer 模型下载见本文件底部注释。
"""
import argparse
import json
import os


def cer(ref: str, hyp: str) -> float:
    r, h = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    dp = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(h) + 1):
            cur = min(dp[j] + 1, dp[j - 1] + 1,
                      prev + (r[i - 1] != h[j - 1]))
            prev, dp[j] = dp[j], cur
    return dp[len(h)] / max(len(r), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav-dir", required=True)
    p.add_argument("--ref-json", required=True)
    p.add_argument("--paraformer-dir", required=True,
                   help="含 model.int8.onnx 与 tokens.txt 的目录")
    p.add_argument("--cer-threshold", type=float, default=0.15)
    args = p.parse_args()

    import sherpa_onnx
    import soundfile as sf
    rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=os.path.join(args.paraformer_dir, "model.int8.onnx"),
        tokens=os.path.join(args.paraformer_dir, "tokens.txt"))

    refs = json.load(open(args.ref_json))
    bad, results = [], {}
    for uid, ref_text in refs.items():
        path = os.path.join(args.wav_dir, f"{uid}.wav")
        audio, sr = sf.read(path, dtype="float32")
        s = rec.create_stream()
        s.accept_waveform(sr, audio)
        rec.decode_stream(s)
        c = cer(ref_text, s.result.text)
        results[uid] = dict(cer=round(c, 4), hyp=s.result.text)
        if c > args.cer_threshold:
            bad.append(uid)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if bad:
        raise SystemExit(f"ASR 门未过（CER>{args.cer_threshold}）：{bad}")
    print(f"ASR 门通过：{len(refs)} 条，阈值 {args.cer_threshold}")


# paraformer 模型（一次性）:
# wget -qO- https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2 | tar xj

if __name__ == "__main__":
    main()
