"""Ref 音频前端：s3tokenizer(v3_25hz, GPU) + campplus(ORT-CPU) + prompt mel。

RefCondition 同时服务 LLM（未截断 token）与 flow（2:1 截断的 token/feat 对）。
speaker cache key = sha256(音频字节)+ref_text（修 triton 版按 ref_text 碰撞的缺陷，
spec §5.2）。M1 offline 每行 ref 各不相同，cache 主要为 M2 server 铺路。
"""
import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from faster_cosyvoice.token2wav.matcha_audio import mel_spectrogram

_mel_fn = partial(mel_spectrogram, n_fft=1920, num_mels=80, sampling_rate=24000,
                  hop_size=480, win_size=1920, fmin=0, fmax=None, center=False)


@dataclass
class RefCondition:
    prompt_tokens_llm: list      # 未截断，进 LLM prompt
    prompt_tokens_flow: list     # 2:1 截断，进 flow
    prompt_feat: torch.Tensor    # (1, 2*len(prompt_tokens_flow), 80)，cpu fp32
    spk_embedding: torch.Tensor  # (192,)，cpu fp32


def truncate_2to1(tokens: list, feat_len: int) -> tuple:
    """prompt feat/token 对齐到 2:1（同 duplex forward() 与 triton BLS）。"""
    token_len = min(feat_len // 2, len(tokens))
    return tokens[:token_len], 2 * token_len


class SpeakerCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._d: OrderedDict = OrderedDict()

    @staticmethod
    def make_key(wav_16k: torch.Tensor, ref_text: str) -> str:
        h = hashlib.sha256(wav_16k.cpu().numpy().tobytes())
        h.update(ref_text.encode())
        return h.hexdigest()

    def get(self, key: str):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key: str, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.max_size:
            self._d.popitem(last=False)


class RefAudioFrontend:
    def __init__(self, campplus_onnx_path: str, device: str = "cuda:0",
                 cache_size: int = 256, campplus_trt: bool = False):
        self.device = device
        # campplus TRT（opt-in；默认 ORT-CPU）。放最前：import/build 失败要在
        # 任何重加载前 fail loud（campplus_trt._import_trt 给出可操作的报错）。
        self._campplus_trt = None
        if campplus_trt:
            from faster_cosyvoice.token2wav import campplus_trt as _ctrt
            device_id = torch.device(device).index or 0
            plan_path = os.path.join(
                os.path.dirname(campplus_onnx_path),
                f"campplus.{device_id}.fp32.plan")
            self._campplus_trt = _ctrt.load_campplus_trt(
                campplus_onnx_path, plan_path, device=device)
            self._spk_embedding_trt = _ctrt.spk_embedding_trt
        import onnxruntime
        import s3tokenizer
        self.audio_tokenizer = s3tokenizer.load_model(
            "speech_tokenizer_v3_25hz").to(device).eval()
        self._s3 = s3tokenizer
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = \
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.spk_model = onnxruntime.InferenceSession(
            campplus_onnx_path, sess_options=option,
            providers=["CPUExecutionProvider"])
        self.cache = SpeakerCache(cache_size)

    @torch.inference_mode()
    def process_batch(self, wavs: list, sample_rates: list) -> list:
        """wavs: list of 1-D float tensor（任意采样率）→ list[RefCondition]。"""
        wavs_16k = [self._resample(w, sr, 16000) for w, sr in zip(wavs, sample_rates)]
        wavs_24k = [self._resample(w, sr, 24000) for w, sr in zip(wavs, sample_rates)]

        # s3tokenizer 批量（唯一真批量的前端模块）
        mels = [self._s3.log_mel_spectrogram(w) for w in wavs_16k]
        mels_pad, mels_lens = self._s3.padding(mels)
        tokens_pad, tokens_lens = self.audio_tokenizer.quantize(
            mels_pad.to(self.device), mels_lens.to(self.device))

        conds = []
        for i, (w16, w24) in enumerate(zip(wavs_16k, wavs_24k)):
            tokens = tokens_pad[i, :tokens_lens[i].item()].tolist()
            mel = _mel_fn(w24.unsqueeze(0)).transpose(1, 2)  # (1, T, 80)
            spk = self._spk_embedding(w16)
            flow_tokens, feat_len = truncate_2to1(tokens, mel.shape[1])
            conds.append(RefCondition(
                prompt_tokens_llm=tokens,
                prompt_tokens_flow=flow_tokens,
                prompt_feat=mel[:, :feat_len].float(),
                spk_embedding=spk))
        return conds

    def process(self, wav: torch.Tensor, sample_rate: int,
                ref_text: str = "") -> RefCondition:
        key = SpeakerCache.make_key(
            self._resample(wav, sample_rate, 16000), ref_text)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        cond = self.process_batch([wav], [sample_rate])[0]
        self.cache.put(key, cond)
        return cond

    def _spk_embedding(self, wav_16k: torch.Tensor) -> torch.Tensor:
        feat = kaldi.fbank(wav_16k.unsqueeze(0), num_mel_bins=80, dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        if self._campplus_trt is not None:
            return self._spk_embedding_trt(
                self._campplus_trt, feat.to(self.device))
        emb = self.spk_model.run(
            None, {self.spk_model.get_inputs()[0].name:
                   feat.unsqueeze(0).cpu().numpy()})[0]
        return torch.from_numpy(emb).flatten().float()

    @staticmethod
    def _resample(wav: torch.Tensor, sr: int, target: int) -> torch.Tensor:
        assert wav.dim() == 1
        if sr == target:
            return wav
        return torchaudio.transforms.Resample(sr, target)(wav.unsqueeze(0)).squeeze(0)
