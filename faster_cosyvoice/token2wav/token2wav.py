"""CosyVoice3Token2Wav：flow+hift 持有者，offline 批量入口（spec §5.2）。

保持 .flow/.hift/.device/.fp16 属性名与 duplex CosyVoice3_Token2Wav 一致，
使 flashinfer_dit.apply_flashinfer 不改即可用。
"""
import torch

from faster_cosyvoice.token2wav.builders import build_flow, build_hift


class CosyVoice3Token2Wav(torch.nn.Module):
    def __init__(self, model_dir: str, device: str = "cuda:0",
                 estimator_mode: str = "flashinfer"):
        super().__init__()
        self.device = device
        self.fp16 = False
        self.flow = build_flow()
        self.flow.load_state_dict(
            torch.load(f"{model_dir}/flow.pt", map_location="cpu",
                       weights_only=True), strict=True)
        self.flow.to(device).eval()
        self.hift = build_hift()
        hift_sd = {k.replace("generator.", ""): v for k, v in torch.load(
            f"{model_dir}/hift.pt", map_location="cpu",
            weights_only=True).items()}
        self.hift.load_state_dict(hift_sd, strict=True)
        self.hift.to(device).eval()

        assert estimator_mode in ("flashinfer", "torch"), estimator_mode
        self.estimator_mode = estimator_mode
        if estimator_mode == "flashinfer":
            from faster_cosyvoice.token2wav.flashinfer_dit import apply_flashinfer
            apply_flashinfer(self)  # flow→fp16、estimator 替换、self.fp16=True

    @torch.inference_mode()
    def offline_batch(self, generated_tokens_list: list, conds: list,
                      max_batch: int = 8) -> list:
        """全量 token → wav（24kHz）。内部按 max_batch 分段防 OOM（spec §5.2）。"""
        wavs = []
        for s in range(0, len(generated_tokens_list), max_batch):
            wavs.extend(self._batch_impl(
                generated_tokens_list[s:s + max_batch], conds[s:s + max_batch]))
        return wavs

    def _batch_impl(self, tokens_list, conds):
        if self.estimator_mode == "flashinfer":
            from faster_cosyvoice.token2wav.flashinfer_dit import flow_inference_batched
            token_list = [c.prompt_tokens_flow + list(t)
                          for c, t in zip(conds, tokens_list)]
            prompt_feat_list = [c.prompt_feat.to(self.device) for c in conds]
            emb = torch.stack([c.spk_embedding for c in conds]).to(self.device)
            mels = flow_inference_batched(
                self.flow, token_list, prompt_feat_list, emb)
        else:
            mels = [self._flow_single(t, c) for t, c in zip(tokens_list, conds)]
        return [self.hift.inference(speech_feat=mel, finalize=True)[0].cpu()
                for mel in mels]

    def _flow_single(self, tokens, cond):
        """torch 逐条路径（同 duplex forward_flow）。"""
        token = torch.tensor([list(tokens)], device=self.device)
        prompt_token = torch.tensor([cond.prompt_tokens_flow], device=self.device)
        prompt_feat = cond.prompt_feat.to(self.device)
        embedding = cond.spk_embedding.unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast(self.fp16):
            mel, _ = self.flow.inference(
                token=token,
                token_len=torch.tensor([token.shape[1]], device=self.device),
                prompt_token=prompt_token,
                prompt_token_len=torch.tensor([prompt_token.shape[1]],
                                              device=self.device),
                prompt_feat=prompt_feat,
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]],
                                             device=self.device),
                embedding=embedding,
                streaming=False, finalize=True)
        return mel

    @torch.inference_mode()
    def stream_step(self, session, plan) -> torch.Tensor:
        """流式单步（spec D3/§5.2；语义同 duplex forward_stream 单步）：
        flow 全前缀重算(streaming=True) → mel 按 token_offset×2 切新段 → 拼
        session.mel_cache → hift 对全量 mel 重跑 → 按 speech_offset 切新音频。
        HiFT 无跨调用状态，共享实例可多 session 交错；重算确定性由
        CausalConditionalCFM 的固定 rand_noise 保证。返回 (1, N) cpu fp32。

        M3 起两种 estimator_mode 都支持：flashinfer 模式（apply_flashinfer 后
        flow 为 fp16、self.fp16=True）经 chunk-causal custom mask 走流式；
        autocast 包装与 _flow_single/duplex forward_stream 对齐（torch 模式
        fp16=False → enabled=False，无行为变化）。flow.inference 返回值恒为
        fp32（vendored flow.py:409 的 feat.float()），故 mel cache / hift /
        speech_offset 逻辑在 autocast 下不变。"""
        assert self.estimator_mode in ("torch", "flashinfer"), \
            self.estimator_mode
        cond = session.cond
        token = torch.tensor([session.tokens[:plan.prefix_len]],
                             device=self.device)
        prompt_token = torch.tensor([cond.prompt_tokens_flow],
                                    device=self.device)
        prompt_feat = cond.prompt_feat.to(self.device)
        embedding = cond.spk_embedding.unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled=self.fp16):
            mel, _ = self.flow.inference(
                token=token,
                token_len=torch.tensor([token.shape[1]], device=self.device),
                prompt_token=prompt_token,
                prompt_token_len=torch.tensor([prompt_token.shape[1]],
                                              device=self.device),
                prompt_feat=prompt_feat,
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]],
                                             device=self.device),
                embedding=embedding,
                streaming=True, finalize=plan.finalize)
        mel = mel[:, :, plan.token_offset * self.flow.token_mel_ratio:]
        if session.mel_cache is not None:
            mel = torch.cat([session.mel_cache, mel], dim=2)
        session.mel_cache = mel
        speech, _ = self.hift.inference(speech_feat=mel,
                                        finalize=plan.finalize)
        new = speech[:, session.speech_offset:]
        session.speech_offset += new.shape[1]
        session.chunk_index += 1
        return new.cpu()
