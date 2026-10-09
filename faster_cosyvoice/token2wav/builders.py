"""Construct Flow and HiFT from the released CosyVoice3 model parameters.

Building the two inference modules directly avoids HyperPyYAML instantiating
unneeded training, dataset, and LLM objects.
"""
from omegaconf import DictConfig

from faster_cosyvoice.token2wav.cosyvoice.flow.DiT.dit import DiT
from faster_cosyvoice.token2wav.cosyvoice.flow.flow import CausalMaskedDiffWithDiT
from faster_cosyvoice.token2wav.cosyvoice.flow.flow_matching import CausalConditionalCFM
from faster_cosyvoice.token2wav.cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
from faster_cosyvoice.token2wav.cosyvoice.hifigan.generator import CausalHiFTGenerator
from faster_cosyvoice.token2wav.cosyvoice.transformer.upsample_encoder import PreLookaheadLayer


def build_flow() -> CausalMaskedDiffWithDiT:
    estimator = DiT(
        dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
        mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
        static_chunk_size=50,          # chunk_size(25) * token_mel_ratio(2)
        num_decoding_left_chunks=-1)
    decoder = CausalConditionalCFM(
        in_channels=240, n_spks=1, spk_emb_dim=80,
        cfm_params=DictConfig({
            "sigma_min": 1e-06, "solver": "euler", "t_scheduler": "cosine",
            "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7,
            "reg_loss_type": "l1"}),
        estimator=estimator)
    return CausalMaskedDiffWithDiT(
        input_size=80, output_size=80, spk_embed_dim=192, output_type="mel",
        vocab_size=6561, input_frame_rate=25, only_mask_loss=True,
        token_mel_ratio=2, pre_lookahead_len=3,
        pre_lookahead_layer=PreLookaheadLayer(
            in_channels=80, channels=1024, pre_lookahead_len=3),
        decoder=decoder)


def build_hift() -> CausalHiFTGenerator:
    return CausalHiFTGenerator(
        in_channels=80, base_channels=512, nb_harmonics=8, sampling_rate=24000,
        nsf_alpha=0.1, nsf_sigma=0.003, nsf_voiced_threshold=10,
        upsample_rates=[8, 5, 3], upsample_kernel_sizes=[16, 11, 7],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1, audio_limit=0.99, conv_pre_look_right=4,
        f0_predictor=CausalConvRNNF0Predictor(
            num_class=1, in_channels=80, cond_channels=512))
