from faster_cosyvoice.token2wav.builders import build_flow, build_hift


def test_build_flow_hyperparams():
    flow = build_flow()
    assert flow.pre_lookahead_len == 3
    assert flow.token_mel_ratio == 2
    assert flow.decoder.estimator.dim == 1024
    assert len(flow.decoder.estimator.transformer_blocks) == 22


def test_build_hift_constructs():
    hift = build_hift()
    assert hift is not None
