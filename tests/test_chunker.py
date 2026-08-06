# tests/test_chunker.py
from faster_cosyvoice.streaming.chunker import ChunkPlanner


def test_no_pad_sequence():
    """prompt 75（15 的倍数 → pad=0）：首块需 15+3=18 可用。"""
    p = ChunkPlanner(prompt_token_len=75)
    assert p.next_chunk(17, finished=False) is None
    c1 = p.next_chunk(18, finished=False)
    assert (c1.prefix_len, c1.token_offset, c1.finalize) == (18, 0, False)
    # hop 翻倍→30：下一块需 15+30+3=48 可用
    assert p.next_chunk(47, finished=False) is None
    c2 = p.next_chunk(48, finished=False)
    assert (c2.prefix_len, c2.token_offset, c2.finalize) == (48, 15, False)
    # hop→60（封顶）：需 45+60+3=108
    c3 = p.next_chunk(108, finished=False)
    assert (c3.prefix_len, c3.token_offset, c3.finalize) == (108, 45, False)
    # hop 保持 60
    c4 = p.next_chunk(168, finished=False)
    assert (c4.prefix_len, c4.token_offset, c4.finalize) == (168, 105, False)


def test_pad_applies_to_first_chunk_only():
    """prompt 71 → pad=4：首块需 15+4+3=22。"""
    p = ChunkPlanner(prompt_token_len=71)
    assert p.next_chunk(21, finished=False) is None
    c1 = p.next_chunk(22, finished=False)
    assert (c1.prefix_len, c1.token_offset, c1.finalize) == (22, 0, False)
    # 消费 19（含 pad），第二块 hop=30 不再加 pad：需 19+30+3=52
    c2 = p.next_chunk(52, finished=False)
    assert (c2.prefix_len, c2.token_offset, c2.finalize) == (52, 19, False)


def test_finalize_flushes_remainder():
    p = ChunkPlanner(prompt_token_len=75)
    p.next_chunk(18, finished=False)          # 消费 15
    c = p.next_chunk(20, finished=True)       # 余 5 个，不足 hop 也 flush
    assert (c.prefix_len, c.token_offset, c.finalize) == (20, 15, True)
    assert p.next_chunk(20, finished=True) is None  # 无余量


def test_finished_with_nothing_left():
    p = ChunkPlanner(prompt_token_len=0)
    assert p.next_chunk(0, finished=True) is None


def test_zero_prompt_no_pad():
    p = ChunkPlanner(prompt_token_len=0)
    c = p.next_chunk(18, finished=False)
    assert (c.prefix_len, c.token_offset, c.finalize) == (18, 0, False)
