# tests/test_chunker.py
from faster_cosyvoice.streaming.chunker import ChunkPlanner


def test_no_pad_sequence():
    """A 75-token prompt is aligned, so the first chunk needs 15+3 tokens."""
    p = ChunkPlanner(prompt_token_len=75)
    assert p.next_chunk(17, finished=False) is None
    c1 = p.next_chunk(18, finished=False)
    assert (c1.prefix_len, c1.token_offset, c1.finalize) == (18, 0, False)
    # The hop doubles to 30; the next boundary is 15+30+3=48.
    assert p.next_chunk(47, finished=False) is None
    c2 = p.next_chunk(48, finished=False)
    assert (c2.prefix_len, c2.token_offset, c2.finalize) == (48, 15, False)
    # The hop reaches its cap of 60; the next boundary is 45+60+3=108.
    c3 = p.next_chunk(108, finished=False)
    assert (c3.prefix_len, c3.token_offset, c3.finalize) == (108, 45, False)
    # The hop remains capped at 60.
    c4 = p.next_chunk(168, finished=False)
    assert (c4.prefix_len, c4.token_offset, c4.finalize) == (168, 105, False)


def test_pad_applies_to_first_chunk_only():
    """A 71-token prompt adds four alignment tokens only to the first chunk."""
    p = ChunkPlanner(prompt_token_len=71)
    assert p.next_chunk(21, finished=False) is None
    c1 = p.next_chunk(22, finished=False)
    assert (c1.prefix_len, c1.token_offset, c1.finalize) == (22, 0, False)
    # The first chunk consumes 19; the next 30-token hop does not repeat padding.
    c2 = p.next_chunk(52, finished=False)
    assert (c2.prefix_len, c2.token_offset, c2.finalize) == (52, 19, False)


def test_finalize_flushes_remainder():
    p = ChunkPlanner(prompt_token_len=75)
    p.next_chunk(18, finished=False)          # Consume 15 tokens.
    c = p.next_chunk(20, finished=True)       # Flush the five-token remainder.
    assert (c.prefix_len, c.token_offset, c.finalize) == (20, 15, True)
    assert p.next_chunk(20, finished=True) is None  # Nothing remains.


def test_finished_with_nothing_left():
    p = ChunkPlanner(prompt_token_len=0)
    assert p.next_chunk(0, finished=True) is None


def test_zero_prompt_no_pad():
    p = ChunkPlanner(prompt_token_len=0)
    c = p.next_chunk(18, finished=False)
    assert (c.prefix_len, c.token_offset, c.finalize) == (18, 0, False)


def test_uniform_mode():
    """A growth factor of one keeps 25-token hops after first-chunk padding."""
    p = ChunkPlanner(prompt_token_len=71, chunk_size=25, scale=1)
    assert p.pad == 4
    # The first chunk needs 25+4 alignment+3 lookahead tokens.
    assert p.next_chunk(31, finished=False) is None
    c1 = p.next_chunk(32, finished=False)
    assert (c1.prefix_len, c1.token_offset, c1.finalize) == (32, 0, False)
    # Every later hop remains 25 because scale=1.
    offsets = [29]
    for _ in range(3):
        need = offsets[-1] + 25 + 3
        assert p.next_chunk(need - 1, finished=False) is None
        c = p.next_chunk(need, finished=False)
        assert (c.prefix_len, c.token_offset, c.finalize) == (
            need, offsets[-1], False)
        offsets.append(offsets[-1] + 25)
    # Finalization flushes the remainder.
    c = p.next_chunk(offsets[-1] + 7, finished=True)
    assert (c.prefix_len, c.token_offset, c.finalize) == (
        offsets[-1] + 7, offsets[-1], True)
