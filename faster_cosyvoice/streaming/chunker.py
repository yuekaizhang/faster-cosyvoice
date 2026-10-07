# faster_cosyvoice/streaming/chunker.py
"""token→chunk 纯数学状态机（vllm-omni talker2code2wav_async_chunk 语义，spec §5.3）。

lookahead 只重发不消费；pad 仅首块；hop 逐块 ×2 封顶 4×chunk_size；
LLM 结束后余量一次 finalize（不要求 lookahead，哪怕不足一个 hop）。
常数 15/3/×2/60 出处：vllm-omni deploy/cosyvoice3.yaml codec_chunk_frames=15。
"""
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkPlan:
    prefix_len: int    # flow 输入 = tokens[:prefix_len]（非 final 含 lookahead）
    token_offset: int  # 本块之前已消费 token 数 = mel 切片起点（×token_mel_ratio）
    finalize: bool


class ChunkPlanner:
    def __init__(self, prompt_token_len: int, chunk_size: int = 15,
                 pre_lookahead: int = 3, scale: int = 2,
                 max_hop: Optional[int] = None):
        self.chunk_size = chunk_size
        self.lookahead = pre_lookahead
        self.scale = scale
        self.max_hop = max_hop if max_hop is not None else 4 * chunk_size
        self.hop = chunk_size
        self.pad = ((math.ceil(prompt_token_len / chunk_size) * chunk_size
                     - prompt_token_len) if prompt_token_len > 0 else 0)
        self.emitted = 0  # 已消费 token 数

    def next_chunk(self, available_total: int,
                   finished: bool) -> Optional[ChunkPlan]:
        """None 有两义：finished=False 时表示"等更多 token"；
        finished=True 时表示"没有余量了"。调用方以
        `finished and plan is None` 作为循环退出条件。"""
        available = available_total - self.emitted
        this_hop = self.hop + (self.pad if self.emitted == 0 else 0)
        if not finished:
            if available < this_hop + self.lookahead:
                return None
            plan = ChunkPlan(self.emitted + this_hop + self.lookahead,
                             self.emitted, False)
            self.emitted += this_hop
            self.hop = min(self.max_hop,
                           max(self.chunk_size, self.hop * self.scale))
            return plan
        if available <= 0:
            return None
        plan = ChunkPlan(available_total, self.emitted, True)
        self.emitted = available_total
        return plan
