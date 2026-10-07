# faster_cosyvoice/streaming/session.py
"""每请求流式状态（spec §5.2/§6.2）。token2wav 模块本身无状态，
多 session 交错依赖本对象承载缓存与 playback credit。"""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from faster_cosyvoice.streaming.chunker import ChunkPlanner

if TYPE_CHECKING:
    from faster_cosyvoice.token2wav.frontend import RefCondition


@dataclass
class StreamSession:
    cond: "RefCondition"
    planner: ChunkPlanner
    tokens: List[int] = field(default_factory=list)
    mel_cache: Optional[torch.Tensor] = None  # (1,80,T) 累计生成 mel（不含 prompt）
    speech_offset: int = 0
    chunk_index: int = 0
    # Nari-style playback credit.  Only publish credit when PCM crosses the
    # serving boundary; the batcher uses it to protect an established stream
    # before the client would run out of buffered audio.
    playback_started_at_s: Optional[float] = None
    emitted_duration_s: float = 0.0

    def mark_pcm_routed(self, sample_count: int, sample_rate: int,
                        routed_at_s: float) -> None:
        if sample_count <= 0:
            return
        if self.playback_started_at_s is None:
            self.playback_started_at_s = routed_at_s
        self.emitted_duration_s += sample_count / sample_rate
