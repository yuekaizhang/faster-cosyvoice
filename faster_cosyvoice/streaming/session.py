"""Per-request state for interleaved streaming synthesis.

The Token2Wav model is shared across requests.  Mutable Mel/audio offsets and
playback credit therefore live on each :class:`StreamSession`.
"""
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
    mel_cache: Optional[torch.Tensor] = None  # Generated Mel, shape (1, 80, T).
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
