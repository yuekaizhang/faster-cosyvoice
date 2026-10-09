"""Queue-selection policies for streaming Token2Wav work.

The policy is intentionally separate from :class:`Token2WavWorker`: it decides
which ready chunk runs next but does not know how GPU work is executed.  This
makes the deadline-aware ordering reusable by another worker implementation.
"""

import asyncio
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class PendingChunk:
    """One queued token-to-wave chunk and the future awaiting its PCM."""

    sequence: int
    chunk_index: int
    session: Any
    plan: Any
    future: asyncio.Future


Candidate = tuple[int, PendingChunk]


class SchedulingPolicy(Protocol):
    """Select an ``(index, job)`` pair from ready queue candidates."""

    def select(self, candidates: Sequence[Candidate]) -> Candidate: ...


class LegacyChunkPolicy:
    """Prefer the lowest chunk index, then FIFO admission order."""

    def select(self, candidates: Sequence[Candidate]) -> Candidate:
        return min(
            candidates,
            key=lambda item: (item[1].chunk_index, item[1].sequence),
        )


class DeadlineAwarePolicy:
    """Protect playback continuity while retaining low startup latency.

    Selection order is:

    1. established streams within ``reserve_s`` of exhausting buffered audio;
    2. streams that have not emitted their first PCM chunk;
    3. established streams by earliest playback deadline;
    4. FIFO as a final fallback.

    A stream publishes its playback start time and emitted duration only after
    PCM crosses the HTTP serving boundary.  Their sum is the time at which the
    client is expected to run out of buffered audio.
    """

    def __init__(
        self,
        reserve_s: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(reserve_s) or reserve_s < 0:
            raise ValueError("reserve_s must be finite and non-negative")
        self.reserve_s = reserve_s
        self.clock = clock

    @staticmethod
    def playback_deadline(job: PendingChunk) -> float | None:
        started_at = getattr(job.session, "playback_started_at_s", None)
        if started_at is None:
            return None
        emitted_duration = getattr(job.session, "emitted_duration_s", 0.0)
        return float(started_at) + float(emitted_duration)

    @staticmethod
    def is_startup(job: PendingChunk) -> bool:
        # StreamSession exposes route-boundary state.  The chunk-index fallback
        # keeps small test doubles and third-party callers compatible.
        if hasattr(job.session, "playback_started_at_s"):
            return job.session.playback_started_at_s is None
        return job.chunk_index == 0

    def select(self, candidates: Sequence[Candidate]) -> Candidate:
        now = self.clock()
        established = [
            (index, job, self.playback_deadline(job))
            for index, job in candidates
            if not self.is_startup(job)
        ]
        urgent = [
            (index, job, deadline)
            for index, job, deadline in established
            if deadline is not None and now >= deadline - self.reserve_s
        ]
        if urgent:
            index, job, _ = min(
                urgent,
                key=lambda item: (item[2], item[1].sequence),
            )
            return index, job

        startup = [
            (index, job) for index, job in candidates if self.is_startup(job)
        ]
        if startup:
            return min(startup, key=lambda item: item[1].sequence)

        if established:
            index, job, _ = min(
                established,
                key=lambda item: (
                    float("inf") if item[2] is None else item[2],
                    item[1].sequence,
                ),
            )
            return index, job

        return min(candidates, key=lambda item: item[1].sequence)


def create_scheduling_policy(
    mode: str,
    reserve_s: float = 0.1,
    clock: Callable[[], float] = time.monotonic,
) -> SchedulingPolicy:
    """Create one of the public scheduling policies by CLI/config name."""
    if not math.isfinite(reserve_s) or reserve_s < 0:
        raise ValueError("reserve_s must be finite and non-negative")
    if mode == "deadline":
        return DeadlineAwarePolicy(reserve_s=reserve_s, clock=clock)
    if mode == "legacy":
        return LegacyChunkPolicy()
    raise ValueError("scheduler mode must be 'deadline' or 'legacy'")
