"""Public sample-clock metadata for hardware synchronization."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DigitalEdge:
    """A digital transition on a device's sample clock within one recording segment.

    Lines are one-based. The sample rate is unknown if a live event arrives
    before its stream metadata. Compare sample numbers within a stream and
    segment, using a shared physical marker to compare between clocks.
    """

    sample_number: int
    line: int
    rising: bool
    sample_rate: float | None
    stream_id: str
    segment: int
    source: str | None = None


@dataclass(frozen=True)
class RHXChunk:
    """A contiguous RHX chunk delivered by the opt-in streaming callback."""

    samples: np.ndarray
    sample_numbers: np.ndarray
    sample_rate: float
    stream_id: str
    segment: int
    missing_samples: int
    received_monotonic: float
    digital_edges: tuple[DigitalEdge, ...]
