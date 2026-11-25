"""Track Intan's 32-bit counters and decode packed digital words."""

import numpy as np

from .sync import DigitalEdge


class SampleClock:
    def __init__(self, signed=True, segment=-1):
        self.segment = segment
        self.previous_raw = None
        self.high = (1 << 31) - 1 if signed else (1 << 32) - 1
        self.low = -(1 << 31) if signed else 0
        self.offset = 0

    def split(self, counters):
        """Yield (slice, unwrapped counters, missing samples before slice)."""
        if not len(counters):
            return
        counters = counters.astype(np.int64, copy=False)
        previous = self.previous_raw
        if previous is None:
            self.segment += 1
            previous = int(counters[0]) - 1
        self.previous_raw = int(counters[-1])
        deltas = np.diff(counters, prepend=previous)
        # Split at gaps, restarts and wrap, working in int64 to avoid overflow.
        starts = np.flatnonzero(deltas[1:] != 1) + 1
        for start, end in zip(np.r_[0, starts], np.r_[starts, len(counters)]):
            delta = int(deltas[start])
            if start:
                previous = int(counters[start - 1])
            current = int(counters[start])
            # Accept a wrap only close to both ends of the counter range.
            # A jump elsewhere in the reverse direction starts a new segment.
            if previous > self.high - 65536 and self.low <= current < self.low + 65536 and delta < 0:
                self.offset += 1 << 32
                delta += 1 << 32
            elif delta <= 0:
                self.segment += 1
                self.offset = 0
            selection = slice(start, end)
            values = counters[selection]
            yield selection, values + self.offset if self.offset else values, max(0, delta - 1)


class LSLClock:
    """Map each device segment to estimated local LSL times."""

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.segment = None
        self.last_time = None

    def first_time(self, segment, counters, receipt_time, samples_to_end):
        first = int(counters[0])
        if segment != self.segment:
            self.segment = segment
            self.anchor_sample = first
            self.anchor_time = receipt_time - samples_to_end / self.sample_rate
            if self.last_time is not None:
                self.anchor_time = max(self.anchor_time, self.last_time + 1 / self.sample_rate)
        timestamp = self.anchor_time + (first - self.anchor_sample) / self.sample_rate
        self.last_time = timestamp + (len(counters) - 1) / self.sample_rate
        return timestamp


class DigitalInputs:
    def __init__(self, lines, sample_rate, stream_id):
        self.mask = sum(1 << (line - 1) for line in lines)
        self.sample_rate = sample_rate
        self.stream_id = stream_id
        self.previous = None
        self.segment = None

    def edges(self, words, counters, segment, missing_samples):
        if not len(words):
            return ()
        if self.previous is None or self.segment != segment or missing_samples:
            previous = int(words[0])
        else:
            previous = self.previous
        changed = np.empty_like(words)
        changed[0] = int(words[0]) ^ previous
        np.bitwise_xor(words[1:], words[:-1], out=changed[1:])
        changed &= self.mask
        self.previous = int(words[-1])
        self.segment = segment
        edges = []
        for index in np.flatnonzero(changed):
            bits = int(changed[index])
            while bits:
                bit = bits & -bits
                edges.append(DigitalEdge(
                    int(counters[index]), bit.bit_length(), bool(int(words[index]) & bit),
                    self.sample_rate, self.stream_id, segment,
                ))
                bits ^= bit
        return tuple(edges)
