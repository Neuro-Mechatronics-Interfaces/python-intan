"""Read Intan RHD digital edges without loading amplifier channels."""

import math
from pathlib import Path
import struct

import numpy as np

from .._sample_clock import DigitalInputs, SampleClock
from ._rhd_block_parser import get_timestamp_signed
from ._rhd_header_parsing import get_bytes_per_data_block, read_header


def _sidecars(path, channels, counter_type):
    time_path = path.with_name("time.dat")
    if not time_path.is_file() or not time_path.stat().st_size:
        raise ValueError("The RHD header has no data. Keep time.dat next to info.rhd.")
    if time_path.stat().st_size % 4:
        raise ValueError("Intan time.dat is truncated.")
    counters = np.memmap(time_path, mode="r", dtype=counter_type)

    def states(file):
        if not file.is_file() or file.stat().st_size != 2 * len(counters):
            raise ValueError(f"{file.name} is missing or its length differs from time.dat.")
        return np.memmap(file, mode="r", dtype="<u2")

    packed_path = path.with_name("digitalin.dat")
    packed = states(packed_path) if packed_path.is_file() else None
    inputs = {} if packed is not None else {
        bit: states(path.with_name(f"board-{name}.dat")) for bit, name in channels.items()
    }
    for start in range(0, len(counters), 131072):
        chunk = slice(start, start + 131072)
        numbers = counters[chunk]
        if packed is not None:
            yield numbers, packed[chunk]
        else:
            words = np.zeros(len(numbers), dtype=np.uint16)
            for bit, values in inputs.items():
                values = values[chunk]
                if np.any(values > 1):
                    raise ValueError(f"board-{channels[bit]}.dat contains invalid digital states.")
                words |= values << bit
            yield numbers, words


def read_digital_edges(path, *, stream_id="intan", segment=0):
    """Return both polarities for enabled RHD digital inputs in recorded order.

    Supports embedded RHD blocks and header/sidecar recordings. Lines are
    one-based. No edge is inferred at the first sample or across a counter
    gap or restart. Counter wraps are unwrapped within the segment. Use
    `segment` to set the first segment number for this recording.
    """
    path = Path(path)
    with path.open("rb") as file:
        try:
            header = read_header(file, verbose=False)
        except (struct.error, UnicodeDecodeError) as exc:
            raise ValueError("The Intan RHD header is incomplete or invalid.") from exc
        data_offset = file.tell()
    rate = float(header["sample_rate"])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("The Intan RHD sampling rate must be positive.")
    digital_channels = header["board_dig_in_channels"]
    channels = {channel["native_order"]: channel["native_channel_name"] for channel in digital_channels}
    if not channels or len(channels) != len(digital_channels) or any(not 0 <= bit < 16 for bit in channels):
        raise ValueError("The Intan RHD header has no usable digital inputs.")
    signed = bool(get_timestamp_signed(header))
    counter_type = "<i4" if signed else "<u4"
    count = header["num_samples_per_data_block"]
    block_size = int(get_bytes_per_data_block(header))
    blocks, remainder = divmod(path.stat().st_size - data_offset, block_size)
    if blocks < 0 or remainder:
        raise ValueError("The Intan RHD has an incomplete data block.")
    if blocks:
        input_offset = block_size - 2 * count * (1 + bool(header["num_board_dig_out_channels"]))
        layout = np.dtype({
            "names": ["numbers", "words"],
            "formats": [(counter_type, count), ("<u2", count)],
            "offsets": [0, input_offset],
            "itemsize": block_size,
        })
        data = np.memmap(path, mode="r", dtype=layout, offset=data_offset, shape=(blocks,))
        batches = ((part["numbers"].reshape(-1), part["words"].reshape(-1))
                   for part in (data[start:start + 1024] for start in range(0, blocks, 1024)))
    else:
        batches = _sidecars(path, channels, counter_type)

    clock = SampleClock(signed=signed, segment=segment - 1)
    inputs = DigitalInputs([bit + 1 for bit in channels], rate, stream_id)
    edges = []
    for counters, words in batches:
        for selection, numbers, missing in clock.split(counters):
            edges.extend(inputs.edges(words[selection], numbers, clock.segment, missing))
    return edges
