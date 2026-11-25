"""Known-pulse fixtures for RHD and RHX waveform synchronization."""

import struct

import numpy as np
import pytest

from intan.interface import IntanRHXDevice, LSLOptions
from intan.interface import _rhx_device
from intan.io import load_rhd_file, read_digital_edges
from intan.sync import DigitalEdge


def _qstring(text):
    raw = text.encode("utf-16-le")
    return struct.pack("<I", len(raw)) + raw


def _rhd_header(version=(2, 0)):
    header = struct.pack("<Ihhf", 0xC6912702, *version, 1000)
    header += bytes(36) + _qstring("") * 3
    if version >= (1, 1):
        header += struct.pack("<h", 1)  # Temperature sensor
    if version >= (1, 3):
        header += struct.pack("<h", 13)
    if version >= (2, 0):
        header += _qstring("n/a")
    header += struct.pack("<h", 2) + _qstring("Disabled") + _qstring("B")
    header += struct.pack("<hhh", 0, 1, 1)
    if version >= (3, 0):
        header += _qstring("B-000") * 2 + struct.pack("<10h2f", *([0] * 12))
    header += _qstring("Signals") + _qstring("S")
    channels = [(0, 0, "A-000"), (4, 3, "DIN-03"), (5, 0, "DOUT-00")]
    header += struct.pack("<hhh", 1, len(channels), 1)
    for kind, bit, name in channels:
        header += _qstring(name) * 2 + struct.pack("<10h2f", bit, bit, kind, 1, *([0] * 8))
    return header


@pytest.mark.parametrize("version", [(1, 0), (1, 3), (2, 0), (3, 5)])
def test_embedded_rhd_ttl_edges(version, tmp_path):
    count = 128 if version >= (2, 0) else 60
    counter_type = "<i4" if version >= (1, 2) else "<u4"
    first = -16 if version >= (1, 2) else 100
    numbers = np.arange(first, first + 2 * count, dtype=counter_type)
    words = np.zeros(2 * count, dtype="<u2")
    words[count - 1: count + 3] = 8
    words[12:28] = 1  # Digital input line 1 was disabled.
    data = bytearray(_rhd_header(version))
    for start in (0, count):
        block = slice(start, start + count)
        data += numbers[block].tobytes() + bytes(2 * (count + (version >= (1, 1))))
        data += words[block].tobytes() + bytes([255] * 2 * count)
    path = tmp_path / "trial.rhd"
    path.write_bytes(data)
    assert read_digital_edges(path, stream_id="trial") == [
        DigitalEdge(first + count - 1, 4, True, 1000, "trial", 0),
        DigitalEdge(first + count + 3, 4, False, 1000, "trial", 0),
    ]
    full = load_rhd_file(path)
    np.testing.assert_array_equal(full["board_dig_in_data"][0], (words & 8) != 0)
    assert full["amplifier_data"].shape == (1, 2 * count)
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError, match="incomplete data block"):
        read_digital_edges(path)


@pytest.mark.parametrize("packed", [False, True])
def test_sidecar_gap_reset_and_wrap(packed, tmp_path):
    path = tmp_path / "info.rhd"
    path.write_bytes(_rhd_header())
    numbers = np.array([2147483646, 2147483647, -2147483648, -2147483647,
                        -2147483644, -2147483643, -2147483642, -2147483639,
                        -2147483638, -2147483637, -2**31, -2**31 + 1], dtype="<i4")
    # Wrap is consecutive, gap and reset are not. The change at index 4
    # or the first sample of a segment must not invent an edge.
    words = np.array([0, 8, 0, 0, 8, 8, 0, 0, 8, 0, 8, 0], dtype="<u2")
    numbers.tofile(tmp_path / "time.dat")
    name = "digitalin.dat" if packed else "board-DIN-03.dat"
    (words if packed else words >> 3).tofile(tmp_path / name)
    actual = [(e.sample_number, e.rising, e.segment) for e in read_digital_edges(path)]
    assert actual == [(2147483647, True, 0), (2147483648, False, 0),
                      (2147483654, False, 0), (2147483658, True, 0),
                      (2147483659, False, 0), (-2147483647, False, 1)]


def _waveform(counters, digital=None):
    payload = bytearray(struct.pack("<I", _rhx_device.MAGIC_NUMBER))
    for index, counter in enumerate(counters):
        payload += struct.pack("<iH", counter, 32768 + index)
        if digital is not None:
            payload += struct.pack("<H", digital[index])
    return payload


class _Outlet:
    def __init__(self):
        self.pushes = []

    def push_chunk(self, samples, ts=None):
        self.pushes.append((samples.copy(), ts))


def _device(monkeypatch, **options):
    def connect(device):
        device.connected = True
        device.command_socket = object()
    monkeypatch.setattr(IntanRHXDevice, "connect", connect)
    return IntanRHXDevice(num_channels=1, sample_rate=1000, **options)


def _receive_waveform(monkeypatch, device, payloads):
    batches = iter(payloads)
    def receive(buffer, size):
        batch = next(batches, None)
        if batch is None:
            device.streaming = False
            return buffer
        return buffer + batch

    modes = []
    monkeypatch.setattr(device, "receive_data", receive)
    monkeypatch.setattr(device, "set_run_mode", modes.append)
    device.streaming = True
    device._streaming_worker()
    assert modes == ["run", "stop"]


def test_live_callback_and_anchored_lsl_preserve_gap(monkeypatch):
    chunks = []
    device = _device(monkeypatch, on_chunk=chunks.append, digital_input_lines=(4, 12),
                     use_lsl=True, lsl_options=LSLOptions(chunk_size=32, device_timestamps=True))
    outlet = device._lsl_numeric = _Outlet()
    monkeypatch.setattr(_rhx_device, "local_clock", lambda: 20.0)
    first_word = np.zeros(128, dtype=np.uint16)
    first_word[-2:] = 8
    second_word = np.full(128, 8, dtype=np.uint16)
    second_word[3:] = 2048
    first = _waveform(range(128), first_word)
    second = _waveform(range(140, 268), second_word)
    _receive_waveform(monkeypatch, device, [first[:2], first[2:], second])
    assert [chunk.missing_samples for chunk in chunks] == [0, 12]
    assert list(chunks[1].sample_numbers[[0, -1]]) == [140, 267]
    assert not chunks[1].samples.flags.writeable
    assert [(e.sample_number, e.line, e.rising) for chunk in chunks for e in chunk.digital_edges] == [
        (126, 4, True), (143, 4, False), (143, 12, True),
    ]
    np.testing.assert_allclose(chunks[0].samples[0], np.arange(128) * 0.195, rtol=1e-6)
    assert outlet.pushes[3][1] == pytest.approx(20)
    assert outlet.pushes[4][1] - outlet.pushes[3][1] == pytest.approx(0.044)


def test_default_waveform_lsl_still_stamps_on_send(monkeypatch):
    device = _device(monkeypatch, use_lsl=True)
    outlet = device._lsl_numeric = _Outlet()
    data = _waveform(range(10, 138))
    parsed, clocks, consumed, synced = device.parse_emg_stream_fast(data)
    reference, reference_clocks, reference_consumed = device.parse_emg_stream(data)
    np.testing.assert_allclose(parsed, reference, rtol=1e-6)
    np.testing.assert_array_equal(clocks, reference_clocks)
    assert synced and consumed == reference_consumed == len(data)
    _receive_waveform(monkeypatch, device, [data])
    assert len(outlet.pushes) == 4 and all(ts is None for _, ts in outlet.pushes)


def test_first_receive_gap_anchors_last_sample_to_receipt(monkeypatch):
    chunks = []
    device = _device(monkeypatch, on_chunk=chunks.append, digital_input_lines=(4,),
                     use_lsl=True, lsl_options=LSLOptions(device_timestamps=True))
    outlet = device._lsl_numeric = _Outlet()
    monkeypatch.setattr(_rhx_device, "local_clock", lambda: 20.0)
    low = np.zeros(128, dtype=np.uint16)
    high = np.full(128, 8, dtype=np.uint16)
    high[3:] = 0
    _receive_waveform(monkeypatch, device, [
        _waveform(range(128), low) + _waveform(range(140, 268), high),
        _waveform(range(268, 396), low),
    ])
    assert [c.missing_samples for c in chunks] == [0, 12, 0]
    assert [(e.sample_number, e.rising) for c in chunks for e in c.digital_edges] == [(143, False)]
    assert outlet.pushes[3][1] == pytest.approx(19.860)
    assert outlet.pushes[7][1] == pytest.approx(20.0)
    assert outlet.pushes[-1][1] == pytest.approx(20.128)


@pytest.mark.parametrize("reset", [False, True], ids=["wrap", "reset"])
@pytest.mark.parametrize("separate_receives", [False, True], ids=["one-receive", "two-receives"])
def test_live_counter_boundary_timing_and_edges(monkeypatch, reset, separate_receives):
    chunks = []
    device = _device(monkeypatch, on_chunk=chunks.append, digital_input_lines=(4,),
                     use_lsl=True, lsl_options=LSLOptions(device_timestamps=True))
    outlet = device._lsl_numeric = _Outlet()
    receipts = iter([19.872, 20.0, 20.0] if separate_receives else [20.0, 20.0])
    monkeypatch.setattr(_rhx_device, "local_clock", lambda: next(receipts))
    first = np.arange(128) if reset else np.arange(2**31 - 128, 2**31)
    second = np.arange(128) if reset else np.arange(-2**31, -2**31 + 128)
    low = np.zeros(128, dtype=np.uint16)
    high = np.full(128, 8, dtype=np.uint16)
    high[1:] = 0
    payloads = [_waveform(first, low), _waveform(second, high)]
    _receive_waveform(monkeypatch, device, payloads if separate_receives else [payloads[0] + payloads[1]])
    assert [c.segment for c in chunks] == [0, int(reset)]
    assert [c.missing_samples for c in chunks] == [0, 0]
    expected_edges = [(1, False)] if reset else [(2**31, True), (2**31 + 1, False)]
    assert [(e.sample_number, e.rising) for c in chunks for e in c.digital_edges] == expected_edges
    assert outlet.pushes[3][1] == pytest.approx(19.872)
    assert outlet.pushes[-1][1] == pytest.approx(20.0)
