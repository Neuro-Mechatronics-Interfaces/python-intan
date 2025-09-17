# intan/interface/_lsl_subscriber.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import threading
import time
from collections import deque
from typing import Callable, Optional, Iterable, Tuple, List

from pylsl import StreamInlet, resolve_byprop, StreamInfo, IRREGULAR_RATE, local_clock, StreamOutlet

resolve_stream = None

def _to_str(x):
    # LSL markers often come as a 1-element list of str/bytes.
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


class OldLSLSubscriber:
    """
    Tiny wrapper around pylsl for string marker streams.

    - Resolves by `type` (default "Markers") or by exact `name` if provided.
    - Creates a StreamInlet with default pylsl settings (no kwargs).
    - `pull()` returns (value, timestamp) where value is a str.
    - Optional background thread with a user callback.
    - Context manager support.
    """

    def __init__(
        self,
        stream_type: str = "Markers",
        name: Optional[str] = None,
        timeout: float = 5.0,
        verbose: bool = False,
    ):
        self.stream_type = stream_type
        self.name = name
        self.timeout = float(timeout)
        self.verbose = verbose

        self._info: Optional[StreamInfo] = None
        self._inlet: Optional[StreamInlet] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[str, float], None]] = None

        self._queue = deque(maxlen=1024)  # if you want to poll without callback

    # ---------- public API ----------

    def start(self) -> None:
        """Resolve and connect (no-op if already connected)."""
        if self._inlet is not None:
            return
        self._info = self._resolve_stream()
        self._inlet = StreamInlet(self._info)  # no kwargs → avoids ctypes issues
        if self.verbose:
            print(f"[LSL] Connected to stream: name='{self._info.name()}', type='{self._info.type()}'")

    def stop(self) -> None:
        """Stop background thread (if any)."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def close(self) -> None:
        """Close inlet and stop thread."""
        self.stop()
        self._inlet = None
        self._info = None

    def set_callback(self, fn: Callable[[str, float], None], poll_hz: float = 50.0) -> None:
        """
        Start a background thread that calls `fn(value, timestamp)` for each sample.
        `poll_hz` controls the pull timeout (lower → less CPU).
        """
        self.start()
        self._callback = fn
        self._running = True
        timeout = 1.0 / max(1.0, float(poll_hz))

        def _loop():
            while self._running:
                try:
                    val_ts = self.pull(timeout=timeout)
                    if val_ts is None:
                        continue
                    val, ts = val_ts
                    if self._callback:
                        self._callback(val, ts)
                except Exception as e:
                    if self.verbose:
                        print(f"[LSL] subscriber error: {e}")
                    time.sleep(0.1)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def pull(self, timeout: float = 0.0) -> Optional[Tuple[str, float]]:
        """
        Pull one sample. Returns (value:str, timestamp:float) or None if no sample in timeout.
        """
        self.start()
        sample, ts = self._inlet.pull_sample(timeout=timeout)  # returns (list, ts) or (None, None)
        if sample is None:
            return None
        # Typical marker is a 1-element list
        val = _to_str(sample[0] if len(sample) else "")
        self._queue.append((val, ts))
        return val, ts

    def pull_chunk(self, max_samples: int = 32, timeout: float = 0.0) -> List[Tuple[str, float]]:
        """
        Pull a small chunk. Returns list of (value:str, timestamp:float).
        """
        self.start()
        data, ts = self._inlet.pull_chunk(max_samples=max_samples, timeout=timeout)
        out: List[Tuple[str, float]] = []
        if data and ts:
            for row, t in zip(data, ts):
                val = _to_str(row[0] if row else "")
                out.append((val, t))
                self._queue.append((val, t))
        return out

    # ---------- context manager ----------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------- internals ----------

    def _resolve_stream(self) -> StreamInfo:
        """
        Try resolve by name first (if provided), else by type.
        """
        if self.name:
            if self.verbose:
                print(f"[LSL] Resolving by name='{self.name}' (timeout={self.timeout}s)...")
            by_name = resolve_byprop("name", self.name, timeout=self.timeout)
            if by_name:
                return by_name[0]
            # fall back to type if name not found
            if self.verbose:
                print(f"[LSL] Name not found; falling back to type='{self.stream_type}'")

        if self.verbose:
            print(f"[LSL] Resolving by type='{self.stream_type}' (timeout={self.timeout}s)...")
        by_type = resolve_byprop("type", self.stream_type, timeout=self.timeout)
        if not by_type:
            raise TimeoutError(
                f"No LSL stream found (type='{self.stream_type}', name='{self.name or ''}') within {self.timeout}s."
            )
        return by_type[0]

@dataclass
class LSLStreamSpec:
    """Selector for a specific LSL stream."""
    name: Optional[str] = None        # e.g., "EMG-128"
    stype: Optional[str] = "EMG"      # LSL 'type' field
    source_id: Optional[str] = None   # device/source identifier
    timeout: float = 3.0              # resolve timeout (seconds)
    verbose: bool = True              # print resolution logs


class LSLSubscriber:
    """
    Numeric (float32) multichannel LSL subscriber with a ring buffer.

    - Resolves stream by: name → source_id → type
    - Threaded inlet loop writes into a circular buffer
    - Consumers can read the most recent window or the full buffer copy
    """

    def __init__(
        self,
        spec: LSLStreamSpec = LSLStreamSpec(),
        *,
        fs_hint: float = 2000.0,
        channels_hint: int = 128,
        buffer_seconds: float = 60.0,
        on_chunk: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.spec = spec
        self.on_chunk = on_chunk

        # runtime characteristics (updated after connecting)
        self.fs: float = float(fs_hint)
        self.n_channels: int = int(channels_hint)

        # ring buffer
        self._buf_len = max(1, int(self.fs * buffer_seconds))
        self._buf = np.zeros((self._buf_len, self.n_channels), dtype=np.float32)
        self._wp = 0
        self._lock = threading.Lock()

        # LSL
        self._inlet: Optional[StreamInlet] = None
        self._t: Optional[threading.Thread] = None
        self._stop = False

    # -------------- public API --------------

    def start(self):
        """Resolve and start receiving samples into the ring buffer."""
        info = self._resolve_stream()
        if self.spec.verbose:
            try:
                print(
                    f"[LSL] Connected: name='{info.name()}', type='{info.type()}', "
                    f"source_id='{info.source_id()}', fs={info.nominal_srate()}, ch={info.channel_count()}"
                )
            except Exception:
                pass

        self._inlet = StreamInlet(info, max_buflen=120)

        # adopt actual fs/ch
        try:
            fs = info.nominal_srate()
            if fs and fs != IRREGULAR_RATE:
                self.fs = float(fs)
        except Exception:
            pass
        try:
            self.n_channels = int(info.channel_count())
        except Exception:
            pass

        # rebuild ring buffer with correct dims but preserve duration
        seconds = self._buf_len / max(1.0, self.fs)
        self._buf_len = max(1, int(self.fs * seconds))
        self._buf = np.zeros((self._buf_len, self.n_channels), dtype=np.float32)
        self._wp = 0

        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self):
        """Stop the reader thread and detach inlet."""
        self._stop = True
        if self._t is not None:
            self._t.join(timeout=1.0)
        self._t = None
        self._inlet = None

    def read_window(self, seconds: float) -> np.ndarray:
        """
        Return the most recent window of samples with shape (T, C).
        If requested window > buffer length, it is truncated to fit.
        """
        n = max(1, int(self.fs * seconds))
        with self._lock:
            n = min(n, self._buf.shape[0])
            idx = (np.arange(n) + (self._wp - n)) % self._buf.shape[0]
            out = self._buf[idx, :].copy()
        return out

    def read_all(self) -> Tuple[np.ndarray, float]:
        """Return a copy of the entire ring buffer and the sampling rate."""
        with self._lock:
            out = self._buf.copy()
        return out, self.fs

    # -------------- internal --------------

    def _loop(self):
        while not self._stop and self._inlet is not None:
            try:
                samples, _ = self._inlet.pull_chunk(timeout=0.05)
            except Exception:
                samples = None
            if not samples:
                continue

            x = np.asarray(samples, dtype=np.float32)
            if x.ndim == 1:
                x = x[None, :]
            x = x[:, :self.n_channels]

            with self._lock:
                n = x.shape[0]
                idx = (np.arange(n) + self._wp) % self._buf.shape[0]
                self._buf[idx, :] = x
                self._wp = (self._wp + n) % self._buf.shape[0]

            if self.on_chunk:
                try:
                    self.on_chunk(x)
                except Exception:
                    pass

    # ---- resolving helpers ----

    def _resolve_stream(self) -> StreamInfo:
        """
        Resolve an LSL StreamInfo using (1) exact name, then (2) source_id,
        then (3) stream type as a fallback. Raises TimeoutError if not found.
        """
        # 1) by name
        if self.spec.name:
            if self.spec.verbose:
                print(f"[LSL] Resolving by name='{self.spec.name}' (timeout={self.spec.timeout}s)…")
            by_name = self._resolve_byprop("name", self.spec.name, timeout=self.spec.timeout)
            if by_name:
                return by_name[0]
            if self.spec.verbose:
                print("[LSL] Name not found; continuing…")

        # 2) by source_id
        if self.spec.source_id:
            if self.spec.verbose:
                print(f"[LSL] Resolving by source_id='{self.spec.source_id}' (timeout={self.spec.timeout}s)…")
            by_src = self._resolve_byprop("source_id", self.spec.source_id, timeout=self.spec.timeout)
            if by_src:
                return by_src[0]
            if self.spec.verbose:
                print("[LSL] source_id not found; continuing…")

        # 3) by type (broadest)
        stype = self.spec.stype or ""
        if self.spec.verbose:
            print(f"[LSL] Resolving by type='{stype}' (timeout={self.spec.timeout}s)…")
        by_type = self._resolve_byprop("type", stype, timeout=self.spec.timeout) if stype else []
        if not by_type:
            raise TimeoutError(
                f"No LSL stream found (type='{stype}', name='{self.spec.name or ''}', "
                f"source_id='{self.spec.source_id or ''}') within {self.spec.timeout}s."
            )
        return by_type[0]

    def _resolve_byprop(self, key: str, value: Optional[str], timeout: float) -> List[StreamInfo]:
        """
        Try pylsl.resolve_byprop when available; otherwise fall back to
        resolve_stream() and filter by StreamInfo properties.
        """
        if not value:
            return []

        # Preferred: native API if present
        if resolve_byprop is not None:
            try:
                return resolve_byprop(key, value, timeout=timeout)
            except Exception:
                pass

        # Fallback: resolve all then filter
        cands = []
        if resolve_stream is not None:
            try:
                cands = resolve_stream(timeout=timeout)
            except Exception:
                cands = []

        def prop(si: StreamInfo, k: str) -> str:
            try:
                if k == "name":      return si.name()
                if k == "type":      return si.type()
                if k == "source_id": return si.source_id()
            except Exception:
                return ""
            return ""

        return [si for si in cands if prop(si, key) == value]


    # Test if main

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="LSL Subscriber Test")
    ap.add_argument("--type", default="Markers", help="LSL stream type to subscribe to.")
    ap.add_argument("--name", default=None, help="LSL stream name to subscribe to.")
    ap.add_argument("--source_id", default=None, help="LSL stream source_id to subscribe to.")
    ap.add_argument("--timeout", type=float, default=5.0, help="Timeout for resolving the stream.")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = ap.parse_args()

    def on_marker(value: str, ts: float):
        print(f"[marker] {value} at {ts:.6f} (local {local_clock():.6f})")

    spec = LSLStreamSpec(
        name=args.name,
        stype=args.type,
        source_id=args.source_id,
        timeout=args.timeout,
        verbose=args.verbose
    )

    with LSLSubscriber(spec=spec) as sub:
        sub.set_callback(on_marker, poll_hz=20.0)
        print("Listening for markers... Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Stopping...")