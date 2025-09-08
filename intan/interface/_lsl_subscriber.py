# intan/interface/_lsl_subscriber.py
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional, Iterable, Tuple, List

from pylsl import StreamInlet, resolve_byprop, StreamInfo


def _to_str(x):
    # LSL markers often come as a 1-element list of str/bytes.
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


class LSLSubscriber:
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
