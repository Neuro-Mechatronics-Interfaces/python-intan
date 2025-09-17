import threading
from collections import deque
import numpy as np
from pylsl import StreamInlet, resolve_byprop, IRREGULAR_RATE
from typing import Optional, Sequence, Tuple, List
from dataclasses import dataclass

@dataclass
class ResolveSpec:
    name: Optional[str] = None
    stype: Optional[str] = "EMG"
    source_id: Optional[str] = None
    timeout: float = 5.0
    verbose: bool = False

# A generic LSL handler that subscribes to LSL streams and provides basic functionality
class OldLSLClient:
    def __init__(self, stream_name=None, stream_type=None, maxlen=10000, auto_start=True, verbose=False):
        if stream_name is None and stream_type is None:
            raise ValueError("Either stream_name or stream_type must be provided.")
        if stream_name is not None and stream_type is not None:
            raise ValueError("Only one of stream_name or stream_type should be provided.")
        self.auto_start = auto_start
        self.verbose = verbose

        streams = None
        if stream_name:
            print(f"[LSLClient] Looking for a stream with name '{stream_name}'...")
            streams = resolve_byprop("name", stream_name, timeout=5)
            if not streams:
                raise RuntimeError(f"No LSL stream with name '{stream_name}' found.")

        if stream_type:
            print(f"[LSLClient] Looking for a stream of type '{stream_type}'...")
            streams = resolve_byprop("type", stream_type, timeout=5)
            if not streams:
                raise RuntimeError(f"No LSL stream with type '{stream_type}' found.")

        print("[LSLClient] Streams found:")
        for s in streams:
            print(f"  Stream name: {s.name()}, type: {s.type()}, id: {s.source_id()}")
        try:
            self.inlet = StreamInlet(streams[0])
            self.info = self.inlet.info()
            self.n_channels = self.info.channel_count()
            self.sampling_rate = self.info.nominal_srate()
            self.name = self.info.name()
            self.type = self.info.type()
            self.channel_labels, self.units = self._get_channel_metadata()
        except Exception as e:
            print(f"[LSLClient] Failed to create inlet or extract metadata: {e}")
            raise

        print(f"[LSLClient] Connected to stream: {self.name}")
        print(f"  Channels: {self.n_channels}, Sample Rate: {self.sampling_rate} Hz")

        self.buffers = [deque(maxlen=maxlen) for _ in range(self.n_channels)]
        self.lock = threading.Lock()
        self.streaming = False
        self.thread = None
        self.total_samples = 0
        self.reconnect_attempts = 0

        if self.verbose:
            self._print_metadata()

        if self.auto_start:
            self.start_streaming()

    def _print_metadata(self):
        print(f"[LSLClient] Connected to stream: '{self.name}'")
        print(f"  Type: {self.type}")
        print(f"  Sampling Rate: {self.sampling_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        print(f"  Channel Labels: {self.channel_labels}")
        #print(f"  Units: {self.units}")
        try:
            desc = self.info.desc()
            created_at = desc.child_value("created_at") or "N/A"
            manufacturer = desc.child_value("manufacturer") or "N/A"
            print(f"  Created At: {created_at}")
            print(f"  Manufacturer: {manufacturer}")
        except Exception:
            print("  No additional metadata found.")

    def _get_channel_metadata(self):
        try:
            ch_info = self.info.desc().child("channels").child("channel")
            labels = []
            units = []
            for _ in range(self.n_channels):
                labels.append(ch_info.child_value("label") or f"Ch{_}")
                units.append(ch_info.child_value("unit") or "unknown")
                ch_info = ch_info.next_sibling()
            return labels, units
        except Exception:
            return [f"Ch{i}" for i in range(self.n_channels)], ["unknown"] * self.n_channels

    def start_streaming(self):
        self.streaming = True
        self.thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.thread.start()

    def stop_streaming(self):
        self.streaming = False
        self.thread.join()

    def _streaming_worker(self):
        while self.streaming:
            sample, timestamp = self.inlet.pull_sample(timeout=0.1)
            if sample:
                with self.lock:
                    for ch, val in enumerate(sample):
                        if ch < self.n_channels:
                            self.buffers[ch].append(val)
                    self.total_samples += 1

    def get_latest_window(self, window_ms):
        n_samples = int(self.sampling_rate * window_ms / 1000.0)
        with self.lock:
            result = np.zeros((self.n_channels, n_samples))
            for ch in range(self.n_channels):
                buf = list(self.buffers[ch])
                if len(buf) < n_samples:
                    buf = [0.0] * (n_samples - len(buf)) + buf
                result[ch, :] = buf[-n_samples:]
        return result

    def get_connection_status(self):
        return {
            "connected": True,
            "total_samples": self.total_samples,
            "reconnect_attempts": self.reconnect_attempts
        }

    def get_metadata(self):
        """Return dictionary of all available stream metadata."""
        return {
            "name": self.name,
            "type": self.type,
            "fs": self.sampling_rate,
            "n_channels": self.n_channels,
            "channel_labels": self.channel_labels,
            "units": self.units,
        }

class LSLClient:
    """
    Fast numeric LSL client with:
      - name/type/source_id resolution (any combination)
      - chunked inlet loop into a NumPy ring buffer (T x C)
      - helper APIs for live viewers: read_window, get_samples, drain_new, latest, fs_estimate
      - context manager support
    """

    def __init__(
        self,
        stream_name: Optional[str] = None,
        stream_type: Optional[str] = "EMG",
        source_id: Optional[str] = None,
        *,
        timeout: float = 5.0,
        max_seconds: float = 10.0,
        fs_hint: float = 2000.0,
        channels_hint: int = 128,
        auto_start: bool = True,
        verbose: bool = False,
    ):
        # Resolve
        self.spec = ResolveSpec(stream_name, stream_type, source_id, timeout, verbose)
        info = self._resolve_info()
        self.inlet = StreamInlet(info, max_buflen=120)

        # Stream properties
        self.name = info.name()
        self.type = info.type()
        self.fs = float(info.nominal_srate() if info.nominal_srate() and info.nominal_srate() != IRREGULAR_RATE else fs_hint)
        self.n_channels = int(info.channel_count() or channels_hint)

        # Channel metadata (best-effort)
        self.channel_labels, self.units = self._get_channel_metadata(info)

        # Ring buffer (T x C)
        self.max_seconds = float(max_seconds)
        self.T = max(1, int(self.fs * self.max_seconds))
        self.n_samples = self.T # Older support
        self._buf = np.zeros((self.T, self.n_channels), dtype=np.float32)
        self._wp = 0                      # write pointer
        self._count = 0                   # total valid samples (<= T)
        self._lock = threading.Lock()

        # Incremental drain timestamping
        self._last_t = 0.0

        # Worker
        self._stop = False
        self._th: Optional[threading.Thread] = None
        self.verbose = verbose
        if auto_start:
            self.start_streaming()

    # ---------- lifecycle ----------
    def start_streaming(self):
        if self._th and self._th.is_alive():
            return
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop_streaming(self):
        self._stop = True
        if self._th:
            self._th.join(timeout=1.0)
        self._th = None

    def close(self):
        self.stop_streaming()
        try:
            self.inlet.close_stream()
        except Exception:
            pass

    # context manager
    def __enter__(self):
        self.start_streaming()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------- viewers API ----------
    def read_window(self, seconds: float) -> np.ndarray:
        """Return the most recent window as (T, C)."""
        n = max(1, int(self.fs * float(seconds)))
        with self._lock:
            n = min(n, self._buf.shape[0])
            idx = (np.arange(n) + (self._wp - n)) % self._buf.shape[0]
            return self._buf[idx, :].copy()

    def get_samples(self, channel: int, n_samples: int) -> List[float]:
        """Minimal API for _realtime_plotter.py."""
        X = self.read_window(n_samples / self.fs)
        if X.size == 0 or channel >= X.shape[1]:
            return []
        return X[-n_samples:, channel].astype(np.float32).tolist()

    def drain_new(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        API for _stacked_plot.py:
          returns (t_new, Y) where t_new is 1D (N,) seconds (relative),
          and Y is (C, N).
        """
        X = self.read_window(0.25)  # ~250 ms updates
        if X.size == 0:
            return np.empty((0,), dtype=np.float64), np.zeros((self.n_channels, 0), np.float32)
        n = X.shape[0]
        dt = 1.0 / self.fs
        t_new = self._last_t + dt * np.arange(1, n + 1, dtype=np.float64)
        self._last_t = t_new[-1]
        return t_new, X.T

    # def latest(self) -> Tuple[np.ndarray, np.ndarray]:
    #     """Return (t_rel, Y) for full ring: t_rel ∈ [-W, 0], Y shape (C, T)."""
    #     X = self.read_window(self.max_seconds)
    #     if X.size == 0:
    #         return np.empty((0,), dtype=np.float64), np.zeros((self.n_channels, 0), np.float32)
    #     T = X.shape[0]
    #     t_rel = np.linspace(-T / self.fs, 0.0, T, dtype=np.float64)
    #     return t_rel, X.T
    def latest(self, window_secs=5.0):
        X = self.read_window(window_secs)  # (T, C) float32
        if X.size == 0:
            return None, None
        T = X.shape[0]
        t_rel = np.linspace(-T / self.fs, 0.0, T, dtype=np.float64)
        return t_rel, X.T  # (C, T)

    def fs_estimate(self) -> float:
        """Simple estimate (we use nominal fs)."""
        return float(self.fs)

    # ---------- internals ----------
    def _loop(self):
        pull_timeout = 0.05  # seconds
        while not self._stop:
            try:
                data, _ts = self.inlet.pull_chunk(timeout=pull_timeout)
            except Exception:
                data = None
            if not data:
                continue

            X = np.asarray(data, dtype=np.float32)
            if X.ndim == 1:
                X = X[None, :]
            X = X[:, :self.n_channels]

            n = X.shape[0]
            with self._lock:
                dst = self._wp % self.T
                first = min(n, self.T - dst)
                self._buf[dst:dst + first, :] = X[:first, :]
                rem = n - first
                if rem > 0:
                    self._buf[:rem, :] = X[first:, :]
                self._wp = (self._wp + n) % self.T
                self._count = min(self._count + n, self.T)

    def _resolve_info(self):
        # try name → source_id → type, accepting any that the user provided
        streams = []
        tried = []
        if self.spec.name:
            tried.append(("name", self.spec.name))
            streams = resolve_byprop("name", self.spec.name, timeout=self.spec.timeout)
            if streams:
                return streams[0]
        if self.spec.source_id:
            tried.append(("source_id", self.spec.source_id))
            streams = resolve_byprop("source_id", self.spec.source_id, timeout=self.spec.timeout)
            if streams:
                return streams[0]
        if self.spec.stype:
            tried.append(("type", self.spec.stype))
            streams = resolve_byprop("type", self.spec.stype, timeout=self.spec.timeout)
            if streams:
                return streams[0]
        raise TimeoutError(f"No LSL stream found after tries: {tried} (timeout={self.spec.timeout}s).")

    @staticmethod
    def _get_channel_metadata(info) -> Tuple[Sequence[str], Sequence[str]]:
        try:
            ch = info.desc().child("channels").child("channel")
            labels, units = [], []
            n = int(info.channel_count())
            for _ in range(n):
                labels.append(ch.child_value("label") or f"Ch{_}")
                units.append(ch.child_value("unit") or "au")
                ch = ch.next_sibling()
            return labels, units
        except Exception:
            n = int(info.channel_count() or 0)
            return [f"Ch{i}" for i in range(n)], ["au"] * n
