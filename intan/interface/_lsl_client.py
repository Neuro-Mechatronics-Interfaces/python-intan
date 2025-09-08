import threading
from collections import deque
import numpy as np
from pylsl import StreamInlet, resolve_byprop


# A generic LSL handler that subscribes to LSL streams and provides basic functionality
class LSLClient:
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
