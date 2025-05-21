import time
import socket
from struct import unpack
import numpy as np
from intan.rhx_interface import _config_options as cfg

MAGIC_NUMBER = 0x2ef07a08
FRAMES_PER_BLOCK = 128


class DataStreamer:
    def get_sample_rate(self):
        return self.sample_rate

    def stream(self, n_frames=None, duration_sec=None, align_to_windows=False, window_size=None, step_size=None, n_windows=None):
        if align_to_windows:
            if n_windows is None or window_size is None or step_size is None:
                raise ValueError("Must specify window_size, step_size, and n_windows when align_to_windows=True")
            n_frames = (n_windows - 1) * step_size + window_size

        if n_frames is None:
            if duration_sec is None:
                raise ValueError("Must specify either n_frames or duration_sec.")
            n_frames = int(self.sample_rate * duration_sec)

        n_channels = len(self.enabled_channels)
        expected_blocks = int(np.ceil(n_frames / FRAMES_PER_BLOCK))
        bytes_per_block = 4 + FRAMES_PER_BLOCK * (4 + 2 * n_channels)
        expected_bytes = expected_blocks * bytes_per_block

        blocks_per_read = getattr(self, "blocks_per_write", 1)
        waveform_buffer_size = int(blocks_per_read * bytes_per_block * 1.5)

        cfg.set_run_mode(self, "run")
        raw_data = b''
        start = time.time()

        while len(raw_data) < expected_bytes and (time.time() - start) < (n_frames / self.sample_rate + 2):
            try:
                chunk = self.waveform_socket.recv(waveform_buffer_size)
                if not chunk:
                    break
                raw_data += chunk
            except socket.timeout:
                break

        cfg.set_run_mode(self, "stop")

        timestamps = []
        channel_data = [[] for _ in range(n_channels)]
        idx = 0
        frame_count = 0

        while idx + 4 <= len(raw_data) and frame_count < n_frames:
            magic = unpack('<I', raw_data[idx:idx + 4])[0]
            if magic != MAGIC_NUMBER:
                idx += 1
                continue

            idx += 4
            for _ in range(FRAMES_PER_BLOCK):
                if frame_count >= n_frames or idx + 4 + 2 * n_channels > len(raw_data):
                    break
                ts = unpack('<i', raw_data[idx:idx + 4])[0]
                timestamps.append(ts * self.timestep)
                idx += 4

                for ch in range(n_channels):
                    val = unpack('<H', raw_data[idx:idx + 2])[0]
                    channel_data[ch].append(0.195 * (val - 32768))
                    idx += 2

                frame_count += 1

        return np.array(timestamps), np.array(channel_data)


    def close_stream(self):
        try:
            self.waveform_socket.close()
        except Exception as e:
            print(f"[DATA] Error closing waveform socket: {e}")
