import time
import socket
from struct import unpack
import numpy as np
from intan.rhx_interface import _config_options as cfg

MAGIC_NUMBER = 0x2ef07a08
FRAMES_PER_BLOCK = 128
WAVEFORM_BUFFER_SIZE = 200000


class DataStreamer:
    def __init__(self, device, verbose=False):
        """ Initialize the DataStreamer with a connected Intan RHX device"""
        if not device.connected:
            raise RuntimeError("Device not connected. Please connect to the device first.")
        self.device = device
        self.verbose = verbose

        self.sample_rate = cfg.get_sample_rate(device)
        self.timestep = 1.0 / self.sample_rate

        # Track enabled wideband channels
        self.enabled_channels = sorted(getattr(device, "enabled_channels", []))
        if not self.enabled_channels:
            raise ValueError(
                "No enabled channels found in device. Did you forget to call `set_channel(..., enable_wide=True)`?")

    def get_sample_rate(self):
        """ Get the sample rate from the device. """
        return self.sample_rate

    def original_stream(self, duration_sec=1):
        """
        Stream data for a specified duration and return timestamps and voltages.
        """
        num_expected_frames = int(self.sample_rate * duration_sec)
        expected_blocks = (num_expected_frames + FRAMES_PER_BLOCK - 1) // FRAMES_PER_BLOCK
        expected_bytes = expected_blocks * (FRAMES_PER_BLOCK * 6 + 4)

        cfg.set_run_mode(self.device, "run")

        raw_data = b''
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            try:
                chunk = self.device.waveform_socket.recv(WAVEFORM_BUFFER_SIZE)
                if not chunk:
                    break
                raw_data += chunk
            except TimeoutError:
                print("[STREAMER] Socket recv timed out — possibly end of stream.")
                break

        cfg.set_run_mode(self.device, "stop")

        if len(raw_data) < 4:
            raise RuntimeError("No waveform data received.")

        timestamps = []
        voltages = []
        idx = 0

        while idx < len(raw_data):
            try:
                magic_number = unpack('<I', raw_data[idx:idx + 4])[0]
                if magic_number != MAGIC_NUMBER:
                    raise ValueError(f"Invalid magic number: {hex(magic_number)}")
                idx += 4

                for _ in range(FRAMES_PER_BLOCK):
                    if idx + 6 > len(raw_data):
                        break
                    ts = unpack('<i', raw_data[idx:idx + 4])[0]
                    val = unpack('<H', raw_data[idx + 4:idx + 6])[0]
                    timestamps.append(ts * self.timestep)
                    voltages.append(0.195 * (val - 32768))
                    idx += 6
            except Exception as e:
                if self.verbose:
                    print(f"[STREAMER] Error parsing block: {e}")
                break

        if len(timestamps) == 0:
            raise RuntimeError("No valid data blocks parsed.")

        return np.array(timestamps), np.array(voltages)

    def stream(self, n_frames=None, duration_sec=None, align_to_windows=False, window_size=None, step_size=None,
               n_windows=None):
        """
        Stream EMG data and return (timestamps, channel_array).

        Args:
            n_frames: number of frames to stream exactly
            duration_sec: fallback duration in seconds
            align_to_windows: set True to stream based on window count
            window_size, step_size, n_windows: used with align_to_windows=True

        Returns:
            timestamps: 1D np.ndarray [samples]
            channel_data: 2D np.ndarray [n_channels, samples]
        """
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

        # Calculate buffer size based on blocks per write
        blocks_per_read = getattr(self.device, "blocks_per_write", 1)  # default = 1 if not set
        waveform_buffer_size = int(blocks_per_read * bytes_per_block * 1.5)  # Multiply by 2 for a factor of safety

        # Start acquisition
        cfg.set_run_mode(self.device, "run")
        raw_data = b''
        start = time.time()
        #print("\nStreaming live TCP EMG data...")

        while len(raw_data) < expected_bytes and (time.time() - start) < (n_frames / self.sample_rate + 2):
            try:
                chunk = self.device.waveform_socket.recv(waveform_buffer_size)
                if not chunk:
                    break
                raw_data += chunk
            except socket.timeout:
                break

        # Stop acquisition
        cfg.set_run_mode(self.device, "stop")

        # === Parse data ===
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

    def close(self):
        try:
            self.device.close()
            if self.verbose:
                print("[DATA] Waveform connection closed.")
        except Exception as e:
            print(f"[DATA] Error closing socket: {e}")
