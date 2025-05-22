"""
intan.rhx_interface._rhx_device

Stream and record EMG data from the Intan RHX system via TCP.

This module:
- Connects to the Intan TCP command and data ports
- Reads, parses, and scales EMG data from the waveform stream
- Manages circular buffers and saves data to disk (optionally)
- Supports context-managed use (`with IntanRHXDevice(...) as dev:`)

Inherits from `RHXConfig` for unified parameter control and data streaming configuration.
"""

import time
import socket
import numpy as np
from struct import unpack
from collections import deque
import threading

from ._rhx_config import RHXConfig

FRAMES_PER_BLOCK = 128  # Software hard coded but will
MAGIC_NUMBER = 0x2ef07a08


class IntanRHXDevice(RHXConfig):
    """
        Class for interacting with the Intan RHX system over TCP/IP.

        Inherits:
            RHXConfig: Provides configuration and command utilities.

        Responsibilities:
            - Connects to command and data ports
            - Streams and parses EMG data
            - Records EMG data into memory

        Parameters:
            host (str): IP address of the RHX server.
            command_port (int): TCP port for command communication.
            data_port (int): TCP port for waveform data.
            num_channels (int): Number of channels to collect.
            sample_rate (float): Expected EMG sample rate.
            verbose (bool): Enable debug logging.
        """
    def __init__(self, host="127.0.0.1", command_port=5000, data_port=5001, num_channels=128, sample_rate=None,
                 buffer_duration_sec=5, verbose=False):
        self.host = host
        self.command_port = command_port
        self.data_port = data_port
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.verbose = verbose
        self.connected = False

        # ---- Streaming buffer for real-time access ----
        self.buffer_duration_sec = buffer_duration_sec  # Seconds of history to keep, adjust as needed
        self.circular_buffer = None
        self.circular_idx = 0
        self.buffer_lock = threading.Lock()
        self.streaming_thread = None
        self.streaming = False

        # Attempt connection to device
        self.connect()

        # Initialize buffer after sample_rate is known
        if self.sample_rate is None:
            self.sample_rate = self.get_sample_rate()
        self.init_circular_buffer()


        # Inherit the commands from the configuration class
        super().__init__(self.command_socket, verbose=verbose)

        if self.sample_rate is None:
            self.sample_rate = self.get_sample_rate()

    def connect(self):
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.command_socket.connect((self.host, self.command_port))

        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.connect((self.host, self.data_port))
        self.data_socket.settimeout(1.0)

    def receive_data(self, buffer, read_size):
        try:
            data = self.data_socket.recv(read_size)
            if data:
                buffer.extend(data)
        except socket.timeout:
            pass
        return buffer

    def init_circular_buffer(self):
        buffer_length = int(self.sample_rate * self.buffer_duration_sec)
        self.circular_buffer = np.zeros((self.num_channels, buffer_length), dtype=np.float32)
        self.circular_idx = 0


    def parse_emg_stream(self, raw_bytes, return_all_timestamps=True):
        idx = 0
        timestamps = []
        channel_data = [[] for _ in range(self.num_channels)]

        bytes_per_sample = 4 + 2 * self.num_channels
        bytes_per_block = 4 + FRAMES_PER_BLOCK * bytes_per_sample

        while idx + bytes_per_block <= len(raw_bytes):
            if unpack('<I', raw_bytes[idx:idx + 4])[0] != MAGIC_NUMBER:
                idx += 1
                continue

            idx += 4
            for _ in range(FRAMES_PER_BLOCK):
                ts = unpack('<i', raw_bytes[idx:idx + 4])[0]
                if return_all_timestamps:
                    timestamps.append(ts)
                last_ts = ts
                idx += 4

                for ch in range(self.num_channels):
                    val = unpack('<H', raw_bytes[idx:idx + 2])[0]
                    voltage = 0.195 * (val - 32768)
                    channel_data[ch].append(voltage)
                    idx += 2

        emg_array = np.array(channel_data, dtype=np.float32)
        bytes_consumed = idx
        if return_all_timestamps:
            return emg_array, np.array(timestamps, dtype=np.int64), bytes_consumed
        else:
            return emg_array, last_ts if 'last_ts' in locals() else None, bytes_consumed

    def _streaming_worker(self):
        rolling_buffer = bytearray()
        buffer_size = 4 + FRAMES_PER_BLOCK * (4 + 2 * self.num_channels)
        self.set_run_mode("run")

        try:
            while self.streaming:
                rolling_buffer = self.receive_data(rolling_buffer, buffer_size)
                emg_data, _, consumed = self.parse_emg_stream(rolling_buffer)
                rolling_buffer = rolling_buffer[consumed:]

                if emg_data is not None:
                    n = emg_data.shape[1]
                    with self.buffer_lock:
                        idx = self.circular_idx
                        buf_len = self.circular_buffer.shape[1]
                        if n >= buf_len:
                            self.circular_buffer[...] = emg_data[:, -buf_len:]
                            self.circular_idx = 0
                        else:
                            end_idx = idx + n
                            if end_idx < buf_len:
                                self.circular_buffer[:, idx:end_idx] = emg_data
                            else:
                                part1 = buf_len - idx
                                self.circular_buffer[:, idx:] = emg_data[:, :part1]
                                self.circular_buffer[:, :n - part1] = emg_data[:, part1:]
                            self.circular_idx = (idx + n) % buf_len
                # Optional: Sleep a tiny bit if you want to reduce CPU usage
                # time.sleep(0.001)
        finally:
            self.set_run_mode("stop")

    def start_streaming(self):
        if self.streaming:
            print("Already streaming")
            return
        self.streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.streaming_thread.start()

    def stop_streaming(self):
        self.streaming = False
        if self.streaming_thread is not None:
            self.streaming_thread.join()

    def get_latest_window(self, duration_ms):
        num_samples = int(self.sample_rate * duration_ms / 1000)
        buf_len = self.circular_buffer.shape[1]
        with self.buffer_lock:
            idx = self.circular_idx
            if num_samples > buf_len:
                raise ValueError("Requested window exceeds buffer size")
            start_idx = (idx - num_samples) % buf_len
            if start_idx < idx:
                window = self.circular_buffer[:, start_idx:idx]
            else:
                # Wrap-around
                window = np.hstack([self.circular_buffer[:, start_idx:], self.circular_buffer[:, :idx]])
        return window


    def record(self, duration_sec=10, verbose=True):
        """
        Record EMG data for a specified duration.

        Parameters:
            duration_sec (int): Number of seconds to record.
            verbose (bool): Whether to print stream rate.

        Returns:
            np.ndarray: EMG data array of shape (channels, samples)
        """
        total_samples = int(self.sample_rate * duration_sec)
        collected_emg = np.zeros((self.num_channels, total_samples), dtype=np.float32)
        write_index = 0
        rolling_buffer = bytearray()
        buffer_size = 4 + FRAMES_PER_BLOCK * (4 + 2 * self.num_channels)

        sample_counter = 0
        last_print = time.time()

        self.set_run_mode("run")
        if verbose:
            print(f"[→] Recording {duration_sec}s of EMG data...")

        try:
            while write_index < total_samples:
                rolling_buffer = self.receive_data(rolling_buffer, buffer_size)
                emg_data, _, consumed = self.parse_emg_stream(rolling_buffer)
                rolling_buffer = rolling_buffer[consumed:]

                if emg_data is not None:
                    n = emg_data.shape[1]
                    store = min(n, total_samples - write_index)
                    collected_emg[:, write_index:write_index + store] = emg_data[:, :store]
                    write_index += store
                    sample_counter += store

                now = time.time()
                if now - last_print >= 1.0 and verbose:
                    rate = sample_counter / (now - last_print)
                    print(f"[📊] Rate: {rate:.2f} samples/sec")
                    last_print = now
                    sample_counter = 0

        finally:
            self.set_run_mode("stop")
            return collected_emg

    def close(self, stop_after_disconnect=True):
        if stop_after_disconnect:
            if self.get_run_mode() == 'run':
                self.set_run_mode("stop")
                if self.verbose:
                    print("Runmode set to stop before closing.")
        self.command_socket.close()
        self.data_socket.close()

    def record_to_file(self, path, duration_sec=10):
        emg = self.record(duration_sec)
        np.savez(path, emg=emg, sample_rate=self.sample_rate)
        if self.verbose:
            print(f"Saved EMG to: {path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

VALID_PREFIXES = ("set ", "get ", "execute ")



# class OldIntanRHXDevice(DataStreamer):
#     def __init__(self, host='localhost', command_port=5000, waveform_port=5001, timeout=10.0, send_delay=0.01,
#                  runmode_stop_on_connect=True, runmode_stop_on_disconnect=True, clear_data_outputs=True, verbose=False):
#         self.host = host
#         self.command_port = command_port
#         self.waveform_port = waveform_port
#         self.send_delay = send_delay
#         self.verbose = verbose
#         self.enabled_channels = []
#         self.blocks_per_write = 1
#         self.connected = False
#
#         self.stream_thread = None
#         self._stream_active = threading.Event()
#
#         # TCP Sockets
#         self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.command_socket.settimeout(timeout)
#         self.waveform_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.waveform_socket.settimeout(timeout)
#
#         # Command buffer thread
#         self.command_queue = queue.Queue()
#         self.command_thread = threading.Thread(target=self._command_worker, daemon=True)
#         self.command_thread.start()
#
#         try:
#             self.command_socket.connect((self.host, self.command_port))
#             self.waveform_socket.connect((self.host, self.waveform_port))
#             self.connected = True
#             if self.verbose:
#                 print(f"[COMMAND] Connected to {self.host}:{self.command_port}")
#                 print(f"[WAVEFORM] Connected to {self.host}:{self.waveform_port}")
#
#             self.runmode_stop_on_disconnect = runmode_stop_on_disconnect
#             self.runmode_stop_on_connect = runmode_stop_on_connect
#
#             if self.runmode_stop_on_connect:
#                 cfg.set_run_mode(self, "stop")
#                 cfg.wait_until_runmode(self, "stop")
#
#             if clear_data_outputs:
#                 cfg.clear_data_outputs(self)
#
#             self.sample_rate = cfg.get_sample_rate(self)
#             self.timestep = 1.0 / self.sample_rate
#
#         except socket.error as e:
#             print(f"[ERROR] Failed to connect: {e}")
#             self.connected = False
#
#     def send(self, message):
#         """Queue a command to be sent to the RHX device."""
#         message = message.strip()  # Remove accidental newlines or whitespace
#         if not message or not message.lower().startswith(VALID_PREFIXES):
#             print(f"[WARN] Invalid or blank command skipped: '{message}'")
#             return
#         #message += "\n"
#         self.command_queue.put(message)
#
#     def recv(self, size=1024):
#         """Receive response from command socket."""
#         try:
#             return self.command_socket.recv(size)
#         except socket.timeout:
#             print("[COMMAND] Receive timed out.")
#             return b""
#         except socket.error as e:
#             print(f"[COMMAND] Receive error: {e}")
#             return b""
#
#     def _command_worker(self):
#         """Background thread that sends queued commands with delays."""
#         while True:
#             try:
#                 command = self.command_queue.get()
#                 if self.verbose:
#                     print(f"[SEND] {command.strip()}")
#                 self.command_socket.sendall(command.encode("utf-8"))
#                 time.sleep(self.send_delay)
#                 self.command_queue.task_done()
#             except Exception as e:
#                 print(f"[COMMAND THREAD ERROR] {e}")
#
#     def stream_and_write(self, n_frames, file=None, buffer=None, callback=None):
#         """
#         Streams EMG data and optionally writes to file or buffer.
#         :param n_frames: Number of samples to collect per channel.
#         :param file: A binary file object opened in 'wb' or 'ab' mode.
#         :param buffer: A preallocated NumPy array or list to write data into.
#         :param callback: A function to call with each (timestamp, data_chunk).
#         :return: tuple (timestamps, data_chunk)
#         """
#         timestamps, data = self.stream(n_frames=n_frames)
#
#         if file is not None:
#             data.T.tofile(file)  # Write as (samples, channels)
#
#         if buffer is not None:
#             buffer.append(data)
#
#         if callback is not None:
#             callback(timestamps, data)
#
#         return timestamps, data
#
#
#     def flush_commands(self):
#         """Block until all queued commands are sent."""
#         self.command_queue.join()
#
#     def get_enabled_channels(self):
#         return self.enabled_channels
#
#     def configure(self, channels, port='a', sample_rate=None, blocks_per_write=None,
#                   enable_wide=True, enable_high=False, enable_low=False, enable_spike=False):
#         """
#         Bulk device configuration helper.
#         Args:
#             channels: list of int
#             port: Port to configure (a, b, c, d)
#             sample_rate: optional new rate
#             blocks_per_write: how many blocks before TCP write
#             enable_*: stream types
#         """
#         if sample_rate is not None:
#             cfg.set_sample_rate(self, sample_rate)
#             self.sample_rate = sample_rate
#             self.timestep = 1.0 / sample_rate
#
#         if blocks_per_write is not None:
#             cfg.set_blocks_per_write(self, blocks_per_write)
#             self.blocks_per_write = blocks_per_write
#
#         self.enabled_channels = list(channels)
#         for ch in channels:
#             cfg.set_channel(
#                 self, port=port, channel=ch,
#                 enable_wide=enable_wide,
#                 enable_high=enable_high,
#                 enable_low=enable_low,
#                 enable_spike=enable_spike
#             )
#
#     def close(self):
#         """Clean shutdown and disconnect."""
#         try:
#             if self.runmode_stop_on_disconnect:
#                 cfg.set_run_mode(self, "stop")
#                 cfg.wait_until_runmode(self, "stop")
#                 if self.verbose:
#                     print("Runmode set to stop before closing.")
#             self.command_socket.close()
#             self.waveform_socket.close()
#             print("[COMMAND] Connection closed.")
#             print("[WAVEFORM] Connection closed.")
#         except Exception as e:
#             print(f"[CLOSE ERROR] {e}")
#
#     def start_background_stream(self, n_frames, interval_sec=0.25, callback=None):
#         def _worker():
#             while self._stream_active.is_set():
#                 self.stream_and_write(n_frames, callback=callback)
#                 time.sleep(interval_sec)
#
#         self._stream_active.set()
#         self.stream_thread = threading.Thread(target=_worker, daemon=True)
#         self.stream_thread.start()
#
#
#     def stop_background_stream(self):
#         self._stream_active.clear()
#         if self.stream_thread:
#             self.stream_thread.join()
#         if self.verbose:
#             print("[STREAM] Background stream stopped.")
#
