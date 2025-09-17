"""
MIT License

Copyright (c) 2025 Neuromechatronics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import time
import socket
import numpy as np
from struct import unpack
from collections import deque
import threading
from typing import Optional
from ._lsl_options import LSLOptions
from ._lsl_publisher import LSLNumericPublisher, LSLMarkerPublisher
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
    def __init__(self,
                 host="127.0.0.1",
                 command_port=5000,
                 data_port=5001,
                 num_channels=128,
                 sample_rate=None,
                 buffer_duration_sec=5,
                 auto_start=False,
                 use_lsl: bool = False,
                 lsl_options: Optional[LSLOptions] = None,
                 verbose=False):
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
        self.connected = False

        # streaming constants (bytes math)
        self.bytes_per_frame = 4 + 2 * self.num_channels  # ts(int32) + C*uint16
        self.bytes_per_block = 4 + FRAMES_PER_BLOCK * self.bytes_per_frame
        self.blocks_per_write = 1
        self.read_size = self.bytes_per_block * 1  # request ≥4 blocks per recv by default
        self._synced = False  # parser sync state across reads

        # Attempt connection to device
        self.connect()

        if self.connected:
            # Initialize buffer after sample_rate is known
            if self.sample_rate is None:
                self.sample_rate = self.get_sample_rate()
            self._sample_counter = 0
            self.effective_fs = float(self.sample_rate)
            self.init_circular_buffer()

            # Inherit the commands from the configuration class
            super().__init__(self.command_socket, verbose=verbose)

        # LSL Setup
        self.use_lsl = use_lsl
        self.lsl = lsl_options or LSLOptions()
        self._lsl_numeric: Optional[LSLNumericPublisher] = None
        self._lsl_markers: Optional[LSLMarkerPublisher] = None
        self._lsl_chunk = int(getattr(self.lsl, "chunk_size", 32) or 32) # For predictable latency

        if auto_start:
            self.start_streaming()

    def _lsl_open_outlets(self):
        """Create LSL outlets using the latest fs/ch. Fail-soft if unavailable."""
        if not self.use_lsl:
            return
        try:
            # Prefer device-reported numbers
            fs = float(self.sample_rate or self.lsl.fs or 2000.0)
            ch = int(self.num_channels or self.lsl.channels or 1)

            # Create numeric outlet if missing
            if self._lsl_numeric is None:
                self._lsl_numeric = LSLNumericPublisher(
                    name=self.lsl.numeric_name,
                    stype=self.lsl.numeric_type,
                    fs=fs,
                    channels=ch,
                    source_id=self.lsl.source_id,
                    channel_labels=self.lsl.channel_labels,
                )

            # (Optional) markers
            if self.lsl.with_markers and self._lsl_markers is None:
                self._lsl_markers = LSLMarkerPublisher(
                    name=self.lsl.marker_name,
                    stype=self.lsl.marker_type,
                    source_id=f"{self.lsl.source_id}-markers",
                )
        except Exception as e:
            # Don’t break acquisition if pylsl is missing or outlet creation fails
            if self.verbose:
                print(f"[LSL] Outlet init failed: {e}")
            self._lsl_numeric = None
            self._lsl_markers = None

    def _lsl_close_outlets(self):
        """Close outlets and clear buffers."""
        try:
            if self._lsl_markers:
                self._lsl_markers.close()
        finally:
            self._lsl_markers = None
        try:
            if self._lsl_numeric:
                self._lsl_numeric.close()
        finally:
            self._lsl_numeric = None

    def _streaming_worker(self):
        rolling_buffer = bytearray()
        #buffer_size = 4 + FRAMES_PER_BLOCK * (4 + 2 * self.num_channels)
        self.set_run_mode("run")

        try:
            while self.streaming:
                rolling_buffer = self.receive_data(rolling_buffer, self.read_size)
                emg_data, timestamps, consumed, self._synced = self.parse_emg_stream_fast(rolling_buffer)
                rolling_buffer = rolling_buffer[consumed:]

                if emg_data is not None:
                    n = emg_data.shape[1]
                    with self.buffer_lock:
                        idx = self.circular_idx
                        buf_len = self.circular_buffer.shape[1]
                        if n >= buf_len:
                            self.circular_buffer[:,:] = emg_data[:, -buf_len:]
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

                    if self.use_lsl and self._lsl_numeric is not None:
                        try:
                            C, N = emg_data.shape
                            cs = self._lsl_chunk  # e.g., 32 samples/packet
                            # precompute timestamps once for this burst (device timebase in seconds)
                            t_all = self._make_timestamps(N)  # shape (N,)

                            for i0 in range(0, N, cs):
                                i1 = min(i0 + cs, N)
                                blk = emg_data[:, i0:i1]  # (C, K)
                                self._lsl_numeric.push_chunk(blk.T)  # stamp on send

                        except Exception as e:
                            if self.verbose:
                                print(f"[LSL] push failed: {e}")
                            # Don’t kill the stream; just drop LSL outlet if it misbehaves
                            self._lsl_numeric = None

                # Optional: Sleep a tiny bit if you want to reduce CPU usage
                # time.sleep(0.001)
        finally:
            self.set_run_mode("stop")

    def _make_timestamps(self, n: int) -> np.ndarray:
        i0 = self._sample_counter
        self._sample_counter += n
        return (i0 + np.arange(n, dtype=float)) / self.effective_fs

    def _update_read_size(self):
        self.bytes_per_frame = 4 + 2 * self.num_channels
        self.bytes_per_block = 4 + FRAMES_PER_BLOCK * self.bytes_per_frame
        #  allow 1 block (previously max(4, …) forced bigger reads → higher latency)
        self.read_size = self.bytes_per_block * max(1, int(getattr(self, "blocks_per_write", 1)))

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            #print(f'kwarg {item}')
            if "blocks_per_write" in key:
                self.set_blocks_per_write(value)
                # after calling set_blocks_per_write(n) elsewhere, remember it
                self.blocks_per_write = max(1, int(value))
                self._update_read_size()                 # <— new

            elif "enable_wide_channel" in key:
                self.enable_wide_channel(value)
                self._update_read_size()                 # <— in case channel count/output set changes

    def connect(self):
        try:
            self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.command_socket.connect((self.host, self.command_port))

            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.connect((self.host, self.data_port))

            self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # 1MB recv buffer
            try:
                self.data_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # lower jitter
            except Exception:
                pass
            self.data_socket.settimeout(0.005)  # short timeout avoids 1 Hz throttling

            #self.data_socket.settimeout(1.0)
            self.connected = True
            self.connection_lost = False
            self.reconnect_attempts = 0

        except Exception as e:
            print("Failed to initialize connection with 'Remote TCP Control'")

    def receive_data(self, buffer: bytearray, read_size: int, max_reads: int = 16):
        """Try to drain multiple chunks from the socket without blocking forever."""
        for _ in range(max_reads):
            try:
                chunk = self.data_socket.recv(read_size)
                if not chunk:
                    break
                buffer.extend(chunk)
                # If kernel returned less than we asked for, likely drained for now.
                if len(chunk) < read_size:
                    break
            except socket.timeout:
                break
        return buffer

    def init_circular_buffer(self):
        buffer_length = int(self.sample_rate * self.buffer_duration_sec)
        self.circular_buffer = np.zeros((self.num_channels, buffer_length), dtype=np.float32)
        self.circular_idx = 0

    def parse_emg_stream_fast(self, raw_bytes: bytearray, synced=True):
        C = self.num_channels
        frames = FRAMES_PER_BLOCK
        # sizes
        bytes_per_frame = 4 + 2 * C  # ts(int32) + C * uint16
        bytes_per_block = 4 + frames * bytes_per_frame  # magic + frames
        mv = memoryview(raw_bytes)

        # try to sync on magic once (cheap search)
        start = 0
        if not synced:
            magic = MAGIC_NUMBER.to_bytes(4, "little")
            pos = raw_bytes.find(magic)
            if pos == -1:
                return None, None, 0, False
            start = pos

        # how many full blocks do we have?
        available = (len(raw_bytes) - start)
        nblocks = available // bytes_per_block
        if nblocks <= 0:
            return None, None, start, True

        # dtype: one frame = {'ts': int32, 'v': (uint16, C)}
        frame_dtype = np.dtype([('ts', '<i4'), ('v', ('<u2', C))])
        block_dtype = np.dtype([('magic', '<u4'), ('data', (frame_dtype, frames))])

        # view all full blocks at once
        blocks = np.frombuffer(mv[start:start + nblocks * bytes_per_block], dtype=block_dtype)

        # sanity: mask blocks with correct magic
        good = (blocks['magic'] == MAGIC_NUMBER)
        blocks = blocks[good]
        if blocks.size == 0:
            return None, None, start + nblocks * bytes_per_block, True

        # stack frames across blocks → (total_frames, C)
        frames_arr = blocks['data'].reshape(-1)  # (nblocks*frames,)
        ts = frames_arr['ts'].astype(np.int64)  # (total_frames,)
        v = frames_arr['v'].astype(np.int32, copy=False)  # (total_frames, C)
        v -= 32768
        v *= 195
        #emg = (0.195 * (v - 32768)).astype(np.float32)  # µV
        emg = (v.astype(np.float32, copy=False) / 1000.0).T

        # rearrange to (C, total_frames) like your current API
        #emg = emg.T
        consumed = start + nblocks * bytes_per_block
        return emg, ts, consumed, True

    def parse_emg_stream(self, raw_bytes, return_all_timestamps=True):
        """Slow reference parser (struct.unpack in Python loops). Prefer parse_emg_stream_fast."""
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

    def start_streaming(self):
        if self.streaming:
            print("Already streaming")
            return

        # Ensure sample_rate/num_channels are known
        if self.sample_rate is None:
            self.sample_rate = self.get_sample_rate()
        self.effective_fs = float(self.sample_rate)
        self.init_circular_buffer()

        # Force low-latency mode (if possible)
        try:
            self.set_blocks_per_write(1)
            self.blocks_per_write = 1
            self._update_read_size()
        except Exception:
            pass

        # LSL outlets (if enabled)
        self._lsl_open_outlets()

        self.streaming = True
        self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self.streaming_thread.start()

    def stop_streaming(self):
        self.streaming = False
        if self.streaming_thread is not None:
            self.streaming_thread.join()
            self.streaming_thread = None

        self._lsl_close_outlets()

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

        sample_counter = 0
        last_print = time.time()

        self.set_run_mode("run")
        if verbose:
            print(f"[→] Recording {duration_sec}s of EMG data...")

        try:
            while write_index < total_samples:
                rolling_buffer = self.receive_data(rolling_buffer, self.read_size)
                emg_data, timestamps, consumed, self._synced = self.parse_emg_stream_fast(
                    rolling_buffer, synced=self._synced
                )
                if consumed:
                    del rolling_buffer[:consumed]

                if emg_data is not None:
                    n = emg_data.shape[1]
                    store = min(n, total_samples - write_index)
                    collected_emg[:, write_index:write_index + store] = emg_data[:, :store]
                    write_index += store
                    sample_counter += store

                    #if timestamps is not None and len(timestamps) > 0:
                    #    time_s[write_index - store:write_index] = timestamps[:store] / self.sample_rate


                now = time.time()
                if now - last_print >= 1.0 and verbose:
                    rate = sample_counter / (now - last_print)
                    print(f"[📊] Rate: {rate:.2f} samples/sec")
                    last_print = now
                    sample_counter = 0

        finally:
            self.set_run_mode("stop")
            #return time_s, collected_emg
            return collected_emg

    def close(self, stop_after_disconnect=True):
        if stop_after_disconnect:
            if self.get_run_mode() == 'run':
                self.set_run_mode("stop")
                if self.verbose:
                    print("Runmode set to stop before closing.")

        self._lsl_close_outlets()
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



