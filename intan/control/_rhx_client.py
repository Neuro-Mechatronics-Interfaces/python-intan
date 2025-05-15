import time
import socket
import numpy as np
from collections import deque
from threading import Thread, Event

MAGIC_NUMBER = 0x2ef07a08
FRAMES_PER_BLOCK = 128
SAMPLE_SCALE_FACTOR = 0.195
ANALOG_SCALE_FACTOR = 312.5e-6


def read_uint32(array, index):
    """ Reads a 32-bit unsigned integer from a byte array."""
    return int.from_bytes(array[index:index + 4], 'little'), index + 4


def read_int32(array, index):
    """ Reads a 32-bit signed integer from a byte array."""
    return int.from_bytes(array[index:index + 4], 'little', signed=True), index + 4


def read_uint16(array, index):
    """ Reads a 16-bit unsigned integer from a byte array."""
    return int.from_bytes(array[index:index + 2], 'little'), index + 2


class IntanRHXClient:
    """    A client for communicating with the Intan RHX system over TCP/IP."""
    def __init__(self, host='localhost', command_port=5000, data_port=5001, timeout=10, n_channels=8, buffer_len=2000, verbose=False):
        self.host = host
        self.command_addr = (host, command_port)
        self.data_addr = (host, data_port)
        self.verbose = verbose

        # Command socket
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.command_socket.settimeout(timeout)
        self.command_socket.connect(self.command_addr)

        # Data socket
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.settimeout(timeout)
        self.data_socket.connect(self.data_addr)

        self.n_channels = n_channels  # will be set during setup
        self.buffers = []  # deque per channel
        self._sampling_thread = None
        self._stop_event = Event()
        self.setup_streaming(n_channels, buffer_len)

        self.is_streaming = False


    def send(self, message):
        """ Sends a message to the RHX system."""
        try:
            self.command_socket.sendall(message.encode('utf-8'))
            time.sleep(0.01)  # Small delay to ensure the message is sent
        except socket.error as e:
            print(f"[COMMAND] Send error: {e}")

    def recv(self, size=1024):
        """ Receives a message from the RHX system."""
        try:
            return self.command_socket.recv(size)
        except socket.timeout:
            print("[COMMAND] Receive timed out.")
            return b""
        except socket.error as e:
            print(f"[COMMAND] Receive error: {e}")
            return b""

    def start(self):
        """ Starts the RHX system."""
        self.send("set runmode run")
        if self.verbose: print("RHX system started.")

    def stop(self):
        """ Stops the RHX system."""
        self.send("set runmode stop")
        if self.verbose: print("RHX system stopped.")

    def clear_data_outputs(self):
        """ Clears all data outputs."""
        self.send("execute clearalldataoutputs")
        if self.verbose: print("All data outputs cleared.")

    def setup_streaming(self, n_channels, buffer_len=2000):
        """ Sets up the streaming configuration. """
        self.n_channels = n_channels
        self.buffer_len = buffer_len
        self.buffers = [deque(maxlen=self.buffer_len) for _ in range(n_channels)]

    def set_channel(self, port='a', channel=0, enable_low=False, enable_high=False, enable_wide=True,
                    enable_spike=False):
        """ Sets the channel configuration.

        Parameters:
            port (str): The port to configure ('a', 'b', 'c', 'd').
            channel (int): The channel number (0-127).
            enable_low (bool): Enable low frequency output.
            enable_high (bool): Enable high frequency output.
            enable_wide (bool): Enable wideband output.
            enable_spike (bool): Enable spike output.
        """
        flags = {
            'tcpdataoutputenabled': enable_wide,
            'tcpdataoutputenabledhigh': enable_high,
            'tcpdataoutputenabledlow': enable_low,
            'tcpdataoutputenabledspike': enable_spike,
        }
        for suffix, value in flags.items():
            msg = f"set {port}-{channel:03d}.{suffix} {'true' if value else 'false'}"
            self.send(msg)

    def enable_digital_input(self, dig_input=1):
        """ Enables a digital input """
        self.send(f"set DIGITAL-IN-{dig_input}.enabled true")
        self.send(f"set DIGITAL-IN-{dig_input}.recordingenabled true")

    def disable_digital_input(self, dig_input=1):
        """ Disables a digital input """
        self.send(f"set DIGITAL-IN-{dig_input}.enabled false")
        self.send(f"set DIGITAL-IN-{dig_input}.recordingenabled false")

    def set_blocks_per_write(self, num_blocks):
        """ Sets the number of blocks per write for the TCP output bufer (Example: 1). """
        self.send(f"set TCPNumberDataBlocksPerWrite {num_blocks}")

    def get_sample_rate(self):
        """ Gets the sample rate in Hz. """
        self.send("get sampleratehertz")
        resp = self.recv().decode('utf-8')
        return float(resp.split()[-1])

    def get_latest_sample(self):
        """Returns the latest sample from each channel."""
        return [buf[-1] if buf else 0 for buf in self.buffers]

    def get_samples(self, channel=0, n_samples=1):
        """Returns up to `n_samples` from the given channel's buffer."""
        if 0 <= channel < self.n_channels:
            return list(self.buffers[channel])[-n_samples:]
        else:
            print(f"Invalid channel: {channel}")
            return []

    def start_streaming(self, ana_ch=0, dig_in=False, blocks=1, rate=2000):
        """Starts a thread to continuously collect and buffer RHX data."""
        self._stop_event.clear()

        def run():
            while not self._stop_event.is_set():
                try:
                    ts, amp_data, _, _ = self.read_waveform_block(
                        num_amplifier_channels=self.n_channels,
                        num_analog_channels=ana_ch,
                        dig_in_present=dig_in,
                        blocks_per_read=blocks,
                        sample_rate=rate
                    )
                    if amp_data is not None:
                        for ch in range(min(self.n_channels, amp_data.shape[0])):
                            self.buffers[ch].extend(amp_data[ch])
                except (socket.timeout, socket.error, Exception) as e:
                    print(f"Error in streaming thread: {e}")
                    self._stop_event.set()
                    break

        self._sampling_thread = Thread(target=run, daemon=True)
        self._sampling_thread.start()
        self.is_streaming = True
        print(f"RHXClient streaming started for {self.n_channels} channels at {rate} Hz.")

    def stop_streaming(self):
        """Stops the streaming thread."""
        self._stop_event.set()
        if self._sampling_thread:
            self._sampling_thread.join()
        self.is_streaming = False
        print("RHXClient streaming stopped.")

    def reconnect(self):
        print("Attempting to reconnect to RHX...")
        self.command_socket.close()
        self.data_socket.close()
        self.__init__(self.host, self.command_addr[1], self.data_addr[1], n_channels=self.n_channels,
                      buffer_len=self.buffer_len)

    def read_waveform_block(self, num_amplifier_channels, num_analog_channels, dig_in_present, blocks_per_read=100,
                            sample_rate=4000):
        """ Reads a waveform block from the RHX system.

        Parameters:
            num_amplifier_channels (int): Number of amplifier channels.
            num_analog_channels (int): Number of analog channels.
            dig_in_present (bool): Whether digital input is present.
            blocks_per_read (int): Number of blocks to read.
            sample_rate (int): Sample rate in Hz.

        Returns:
            tuple: A tuple containing:
                - timestamps (numpy.ndarray): Timestamps of the data.
                - amp_data (numpy.ndarray): Amplifier data.
                - ana_data (numpy.ndarray): Analog data.
                - dig_data (numpy.ndarray): Digital data.
        """

        timestep = 1 / sample_rate
        frame_size = 4 + 2 * (num_amplifier_channels + num_analog_channels + (1 if dig_in_present else 0))
        block_size = FRAMES_PER_BLOCK * frame_size + 4
        total_bytes = blocks_per_read * block_size

        try:
            raw_data = self.data_socket.recv(total_bytes)
            print(f" Raw data length: {len(raw_data)}")
            print(f" Total bytes expected: {total_bytes}")
            if len(raw_data) < total_bytes:
                print("Insufficient data received.")
                return None, None, None, None

            timestamps, amp_data, ana_data, dig_data = [], [], [], []
            index = 0
            for _ in range(blocks_per_read):
                magic, index = read_uint32(raw_data, index)
                if magic != MAGIC_NUMBER:
                    print("Invalid magic number, skipping block")
                    index += FRAMES_PER_BLOCK * frame_size
                    continue

                block = raw_data[index:index + FRAMES_PER_BLOCK * frame_size]
                index += FRAMES_PER_BLOCK * frame_size
                ts, amp, ana, dig = self._process_block(
                    block, num_amplifier_channels, num_analog_channels, dig_in_present, sample_rate
                )
                timestamps.extend(ts)
                amp_data.append(amp)
                if ana is not None: ana_data.append(ana)
                if dig is not None: dig_data.append(dig)

            return (
                np.array(timestamps),
                np.concatenate(amp_data, axis=1),
                np.concatenate(ana_data, axis=1) if ana_data else None,
                np.concatenate(dig_data, axis=1) if dig_data else None,
            )
        except socket.timeout:
            #print("Socket timeout: no data received.")
            return None, None, None, None
        except Exception as e:
            print(f"Error reading waveform block: {e}")
            return None, None, None, None

    def _process_block(self, data, n_amp, n_ana, dig_in, sample_rate):
        """ Processes a block of data.

        Parameters:
            data (bytes): The block of data.
            n_amp (int): Number of amplifier channels.
            n_ana (int): Number of analog channels.
            dig_in (bool): Whether digital input is present.
            sample_rate (int): Sample rate in Hz.

        Returns:
            tuple: A tuple containing:
                - timestamps (numpy.ndarray): Timestamps of the data.
                - amp_data (numpy.ndarray): Amplifier data.
                - ana_data (numpy.ndarray): Analog data.
                - dig_data (numpy.ndarray): Digital data.
        """
        timestep = 1 / sample_rate
        frames = len(data) // (4 + 2 * (n_amp + n_ana + (1 if dig_in else 0)))
        ts = np.zeros(frames)
        amp = np.zeros((n_amp, frames))
        ana = np.zeros((n_ana, frames)) if n_ana > 0 else None
        dig = np.zeros((1, frames)) if dig_in else None

        idx = 0
        for i in range(frames):
            timestamp, idx = read_int32(data, idx)
            ts[i] = timestamp * timestep
            for ch in range(n_amp):
                val, idx = read_uint16(data, idx)
                amp[ch, i] = SAMPLE_SCALE_FACTOR * (val - 32768)
            if n_ana > 0:
                for ch in range(n_ana):
                    val, idx = read_uint16(data, idx)
                    ana[ch, i] = ANALOG_SCALE_FACTOR * (val - 32768)
            if dig_in:
                val, idx = read_uint16(data, idx)
                dig[0, i] = val

        return ts, amp, ana, dig

    def close(self):
        """ Closes the connection to the RHX system. """
        self.command_socket.close()
        self.data_socket.close()
        print("RHXClient connection closed.")

