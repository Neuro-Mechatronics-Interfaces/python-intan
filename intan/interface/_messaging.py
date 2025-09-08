import time
import socket
import numpy as np
import collections

            
class TCPClient:
    """ Class for managing TCP connections to the Intan system."""
    def __init__(self, name, host, port, buffer=1024):
        """Initializes the TCPClient.

        Args:
            name (str): Name of the client.
            host (str): The IP address of the host to connect to.
            port (int): The port number to connect to.
            buffer (int): The buffer size for receiving data.
        """
        self.name = name
        self.host = host
        self.port = port
        self.buffer = buffer
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(5)  # Timeout after 5 seconds if no data received

    def connect(self):
        """Connects to the host server."""
        self.s.setblocking(True)
        self.s.connect((self.host, self.port))
        self.s.setblocking(False)

    def send(self, data, wait_for_response=False):
        """Sends data to the host server and optionally waits for a response."""
        # convert data to bytes if it is not already
        if not isinstance(data, bytes):
            data = data.encode()
        self.s.sendall(data)
        time.sleep(0.01)

        if wait_for_response:
            return self.read()

    def read(self, bytes=None):
        """ Reads and returns bytes by the buffer size unless specified """
        if bytes is None:
            return self.s.recv(self.buffer)
        else:
            return self.s.recv(bytes)

    def close(self):
        self.s.close()


class RingBuffer:
    """Fixed-size ring buffer for storing recent data up to max number of samples."""

    def __init__(self, num_channels, size_max=4000):
        self.max = size_max
        self.samples = collections.deque(maxlen=size_max)  # Stores (timestamp, data)
        self.num_channels = num_channels

    def append(self, t, x):
        """Adds a new sample to the buffer, automatically removing the oldest if full."""
        x = np.array(x, dtype=np.float32).reshape(1, -1)  # Ensure it remains multi-channel
        self.samples.append((t, x))

    def get_samples(self, n=1):
        """Returns the last n samples from the buffer as NumPy arrays."""
        if len(self.samples) < n:
            raise ValueError("Requested more samples than available in the buffer.")

        recent_samples = list(self.samples)[-n:]  # Get last n elements
        timestamps, data = zip(*recent_samples)  # Separate timestamps and data

        # Convert to NumPy arrays and ensure shape is correct
        data_array = np.vstack(data)  # Stack samples to shape (n, num_channels)

        return data_array, np.array(timestamps)

    def is_full(self):
        """Checks if the buffer is at max capacity."""
        return len(self.samples) == self.max