import time
import socket
import serial
import numpy as np
import collections
from statistics import mode, StatisticsError
import collections


class PicoMessager:
    """Class for managing serial communication with a Raspberry Pi Pico."""

    def __init__(self, port='COM13', baudrate=9600, buffer_size=1, verbose=False):
        """Initializes the PicoMessager.

        Args:
            port (str): The serial port to connect to (e.g., 'COM13').
            baudrate (int): The baud rate for serial communication.
            buffer_size (int): The number of past gestures to keep in the buffer.
            verbose (bool): Whether to print incoming messages automatically.
        """
        self.port = port
        self.baudrate = baudrate
        self.buffer = collections.deque(maxlen=buffer_size)
        self.current_gesture = None  # Keep track of the current gesture being sent
        self.verbose = verbose
        self.running = True  # To control the connection

        # Connect to the Pico via serial
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to Pico on {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"Error connecting to Pico: {e}")
            self.serial_connection = None

    def dump_output(self, mute=False):
        """Reads all available bytes from the serial connection and prints them.

        This function reads all incoming messages from the Pico until there are no more bytes left.
        """
        if self.serial_connection and self.serial_connection.is_open:
            try:
                if self.serial_connection.in_waiting > 0:
                    incoming_message = self.serial_connection.readline().decode().strip()
                    if incoming_message and not mute:
                        print(f"Message from Pico: {incoming_message}")
            except serial.SerialException as e:
                print(f"Error reading message: {e}")

    def update_gesture(self, new_gesture):
        """Updates the gesture buffer and sends the most common gesture if it changes.

        Args:
            new_gesture (str): The newly detected gesture.
        """
        # Update the gesture buffer
        self.buffer.append(new_gesture)

        # Find the most common gesture in the buffer
        try:
            most_common_gesture = mode(self.buffer)
        except StatisticsError:
            # If mode cannot be determined, continue without change
            most_common_gesture = None

        # If the most common gesture changes, update current_gesture and send the new message
        if most_common_gesture and most_common_gesture != self.current_gesture:
            self.current_gesture = most_common_gesture
            self.send_message(self.current_gesture)

    def send_message(self, message):
        """Sends a message to the Pico over serial.

        Args:
            message (str): The message to send.
        """
        if self.serial_connection and self.serial_connection.is_open:
            try:
                formatted_message = f"{message};"  # Add terminator character to the message
                self.serial_connection.write(formatted_message.encode())
                print(f"Sent message to Pico: {formatted_message}")
            except serial.SerialException as e:
                print(f"Error sending message: {e}")
        else:
            print("Serial connection not available or not open.")

    def close_connection(self):
        """Closes the serial connection to the Pico and stops the background listener."""
        # Stop the background thread
        self.running = False

        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("Closed connection to Pico.")
            
            
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

class OldRingBuffer:
    """Fixed-size ring buffer for storing recent data up to max number of samples. """
    def __init__(self, num_channels, size_max=4000):
        self.max = size_max
        self.samples = np.zeros((size_max, num_channels), dtype=np.float32)
        self.timestamp = np.zeros(size_max, dtype=np.float32)
        self.cur = 0
        self.size = 0

    def append(self, t, x):
        """Adds a new sample to the buffer, removing the oldest sample if full."""
        self.samples[self.cur] = x
        self.timestamp[self.cur] = t
        self.cur = (self.cur + 1) % self.max
        self.size = min(self.size + 1, self.max)

    def get_samples(self, n=1):
        """ Returns the current data in the ring buffer. """
        #return self.samples
        if n > self.size:
            raise ValueError("Requested more samples than available in the buffer.")
        end_idx = self.cur if self.size == self.max else self.size
        start_idx = (end_idx - n) % self.max
        if start_idx < end_idx:
            return self.samples[start_idx:end_idx], self.timestamp[start_idx:end_idx]
        else:
            # When the wrap-around occurs
            return np.vstack((self.samples[start_idx:], self.samples[:end_idx])), \
                np.hstack((self.timestamp[start_idx:], self.timestamp[:end_idx]))

    def is_full(self):
        return self.size == self.max


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