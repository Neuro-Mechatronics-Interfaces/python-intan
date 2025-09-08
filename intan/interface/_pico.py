import serial
import collections
from statistics import mode, StatisticsError


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
