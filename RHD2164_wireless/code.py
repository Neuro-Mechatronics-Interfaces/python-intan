import os
import time
import math
import wifi
import socketpool
import microcontroller

import board
import busio
import digitalio

from rgbled import RGBLED


# ------------- Helper Functions ------------------------------------------------

def adc_to_mV(adc_value):
    """
    Converts the 16-bit output of the Intan RHD2164 ADC to millivolts.
    The chip is configured for ±5 mV range, giving:
      -32768 => -5.0 mV
      +32767 => +5.0 mV
    """
    return (adc_value / 32768.0) * 5.0


def clamp_if_outlier(value_mV, max_range=20.0):
    """
    If |value_mV| > 20 mV (well outside ±5mV), it's almost surely a corrupted sample.
    We clamp to 0.0 to keep filters from blowing up.
    """
    if abs(value_mV) > max_range:
        return 0.0
    return value_mV


def highpass_coefficients(cutoff_freq, fs):
    """
    Returns the alpha coefficient for a simple first-order high-pass filter.
    alpha ~ sets how quickly offset is removed.
    """
    RC = 1.0 / (2 * math.pi * cutoff_freq)
    dt = 1.0 / fs
    return RC / (RC + dt)


class HighPassFilter:
    """
    First-order Butterworth high-pass filter to remove DC drift/baseline wander.
    """

    def __init__(self, cutoff_freq, fs):
        self.alpha = highpass_coefficients(cutoff_freq, fs)
        self.prev_input = 0.0
        self.prev_output = 0.0

    def filter(self, new_input):
        """
        y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        """
        y = self.alpha * (self.prev_output + new_input - self.prev_input)
        self.prev_input = new_input
        self.prev_output = y
        return y


# --------------------------------------------------------------------------------
# A Combined Class for Wi-Fi + EMG SPI + TCP Socket
# --------------------------------------------------------------------------------
class EMGClient:
    """
    This class manages:
     - Wi-Fi connection using credentials in CIRCUITPY_WIFI_SSID/PASSWORD
     - TCP socket connection to a server running on your laptop
     - SPI-based reading of the Intan RHD2164 for EMG signals
     - Optional high-pass filtering

    Usage:
     1. Create an EMGClient instance with server IP/port, filter cutoff, etc.
     2. call client.run() to:
        a) Connect Wi-Fi
        b) Connect server
        c) Calibrate the Intan chip
        d) Continuously read data & send to server
    """

    def __init__(self, server_host: str, server_port: int, HP_CUTOFF: float, FS: int):
        """
        server_host: IP address of your laptop or external server
        server_port: TCP port number the server is listening on
        HP_CUTOFF: High-pass filter cutoff frequency (Hz)
        FS: sampling frequency (Hz) for the high-pass filter
        """
        self.server_host = server_host
        self.server_port = server_port
        self.sock = None
        self.ip = None

        # Increase CPU frequency (optional). 
        # Check your board's docs to confirm available frequencies.
        microcontroller.cpu.frequency = 200_000_000  # 200MHz for Pico W

        # SPI pin assignments for Pi Pico:
        self.SCLK = board.GP2
        self.MOSI = board.GP3
        self.MISO = board.GP4
        self.CS = board.GP5

        # Configure Chip Select (active LOW)
        self.cs = digitalio.DigitalInOut(self.CS)
        self.cs.direction = digitalio.Direction.OUTPUT
        self.cs.value = True  # Deselected by default

        # Initialize the SPI bus
        self.spi = busio.SPI(clock=self.SCLK, MOSI=self.MOSI, MISO=self.MISO)

        # Optional RGB LED on pins GP10-13 for status indication
        self.LED_OUT = board.GP10
        self.RED = board.GP11
        self.GREEN = board.GP12
        self.BLUE = board.GP13
        self.rgb = RGBLED(rpin=self.RED, gpin=self.GREEN, bpin=self.BLUE,
                          led_out=self.LED_OUT, set_color=[0, 0, 0])

        # Initialize high-pass filter
        self.hp_filter = HighPassFilter(HP_CUTOFF, FS)

    def connect_wifi(self, max_retries=5):
        """
        Connects to Wi-Fi using credentials stored in settings.toml
          [env]
          CIRCUITPY_WIFI_SSID=YourNetwork
          CIRCUITPY_WIFI_PASSWORD=YourPassword
        Retries if connection fails.
        """
        ssid = os.getenv('HOTSPOT_SSID')
        password = os.getenv('HOTSPOT_PASSWORD')

        if not ssid or not password:
            raise RuntimeError("Wi-Fi SSID or password not found in settings.toml!")

        print("\nConnecting to Wi-Fi...")
        for attempt in range(1, max_retries + 1):
            try:
                wifi.radio.connect(ssid, password)
                # Wait for an IP address
                while not wifi.radio.ipv4_address:
                    print("Waiting for IP...")
                    time.sleep(1)
                print(f"Connected to Wi-Fi! IP address: {wifi.radio.ipv4_address}")
                print("My MAC addr:", [hex(i) for i in wifi.radio.mac_address])
                self.ip = str(wifi.radio.ipv4_address)

                # Brief LED blink to indicate success
                self.rgb.set_color([0, 30, 0])
                time.sleep(0.4)
                self.rgb.set_color([0, 0, 0])
                time.sleep(0.4)
                self.rgb.set_color([0, 30, 0])
                time.sleep(0.4)
                self.rgb.set_color([0, 0, 0])
                time.sleep(0.4)
                self.rgb.set_color([0, 30, 0])
                time.sleep(0.4)
                self.rgb.set_color([0, 0, 0])
                return  # success
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")
                time.sleep(2)

        raise ConnectionError(f"Could not connect to Wi-Fi after {max_retries} attempts.")

    def connect_server(self, max_retries=5):
        """
        Opens a TCP socket to the specified server with simple retry logic.
        """
        print(f"\nConnecting to server at {self.server_host}:{self.server_port}...")
        pool = socketpool.SocketPool(wifi.radio)

        for attempt in range(1, max_retries + 1):
            try:
                self.sock = pool.socket(pool.AF_INET, pool.SOCK_STREAM)
                self.sock.connect((self.server_host, self.server_port))
                print("Connected to server!")
                self.rgb.set_color([0, 50, 0])
                time.sleep(1)
                return  # success
            except Exception as e:
                print(f"Server connection attempt {attempt} failed: {e}")
                time.sleep(2)

        raise ConnectionError(f"Could not connect to server after {max_retries} attempts.")

    # ---------------- SPI Helpers ----------------
    def _spi_lock_and_select(self):
        """Helper to lock SPI bus and pull CS low."""
        while not self.spi.try_lock():
            pass
        self.cs.value = False

    def _spi_unlock_and_deselect(self):
        """Helper to release SPI bus and pull CS high."""
        self.cs.value = True
        self.spi.unlock()

    def send_read_command(self, regnum: int) -> int:
        """
        Example: read from a register on the RHD chip.
        This might vary depending on which Intan commands you need.
        """
        command = (0b11000000 << 8) | (regnum << 8)
        buf = command.to_bytes(2, 'big')
        result = bytearray(2)

        self._spi_lock_and_select()
        self.spi.write_readinto(buf, result)
        self._spi_unlock_and_deselect()

        return int.from_bytes(result, 'big')

    def send_write_command(self, regnum: int, data: int):
        """
        Example: write a register on the RHD chip (e.g., amplifier power control).
        """
        command = (0b10000000 << 8) | (regnum << 8) | data
        buf = command.to_bytes(2, 'big')

        self._spi_lock_and_select()
        self.spi.write(buf)
        self._spi_unlock_and_deselect()

    def send_spi_command(self, command):
        """
        Sends a 16-bit command and reads 4 bytes (32 bits) back.
        This is a naive approach to read the RHD2164's DDR SPI data in standard SPI mode
        at a slower clock rate. The first 16 bits are module A data, the second 16 bits are
        module B data.
        """
        buf = command.to_bytes(2, 'big')
        result = bytearray(4)

        self._spi_lock_and_select()
        self.spi.write(buf)
        self.spi.readinto(result)  # Attempt to capture both half-words
        self._spi_unlock_and_deselect()

        return result

    def calibrate_chip(self):
        """
        Sends the CALIBRATE command, then issues dummy read commands while the chip calibrates.
        """
        print("\nCalibrating chip...")
        calibration_cmd = 0b0101010100000000
        buf = calibration_cmd.to_bytes(2, 'big')

        self._spi_lock_and_select()
        self.spi.write(buf)
        self._spi_unlock_and_deselect()

        time.sleep(0.01)

        # Typically 9 read cycles to let calibration settle
        for _ in range(9):
            self.send_read_command(40)
        print("Calibration complete.")

    # ---------------- EMG Sampling Logic ----------------
    def read_raw_mV(self, channel=0):
        """
        Reads raw voltage in mV from the specified amplifier channel (0-63),
        using single SPI transaction.
        """
        cmd = (channel & 0x3F) << 8
        response = self.send_spi_command(cmd)

        # response[:2] => 16 bits for the A module data
        raw_16bit = int.from_bytes(response[:2], 'big')
        raw_adc = raw_16bit - 32768  # shift midpoint
        return adc_to_mV(raw_adc)

    def read_emg_sample(self) -> float:
        """
        Reads EMG data for one channel, then processes it (clamping & HP filter).
        Modify or expand as needed if you want multiple channels.
        """
        # 1) Raw read in mV
        raw_mv = self.read_raw_mV(channel=0)

        # 2) Optionally clamp outliers
        safe_mv = clamp_if_outlier(raw_mv, max_range=20.0)

        # 3) High-pass filter to remove DC drift
        hp_mv = self.hp_filter.filter(safe_mv)
        return raw_mv

    def run(self):
        """
        Main routine:
          1) Connect to Wi-Fi
          2) Connect to server
          3) Configure & calibrate the RHD chip
          4) Continuously sample EMG & send to server
        """
        self.connect_wifi()
        self.connect_server()

        # Configure the SPI bus once at the start (slower speed recommended ~100k-200k)
        while not self.spi.try_lock():
            pass
        self.spi.configure(baudrate=115200, phase=0, polarity=0)
        self.spi.unlock()

        self.calibrate_chip()

        print("\nEntering main data loop. Press Ctrl+C to stop (via serial).")

        try:
            self.rgb.set_color([0, 0, 30])  # Blue LED means "sampling"
            while True:
                # Retrieve a single EMG sample
                emg_val = self.read_emg_sample()

                # Convert to string + newline for the server
                message = f"{emg_val}\n"

                # Send over the TCP socket
                self.sock.send(bytes(message, "utf-8"))
                print(f"Sent EMG value: {emg_val:.3f} mV")

                # Wait ~0.05s => 20 Hz data rate. Adjust as desired.
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("Stopped by user.")
        except Exception as e:
            print(f"Error during run: {e}")
        finally:
            if self.sock:
                self.sock.close()
                print("Socket closed.")


# -----------------------------
# Typical usage: code.py
# -----------------------------
if __name__ == "__main__":
    # Replace with the IP address your laptop server uses.
    # Example: "192.168.137.1" if your Pico is connected to your laptop's hotspot.
    SERVER_HOST = "192.168.137.1"
    SERVER_PORT = 5001

    # Pin connection
    # SCLK = GP2, MOSI = GP3, MISO = GP4, CS = GP5


    # High-pass filter cutoff at 10 Hz, sampling freq 100 Hz (for demonstration).
    # Adjust these based on your real-time needs or hardware constraints.
    client = EMGClient(server_host=SERVER_HOST, server_port=SERVER_PORT,
                       HP_CUTOFF=10, FS=100)
    client.run()
