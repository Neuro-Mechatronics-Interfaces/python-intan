import os
import sys
import keras
import argparse
import numpy as np
import pandas as pd
import socket
import time
import torch
import asyncio
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Get utilities
import utilities.emg_processing as emg_proc
from utilities.messaging_utilities import TCPClient, RingBuffer, PicoMessager
from utilities.ml_utilities import EMGCNN

# Constants for parsing and data handling
FRAMES_PER_BLOCK = 128  # Number of frames per data block from Intan (constant)
SAMPLE_SCALE_FACTOR = 0.195  # Microvolts per bit (used for scaling raw samples)


# Byte Parsing Functions
def readUint32(array, arrayIndex):
    variableBytes = array[arrayIndex: arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    arrayIndex += 4
    return variable, arrayIndex


def readInt32(array, arrayIndex):
    variableBytes = array[arrayIndex: arrayIndex + 4]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=True)
    arrayIndex += 4
    return variable, arrayIndex


def readUint16(array, arrayIndex):
    variableBytes = array[arrayIndex: arrayIndex + 2]
    variable = int.from_bytes(variableBytes, byteorder='little', signed=False)
    arrayIndex += 2
    return variable, arrayIndex


class IntanEMG:
    """Class for managing connection and data sampling from the Intan RHD recorder system.

    Args:
        ring_buffer_size (int): The size of the ring buffer for storing data samples.
        waveform_buffer_size (int): The size of the buffer for receiving waveform data.
        channels (list): List of channels to sample data from. Default is all 64 channels.
        show_plot (bool): Whether to display a real-time plot of the data.
        verbose (bool): Whether to display verbose output.
    """

    def __init__(self, model_path='emg_cnn_model.pth',  # model_path='model.keras',
                 gesture_labels_filepath=None,
                 channels=None,
                 ring_buffer_size=4000,
                 waveform_buffer_size=175000,
                 command_buffer_size=1024,
                 use_serial=False,
                 COM_PORT='COM13',
                 BAUDRATE=9600,  # Baud rate set to match Pico's configuration
                 show_plot=False,
                 verbose=False
                 ):
        self.channels = channels if channels is not None else list(range(128))  # Default to all 64 channels
        self.show_plot = show_plot
        self.verbose = verbose
        self.sample_rate = None  # Sample rate of the Intan system, gets set during initialization
        self.all_stop = False
        self.ring_buffer = RingBuffer(len(self.channels), ring_buffer_size)
        print(f" Size of ring buffer: {ring_buffer_size} samples")
        self.pico = None
        self.last_gesture = None
        if use_serial:
            self.pico = PicoMessager(port=COM_PORT, baudrate=BAUDRATE)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)

        # Load the gesture labels
        self.gesture_labels_dict = self._load_gesture_labels(gesture_labels_filepath)

        # Setup TCP clients
        self.command_tcp = TCPClient(name='Command', host='127.0.0.1', port=5000, buffer=command_buffer_size)  # Socket to handle sending commands
        self.waveform_tcp = TCPClient('Waveform', '127.0.0.1', 5001, waveform_buffer_size * len(self.channels))  # Socket to handle receiving waveform data

        # Set up plotting
        self.fig, self.ax = plt.subplots(len(self.channels), 1, figsize=(10, 6), sharex=True)
        if len(self.channels) == 1:
            self.ax = [self.ax]  # Make it iterable for single channel case

        self._initialize_intan()

    def _load_model(self, model_path):
        """Loads the trained PyTorch model for gesture classification."""
        model = EMGCNN(num_classes=8, input_channels=128)  # Adjust based on trained model
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()  # Set model to evaluation mode
        print("Loaded PyTorch model successfully.")
        return model

    def _load_gesture_labels(self, gesture_labels_filepath):
        """Loads gesture labels from a CSV file."""
        if gesture_labels_filepath:
            gesture_labels = pd.read_csv(gesture_labels_filepath)
            gesture_labels_dict = dict(zip(gesture_labels['Numerical Value'], gesture_labels['Gesture']))
            print(f"Loaded Gesture Labels: {gesture_labels_dict}")
            return gesture_labels_dict
        return {}

    def _initialize_intan(self):
        """Connect and configures Intan system for data streaming."""
        if not self._connect(): return

        # Ensure RHX is in "stop" mode to configure settings
        resp = str(self.command_tcp.send('get runmode', wait_for_response=True), 'utf-8')
        if "RunMode Stop" not in resp:
            self.command_tcp.send('set runmode stop')

        # Clear all data outputs
        self.command_tcp.send('execute clearalldataoutputs')

        # Set up sample rate and channels
        sample_rate_response = self.command_tcp.send('get sampleratehertz', True).decode()
        self.sample_rate = float(sample_rate_response.split()[-1])
        print(f"Sample rate: {self.sample_rate} Hz")

        # Enable TCP data output for specified channels
        for channel in self.channels:
            print("Setting up channel: ", channel)
            channel_cmd = f'set a-{channel:03}.tcpdataoutputenabled true'.encode()
            self.command_tcp.send(channel_cmd)

        # Set the number of data blocks to write to the TCP output buffer
        print("Setting up data block size to 1...")
        self.command_tcp.send('set TCPNumberDataBlocksPerWrite 1')
        print("Setup complete.")

    def _connect(self):
        """Connects to Intan TCP servers and sets up the sample rate and selected channels."""
        print("Connecting to Intan TCP servers...")
        try:
            self.command_tcp.connect()  # Command server
            self.waveform_tcp.connect()  # Waveform data server
            print("Connected to Intan TCP servers.")
            return True

        except ConnectionRefusedError:
            print("Failed to connect to Intan TCP servers.")
            return False

    def _disconnect(self):
        """Stops data streaming and cleans up connections."""
        self.command_tcp.send('set runmode stop')

        # Clear TCP data output to ensure no TCP channels are enabled.
        self.command_tcp.send('execute clearalldataoutputs')

        self.command_tcp.close()
        self.waveform_tcp.close()
        print("Connections closed and resources cleaned up.")

    async def _sample_intan(self):
        """Receives and processes a block of data from the waveform server."""
        while not self.all_stop:
            start_t = time.time()
            try:
                raw_data = self.waveform_tcp.read()
                if not raw_data:
                    print("No data received.")
                    await asyncio.sleep(0.1)
                    continue
                if self.verbose: print("length of bytes received: ", len(raw_data))
                raw_index = 0
                num_blocks = len(raw_data) // (FRAMES_PER_BLOCK * (4 + (2 * len(self.channels))) + 4)
                if self.verbose: print("Number of blocks: ", num_blocks)

                # Process each block of data
                for block in range(num_blocks):
                    # Expect 4 bytes to be TCP Magic Number as uint32
                    magic_number, raw_index = readUint32(raw_data, raw_index)
                    if magic_number != 0x2ef07a08:
                        if self.verbose: print("Unexpected magic number; skipping block.")
                        continue

                    # Each block should contain 128 frames of data - process each of these one-by-one
                    for frame in range(FRAMES_PER_BLOCK):
                        # Expect 4 bytes to be the timestamp as int32.
                        raw_timestamp, raw_index = readInt32(raw_data, raw_index)  # Read timestamp (microseconds)
                        timestamp = raw_timestamp / self.sample_rate  # Convert to seconds

                        frame_data = np.zeros(len(self.channels))
                        for ch in range(len(self.channels)):
                            raw_sample, raw_index = readUint16(raw_data, raw_index)
                            sample = SAMPLE_SCALE_FACTOR * (raw_sample - 32768)  # Scale to microvolts
                            # Update the frame data by channel
                            frame_data[ch] = sample

                        self.ring_buffer.append(timestamp, frame_data)

                end_t = time.time()
                if self.verbose:
                    print(f"Buffer length: {len(self.ring_buffer.samples)} samples.")
                if self.verbose:
                    print(f"Intan data sampled in: {end_t - start_t:.4f} seconds")
                await asyncio.sleep(0)

            except BlockingIOError:
                # No data available immediately, retry later
                await asyncio.sleep(0.1)
            except socket.timeout:
                print("Read timed out, server may not be streaming.")
                await asyncio.sleep(0.1)  # Wait before retrying
            except Exception as e:
                print(f"Unexpected error in _sample_intan: {e}")
                await asyncio.sleep(0.1)  # Wait before retrying

    def get_samples(self, n=1):
        try:
            emg_data, t = self.ring_buffer.get_samples(n)
            return emg_data, t
        except ValueError as e:
            print(f"Error: {e}")
            return None, None

    def start(self):
        """Make a call to the asyncronous library to run the main routine"""
        asyncio.run(self._main())  # Need to pass the async function into the run method to start

    async def _main(self):
        """ Start main tasks and coroutines in a single function"""
        # Start the data sampling task
        asyncio.create_task(self._sample_intan())

        # Start the decoding task
        asyncio.create_task(self.decode_gesture())

        # Can start other tasks in the background here if needed...

        while not self.all_stop:
            await asyncio.sleep(0)  # Calling sleep with 0 makes it run as fast as possible

    async def decode_gesture(self):
        """Begins data streaming from Intan and processes the data in real-time."""
        try:
            # Start streaming data
            last_msg_time = None
            self.command_tcp.send('set runmode run')
            print("Sampling started.")

            if self.show_plot:
                plt.show(block=False)

            while not self.all_stop:
                start_t = time.time()

                if not self.ring_buffer.is_full():
                    await asyncio.sleep(0.01)
                    continue

                try:
                    emg_data, t = self.get_samples(400)  # Pass in the number of samples (0.25 seconds)

                    #print(f"Shape of emg_data: {emg_data.shape}")  # Should be (2000, num_channels)


                    if emg_data is None or len(emg_data) == 0:
                        await asyncio.sleep(0.1)
                        continue

                    # Visualization shows down the update rate
                    if self.show_plot:
                        # Update each channels subplot
                        for i, channel in enumerate(self.channels):
                            y = emg_data[:, i]
                            self.ax[i].cla()
                            self.ax[i].plot(t, y)
                            self.ax[i].set_ylim(-1000, 1000)
                            self.ax[i].grid()
                            self.ax[i].set_ylabel(f'CH{channel}')

                        plt.draw()
                        plt.pause(0.001)

                    # ===== EMG Processing ======
                    emg_data = emg_data.T  # Transpose to have shape (num_channels, num_samples)
                    #print("Shape of emg_data: ", emg_data.shape)
                    filtered_data = emg_proc.notch_filter(emg_data, fs=self.sample_rate, f0=60)  # 60Hz notch filter
                    #print("Shape of filtered data: ", filtered_data.shape)
                    filtered_data = emg_proc.butter_bandpass_filter(filtered_data, lowcut=20, highcut=400,
                                                                    fs=self.sample_rate, order=2,
                                                                    axis=1)  # bandpass filter
                    #print("Shape of bandpass filtered data: ", filtered_data.shape)
                    bin_size = int(0.1 * self.sample_rate)  # 400ms bin size
                    rms_features = emg_proc.calculate_rms(filtered_data, bin_size)  # Calculate RMS feature with 400ms sampling bin
                    #print("Shape of RMS features: ", rms_features.shape)

                    # Ensure RMS features have exactly 128 channels
                    if rms_features.shape[0] < 128:
                        print(f"Warning: Expected 128 channels but got {rms_features.shape[0]}. Padding with zeros.")
                        pad_size = 128 - rms_features.shape[0]
                        rms_features = np.pad(rms_features, ((0, pad_size), (0, 0)), mode='constant')

                    elif rms_features.shape[0] > 128:
                        print(
                            f"Warning: Expected 128 channels but got {rms_features.shape[0]}. Trimming extra channels.")
                        rms_features = rms_features[:128, :]

                    if rms_features.shape[1] > 1:
                        print(f"Warning: Expected 1 sample but got {rms_features.shape[1]}. Taking first sample.")
                        rms_features = rms_features[:, 0]

                    # Ensure data shape (1, 128, 1) before passing to model
                    feature_tensor = torch.tensor(rms_features.T, dtype=torch.float32).unsqueeze(-1).to(self.device)
                    #feature_tensor = torch.tensor(rms_features.T, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)

                    # Predict the gesture
                    if self.model:
                        # if self.verbose: print("Predicting gesture...")
                        try:
                            #print(f"Feature tensor shape before model: {feature_tensor.shape}")  # Debugging

                            # === Model Prediction ===
                            with torch.no_grad():
                                predictions = self.model(feature_tensor)  # Get logits
                                pred_idx = torch.argmax(predictions, dim=1).item()  # Get class index
                                gesture_str = self.gesture_labels_dict.get(pred_idx, "Unknown")
                                print(f"Predicted Gesture: {gesture_str}")

                            # === Handle Gesture Output ===
                            if self.last_gesture != gesture_str:
                                self.last_gesture = gesture_str
                                print(f"Predicted Gesture: {gesture_str}")

                                if self.pico:
                                    self.pico.update_gesture(gesture_str)

                            #p_gestures = self.model.predict(feature_data, verbose=0)  # Use .predict() for Keras models
                            #pred_idx = np.argmax(p_gestures[0])
                            #gesture_str = self.gesture_labels_dict[pred_idx]
                            #print(f"Predicted gesture: {gesture_str}")
                            #if self.last_gesture != gesture_str:
                            #    self.last_gesture = gesture_str
                            #    self.pico.update_gesture(gesture_str)

                                # Update PicoMessager with the detected gesture
                                # if last_msg_time is None or time.time() - last_msg_time > 1:
                                #    last_msg_time = time.time()
                                #    if self.pico:
                                #        self.pico.update_gesture(gesture_str)
                                # self.pico.dump_output()

                            # print(f"Predicted gesture: {gesture_str}")

                            # Update PicoMessager with the detected gesture
                            # self.pico.update_gesture(gesture_str)
                            # self.pico.dump_output(mute=True)

                        except Exception as e:
                            print(f"Error predicting gesture: {e}")

                    end_t = time.time()
                    if self.verbose:
                        print(f"Loop time: {end_t - start_t:.4f} seconds\n\n")

                    await asyncio.sleep(0)

                except BlockingIOError:
                    # No data available immediately, retry later
                    await asyncio.sleep(0.1)
                except socket.timeout:
                    print("Read timed out, server may not be streaming.")
                    await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"Unexpected error in decode_gesture: {e}")
                    await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print("Sampling stopped by user.")
        finally:
            print("Stopping data streaming...")
            self.all_stop = True
            self._disconnect()
            self.pico.close_connection()


# Main Execution
if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Real-time EMG gesture decoding using a trained model.')
    args.add_argument('--config_path', type=str, default='../config.txt', help='Path to the config file containing the directory of .rhd files.')
    args.add_argument('--channels', type=str, nargs='+', default='1:9', help='List of channels to sample data from.')
    args.add_argument('--use_serial', type=bool, default=False, help='Use serial communication with PicoMessager.')
    args.add_argument('--port', type=str, default='/dev/ttyACM0', help='COM port for PicoMessager. Windows machines use COMXX')
    args.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    args = args.parse_args()

    # Parse the channel ranges
    #chs = emg_proc.parse_channel_ranges(args.channels)
    #print("Channels to sample: ", chs)
    chs = None # Defaults to 128 channels

    # Read the configuration file
    cfg = emg_proc.read_config_file(args.config_path)

    gesture_labels_filepath = os.path.join(cfg['root_directory'], 'gesture_labels.csv')
    model_path = os.path.join(cfg['root_directory'], "emg_cnn_model.pth")
    intan = IntanEMG(model_path=model_path,
                     gesture_labels_filepath=gesture_labels_filepath,
                     channels=chs,
                     show_plot=False,
                     use_serial=args.use_serial,
                     COM_PORT=args.port,
                     verbose=args.verbose)
    intan.start()
