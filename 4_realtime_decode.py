import socket
import keras
import numpy as np
import time
import asyncio
import matplotlib.pyplot as plt
import utilities.emg_processing as emg_proc

# Constants for parsing and data handling
FRAMES_PER_BLOCK = 128 # Number of frames per data block from Intan (constant)
SAMPLE_SCALE_FACTOR = 0.195 # Microvolts per bit (used for scaling raw samples)

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

class TCPClient:
    """ Class for managing TCP connections to the Intan system."""
    def __init__(self, name, host, port, buffer=1024):
        self.name = name
        self.host = host
        self.port = port
        self.buffer = buffer
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(5)  # Timeout after 5 seconds if no data received

    def connect(self):
        self.s.setblocking(True)
        self.s.connect((self.host, self.port))
        self.s.setblocking(False)

    def send(self, data, wait_for_response=False):
        # convert data to bytes if it is not already
        if not isinstance(data, bytes):
            data = data.encode()
        self.s.sendall(data)
        time.sleep(0.01)

        if wait_for_response:
            return self.read()

    def read(self):
        return self.s.recv(self.buffer)

    def close(self):
        self.s.close()

class RingBuffer:
    """Fixed-size ring buffer for storing recent data up to max number of samples. """
    def __init__(self, num_channels, size_max=4000):
        self.max = size_max
        self.samples = np.zeros((size_max, num_channels), dtype=np.float32)
        self.timestamp = np.zeros(size_max, dtype=np.float32)
        self.cur = 0
        self.size = 0

    def append(self, t, x):
        """Adds a new sample to the buffer, removing the oldest sample if full."""
        #if len(self.samples) >= self.max:
        #    self.samples.pop(0)
        #    self.timestamp.pop(0)
        #self.samples.append(x)
        #self.timestamp.append(t)
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

    #def get_timestamps(self):
    #    """ Returns the current timestamps in the ring buffer. """
    #    return self.timestamp

    #def get_channel(self, ch):
    #    """ Returns the current data for a specific channel in the ring buffer. """
    #    return [x[ch] for x in self.samples]

class IntanEMG:
    """Class for managing connection and data sampling from the Intan RHD recorder system.

    Args:
        ring_buffer_size (int): The size of the ring buffer for storing data samples.
        waveform_buffer_size (int): The size of the buffer for receiving waveform data.
        channels (list): List of channels to sample data from. Default is all 64 channels.
        show_plot (bool): Whether to display a real-time plot of the data.
        verbose (bool): Whether to display verbose output.
    """
    def __init__(self, model_path='model.keras',
                       ring_buffer_size=4000,
                       waveform_buffer_size=175000,
                       command_buffer_size=1024,
                       channels=None,
                       show_plot=False,
                       verbose=False
                 ):
        self.channels = channels if channels is not None else list(range(64))  # Default to all 64 channels
        self.show_plot = show_plot
        self.verbose = verbose
        self.sample_rate = None  # Sample rate of the Intan system, gets set during initialization
        self.all_stop = False
        self.ring_buffer = RingBuffer(len(self.channels), ring_buffer_size)

        # Load the model
        try:
            self.model = keras.models.load_model(model_path)
            print(self.model.summary())
            print("Model Input Shape:", self.model.input_shape)
            print("Model Output Shape:", self.model.output_shape)
        except FileNotFoundError:
            print(f"Model file not found at {model_path}")
            self.model = None

        # Setup TCP clients
        self.s_command = TCPClient('Command',
                                   host='127.0.0.1', port=5000,
                                   buffer=command_buffer_size) # Socket to handle sending commands
        self.s_waveform = TCPClient('Waveform', '127.0.0.1', 5001, waveform_buffer_size * len(self.channels)) # Socket to handle receiving waveform data

        # Set up plotting
        self.fig, self.ax = plt.subplots(len(self.channels), 1, figsize=(10, 6), sharex=True)
        if len(self.channels) == 1:
            self.ax = [self.ax] # Make it iterable for single channel case

        self._initialize_intan()

    def _initialize_intan(self):
        """Connect and configures Intan system for data streaming."""
        if not self._connect(): return

        # Ensure RHX is in "stop" mode to configure settings
        resp = str(self.s_command.send('get runmode', wait_for_response=True), 'utf-8')
        if "RunMode Stop" not in resp:
            self.s_command.send('set runmode stop')

        # Clear all data outputs
        self.s_command.send('execute clearalldataoutputs')

        # Set up sample rate and channels
        sample_rate_response = self.s_command.send('get sampleratehertz', True).decode()
        self.sample_rate = float(sample_rate_response.split()[-1])
        print(f"Sample rate: {self.sample_rate} Hz")

        # Enable TCP data output for specified channels
        for channel in self.channels:
            print("Setting up channel: ", channel)
            channel_cmd = f'set a-{channel:03}.tcpdataoutputenabled true'.encode()
            self.s_command.send(channel_cmd)

        # Set the number of data blocks to write to the TCP output buffer
        print("Setting up data block size to 1...")
        self.s_command.send('set TCPNumberDataBlocksPerWrite 1')
        print("Setup complete.")

    def _connect(self):
        """Connects to Intan TCP servers and sets up the sample rate and selected channels."""
        print("Connecting to Intan TCP servers...")
        try:
            self.s_command.connect()  # Command server
            self.s_waveform.connect()  # Waveform data server
            print("Connected to Intan TCP servers.")
            return True

        except ConnectionRefusedError:
            print("Failed to connect to Intan TCP servers.")
            return False

    def _disconnect(self):
        """Stops data streaming and cleans up connections."""
        self.s_command.send('set runmode stop')

        # Clear TCP data output to ensure no TCP channels are enabled.
        self.s_command.send('execute clearalldataoutputs')

        self.s_command.close()
        self.s_waveform.close()
        print("Connections closed and resources cleaned up.")

    async def _sample_intan(self):
        """Receives and processes a block of data from the waveform server."""
        while not self.all_stop:
            start_t = time.time()
            try:
                raw_data = self.s_waveform.read()
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
                #if self.verbose:
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
        asyncio.run(self._main()) # Need to pass the async function into the run method to start

    async def _main(self):
        """ Start main tasks and coroutines in a single function"""
        # Start the data sampling task
        asyncio.create_task(self._sample_intan())

        # Start the decoding task
        asyncio.create_task(self.decode_gesture())

        # Can start other tasks in the background here if needed...

        while not self.all_stop:
            await asyncio.sleep(0) # Calling sleep with 0 makes it run as fast as possible

    async def decode_gesture(self):
        """Begins data streaming from Intan and processes the data in real-time."""
        try:
            # Start streaming data
            self.s_command.send('set runmode run')
            print("Sampling started.")

            if self.show_plot:
                plt.show(block=False)

            while not self.all_stop:
                start_t = time.time()
                emg_data, t = self.get_samples(100) # Pass in the number of samples (0.25 seconds)
                if emg_data is None or len(emg_data) == 0:
                    await asyncio.sleep(0.1)
                    continue

                # Visualization shows down the update rate
                if self.show_plot:
                     # Update each channels subplot
                     for i, channel in enumerate(self.channels):
                         y = emg_dtata[:, i]
                         self.ax[i].cla()
                         self.ax[i].plot(t, y)
                         self.ax[i].set_ylim(-1000, 1000)
                         self.ax[i].grid()
                         self.ax[i].set_ylabel(f'CH{channel}')

                     plt.draw()
                     plt.pause(0.001)

                # Process data: bp filter, envelope, normalize, PCA
                bandpass_filtered = emg_proc.filter_emg(emg_data, 'bandpass', lowcut=30, highcut=500, fs=self.sample_rate, order=5)
                smoothed_emg = emg_proc.envelope_extraction(bandpass_filtered, method='hilbert')
                norm_emg = emg_proc.z_score_norm(smoothed_emg)
                pca_data, _ = emg_proc.apply_pca(norm_emg.T, num_components=31)
                pca_data = pca_data.T

                # Extract a single sample (most recent)
                pca_data = pca_data[-1]

                # Add an extra dimension to pca_data to make it compatible with the model's expected input shape
                pca_data = np.expand_dims(pca_data, axis=0)  # Now pca_data has shape (1, 31)

                # Predict the gesture
                if self.model:
                    print("Predicting gesture...")
                    try:
                        p_gestures = self.model.predict(pca_data)  # Use .predict() for Keras models
                        gesture_idx = np.argmax(p_gestures)# Get the gesture index with the highest probability
                        print("Gesture Probabilities/: ")
                        for i, p in enumerate(p_gestures[0]):
                            print(f"{GESTURES[i]}: {p:.12f}")

                        #print(f"Predicted gesture: {GESTURES[gesture_idx]}")
                    except Exception as e:
                        print(f"Error predicting gesture: {e}")

                end_t = time.time()
                print(f"Loop time: {end_t - start_t:.4f} seconds")
                await asyncio.sleep(0)

        except KeyboardInterrupt:
            print("Sampling stopped by user.")
        finally:
            print("Stopping data streaming...")
            self.all_stop = True
            self._disconnect()


# Main Execution
if __name__ == "__main__":

    chs = list(range(128))

    GESTURES = {
        0: '3_Finger',
        1: 'Hand_Open',
        2: 'Index_Finger',
        3: 'Middle_Finger',
        4: 'Pinch',
        5: 'Pinky_Finger',
        6: 'Power_Grip',
        7: 'Rest',
        8: 'Ring_Finger',
        9: 'Thumb',
        10: 'Wrist_Extension',
        11: 'Wrist_Flexion'
    }

    # Specify buffer size and desired channels, e.g., channels=[0, 1, 4, 5]
    intan = IntanEMG(model_path='cnn_model.keras',
                     ring_buffer_size=4000,
                     waveform_buffer_size=100000,
                     channels=chs,
                     show_plot=False,
                     verbose=False
                    )
    intan.start()
