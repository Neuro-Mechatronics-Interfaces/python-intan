"""
 Script that preprocesses the data. It looks in a specified directory for all .rhd files and loads the data from each
 file. Every file is associated with a single gesture with repetitive movements.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utilities.rhd_utilities as rhd_utils
import utilities.emg_processing as emg_proc


class PreProcess:
    """ PreProcess class which handles the preprocessing of EMG data by extracting timing of hand gestures.
    """

    def __init__(self, directory,  # directory containing the .rhd files
                 metrics_filename='data_metrics.csv',  # path to save the start times
                 trigger_channel=None,  # trigger channel to detect the rising edge
                 verbose=False  # print debug messages
                 ):
        self.directory = directory
        self.trigger_channel = trigger_channel
        self.metrics_filepath = os.path.join(directory, metrics_filename)
        self.verbose = verbose
        self.vline = None
        self.hline = None
        self.start_times = []
        self.stop_channel_loop = False
        self.fig = None
        self.ax = None
        self.selected_channel_data = []

    def detect_rising_edge(self, trigger_signal, sampling_rate):
        """
        Detects the rising edge in the trigger signal and calculates the number of trials and trial interval.

        Args:
            trigger_signal: 1D numpy array containing the digital trigger signal (0 or 1 values).
            sampling_rate: Sampling rate of the data in Hz.

        Returns:
            first_rising_edge_time: The time (in seconds) of the first detected rising edge.
            N_trials: The number of detected rising edges (trials).
            trial_interval: The time interval (in seconds) between consecutive rising edges (if more than one).
        """
        # Convert the signal to a boolean array to detect transitions from 0 to 1
        rising_edges = np.where(np.diff(trigger_signal.astype(int)) == 1)[0]

        # Check if any rising edges are detected
        if len(rising_edges) > 0:
            # First rising edge time (in samples)
            first_rising_edge_sample = rising_edges[0]
            first_rising_edge_time = first_rising_edge_sample / sampling_rate

            # Calculate the number of trials based on the number of rising edges
            N_trials = len(rising_edges)

            # If there is more than one trial, calculate the average trial interval
            if N_trials > 1:
                trial_intervals = np.diff(rising_edges) / sampling_rate
                trial_interval = np.mean(trial_intervals)  # Average time between trials
            else:
                trial_interval = None  # No interval if there's only one trial
        else:
            # No rising edge detected, set values to None or defaults
            first_rising_edge_time = None
            N_trials = 0
            trial_interval = None

        return first_rising_edge_time, N_trials, trial_interval

    def onkeypress(self, event, channel_index):
        """
        Event handler for key press.

        Args:
            event: Key press event object.
            channel_index: Index of the current channel being plotted.
        """
        if event.key == ' ':
            print(f"Skipping channel {channel_index}")

    def onclick(self, event, channel_data, channel_name, sampling_rate, file_name):
        """
        Event handler for mouse click.

        Args:
            event: Mouse event object.
            channel_data: Current EMG data for the plotted channel.
            channel_index: Index of the current channel being plotted.
            sampling_rate: Sampling rate of the EMG data.
            file_name: Name of the file being processed.
        """

        # Get the x-coordinate of the mouse click (this corresponds to the index)
        if event.button == 1:  # Left-click
            start_time = event.xdata
            print(f"Start time: {start_time:.2f}s")

            start_index = int(start_time * sampling_rate)  # Convert index to time
            print(f"Start index: {start_index}")

            amplitude = event.ydata  # Get the amplitude at the clicked point
            print(f"Amplitude: {amplitude:.2f} μV")

            # Save the selected data temporarily
            self.selected_channel_data = []
            self.selected_channel_data.append({
                'file_name': file_name,
                'channel_name': channel_name,
                'start_index': start_index,
                'start_time': start_time,
                'amplitude': amplitude
            })
            self.stop_channel_loop = True
            plt.close()  # Close the plot to move to the next channel
            self.fig = None
            self.ax = None

    def plot_emg_channel(self, emg_data, time_vector, channel_name, sampling_rate, file_name):
        """
        Plots the EMG data for a specific channel and waits for a mouse click.

        Args:
            emg_data: 2D numpy array containing the EMG data (channels, samples).
            time_vector: Time vector for the data.
            channel_index: Index of the channel to be plotted.
            sampling_rate: Sampling rate of the EMG data.
            file_name: Name of the file being processed.
        """
        # if fig is closed or does not exist, create a new figure
        if not self.fig or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(12, 6))  # Initialize the plot with a wider figure

        self.ax.cla()  # Clear the current axis
        self.ax.plot(time_vector, emg_data)
        self.ax.set_title(f"{file_name} | Channel {channel_name}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude (μV)")

        # Create horizontal and vertical lines that will follow the cursor
        self.hline = self.ax.axhline(color='red', lw=1)  # Horizontal line
        self.vline = self.ax.axvline(color='blue', lw=1)  # Vertical line

        # Function to update the position of the lines
        def update_lines(event):
            if event.inaxes == self.ax:
                # Update horizontal and vertical line position
                self.hline.set_ydata([event.ydata, event.ydata])  # Fix: Set ydata as a sequence with two points
                self.vline.set_xdata([event.xdata, event.xdata])  # Fix: Set xdata as a sequence with two points

                # Redraw the figure with the updated line positions
                self.fig.canvas.draw_idle()

        # Connect motion event to the update function
        self.fig.canvas.mpl_connect('motion_notify_event', update_lines)

        # Connect the onclick event handler
        self.fig.canvas.mpl_connect('button_press_event',
                                    lambda event: self.onclick(event, emg_data, channel_name, sampling_rate, file_name)
                                    )

        # Connect the onkeypress event handler for the space bar
        self.fig.canvas.mpl_connect('key_press_event', lambda event: self.onkeypress(event, channel_name))

        self.fig.canvas.draw()  # Update the plot with the new channel's data
        plt.show(block=False)

        print("Waiting for user interaction (left-click or spacebar)...")
        plt.waitforbuttonpress()  # Wait for user input

    def save_start_times(self, file_name='file_metrics.csv'):
        """
        Saves the start time data to a CSV file. If an entry for the file already exists, it will be overwritten.

        Args:
            file_name: Name of the CSV file to save the data.
        """
        headers = ['File Name', 'Channel Index', 'Start Index', 'Start Time (s)', 'Amplitude (uV)', 'N_trials',
                   'Trial Interval (s)']

        if not self.start_times:
            return  # If there is no data, do not save

        # Convert start_times to a DataFrame
        new_data = pd.DataFrame(self.start_times, columns=headers)

        if os.path.isfile(file_name):
            existing_data = pd.read_csv(file_name)

            # Filter out any rows corresponding to the files we're about to write (to replace them)
            existing_data = existing_data[~existing_data['File Name'].isin(new_data['File Name'])]

            # Concatenate the new data with the filtered existing data
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            updated_data = new_data

        # Write the updated data back to the CSV file (overwriting it)
        updated_data.to_csv(file_name, index=False)

    def preprocess_data(self):
        """
        Preprocesses all .rhd files in the specified directory. This is necessary to extract the time indices for the
        gestures. Since there are 10x-20x repetitions of the gesture spaced 1 second apart we want to find the most accurate
        estimation of the start time. To do that, we will process the data and look for the channel that provides the
        greatest SNR, compute the time index of an empirical threshold crossing, and use that as the start time.

        """
        # Step 1: Get all .rhd file paths in the directory
        print("Searching in directory:", self.directory)
        file_paths = emg_proc.get_rhd_file_paths(self.directory)
        print(f"Found {len(file_paths)} .rhd files")

        # Step 1.5, load the metrics file if it exists
        file_names = None
        if os.path.isfile(self.metrics_filepath):
            metrics_file = pd.read_csv(self.metrics_filepath)
            file_names = metrics_file['File Name'].tolist()

        # Step 2: Load the data from each file
        for file in file_paths:
            filename = Path(file).name
            self.stop_channel_loop = False  # Reset the flag for each file
            self.start_times = []  # Clear start times for each file

            # Check if the file is already in the metrics file
            if metrics_file is not None and filename in file_names:
                print(f"File {filename} already processed. Skipping...")
                continue

            # Load the data from the file
            result, data_present = rhd_utils.load_file(file)
            if not data_present:
                print(f"No data found in {file}. Skipping...")
                continue

            print(f"Processing file: {filename}")

            if self.trigger_channel is not None:
                if 'board_dig_in_data' in result and result['board_dig_in_data'].shape[0] > self.trigger_channel:
                    trigger_signal = result['board_dig_in_data'][self.trigger_channel, :]
                    sampling_rate = result['frequency_parameters']['amplifier_sample_rate']
                    first_rising_edge_time, N_trials, trial_interval = self.detect_rising_edge(trigger_signal,
                                                                                               sampling_rate)

                    if first_rising_edge_time is not None:
                        print(f"Detected first rising edge at {first_rising_edge_time:.2f}s")
                        print(f"N_trials: {N_trials}, Trial Interval: {trial_interval:.2f}s")
                        self.start_times.append(
                            [file, self.trigger_channel, None, first_rising_edge_time, None, N_trials, trial_interval])
                        self.save_start_times(self.metrics_filepath)
                        continue
                    else:
                        print("No rising edge detected on the trigger channel. Moving to manual selection...")

            # =========== Manual method if no trigger is detected ===========
            # Extract amplifier data (EMG data) and the sampling rate
            emg_data = result['amplifier_data']  # Shape: (num_channels, num_samples)
            sampling_rate = result['frequency_parameters']['amplifier_sample_rate']  # Sampling rate of the data
            time_vector = result['t_amplifier']  # Time vector for the data
            channel_names = [ch['custom_channel_name'] for ch in result['amplifier_channels']]

            # Step 3-6: Apply processing to EMG data (filtering, rectifying, etc.)
            print("Filtering data...")
            filt_data = emg_proc.filter_emg(emg_data, filter_type='bandpass', lowcut=30, highcut=500, fs=sampling_rate, verbose=True)
            print("Subtracting common average reference...")
            car_data = emg_proc.common_average_reference(filt_data)
            print("Rectifying data...")
            rect_data = emg_proc.rectify_emg(car_data)
            print("Applying RMS window...")
            rms_data = emg_proc.window_rms(rect_data, window_size=800)

            # Interactive channel plotting and selection
            print("Instructions: Left-click to select start time, spacebar to skip channel.")
            for channel_index, channel_name in enumerate(channel_names):
                if self.stop_channel_loop:
                    self.stop_channel_loop = False  # Reset the flag for the next channel
                    break
                print(f"Plotting channel {channel_name}...")
                self.plot_emg_channel(rms_data[channel_index, :], time_vector, channel_name, sampling_rate, filename)

            # === Prompt for N_trials and trial_interval after the loop stops ===
            if self.selected_channel_data:
                try:
                    N_trials = int(input("Enter the number of trial repetitions: "))
                    trial_interval = float(input("Enter the time interval between trials (s): "))
                except ValueError:
                    print("Invalid input. Using default values of N_trials=1, trial_interval=0.")
                    N_trials = 1
                    trial_interval = 0.0

                # Add the trial information to the selected channels and save
                for data in self.selected_channel_data:
                    data.update({'N_trials': N_trials, 'Trial Interval (s)': trial_interval})
                    self.start_times.append([data['file_name'], data['channel_name'], data['start_index'],
                                             data['start_time'], data['amplitude'], N_trials, trial_interval])

                # Save the data
                self.selected_channel_data = []  # Clear the selected data
                self.save_start_times(self.metrics_filepath)
                print(f"Start times for {filename} recorded. Moving to the next file...")

        print("Processing complete.")
        return


if __name__ == "__main__":

    # Grab the paths from the config file, returning dictionary of paths
    cfg = emg_proc.read_config_file('CONFIG.txt')
    trigger_name = None  # uncomment and change to name of the trigger channel if exists (ex: 1)

    # Create an instance of the PreProcess class and run the preprocess method. If no trigger data is found,
    # the user will be prompted to manually select the start time for each channel.
    preprocessor = PreProcess(directory=cfg['raw_data_path'],
                              metrics_filename=cfg['metrics_file_path'],
                              trigger_channel=trigger_name,
                              verbose=True,
                   )
    preprocessor.preprocess_data()

    # Note that with this version of code, after the data metrics file is created, a new column for the gesture label
    # needs to be manually created. Open the .csv file and Fill in the rows for 'Gesture' with whatever gesture is being
    # performed in the corresponding file. Save the file and proceed to the next step.
