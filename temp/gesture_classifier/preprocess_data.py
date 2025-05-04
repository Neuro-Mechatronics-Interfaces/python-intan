"""
 Script that preprocesses the data. It looks in a specified directory for all .rhd files and loads the data from each
 file. Every file is associated with a single gesture with repetitive movements.

 Author: Jonathan Shulgach
 Last Modified: 11/15/2024
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utilities.intan_utilities as rhd_utils
import utilities.emg_processing as emg_proc
import utilities.plotting_utilities as plot_utils


class PreProcess:
    """ PreProcess class which handles the preprocessing of EMG data by extracting timing of hand gestures.
    """

    def __init__(self, config_dir,  # filepath to config file, containing path to the .rhd files
                 trigger_channel=None,  # trigger channel to detect the rising edge
                 verbose=False  # print debug messages
                 ):
        self.cfg = config_dir
        self.trigger_channel = trigger_channel
        self.verbose = verbose
        self.vline = None
        self.hline = None
        self.start_times = []
        self.stop_channel_loop = False
        self.fig = None
        self.ax = None
        self.selected_channel_data = []

    def detect_edges(self, trigger_signal, sampling_rate, min_stable_samples=10, show_plot=True, wait_time=0):
        """
        Detects both rising and falling edges in the trigger signal and calculates the number of trials and trial interval.
        Ensures that edges are stable for at least 'min_stable_samples' to be considered valid.
        Adds an optional wait time before starting edge detection to avoid initial transients.

        Args:
            trigger_signal: 1D numpy array containing the digital trigger signal (0 or 1 values).
            sampling_rate: Sampling rate of the data in Hz.
            min_stable_samples: Minimum number of samples that need to be stable after a detected edge for it to be considered valid.
            show_plot: Boolean indicating whether to show a plot of the trigger signal.
            wait_time: Time in seconds to wait before starting edge detection. Default is 0 seconds.

        Returns:
            edge_times: A 1D numpy array containing the times (in seconds) of the detected edges.
            data: A dictionary containing the following information:
                first_edge_time: The time (in seconds) of the first detected edge (rising or falling).
                N_edges: The total number of detected edges.
                average_edge_interval: The average time interval (in seconds) between consecutive edges (if more than one).
        """
        # Calculate the number of samples to skip based on the wait time
        samples_to_skip = int(wait_time * sampling_rate)

        # Skip the initial samples to wait for the specified period
        if samples_to_skip > 0:
            print(f"| Skipping the first {samples_to_skip} samples ({wait_time} seconds) to avoid initial transients.")
            trigger_signal = trigger_signal[samples_to_skip:]

        # Convert the signal to detect transitions from 0 to 1 (rising edges) and 1 to 0 (falling edges)
        rising_edges = np.where(np.diff(trigger_signal.astype(int)) == 1)[0]
        falling_edges = np.where(np.diff(trigger_signal.astype(int)) == -1)[0]

        # Filter the edges to ensure they are stable for at least 'min_stable_samples' samples
        def filter_edges(edges, signal, stable_samples, expected_value):
            valid_edges = []
            for edge in edges:
                if edge + stable_samples < len(signal) and np.all(signal[edge + 1: edge + 1 + stable_samples] == expected_value):
                    valid_edges.append(edge)

            valid_edges = np.array(valid_edges) # Convert valid edges to numpy array for easier handling
            return valid_edges

        valid_rising_edges = filter_edges(rising_edges, trigger_signal, min_stable_samples, 1)
        valid_falling_edges = filter_edges(falling_edges, trigger_signal, min_stable_samples, 0)

        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(trigger_signal, label="Trigger Signal")
            # Plot the valid rising edges as vertical red lines
            for edge in valid_rising_edges:
                plt.axvline(edge, color='red', linestyle='--',
                            label='Rising Edge' if edge == valid_rising_edges[0] else "")

            # Plot the valid falling edges as vertical blue lines
            for edge in valid_falling_edges:
                plt.axvline(edge, color='blue', linestyle='--',
                            label='Falling Edge' if edge == valid_falling_edges[0] else "")

            plt.title("Trigger Signal with Detected Edges")
            plt.xlabel("Sample Index")
            plt.ylabel("Signal Value")
            plt.legend()
            plt.show()

        # Let the user decide whether to keep only rising or falling edges
        edge_type_selection = input(
            "Would you like to keep only rising edges or falling edges? ([rising]/falling): ").strip().lower()

        if edge_type_selection == 'rising':
            print("Keeping only rising edges.")
            edge_times = valid_rising_edges / sampling_rate # Times for valid edges
        elif edge_type_selection == 'falling':
            print("Keeping only falling edges.")
            edge_times = valid_falling_edges / sampling_rate
        else:
            print("Defaulting to keeping rising edges.")
            edge_times = valid_rising_edges / sampling_rate

        # Add back the skipped time
        edge_times += wait_time

        # Calculate first edge time, number of edges, and average edge interval
        first_edge_time = edge_times[0] if len(edge_times) > 0 else None
        N_edges = len(edge_times) if len(edge_times) > 0 else 0
        edge_intervals = np.diff(edge_times) if len(edge_times) > 1 else None
        average_edge_interval = np.mean(edge_intervals) if edge_intervals is not None else None

        # Display the final detected values after selection
        print(f"Final detected values:\n"
              f"First Edge Time: {first_edge_time}\n"
              f"Number of Edges (N_edges): {N_edges}\n"
              f"Average Edge Interval: {average_edge_interval}\n")

        # Prompt user to manually define number of edges and interval if needed
        manual_input = input(
            "Would you like to manually define the number of edges and edge interval? (y/[n]): ").strip().lower()

        if manual_input == 'y':
            N_edges = int(input("Enter the number of edge events: "))
            average_edge_interval = float(input("Enter the time interval between edges (in seconds): "))
            edge_times = np.arange(first_edge_time, first_edge_time + N_edges * average_edge_interval, average_edge_interval)

        data = {'edge_times': edge_times}
        data['first_edge_time'] = first_edge_time
        data['N_trials'] = N_edges
        data['trial_interval'] = average_edge_interval
        return edge_times, data

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
        cfg = emg_proc.read_config_file(self.cfg)
        file_paths = rhd_utils.get_rhd_file_paths(rhd_utils.adjust_path(cfg['root_directory']), True)

        # Step 1.5, load the metrics file if it exists
        file_names = None
        metrics_filepath = rhd_utils.adjust_path(os.path.join(cfg['root_directory'], cfg['metrics_filename']))
        metrics_file = emg_proc.get_metrics_file(metrics_filepath, verbose=True)
        if metrics_file is not None:
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

            # Extract amplifier data (EMG data) and the sampling rate
            emg_data = result['amplifier_data']  # Shape: (num_channels, num_samples)
            time_vector = result['t_amplifier']  # Time vector for the data
            channel_names = [ch['custom_channel_name'] for ch in result['amplifier_channels']]
            sampling_rate = result['frequency_parameters']['amplifier_sample_rate'] # Sampling rate of the data
            do_manual = False if self.trigger_channel is not None else True

            # Step 3: Apply processing to EMG data (filtering, CAR, RMS, etc.)
            print(f"Processing file: {filename}")
            emg_data = emg_proc.notch_filter(emg_data, sampling_rate, 60)
            filt_data = emg_proc.filter_emg(emg_data, filter_type='bandpass', lowcut=30, highcut=500, fs=sampling_rate, verbose=True)
            car_data = emg_proc.common_average_reference(filt_data, True)
            grid_data = emg_proc.compute_grid_average(car_data, 8, 0)
            grid_ch = list(range(grid_data.shape[0]))
            rms_data = emg_proc.window_rms(car_data, window_size=800, verbose=True)

            if self.trigger_channel is not None:
                if 'board_dig_in_data' not in result:
                    print("No digital input data found. Switching to manual...")
                    do_manual=True

            if not do_manual:
                # Handle trial indexing if trigger is specified and present
                print(f"Selecting trigger channel {self.trigger_channel}")
                trigger_signal = result['board_dig_in_data'][self.trigger_channel, :]
                edges, edge_data = self.detect_edges(trigger_signal, sampling_rate, show_plot=True)
                self.start_times.append([filename, self.trigger_channel, None, edge_data['first_edge_time'], None, edge_data['N_trials'], edge_data['trial_interval']])

            else:
                print("Instructions: Left-click to select start time, spacebar to skip channel.")
                #for channel_index, channel_name in enumerate(channel_names):
                for channel_index, channel_name in enumerate(grid_ch):

                    if self.stop_channel_loop:
                        self.stop_channel_loop = False  # Reset the flag for the next channel
                        break
                    print(f"Plotting channel {channel_name}...")
                    #self.plot_emg_channel(car_data[channel_index, :], time_vector, channel_name, sampling_rate, filename)
                    self.plot_emg_channel(grid_data[channel_index, :], time_vector, channel_name, sampling_rate, filename)

                # === Prompt for N_trials and trial_interval after the loop stops ===
                if self.selected_channel_data:
                    N_trials = int(input("Enter the number of trial repetitions: "))
                    trial_interval = float(input("Enter the time interval between trials (s): "))
                    data = self.selected_channel_data[0]
                    data.update({'N_trials': N_trials, 'Trial Interval (s)': trial_interval})
                    self.start_times.append([data['file_name'], data['channel_name'], data['start_index'],
                                             data['start_time'], data['amplitude'], N_trials, trial_interval])
                    self.selected_channel_data = []  # Clear the selected data
                    edges = np.arange(data['start_time'], data['start_time'] + N_trials * trial_interval, trial_interval)

            # Plot the waterfall plot with the trial indices plotted as vertical red lines
            plot_utils.waterfall_plot_old(car_data, emg_data.shape[0], time_vector, edges=edges, plot_title=filename, line_width=0.4, colormap='hsv')

            self.save_start_times(metrics_filepath) # Save the data
            print(f"Start times for {filename} recorded. Moving to the next file...")

        print("Processing complete.")
        return


if __name__ == "__main__":

    # Set up argument parser to let user provide path to config file, using teh current path as default
    parser = argparse.ArgumentParser(description='Preprocess EMG data to extract gesture timings.')
    parser.add_argument('--config_path', type=str, default='config.txt', help='Path to the config file containing the directory of .rhd files.')
    parser.add_argument('--trigger_channel', type=int, default=None, help='Index of the trigger channel to detect rising edges.')
    parser.add_argument('--verbose', action='store_true', help='Print debug messages.')
    args = parser.parse_args()

    # Create an instance of the PreProcess class and run the preprocess method.
    preprocessor = PreProcess(args.config_path, args.trigger_channel, args.verbose)
    preprocessor.preprocess_data()

    # Note that with this version of code, after the data metrics file is created, a new column for the gesture label
    # needs to be manually created. Open the .csv file and Fill in the rows for 'Gesture' with whatever gesture is being
    # performed in the corresponding file. Save the file and proceed to the next step.
    print("Step 1: preprocessing done! Remember to create the Gesture column and define the gestures for each file in the metrics file.")
