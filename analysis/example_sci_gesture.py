"""
This example script analyzes data collected during an experiment where a subject performed gestures during indictated periods.

Author: Jonathan Shulgach
Last Modified: 12/30/2024

"""

import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# Load the Intan utilities
import utilities.rhd_utilities as rhd_utils
from utilities import plotting_utilities as plot_utils
from utilities import emg_processing as emg_utils
from utilities.plotting_utilities import waterfall_plot



def load_gesture_data(data_dir, gesture, verbose=False):
    """
    Load all trials for a specific gesture and store data efficiently.

    Args:
        data_dir (str): Path to the data directory.
        gesture (str): Gesture keyword to search for.
        verbose (bool): Whether to print verbose output.

    Returns:
        tuple: (gesture_data, time_vector)
            gesture_data: NumPy array of shape (n_trials, n_channels, n_samples)
            time_vector: NumPy array of time points (common for all trials if consistent)
    """
    all_data = []  # List to accumulate data for each trial
    all_time_vectors = []  # List to accumulate time vectors
    print(f"Processing gesture: {gesture}")

    for folder in os.listdir(data_dir):
        #if verbose: print(f"|  Looking for folder with keyword: '{gesture}'...", end='')
        if gesture in folder:
            if verbose: print(f"found '{folder}'")
            trial_dir = os.path.join(data_dir, folder)
            rhd_file = os.path.join(trial_dir, f"{folder}.rhd")
            if os.path.exists(rhd_file):
                print(f"|  Loading data from {rhd_file}")
                result, data_present = rhd_utils.load_file(rhd_file) # Load the rhd data

                # display the keys in the result dictionary
                for key, val in result.items():
                    print(key)
                    print(val)

                if not data_present:
                    print(f"No data found in {rhd_file}. Skipping...")
                    continue

                # Add data and time vector to the lists
                all_data.append(result['amplifier_data'])  # Shape: (n_channels, n_samples)
                all_time_vectors.append(result['t_amplifier'])  # Assuming time vector is consistent across files
                break

    # if len(all_data) == 0:
    #     print(f"No data found for gesture: {gesture}")
    #     return None, None
    #
    # # Stack all trial data into a single array
    # # Shape: (n_trials, n_channels, n_samples)
    # print(all_data)
    # gesture_data = np.stack(all_data, axis=0)
    #
    # # Check if all time vectors are consistent; otherwise, handle appropriately
    # if all(np.array_equal(all_time_vectors[0], tv) for tv in all_time_vectors):
    #     time_vector = all_time_vectors[0]  # Use the first time vector if consistent
    # else:
    #     print("Warning: Time vectors are inconsistent across trials.")
    #     time_vector = np.array(all_time_vectors)  # Store all time vectors for inspection

    return gesture_data, time_vector


def process_and_plot_data(gesture, gesture_data):
    """
    Process and visualize data for a specific gesture.
    """
    # Combine all trials
    combined_data = np.concatenate(gesture_data, axis=1)

    # Filter the data
    filtered_data = emg_utils.filter_emg(combined_data, filter_type='bandpass', lowcut=LOWCUT, highcut=HIGHCUT, fs=SAMPLING_RATE)

    # Compute the average waveform across trials
    average_waveform = np.mean(filtered_data, axis=1)

    # Generate time vector
    time_vector = np.arange(combined_data.shape[1]) / SAMPLING_RATE

    # Plot raw waveforms
    plt.figure(figsize=(10, 6))
    plt.title(f"Raw Waveforms for {gesture}")
    plt.plot(time_vector, combined_data.T)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.show()

    # Plot average waveform
    plt.figure(figsize=(10, 6))
    plt.title(f"Average Waveform for {gesture}")
    plt.plot(time_vector[:len(average_waveform)], average_waveform)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.show()

    # Waterfall plot
    plot_utils.waterfall_plot(
        data=filtered_data,
        channel_indices=range(filtered_data.shape[0]),
        time_vector=time_vector,
        plot_title=f"Filtered Waveforms for {gesture}",
    )

def load_data(root_dir, keyword=None, trial=None, verbose=False):
    """ Loads data form specified directory. Uses the keyword or trial number if specified.

    Args:
        root_dir (str): Path to the data directory.
        keyword (str): Keyword to search for in the folder names.
        trial (int): Trial number to load.
        verbose (bool): Whether to print verbose output.
    """

    for folder in os.listdir(root_dir):
        if keyword is not None and keyword in folder:
            trial_dir = os.path.join(root_dir, folder)
            rhd_file = os.path.join(trial_dir, f"{folder}.rhd")
            if os.path.exists(rhd_file):
                if verbose: print(f"Loading data from {rhd_file}")
                result, data_present = rhd_utils.load_file(rhd_file)
                if not data_present:
                    print(f"No data found in {rhd_file}. Skipping...")
                    continue

        elif trial is not None and trial in folder:
            trial_dir = os.path.join(root_dir, folder)
            rhd_file = os.path.join(trial_dir, f"{folder}.rhd")
            if os.path.exists(rhd_file):
                if verbose: print(f"Loading data from {rhd_file}")
                result, data_present = rhd_utils.load_file(rhd_file)
                if not data_present:
                    print(f"No data found in {rhd_file}. Skipping...")
                    continue

    if verbose:
        print(f"Data loaded successfully from {rhd_file}")
        print("|  File contains fields:")
        for key in result.keys():
            print(f"|  |  {key}")

    return result


if __name__ == "__main__":

    # Configurations
    #DATA_DIR = '/mnt/g/Shared drives/NML_shared/DataShare/HDEMG_SCI/MCP01_NML-EMG/2024_12_10'
    #GESTURES = ["extend"]  # , "wflex", "rest", "supinate", "tripod", "wextend", "grip", "MVC", "pronate", "abduct", "adduct"]
    #LOWCUT = 30  # Bandpass filter low cutoff frequency
    #HIGHCUT = 500  # Bandpass filter high cutoff frequency
    #SAMPLING_RATE = 20000  # Default sampling rate (adjust if needed)

    #gesture = GESTURES[0] # Process one gesture at a time

    file_dir = '/mnt/g/Shared drives/NML_shared/DataShare/HDEMG_SCI/MCP01_NML-EMG/2024_12_10/extend_p1_0_241210_172602/extend_p1_0_241210_172602.rhd'

    #gesture_data = load_data(DATA_DIR, keyword=gesture, verbose=True)
    results, _ = rhd_utils.load_file(file_dir)

    # display the keys in the result dictionary
    for key, val in results.items():
        print(key)
        print(val)

    #print(type(results['t_aux_input']))
    #print(results['aux_input_data'])

    # plot the t_aux_input data
    #plt.figure()
    #plt.plot(results['t_aux_input'])
    #plt.plot(results['aux_input_data'][3])
    #plt.show()



    #if gesture_data is not None:
    #    print("Gesture data shape:", gesture_data.shape)  # Expected shape: (n_trials, n_channels, n_samples)
    #    print("Time vector length:", len(time_vector))
    #else:
    #    print("No data loaded.")


