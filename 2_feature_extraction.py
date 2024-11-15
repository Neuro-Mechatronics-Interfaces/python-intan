"""
This is the high-level script that will first lead the .rdh files as well as the data metrics file, and then
collect features to build the final dataset to train our model with

Author: Jonathan Shulgach
Last Modified: 11/15/24
"""
import os
import yaml
import time
import argparse
import numpy as np
import pandas as pd
import utilities.rhd_utilities as rhd_utils
import utilities.plotting_utilities as plot_utils
import utilities.emg_processing as emg_proc


def feature_extraction(config_dir, channels, PCA_comp=8, visualize_pca_results=False):
    """
    This function processes EMG data, performs PCA, and optionally saves the processed feature data.
    """
    # Step 1: Get all .rhd file paths in the directory
    cfg = emg_proc.read_config_file(config_dir)
    file_paths = rhd_utils.get_rhd_file_paths(rhd_utils.adjust_path(cfg['root_directory']), verbose=True)

    # Step 2: Load the metrics file if it exists
    adjusted_metrics_filepath = rhd_utils.adjust_path(os.path.join(cfg['root_directory'], cfg['metrics_filename']))
    metrics_file = emg_proc.get_metrics_file(adjusted_metrics_filepath, verbose=True)
    if metrics_file is None:
        return

    # Initialize Dataframe for PCA data
    EMG_PCA_data_df = pd.DataFrame()
    for file in file_paths:

        # Check if the file is already contained in the file_names list, if not continue
        filename, is_present = rhd_utils.check_file_present(file, metrics_file)
        if not is_present:
            print(f"File {filename} not found in metrics file. Skipping...")
            continue

        result, data_present = rhd_utils.load_file(file, verbose=False)
        if not data_present:
            print(f"No data found in {file}. Skipping...")
            continue

        full_emg_data = result['amplifier_data']  # Extract EMG data shape: (n_channels, n_samples)
        print(f"Full data shape: {full_emg_data.shape}")
        sample_rate = int(result['frequency_parameters']['board_dig_in_sample_rate'])

        # Assuming full_emg_data has shape (128, 92160)
        # Select the subset of rows corresponding to the selected channels
        print(f"Selecting channels: {channels}")
        full_emg_data = full_emg_data[channels, :]
        print(f"Selected data shape: {full_emg_data.shape}")

        # Find matching row in data_metrics for teh current file
        matching_row = metrics_file[metrics_file['File Name'] == filename]
        if matching_row.empty:
            print(f"No matching row found for file: {file}. Skipping...")
            continue

        # Step 3: parse the data metrics and extract the gesture periods
        row_data = matching_row.iloc[0]
        start_idx = row_data['Start Index']
        n_trials = row_data['N_trials']
        trial_interval = row_data['Trial Interval (s)']
        gesture = row_data['Gesture']

        # Iterate over each trial and assign the gesture label to the corresponding samples
        emg_data_segments = []
        for i in range(n_trials):
            # Get start and end indices for the flex (gesture) and relax
            start_flex = start_idx + i * sample_rate * trial_interval
            end_flex = start_flex + sample_rate * trial_interval  # gesture is half of interval
            emg_trial_data = full_emg_data[:, start_flex:end_flex]
            emg_data_segments.append(emg_trial_data)

        emg_data = np.concatenate(emg_data_segments, axis=1)
        print(f"EMG data shape: {emg_data.shape}")

        # Step 4: Process the EMG data

        # ====== Method 1 =====
        # Sliding window approach
        #feature_list = []
        #n_channels, num_samples = emg_data.shape
        #for i in range(n_channels):
        #    window_size = 256
        #    step_size = 128
        #    for start in range(0, num_samples - window_size, step_size):
        #        window = emg_data[i][start:start + window_size]
        #        features = emg_proc.extract_features(window)
        #        feature_list.append(features)
        #features_array = np.array(feature_list)
        # Apply PCA
        #pca_data, explained_variance = emg_proc.apply_pca(features_array, num_components=PCA_comp)

        # ====== Method 2 =====

        #bandpass_filtered = emg_proc.filter_emg(emg_data, 'bandpass', lowcut=30, highcut=500, fs=sample_rate, order=5)
        #smoothed_emg = emg_proc.envelope_extraction(bandpass_filtered, method='hilbert')
        #norm_emg = emg_proc.z_score_norm(smoothed_emg)
        #print(f"Normalized EMG shape: {norm_emg.shape}")
        #pca_data, explained_variance = emg_proc.apply_pca(norm_emg, num_components=PCA_comp)

        # ====== Method 3 =====
        #pca_data = emg_proc.extract_wavelet_features(emg_data.T)

        # ====== Method 4 =====
        filtered_data = emg_proc.notch_filter(emg_data, fs=sample_rate, f0=60) # 60Hz notch filter
        filtered_data = emg_proc.butter_bandpass_filter(filtered_data, lowcut=20, highcut=400, fs=sample_rate, order=2, axis=1, verbose=True) # bandpass filter
        bin_size = int(0.1*sample_rate) # 400ms bin size
        rms_features = emg_proc.calculate_rms(filtered_data, bin_size, verbose=True) # Calculate RMS feature with 400ms sampling bin
        lagged_features = emg_proc.create_lagged_features(rms_features, n_lags=4, verbose=True)

        final_data = lagged_features

        # We can confirm the number of components is good by checking the increasing explained variance
        #print("Highest explained variance in percentage: {}".format(np.cumsum(explained_variance)[-1]*100))

        # We can also plot it to see the elbow point
        #if visualize_pca_results:
        #    plot_utils.plot_figure(
        #        x=range(N_var),
        #        y=np.cumsum(explained_variance),
        #        x_label="Number of components",
        #        y_label="Explained variance",
        #        title="Explained variance vs. Number of components"
        #    )

        # Convert processed data to a DataFrame and add time index
        processed_data = pd.DataFrame(final_data, columns=[f'PC_{i+1}' for i in range(final_data.shape[1])])
        processed_data['Gesture'] = gesture

        # Find matching row in data_metrics for the current file
        matching_row = metrics_file[metrics_file['File Name'] == filename]
        if matching_row.empty:
            print(f"No matching row found for file: {file}. Skipping...")
            continue

        # Add the 'gesture' label to the PCA data as a new column
        print(f"Adding gesture label: {row_data['Gesture']} to the processed data...")
        updated_df = emg_proc.apply_gesture_label(processed_data, sample_rate, row_data)
        #show the unique gestures
        print(f"Unique gestures: {updated_df['Gesture'].unique()}")

        # Append EMG data to the main dataframe
        EMG_PCA_data_df = pd.concat([EMG_PCA_data_df, processed_data], ignore_index=True)
        print(f"EMG_PCA_data_df shape: {EMG_PCA_data_df.shape}")
        #EMG_PCA_data_df = pd.concat([EMG_PCA_data_df, updated_df], ignore_index=True)

        # Trim the data to only get the first 10 seconds
        #EMG_PCA_data_df = EMG_PCA_data_df.iloc[:int(10 * sample_rate), :]

        # Downsample the data to 2kHz
        #EMG_PCA_data_df = EMG_PCA_data_df.iloc[::int(sample_rate/2000), :]

    # Save the processed EMG data to the CSV file
    try:
        processed_filepath = os.path.join(cfg['root_directory'], cfg['processed_data_filename'])
        adjusted_processed_filepath = rhd_utils.adjust_path(processed_filepath)
        print(f"Saving processed feature data to:\n {adjusted_processed_filepath}")
        EMG_PCA_data_df.to_csv(adjusted_processed_filepath, index=False)
        print("Data successfully saved!")
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess EMG data to extract gesture timings.')
    parser.add_argument('--config_path', type=str, default='config.txt', description='Path to the config file containing the directory of .rhd files.')
    parser.add_argument('--emg_channels', type=list, default=list(range(128)), description='Index of the trigger channel to detect rising edges.')
    args = parser.parse_args()

    feature_extraction(args.config_path, args.emg_channels,      # ==== May delete below ====
                       PCA_comp=3,                               # Specify the number of principal components to return from PCA
                       visualize_pca_results=False,              # Whether to visualize PCA results
    )
    print("Step 2: feature extraction done!")
