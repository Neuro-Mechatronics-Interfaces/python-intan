"""
This is the high-level script that will first lead the .rdh files as well as the data metrics file, and then
collect features to build the final dataset to train our model with

Author: Jonathan Shulgach
Last Modified: 10/28/24
"""
import yaml
import time
import numpy as np
import pandas as pd
import utilities.rhd_utilities as rhd_utils
import utilities.plotting_utilities as plot_utils
import utilities.emg_processing as emg_proc


def feature_extraction(data_dir, metrics_filepath, processed_filepath, channels, PCA_comp=8, visualize_pca_results=False, save_df=True):
    """
    This function processes EMG data, performs PCA, and optionally saves the processed feature data.
    """

    # Step 1: Get all .rhd file paths in the directory
    adjusted_data_dir = rhd_utils.adjust_path(data_dir)
    file_paths = rhd_utils.get_rhd_file_paths(adjusted_data_dir, verbose=True)

    # Step 2: Load the metrics file if it exists
    adjusted_metrics_filepath = rhd_utils.adjust_path(metrics_filepath)
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
        print(f"Filtered data shape: {filtered_data.shape}")
        filtered_data = emg_proc.butter_bandpass_filter(filtered_data, lowcut=20, highcut=400, fs=sample_rate, order=2, axis=1) # bandpass filter
        print(f"Bandpass filtered data shape: {filtered_data.shape}")
        bin_size = int(0.1*sample_rate) # 400ms bin size
        rms_features = emg_proc.calculate_rms(filtered_data, bin_size) # Calculate RMS feature with 400ms sampling bin
        print(f"RMS features shape: {rms_features.shape}")

        # Create lagged features by concatenating 4 preceding bins with the current bin
        num_bins = rms_features.shape[1]  # Total number of bins
        lagged_features = []
        for i in range(4, num_bins):
            # Concatenate the current bin with the previous 4 bins
            current_features = rms_features[:, (i - 4):i + 1].flatten()  # Flatten to create feature vector
            lagged_features.append(current_features)

        lagged_features = np.array(lagged_features)
        print(f"Lagged features shape: {lagged_features.shape}")

        pca_data = lagged_features # no PCA
        #pca_data = rms_features # no PCA
        print(f"PCA data shape: {pca_data.shape}")

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
        #print(pca_data)
        processed_data = pd.DataFrame(pca_data, columns=[f'PC_{i+1}' for i in range(pca_data.shape[1])])
        #processed_data['Time (s)'] = result['t_amplifier']
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

    if save_df and processed_filepath:
        adjusted_processed_filepath = rhd_utils.adjust_path(processed_filepath)
        # Save the processed EMG data to the CSV file
        print(f"Saving processed feature data to:\n {adjusted_processed_filepath}")
        try:
            EMG_PCA_data_df.to_csv(adjusted_processed_filepath, index=False)
            print("Data successfully saved!")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        return EMG_PCA_data_df


if __name__ == "__main__":

    # Grab the paths from the config file, returning dictionary of paths
    cfg = emg_proc.read_config_file('CONFIG.txt')

    # Select channels 1 through 8 in a list
    chs = list(range(0, 8)) + list(range(64, 72))  # Channels 1-8 and 65-72 (Python indexing starts at 0)
    #chs = [1]

    # Do feature extraction
    print("Starting feature extraction...")
    feature_extraction(
            data_dir=cfg['raw_data_path'],                 # Path to the raw data files
            metrics_filepath=cfg['metrics_file_path'],     # Path to the metrics CSV that contains file names, gestures, etc.
            processed_filepath=cfg['processed_file_path'], # Path to output file where processed feature data will be saved
            channels=chs,
            PCA_comp=3,                                   # Specify the number of principal components to return from PCA
            visualize_pca_results=False,                   # Whether to visualize PCA results
            save_df=True                                   # Whether to save the processed data to a file
    )
    print("done!")
