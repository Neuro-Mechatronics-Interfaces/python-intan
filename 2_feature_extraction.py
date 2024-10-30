"""
This is the high-level script that will first lead the .rdh files as well as the data metrics file, and then
collect features to build the final dataset to train our model with

Author: Jonathan Shulgach
Last Modified: 10/28/24
"""

import pandas as pd
import utilities.rhd_utilities as rhd_utils
import utilities.plotting_utilities as plot_utils
import utilities.emg_processing as emg_proc


def feature_extraction(data_dir, metrics_filepath, processed_filepath, PCA_comp=8, visualize_pca_results=False, save_df=True):
    """
    This function processes EMG data, performs PCA, and optionally saves the processed feature data.
    """

    # Step 1: Get all .rhd file paths in the directory
    file_paths = rhd_utils.get_rhd_file_paths(data_dir, verbose=True)

    # Step 2: Load the metrics file if it exists
    metrics_file = emg_proc.get_metrics_file(metrics_filepath, verbose=True)
    if metrics_file is None:
        return

    # Process the EMG data in each file
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

        # Step 3: Process the EMG data to remove noise. PCA is highly sensitive to noise
        cleaned_emg = emg_proc.process_emg_pipeline(result)
        sample_rate = int(result['frequency_parameters']['board_dig_in_sample_rate'])  # Extract sampling rate

        # Step 4: There are a lot of channels, so we need to use PCA to get the most important features
        pca_data, explained_variance = emg_proc.apply_pca(cleaned_emg, num_components=PCA_comp)
        #pca_data = cleaned_emg

        # We can confirm the number of components is good by checking the increasing explained variance
        explained_variance_sum = [0]
        N_var = len(explained_variance)
        for i in range(N_var):
            explained_variance_sum.append(explained_variance_sum[i] + explained_variance[i])
        print("Highest explained variance in percentage: {}".format(explained_variance_sum[-1]*100))

        # We can also plot it to see the elbow point
        if visualize_pca_results:
            plot_utils.plot_figure(x=range(N_var), y=explained_variance_sum,
                                   x_label="Number of components",
                                   y_label="Explained variance",
                                   title="Explained variance vs. Number of components")

        # Convert processed data to a DataFrame
        processed_data = pd.DataFrame(pca_data.T, columns=[f'Channel_{i}' for i in range(pca_data.shape[0])])

        # Add the time index to the DataFrame
        processed_data['Time (s)'] = result['t_amplifier']

        # Step 5: Add labels to data
        # Find matching row in data_metrics for teh current file
        matching_row = metrics_file[metrics_file['File Name'] == filename]
        if not matching_row.empty:
            # Add the 'gesture' label to the PCA data as a new column
            row_data = matching_row.iloc[0]
            updated_df = emg_proc.apply_gesture_label(processed_data, sample_rate, row_data)

            # Append EMG data to the main dataframe
            EMG_PCA_data_df = pd.concat([EMG_PCA_data_df, updated_df], ignore_index=True)

            # Trim the data to only get the first 10 seconds
            #EMG_PCA_data_df = EMG_PCA_data_df.iloc[:int(10 * sample_rate), :]

            # Downsample the data to 2kHz
            EMG_PCA_data_df = EMG_PCA_data_df.iloc[::int(sample_rate/2000), :]

        else:
            print(f"No matching row found for file: {file}. Skipping...")

        #break  # To get just the first file info for testing

    if save_df and processed_filepath:
        # Save the processed EMG data to the CSV file
        print(f"Saving processed feature data to:\n {processed_filepath}")
        try:
            EMG_PCA_data_df.to_csv(processed_filepath, index=False)
            print("Data successfully saved!")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        return EMG_PCA_data_df


if __name__ == "__main__":

    # Grab the paths from the config file, returning dictionary of paths
    cfg = emg_proc.read_config_file('CONFIG.txt')

    # Do feature extraction
    print("Starting feature extraction...")
    feature_extraction(
            data_dir=cfg['raw_data_path'],                 # Path to the raw data files
            metrics_filepath=cfg['metrics_file_path'],     # Path to the metrics CSV that contains file names, gestures, etc.
            processed_filepath=cfg['processed_file_path'], # Path to output file where processed feature data will be saved
            PCA_comp=30,                                   # Specify the number of principal components to return from PCA
            visualize_pca_results=False,                   # Whether to visualize PCA results
            save_df=True                                   # Whether to save the processed data to a file
    )
    print("done!")
