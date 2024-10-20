"""
This script will load the data metrics from the data_metrics file as well as load the data from the RHD files, then
collect the features to train our model with
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import utilities.rhd_utilities as rhd_utils
import utilities.emg_processing as emg_proc


def feature_extraction(data_dir, metrics_path, PCA_comp=8, visualize_pca_results=False):

    print("Starting feature extraction...")

    # Load data metrics file
    data_metrics = pd.read_csv(metrics_path)
    print(f"Loaded data metrics file with {len(data_metrics)} rows")

    # Load all the .rhd file paths in the root directory and specified in the metric file
    EMG_PCA_data_df = pd.DataFrame()
    file_paths = emg_proc.get_rhd_file_paths(data_dir)

    # We only want to process files that are contained in the data_matrics['File Name'] column
    # This is to ensure that we only process files that have been annotated
    # Normalize the file paths to ensure consistent format
    file_paths = [os.path.normpath(file) for file in file_paths]
    data_metrics['File Name'] = data_metrics['File Name'].apply(os.path.normpath)

    # Find the intersection of the file paths and the data_metrics['File Name'] column
    file_paths = list(set(file_paths).intersection(set(data_metrics['File Name'].values)))

    print(f"Found {len(file_paths)} files to process")
    # Process the EMG data in each file
    for file in file_paths:
        result, data_present = rhd_utils.load_file(file, verbose=False)
        if not data_present:
            print(f"No data found in {file}. Skipping...")
            continue

        print(f"Processing file: {file}")

        # Process the EMG data to remove noise. PCA is highly sensitive to noise
        cleaned_emg = emg_proc.process_emg_pipeline(result)
        sample_rate = int(result['frequency_parameters']['board_dig_in_sample_rate'])  # Extract sampling rate

        # There are a lot of channels, so we need to use PCA to get the most important features
        pca_data, explained_variance = emg_proc.apply_pca(cleaned_emg, num_components=PCA_comp)

        # We can confirm the number of components is good by checking the increasing explained variance
        explained_variance_sum = [0]
        for i in range(len(explained_variance)):
            explained_variance_sum.append(explained_variance_sum[i] + explained_variance[i])
        print("Highest explained variance in percentage: {}".format(explained_variance_sum[-1]*100))

        # We can also plot it
        if visualize_pca_results:
            plt.figure(figsize=(8, 6))
            plt.plot(explained_variance_sum)
            plt.xlabel("Number of components")
            plt.ylabel("Explained variance")
            plt.title("Explained variance vs. Number of components")
            plt.show()

        # Convert processed data to a DataFrame
        processed_data = pd.DataFrame(pca_data.T, columns=[f'Channel_{i}' for i in range(pca_data.shape[0])])

        # Find matching row in data_metrics for teh current file
        matching_row = data_metrics[data_metrics['File Name'] == file]
        if not matching_row.empty:
            # Add the 'gesture' label to the PCA data as a new column
            row_data = matching_row.iloc[0]
            updated_df = emg_proc.apply_gesture_label(processed_data, sample_rate, row_data)

            # Append EMG data to the main dataframe
            EMG_PCA_data_df = pd.concat([EMG_PCA_data_df, updated_df], ignore_index=True)
        else:
            print(f"No matching row found for file: {file}. Skipping...")

    return EMG_PCA_data_df


if __name__ == "__main__":

    # Specify the root data path
    root_dir = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\intan_HDEMG_sleeve\raw'

    # Specify the data metrics file path
    data_metrics_path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\intan_HDEMG_sleeve\raw_data_metrics.csv'

    # Specify the save path for the processed data
    save_path = r'G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\intan_HDEMG_sleeve\processed_data.csv'

    # Specify the number of principal components to return from PCA
    pc = 30

    # Do feature extraction
    df = feature_extraction(root_dir, data_metrics_path, pc)

    # Save the processed EMG data to a CSV file
    df.to_csv(save_path, index=False)

