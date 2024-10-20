"""
This script will load the data metrics from the data_metrics file as well as load the data from the RHD files, then
collect the features to train our model with
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import utilities.rhd_utilities as rhd_utils
import utilities.emg_processing as emg_proc


def feature_extraction(data_dir, 
                       metrics_file_name='raw_data_metrics.csv', 
                       save_filename='processed_data.csv', 
                       PCA_comp=8, 
                       visualize_pca_results=False, 
                       save_df=False
                       ):
    """
    This function processes EMG data, performs PCA, and optionally saves the processed feature data.

    Parameters:
    - data_dir: Directory containing raw data and the metrics file.
    - metrics_file_name: File name of the metrics CSV that contains file paths and gestures.
    - save_filename: Name of the output file where processed features will be saved (optional).
    - PCA_comp: Number of principal components to return from PCA.
    - visualize_pca_results: Whether to plot the explained variance.
    - save_df: Whether to save the processed DataFrame to a CSV file.
    """

    # Load data metrics file
    metrics_path = os.path.join(data_dir, metrics_file_name)
    data_metrics = pd.read_csv(metrics_path)
    print(f"Loaded data metrics file with {len(data_metrics)} rows")
    
    # The 'File Name' column contains paths that may start with 'raw'. 
    data_metrics['Absolute File Path'] = data_metrics['File Name'].apply(
        lambda x: os.path.join(data_dir, 'raw', x.lstrip("raw\\/"))  # Remove 'raw/' prefix if it exists
    )

    # Normalize the file paths to use consistent forward slashes for WSL2 compatibility
    data_metrics['Absolute File Path'] = data_metrics['Absolute File Path'].apply(lambda x: os.path.normpath(x).replace("\\", "/"))

    # Get the absolute paths from the newly created column
    file_paths = data_metrics['Absolute File Path'].tolist()
    print(f"Found {len(file_paths)} files to process")
    #print(file_paths)  # Print out paths for debugging
    
    # Process the EMG data in each file
    EMG_PCA_data_df = pd.DataFrame()
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
        matching_row = data_metrics[data_metrics['Absolute File Path'] == file]
        if not matching_row.empty:
            # Add the 'gesture' label to the PCA data as a new column
            row_data = matching_row.iloc[0]
            updated_df = emg_proc.apply_gesture_label(processed_data, sample_rate, row_data)

            # Append EMG data to the main dataframe
            EMG_PCA_data_df = pd.concat([EMG_PCA_data_df, updated_df], ignore_index=True)
        else:
            print(f"No matching row found for file: {file}. Skipping...")

    if save_df and save_filename:
        # Save the processed EMG data to the CSV file
        #save_path = os.path.join(data_dir, save_filename)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(current_dir, save_filename) # Saving locally
        print(f"Saving processed feature data to:\n {save_path}")
        try:
            EMG_PCA_data_df.to_csv(save_path, index=False)
            print("Data successfully saved!")
        except Exception as e:
            print(f"Error saving file: {e}")
    else:
        return EMG_PCA_data_df

if __name__ == "__main__":

    # Define the root data path once
    root_dir = '/mnt/g/path/to/folder/with/raw/data'

    # Do feature extraction
    print("Starting feature extraction...")
    feature_extraction(
            data_dir=root_dir, 
            metrics_file_name='raw_data_metrics.csv',  # Data metrics file name
            save_filename='processed_data.csv',        # Saved feature data file name
            PCA_comp=30,                               # Specify the number of principal components to return from PCA
            visualize_pca_results=False,               # Whether to visualize PCA results
            save_df=True                               # Whether to save the processed data to a file
    )
    print("done!")

