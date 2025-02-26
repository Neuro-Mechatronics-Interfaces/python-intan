""" Script to train a new classification model for gestures using EMG data.

Usage:
  python train.py --config_path "/mnt/c/Users/NML/Desktop/hdemg_test/Jonathan/2025_02_25/CONFIG.txt"
"""

import os
import sys
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from utilities import emg_processing as emg_proc
from utilities import ml_utilities as ml_utils
from utilities import rhd_utilities as rhd_utils

def process_emg_data(file_paths, metrics_data, gesture_map, verbose):
    X_list, y_list = [], []
    for file in file_paths:
        # Load EMG Data
        result, data_present = rhd_utils.load_file(file, verbose=verbose)
        if not data_present:
            continue

        emg_data = result['amplifier_data']
        sample_rate = int(result['frequency_parameters']['board_dig_in_sample_rate'])

        # Preprocess the EMG signal, output should be rms features
        rms_features = emg_proc.preprocess_emg(emg_data, sample_rate)

        # Retrieve gesture label
        file_name = os.path.basename(file)
        if file_name not in metrics_data['File Name'].values:
            print(f"⚠️ Warning: No entry found for {file_name} in metrics data. Skipping.")
            continue

        gesture = metrics_data[metrics_data['File Name'] == file_name]['Gesture'].values[0]

        # Append to lists
        X_list.append(rms_features.T)  # Shape (N_samples, 128)
        y_list.append(np.full(rms_features.shape[1], gesture_map[gesture]))

    return X_list, y_list

def train_emg_classifier(config_path, epochs, verbose):
    """ Main function that trains a new classification model for gestures using EMG data. """
    # hard code the path to the configuration file. Will use the argparse module to pass this in the future
    cfg = emg_proc.read_config_file(config_path)

    # Load the metrics data
    metrics_filepath = os.path.join(cfg['root_directory'], cfg['metrics_filename'])
    metrics_data, gesture_map = emg_proc.load_metrics_data(metrics_filepath)

    # Load and process the EMG data
    file_paths = emg_proc.get_file_paths(cfg['root_directory'], file_type='.rhd', verbose=verbose)
    X_list, y_list = process_emg_data(file_paths, metrics_data, gesture_map, verbose=verbose)

    # Convert EMG data and labels into tensors
    X_tensor, y_tensor = ml_utils.convert_lists_to_tensors(X_list, y_list)
    num_classes = torch.unique(y_tensor).shape[0]
    print(f"Final unique labels in y_tensor: {torch.unique(y_tensor)}")

    # Prepare Datasets into training and validation
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))  # 80% training, 20% testing
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model training
    model = ml_utils.EMGCNN(num_classes=num_classes, input_channels=X_tensor.shape[1])
    train_losses, val_losses, val_accuracies = ml_utils.train_pytorch_model(model, train_loader, val_loader, num_epochs=epochs)

    # Save trained model and training metrics
    model_savepath = os.path.join(cfg['root_directory'], cfg["model_filename"])
    model.save(model_savepath)

    # training results file
    results_savepath = os.path.join(cfg['root_directory'], "training_results.txt")
    with open(results_savepath, 'w') as f:
        f.write(f"Train Losses: {train_losses}\n")
        f.write(f"Validation Losses: {val_losses}\n")
        f.write(f"Validation Accuracies: {val_accuracies}\n")
    print(f"Saved training results to {results_savepath}")

    # Plot the training and validation losses
    ml_utils.plot_training_metrics(train_losses, val_losses, val_accuracies)


if __name__ == "__main__":

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess EMG data to extract gesture timings.')
    parser.add_argument('--config_path', type=str, default='config.txt', help='Path to the config file containing the directory of .rhd files.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training the model.')
    parser.add_argument('--do_kfold', type=bool, default=False, help='Whether to use K-Fold cross-validation.')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of data splits to train for cross-validation.')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to print verbose output.')
    args = parser.parse_args()

    train_emg_classifier(args.config_path, args.epochs, args.verbose)
