"""
intan.processing._metrics_utils

Utilities for loading gesture metrics and creating gesture-label mappings.

This module:
- Loads EMG trial classification metadata from CSV or TXT
- Parses gesture names into integer class mappings
- Provides helper functions for checking or retrieving metrics files

Used in training/testing pipelines that require alignment between gesture labels
and EMG signal segments.
"""

import os
import pandas as pd


def load_metrics_data(metrics_filepath, verbose=True):
    """ Loads the metrics data from the specified file path and returns the data along with the gesture mapping.

    Parameters:
        metrics_filepath (str): The path to the metrics data file.
        verbose    (bool): Whether to print the loaded data and gesture mapping.

    Returns:
        tuple: A tuple containing the metrics data as a pandas DataFrame and the gesture mapping as a dictionary.

    """
    if not os.path.isfile(metrics_filepath):
        print(f"Metrics file not found: {metrics_filepath}. Please correct file path or generate the metrics file.")
        return None
    metrics_data = pd.read_csv(metrics_filepath)
    if verbose:
        print(f"Loaded metrics data from {metrics_filepath}: unique labels {metrics_data['Gesture'].unique()}")
        print(metrics_data)

    # Generate gesture mapping
    gestures = metrics_data['Gesture'].unique()
    gesture_map = {gesture: i for i, gesture in enumerate(gestures)}
    if verbose:
        print(f"Gesture mapping: {gesture_map}")

    return metrics_data, gesture_map


def get_metrics_file(metrics_filepath, verbose=False):
    """
    Checks if the metrics file exists at the specified path. If it does, loads the data and returns it.

    Parameters:
        metrics_filepath (str): The path to the metrics data file.
        verbose    (bool): Whether to print the loaded data.

    Returns:
        pd.DataFrame: The loaded metrics data as a pandas DataFrame.
    """
    if os.path.isfile(metrics_filepath):
        if verbose:
            print("Metrics file found.")
        return pd.read_csv(metrics_filepath)
    else:
        print(f"Metrics file not found: {metrics_filepath}. Please correct file path or generate the metrics file.")
        return None
