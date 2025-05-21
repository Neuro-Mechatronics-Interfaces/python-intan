"""
intan.processing._features

Feature extraction utilities for EMG signal segments.

Includes common time-domain features used in classification pipelines:
- Mean Absolute Value (MAV)
- Zero Crossings (ZC)
- Slope Sign Changes (SSC)
- Waveform Length (WL)
- Root Mean Square (RMS)

Also provides a registry of available features and a function to extract them from
multichannel segments using a flexible callable or string-based interface.
"""

import numpy as np


def mean_absolute_value(data):
    """ Computes the Mean Absolute Value (MAV) of the input data.

    Parameters:
        data (np.ndarray) shape (n_channels, n_samples): Input data for which to compute the MAV.

    Returns:
        np.ndarray: MAV value for each channel.
    """
    return np.mean(np.abs(data))


def zero_crossings(ch, threshold=0.01):
    """ Computes the number of zero crossings in the input data.

    Parameters:
        ch (np.ndarray): Input data for which to compute the zero crossings.
        threshold (float): Threshold for detecting significant changes.

    Returns:
        int: Number of zero crossings.
    """
    return np.sum((np.diff(np.sign(ch)) != 0) & (np.abs(np.diff(ch)) > threshold))


def slope_sign_changes(ch, threshold=0.01):
    """ Computes the number of slope sign changes in the input data.

    Parameters:
        ch (np.ndarray): Input data for which to compute the slope sign changes.
        threshold (float): Threshold for detecting significant changes.

    Returns:
        int: Number of slope sign changes.
    """
    return np.sum((np.diff(np.sign(np.diff(ch))) != 0) & (np.abs(np.diff(np.diff(ch))) > threshold))


def waveform_length(data):
    """ Computes the waveform length of the input data.

    Parameters:
        data (np.ndarray) shape (n_channels, n_samples): Input data for which to compute the waveform length.

    Returns:
        np.ndarray: Waveform length for each channel.
    """
    return np.sum(np.abs(np.diff(data)))


def root_mean_square(data):
    """ Computes the Root MEan Square (RMS) of the input data.

    Parameters:
        data (np.ndarray) shape (n_channels, n_samples): Input data for which to compute the RMS.

    Returns:
        np.ndarray: RMS value for each channel.
    """

    return np.sqrt(np.mean(data ** 2))


FEATURE_REGISTRY = {
    'mean_absolute_value': mean_absolute_value,
    'zero_crossings': zero_crossings,
    'slope_sign_changes': slope_sign_changes,
    'waveform_length': waveform_length,
    'root_mean_square': root_mean_square,
}


def extract_features(segment, feature_fns=None):
    """
    Extracts features from a multichannel segment.

    Parameters:
        segment: np.ndarray (n_channels, n_samples)
        feature_fns: list of strings or callables

    Returns:
        1D np.ndarray: flattened feature vector
    """
    if feature_fns is None:
        feature_fns = list(FEATURE_REGISTRY.values())

    # Resolve string names to callables
    resolved_fns = []
    for fn in feature_fns:
        if isinstance(fn, str):
            if fn not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown feature name: {fn}")
            resolved_fns.append(FEATURE_REGISTRY[fn])
        else:
            resolved_fns.append(fn)

    feats = []
    for ch in segment:
        for fn in resolved_fns:
            feats.append(fn(ch))
    return np.array(feats)

