#!/usr/bin/env python3
"""
3_calibrate_and_normalize.py - EMG Calibration and Normalization

STEP 6 (OPTIONAL): Create calibration files for session-to-session EMG normalization.

This script implements calibration and normalization strategies to handle
session-to-session variability caused by:
- Electrode placement differences
- Postural changes (elbow position, forearm orientation)
- Co-contraction of stabilizing muscles
- Baseline EMG activity differences

Two main approaches:
1. MVC Normalization: Normalize EMG by maximum voluntary contraction
2. Resting Baseline Subtraction: Remove session-specific baseline activity
3. Session Calibration: Record known poses and adjust predictions

Usage:
    # Normalize existing dataset
    python 5_calibrate_and_normalize.py normalize --dataset_path dataset.npz
    
    # Create calibration file from recording
    python 5_calibrate_and_normalize.py calibrate --file_path calibration.rhd
    
    # Predict with calibration
    python 4_predict.py file --file_path test.rhd --calibration_file calibration.npz
"""

import os
import sys
import json
import argparse
import logging
from typing import Optional, Tuple, Dict

import numpy as np
from scipy.interpolate import interp1d

from intan.io import load_single_file, load_config_file
from intan.processing import bandpass_filter, notch_filter, extract_features_sliding_window


def compute_mvc_normalization(
    emg_data: np.ndarray,
    fs: float,
    percentile: float = 99.0,
    window_ms: float = 200.0
) -> np.ndarray:
    """
    Compute Maximum Voluntary Contraction (MVC) normalization factors.
    
    For each channel, compute the 99th percentile of RMS values across
    a sliding window. This represents near-maximum activation.
    
    Args:
        emg_data: Raw EMG (channels, samples)
        fs: Sampling rate
        percentile: Percentile to use for MVC estimate (default 99%)
        window_ms: Window size for RMS computation (ms)
        
    Returns:
        mvc_factors: MVC value for each channel (channels,)
    """
    n_channels = emg_data.shape[0]
    window_samples = int(window_ms * fs / 1000)
    
    mvc_factors = np.zeros(n_channels)
    
    for ch_idx in range(n_channels):
        ch_data = emg_data[ch_idx, :]
        
        # Compute RMS in sliding windows
        n_windows = len(ch_data) - window_samples + 1
        rms_values = np.zeros(n_windows)
        
        for i in range(n_windows):
            window = ch_data[i:i+window_samples]
            rms_values[i] = np.sqrt(np.mean(window ** 2))
        
        # Use high percentile as MVC estimate
        mvc_factors[ch_idx] = np.percentile(rms_values, percentile)
    
    # Avoid division by zero
    mvc_factors[mvc_factors < 1e-6] = 1.0
    
    return mvc_factors


def compute_resting_baseline(
    emg_data: np.ndarray,
    fs: float,
    duration_sec: float = 5.0,
    method: str = 'median'
) -> np.ndarray:
    """
    Compute resting baseline EMG activity.
    
    Uses the first N seconds of recording (assumed to be rest) to compute
    baseline activity for each channel.
    
    Args:
        emg_data: Raw EMG (channels, samples)
        fs: Sampling rate
        duration_sec: Duration to use for baseline (seconds)
        method: 'mean' or 'median' or 'percentile10'
        
    Returns:
        baseline: Baseline value for each channel (channels,)
    """
    n_samples = int(duration_sec * fs)
    n_samples = min(n_samples, emg_data.shape[1])
    
    baseline_segment = emg_data[:, :n_samples]
    
    if method == 'mean':
        baseline = np.mean(np.abs(baseline_segment), axis=1)
    elif method == 'median':
        baseline = np.median(np.abs(baseline_segment), axis=1)
    elif method == 'percentile10':
        baseline = np.percentile(np.abs(baseline_segment), 10, axis=1)
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    
    return baseline


def normalize_dataset(
    dataset_path: str,
    output_path: Optional[str] = None,
    method: str = 'mvc',
    overwrite: bool = False,
    verbose: bool = False
):
    """
    Normalize an existing training dataset.
    
    This computes normalization factors from the dataset itself and
    applies them to both features and saves the factors for prediction time.
    
    Args:
        dataset_path: Path to input .npz dataset
        output_path: Path to save normalized dataset (default: add _normalized suffix)
        method: 'mvc' or 'baseline' or 'both'
        overwrite: Overwrite existing output
        verbose: Verbose logging
    """
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl, force=True)
    
    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(dataset_path)
        output_path = f"{base}_normalized{ext}"
    
    if os.path.exists(output_path) and not overwrite:
        logging.info(f"[OK] Normalized dataset exists: {output_path}")
        return
    
    logging.info(f"[INFO] Normalizing dataset: {os.path.basename(dataset_path)}")
    logging.info(f"   Method: {method}")
    
    # Load dataset
    data = np.load(dataset_path)
    X = data['X']
    y = data['y']
    
    logging.info(f"   Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Note: We can't directly normalize the features because they're already
    # extracted. We need to normalize at the raw EMG level during dataset building.
    # For now, we'll implement z-score normalization on the features themselves.
    
    # Compute per-feature statistics
    feature_mean = np.mean(X, axis=0)
    feature_std = np.std(X, axis=0)
    feature_std[feature_std < 1e-6] = 1.0  # Avoid division by zero
    
    # Normalize
    X_normalized = (X - feature_mean) / feature_std
    
    logging.info(f"   Feature normalization:")
    logging.info(f"      Mean range: [{feature_mean.min():.2f}, {feature_mean.max():.2f}]")
    logging.info(f"      Std range: [{feature_std.min():.2f}, {feature_std.max():.2f}]")
    
    # Save normalized dataset
    np.savez(
        output_path,
        X=X_normalized,
        y=y,
        feature_mean=feature_mean,
        feature_std=feature_std,
        normalization_method=method,
        **{k: data[k] for k in data.files if k not in ['X', 'y']}
    )
    
    logging.info(f"[OK] Saved normalized dataset: {output_path}")


def create_calibration_file(
    root_dir: str,
    file_path: str,
    output_path: Optional[str] = None,
    rest_duration_sec: float = 5.0,
    mvc_duration_sec: float = 5.0,
    verbose: bool = False
):
    """
    Create a calibration file from a recording.
    
    Expected recording structure:
    - First N seconds: Rest (hands relaxed)
    - Next M seconds: Maximum voluntary contraction (MVC) - squeeze fist hard
    
    Args:
        root_dir: Project root directory
        file_path: Path to calibration recording
        output_path: Path to save calibration .npz file
        rest_duration_sec: Duration of rest period (seconds)
        mvc_duration_sec: Duration of MVC period (seconds)
        verbose: Verbose logging
    """
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl, force=True)
    
    logging.info(f"[INFO] Creating calibration file from: {os.path.basename(file_path)}")
    
    # Determine output path
    if output_path is None:
        base, _ = os.path.splitext(file_path)
        output_path = f"{base}_calibration.npz"
    
    # Load EMG data
    data = load_single_file('rhd', file_path, root_dir, verbose)
    emg_data = data['amplifier_data']
    fs = data['frequency_parameters']['amplifier_sample_rate']
    
    logging.info(f"   EMG: {emg_data.shape[0]} channels, {emg_data.shape[1]} samples @ {fs:.0f} Hz")
    
    # Preprocess
    logging.info(f"   Preprocessing...")
    emg_data = notch_filter(emg_data, fs=fs, f0=60)
    emg_data = bandpass_filter(emg_data, lowcut=20, highcut=450, fs=fs)
    
    # Extract rest period (first N seconds)
    rest_samples = int(rest_duration_sec * fs)
    rest_samples = min(rest_samples, emg_data.shape[1])
    rest_data = emg_data[:, :rest_samples]
    
    # Compute resting baseline
    baseline = np.median(np.abs(rest_data), axis=1)
    
    logging.info(f"   Resting baseline (median): {np.mean(baseline):.2f} µV")
    
    # Extract MVC period (next M seconds after rest)
    mvc_start = rest_samples
    mvc_end = min(mvc_start + int(mvc_duration_sec * fs), emg_data.shape[1])
    
    if mvc_end > mvc_start:
        mvc_data = emg_data[:, mvc_start:mvc_end]
        mvc_factors = compute_mvc_normalization(mvc_data, fs)
        logging.info(f"   MVC factors: mean={np.mean(mvc_factors):.2f} µV, std={np.std(mvc_factors):.2f}")
    else:
        logging.warning(f"   [WARNING] No MVC data found, using baseline only")
        mvc_factors = np.ones(emg_data.shape[0])
    
    # Save calibration file
    np.savez(
        output_path,
        baseline=baseline,
        mvc_factors=mvc_factors,
        fs=fs,
        n_channels=emg_data.shape[0],
        rest_duration_sec=rest_duration_sec,
        mvc_duration_sec=mvc_duration_sec
    )
    
    logging.info(f"[OK] Calibration file saved: {output_path}")
    logging.info(f"   Use with: python 4_predict.py file --calibration_file {output_path}")


def apply_calibration_to_emg(
    emg_data: np.ndarray,
    calibration_path: str,
    method: str = 'both'
) -> np.ndarray:
    """
    Apply calibration normalization to raw EMG data.
    
    Args:
        emg_data: Raw EMG data (channels, samples)
        calibration_path: Path to calibration .npz file
        method: 'baseline', 'mvc', or 'both'
        
    Returns:
        emg_normalized: Calibrated EMG data
    """
    # Load calibration
    calib = np.load(calibration_path)
    baseline = calib['baseline']
    mvc_factors = calib['mvc_factors']
    
    emg_normalized = emg_data.copy()
    
    # Subtract baseline (remove resting activity)
    if method in ['baseline', 'both']:
        emg_normalized = emg_normalized - baseline[:, np.newaxis]
    
    # Divide by MVC (normalize to percentage of max)
    if method in ['mvc', 'both']:
        emg_normalized = emg_normalized / mvc_factors[:, np.newaxis]
    
    return emg_normalized


def main():
    parser = argparse.ArgumentParser(
        description="EMG Calibration and Normalization Tools"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Normalize command
    normalize_parser = subparsers.add_parser('normalize', help='Normalize existing dataset')
    normalize_parser.add_argument('--dataset_path', type=str, required=True,
                                 help='Path to input dataset .npz file')
    normalize_parser.add_argument('--output_path', type=str, default=None,
                                 help='Path to save normalized dataset')
    normalize_parser.add_argument('--method', type=str, default='zscore',
                                 choices=['zscore', 'mvc', 'baseline'],
                                 help='Normalization method')
    normalize_parser.add_argument('--overwrite', action='store_true',
                                 help='Overwrite existing output')
    normalize_parser.add_argument('--verbose', action='store_true',
                                 help='Verbose logging')
    
    # Calibrate command
    calibrate_parser = subparsers.add_parser('calibrate', help='Create calibration file')
    calibrate_parser.add_argument('--root_dir', type=str, default='.',
                                 help='Project root directory')
    calibrate_parser.add_argument('--file_path', type=str, required=True,
                                 help='Path to calibration recording')
    calibrate_parser.add_argument('--output_path', type=str, default=None,
                                 help='Path to save calibration file')
    calibrate_parser.add_argument('--rest_duration', type=float, default=5.0,
                                 help='Duration of rest period (seconds)')
    calibrate_parser.add_argument('--mvc_duration', type=float, default=5.0,
                                 help='Duration of MVC period (seconds)')
    calibrate_parser.add_argument('--verbose', action='store_true',
                                 help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.command == 'normalize':
        normalize_dataset(
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            method=args.method,
            overwrite=args.overwrite,
            verbose=args.verbose
        )
    
    elif args.command == 'calibrate':
        create_calibration_file(
            root_dir=args.root_dir,
            file_path=args.file_path,
            output_path=args.output_path,
            rest_duration_sec=args.rest_duration,
            mvc_duration_sec=args.mvc_duration,
            verbose=args.verbose
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
