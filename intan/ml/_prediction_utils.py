"""
Shared utilities for EMG gesture prediction across different input sources.
"""

import logging
import numpy as np
from typing import Tuple, Optional

from intan.processing import EMGPreprocessor
from intan.ml import ModelManager


def extract_features_from_emg(
    emg: np.ndarray,
    fs: float,
    window_ms: int,
    step_ms: int,
    env_cut: float,
    verbose: bool = False,
    progress: bool = True,
    desc: str = "Extracting features"
) -> np.ndarray:
    """
    Preprocess EMG and extract features using training-locked parameters.
    
    Args:
        emg: EMG data (channels, samples)
        fs: Sampling frequency in Hz
        window_ms: Feature window size in milliseconds
        step_ms: Window step/hop size in milliseconds
        env_cut: Envelope cutoff frequency in Hz
        verbose: Enable verbose logging
        progress: Show progress bar
        desc: Progress bar description
        
    Returns:
        Feature matrix (n_windows, n_features)
    """
    # Try both naming variants for envelope args
    try:
        pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)
    except TypeError:
        pre = EMGPreprocessor(fs=fs, env_cut=env_cut)
    
    emg_pp = pre.preprocess(emg)
    X = pre.extract_emg_features(
        emg_pp,
        window_ms=window_ms,
        step_ms=step_ms,
        progress=progress,
        tqdm_kwargs={"desc": desc, "unit": "win", "leave": False, "ascii": True}
    )
    
    logging.info(f"Extracted feature matrix: {X.shape}")
    return X


def compute_window_starts(
    n_windows: int,
    step_ms: int,
    fs: float,
    t0: float = 0.0
) -> np.ndarray:
    """
    Compute window start sample indices for alignment with events.
    
    Args:
        n_windows: Number of feature windows
        step_ms: Window step size in milliseconds
        fs: Sampling frequency in Hz
        t0: Start time offset in seconds
        
    Returns:
        Array of window start sample indices
    """
    start_index = int(round(t0 * fs))
    step_samples = int(round(step_ms / 1000.0 * fs))
    window_starts = np.arange(n_windows, dtype=int) * step_samples + start_index
    
    logging.debug(
        f"Window alignment: start_index={start_index}, "
        f"step={step_samples} samples, n_windows={n_windows}"
    )
    
    return window_starts


def predict_with_model(
    X: np.ndarray,
    root_dir: str,
    label: str = "",
    verbose: bool = False
) -> np.ndarray:
    """
    Load trained model and predict on feature matrix.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        root_dir: Directory containing trained model
        label: Model label/tag
        verbose: Enable verbose logging
        
    Returns:
        Predicted class labels
        
    Raises:
        ValueError: If feature dimensions don't match trained model
    """
    from intan.ml import EMGClassifier
    
    manager = ModelManager(
        root_dir=root_dir,
        label=label,
        model_cls=EMGClassifier,
        config={"verbose": verbose}
    )
    manager.load_model()
    
    n_expected = len(manager.scaler.mean_)
    if X.shape[1] != n_expected:
        raise ValueError(
            f"Feature dimension mismatch: got {X.shape[1]}, expected {n_expected}. "
            f"Ensure channels and feature extraction match training."
        )
    
    logging.info(f"Running prediction on {X.shape[0]} windows...")
    y_pred = manager.predict(X)
    
    return y_pred


def predict_rhd_file(
    rhd_path: str,
    root_dir: str,
    label: str,
    window_ms: int,
    step_ms: int,
    env_cut: float,
    trained_channel_names: list[str],
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete prediction pipeline for RHD file.
    
    Args:
        rhd_path: Path to .rhd file
        root_dir: Directory containing trained model
        label: Model label/tag
        window_ms: Feature window size in ms
        step_ms: Window step size in ms
        env_cut: Envelope cutoff frequency in Hz
        trained_channel_names: Channel names in training order
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (window_starts, predictions)
    """
    from intan.io import load_rhd_file, select_training_channels_by_name
    
    # Load RHD file
    data = load_rhd_file(rhd_path, verbose=verbose)
    emg = data["amplifier_data"]
    emg_fs = data['frequency_parameters']['amplifier_sample_rate']
    raw_channel_names = list(data.get("channel_names", [])) or [
        f"CH{i}" for i in range(emg.shape[0])
    ]
    
    if "t_amplifier" in data and data["t_amplifier"].size:
        emg_t = data["t_amplifier"]
    else:
        emg_t = np.arange(emg.shape[1], dtype=float) / emg_fs
    
    t0 = float(emg_t[0])
    dur_s = emg.shape[1] / emg_fs
    
    logging.info(
        f"RHD: fs={emg_fs:.1f} Hz, shape={emg.shape}, "
        f"duration={dur_s:.2f}s, t0={t0:.3f}s"
    )
    
    # Reorder channels to match training
    emg, sel_idx = select_training_channels_by_name(
        emg, raw_channel_names, trained_channel_names
    )
    logging.info(f"Selected {len(sel_idx)} channels in training order")
    
    # Extract features
    X = extract_features_from_emg(
        emg, emg_fs, window_ms, step_ms, env_cut,
        verbose=verbose, progress=True
    )
    
    # Compute window alignment
    window_starts = compute_window_starts(X.shape[0], step_ms, emg_fs, t0)
    
    # Predict
    y_pred = predict_with_model(X, root_dir, label, verbose)
    
    return window_starts, y_pred
