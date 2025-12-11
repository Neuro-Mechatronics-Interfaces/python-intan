"""
Dataset building utilities for EMG gesture classification.

This module provides specialized functions for building training datasets
from electrophysiology recordings. These functions combine file I/O,
preprocessing, feature extraction, and label management into reusable
workflow components.
"""

import os
import re
import json
import logging
from typing import List, Tuple, Optional

import numpy as np

from intan.io import (
    labels_from_events,
    build_indices_from_mapping,
    find_event_for_file,
)
from intan.processing import EMGPreprocessor, aggregate_imu_features, append_imu_features


def select_channels(
    raw_names: List[str],
    channels: List[int] | None,
    channel_map: str | None,
    channel_map_file: str,
    non_strict: bool = False
) -> Tuple[List[int], List[str]]:
    """
    Select channels by mapping or explicit indices.
    
    Args:
        raw_names: List of raw channel names from recording
        channels: Optional explicit list of channel indices
        channel_map: Optional name of channel mapping from JSON file
        channel_map_file: Path to JSON file containing channel mappings
        non_strict: If True, allow missing channels in mapping (default: False)
        
    Returns:
        tuple: (channel_indices, channel_names)
        
    Raises:
        KeyError: If channel_map not found in mapping file
        
    Example:
        >>> indices, names = select_channels(
        ...     raw_names, None, "8-8-L", "mappings.json"
        ... )
    """
    if channel_map:
        with open(channel_map_file) as f:
            mappings = json.load(f)
        if channel_map not in mappings:
            raise KeyError(f"Mapping '{channel_map}' not in {channel_map_file}")
        mapping_names = list(mappings[channel_map])
        indices = build_indices_from_mapping(raw_names, mapping_names, strict=not non_strict)
        return indices, [raw_names[i] for i in indices]
    
    if channels:
        return channels, [raw_names[i] for i in channels]
    
    return list(range(len(raw_names))), raw_names


def process_recording(
    data: dict,
    file_path: str,
    root_dir: str,
    events_file: str | None,
    window_ms: int,
    step_ms: int,
    paper_style: bool = False,
    channels: List[int] | None = None,
    modality: str = "emg",
    imu_features: str = "rich",
    imu_norm: str = "zscore",
    ignore_labels: List[str] | None = None,
    ignore_case: bool = False,
    keep_trial: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Process single recording: preprocess, extract features, load labels.
    
    This function performs the complete pipeline for preparing a single
    recording for machine learning:
    1. Extract EMG data and metadata
    2. Apply preprocessing (filtering, envelope)
    3. Extract features using sliding windows
    4. Load and align labels from events
    5. Filter unwanted labels
    6. Optionally integrate IMU features
    
    Args:
        data: Data dictionary from load_rhd_file, load_npz_file, etc.
        file_path: Path to the data file (for finding events)
        root_dir: Root directory for searching event files
        events_file: Optional explicit path to event file
        window_ms: Window size in milliseconds
        step_ms: Step size in milliseconds
        paper_style: If True, use paper preprocessing (120Hz highpass, RMS only)
        channels: Optional list of channel indices to select
        modality: Feature modality ('emg', 'imu', 'both')
        imu_features: IMU feature mode ('mean' or 'rich')
        imu_norm: IMU normalization method ('zscore' or 'robust')
        ignore_labels: Optional list of labels to exclude
        ignore_case: If True, ignore case when filtering labels
        keep_trial: If True, keep trial numbers in labels (e.g., 'fist_3')
        
    Returns:
        tuple: (X, y, metadata)
            - X: Feature matrix (n_windows, n_features)
            - y: Label array (n_windows,)
            - metadata: dict with 'fs', 'selected_channels', 'channel_names'
            
    Example:
        >>> X, y, meta = process_recording(
        ...     data, "rec.rhd", "/data", None,
        ...     window_ms=200, step_ms=50, paper_style=True
        ... )
    """
    
    # Extract EMG data
    emg_fs = float(data.get("frequency_parameters", {}).get("amplifier_sample_rate") or 
                   data.get("sample_rate", 4000))
    emg = data["amplifier_data"]
    emg_t = data["t_amplifier"]
    raw_names = data.get("channel_names", [f"CH{i}" for i in range(emg.shape[0])])
    
    if channels:
        emg = emg[channels]
    
    # Preprocessing
    if paper_style:
        pre = EMGPreprocessor(fs=emg_fs, highpass_cutoff=120.0, apply_envelope=False)
    else:
        pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=5.0)
    
    emg_pp = pre.preprocess(emg)
    
    # Features
    feature_fns = ["rms"] if paper_style else None
    X = pre.extract_emg_features(emg_pp, window_ms=window_ms, step_ms=step_ms, 
                                   feature_fns=feature_fns, progress=True)
    
    # Window indices
    start_idx = int(round(emg_t[0] * emg_fs))
    step_samples = int(round(step_ms / 1000.0 * emg_fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_idx
    
    # Labels
    if events_file is None:
        events_dir = os.path.join(root_dir, "events")
        events_file = find_event_for_file(events_dir, file_path)
        if not events_file:
            raise FileNotFoundError(f"No event file for {file_path}")
    
    logging.info(f"             Loading labels from: {os.path.basename(events_file)}")
    y = labels_from_events(events_file, window_starts)
    
    # Debug: show unique labels before filtering
    unique_before = sorted(set(y))
    logging.debug(f"             Labels before filter: {unique_before}")
    
    # Filter labels
    mask = np.ones(len(y), dtype=bool)
    if ignore_labels:
        if ignore_case:
            ignore_set = {lbl.lower() for lbl in ignore_labels}
            mask &= ~np.array([lbl.lower() in ignore_set for lbl in y])
            logging.info(f"             Filtering (case-insensitive): {ignore_labels}")
        else:
            mask &= ~np.isin(y, ignore_labels)
            logging.info(f"             Filtering (case-sensitive): {ignore_labels}")
    
    if not keep_trial:
        y = np.array([re.sub(r'_\d+$', '', lbl) for lbl in y])
    
    X, y = X[mask], y[mask]
    
    # Debug: show unique labels after filtering
    unique_after = sorted(set(y))
    logging.info(f"             Labels after filter: {unique_after} ({len(unique_after)} classes)")
    
    # IMU features (CSV only)
    if modality in ("both", "imu") and "board_adc_data" in data:
        window_samples = int(round(window_ms / 1000.0 * emg_fs))
        imu_win, imu_cols = aggregate_imu_features(
            data["board_adc_data"], data.get("board_adc_channels"), 
            window_starts, window_samples, mode=imu_features
        )
        
        if modality == "both":
            X, _ = append_imu_features(X, y, imu_win, imu_cols, mask, imu_norm)
        elif modality == "imu" and imu_win is not None:
            X = imu_win[mask]
    
    metadata = {
        "fs": emg_fs,
        "selected_channels": channels,
        "channel_names": raw_names if channels is None else [raw_names[i] for i in channels]
    }
    
    return X, y, metadata


def save_dataset(
    save_path: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata: dict,
    window_ms: int,
    step_ms: int,
    channel_map: str | None = None,
    channel_map_file: str = "custom_channel_mappings.json",
    modality: str = "emg",
):
    """
    Save dataset to NPZ with metadata.
    
    Creates a comprehensive NPZ file containing features, labels, and all
    metadata needed to understand and reproduce the dataset.
    
    Args:
        save_path: Path to save NPZ file
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
        metadata: dict with 'fs', 'selected_channels', 'channel_names'
        window_ms: Window size used for feature extraction
        step_ms: Step size used for feature extraction
        channel_map: Optional name of channel mapping used
        channel_map_file: Path to channel mapping JSON file
        modality: Feature modality ('emg', 'imu', 'both')
        
    Example:
        >>> save_dataset(
        ...     "dataset.npz", X, y, meta,
        ...     window_ms=200, step_ms=50,
        ...     channel_map="8-8-L"
        ... )
    """
    from intan.processing import FEATURE_REGISTRY
    
    class_names = sorted(set(y))
    label_to_id = {c: i for i, c in enumerate(class_names)}
    
    np.savez(
        save_path,
        X=X, y=y,
        emg_fs=metadata["fs"],
        class_names=np.array(class_names, dtype=object),
        label_to_id_json=np.array(json.dumps(label_to_id), dtype=object),
        window_ms=window_ms,
        step_ms=step_ms,
        selected_channels=np.array(metadata.get("selected_channels", []), dtype=int),
        channel_names=np.array(metadata.get("channel_names", []), dtype=object),
        feature_spec=json.dumps({"feature_names": list(FEATURE_REGISTRY.keys()), 
                                  "n_features": X.shape[1]}),
        channel_mapping_name=np.array(channel_map or "", dtype=object),
        channel_mapping_file=np.array(channel_map_file, dtype=object),
        modality=np.array(modality, dtype=object),
    )
    
    logging.info(f"Saved: {save_path} (X={X.shape}, y={y.shape}, classes={len(class_names)})")
