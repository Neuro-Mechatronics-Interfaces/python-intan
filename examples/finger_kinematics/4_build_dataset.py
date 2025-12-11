#!/usr/bin/env python3
"""
4_build_dataset.py - Finger Kinematics Dataset Builder

STEP 4: Build a regression dataset mapping EMG features to continuous joint angles.

Prerequisites:
    1. Run 1_synchronize_emg_video.py (creates sync/ directory)
    2. Run 2_extract_joint_angles.py (creates joint angle CSVs)

This script:
1. Loads EMG data from RHD/NPZ/CSV files
2. Loads corresponding joint angle data from CSV files
3. **Applies synchronization offsets from sync/ directory**
4. Extracts EMG features using sliding windows
5. Interpolates joint angles to match EMG window timestamps (with sync correction)
6. Saves combined dataset for regression training

Expected file structure:
    root_dir/
        raw/               # EMG recordings (.rhd, .npz, etc.)
        joint_angles/      # Joint angle CSV files
        dataset/           # Output .npz dataset

Joint angle CSV format:
    timestamp,joint_0,joint_1,joint_2,...
    0.000,0.5,1.2,0.8,...
    0.010,0.6,1.3,0.9,...
    
Examples:
    # Single file with explicit paths
    python 1_build_dataset.py --root_dir /data --file_path recording.rhd --angles_file angles.csv
    
    # Multi-file auto-discovery
    python 1_build_dataset.py --root_dir /data --multi_file
    
    # With channel selection
    python 1_build_dataset.py --root_dir /data --multi_file --channels 0:64 --overwrite
"""

import os
import sys
import json
import argparse
import logging
from time import time
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from intan.io import (
    load_config_file,
    load_simple_config,
    save_simple_config,
    prompt_directory,
    prompt_file,
    get_or_prompt_value,
    parse_channels_spec,
    discover_and_group_files,
    load_single_file,
    load_files_merged,
    select_channels,
    save_dataset,
)
from intan.processing import extract_features_sliding_window, bandpass_filter, notch_filter, extract_features, FEATURE_REGISTRY
from tqdm import tqdm
import re


def strip_timestamp_suffix(filename: str) -> str:
    """
    Remove timestamp suffix from filename (e.g., '_YYMMDD_HHMMSS').
    
    Args:
        filename: Base filename without extension
        
    Returns:
        Filename with timestamp removed
        
    Examples:
        'train_dynamic_251110_130130' -> 'train_dynamic'
        'train_dynamic2_251110_130815' -> 'train_dynamic2'
        'train_dynamic' -> 'train_dynamic'
    """
    # Pattern: underscore followed by 6 digits (YYMMDD), underscore, 6 digits (HHMMSS)
    pattern = r'_\d{6}_\d{6}$'
    return re.sub(pattern, '', filename)


def find_matching_angle_file(emg_filename: str, angles_dir: str, verbose: bool = False) -> Optional[str]:
    """
    Find matching angle CSV file for EMG recording, with fuzzy matching.
    
    Tries multiple strategies:
    1. Exact match: <emg_filename>_angles.csv or <emg_filename>.csv
    2. Without timestamp: Strip timestamp suffix and match
    3. Fuzzy match: Find files with similar base name
    
    Args:
        emg_filename: EMG filename stem (without extension)
        angles_dir: Directory containing angle CSV files
        verbose: Enable debug logging
        
    Returns:
        Path to matching angle file, or None if not found
    """
    if not os.path.exists(angles_dir):
        return None
    
    # Strategy 1: Exact match
    exact_patterns = [
        f"{emg_filename}_angles.csv",
        f"{emg_filename}.csv",
    ]
    
    for pattern in exact_patterns:
        path = os.path.join(angles_dir, pattern)
        if os.path.exists(path):
            if verbose:
                logging.debug(f"Found exact match: {pattern}")
            return path
    
    # Strategy 2: Strip timestamp and try again
    base_name = strip_timestamp_suffix(emg_filename)
    if base_name != emg_filename:
        timestamp_patterns = [
            f"{base_name}_angles.csv",
            f"{base_name}.csv",
        ]
        
        for pattern in timestamp_patterns:
            path = os.path.join(angles_dir, pattern)
            if os.path.exists(path):
                if verbose:
                    logging.debug(f"Found match without timestamp: {pattern}")
                return path
    
    # Strategy 3: Fuzzy match - find files starting with base name
    try:
        csv_files = [f for f in os.listdir(angles_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            csv_stem = os.path.splitext(csv_file)[0].replace('_angles', '')
            if csv_stem == base_name or emg_filename.startswith(csv_stem):
                path = os.path.join(angles_dir, csv_file)
                if verbose:
                    logging.debug(f"Found fuzzy match: {csv_file}")
                return path
    except OSError:
        pass
    
    return None


def extract_features_with_progress(data: np.ndarray, fs: float, window_ms: float, step_ms: float, feature_fns=None, desc="Extracting features") -> np.ndarray:
    """
    Extract features with progress bar.
    
    Args:
        data: (n_channels, n_samples) array of EMG data
        fs: sampling rate in Hz
        window_ms: window length in ms
        step_ms: step between windows in ms
        feature_fns: list of feature functions (None = use default)
        desc: progress bar description
        
    Returns:
        (n_windows, n_features) array
    """
    # Compute sizes in samples
    w = int(window_ms/1000 * fs)
    s = int(step_ms/1000 * fs)
    n_samples = data.shape[1]
    n_windows = 1 + (n_samples - w)//s
    
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
    for i in tqdm(range(n_windows), desc=desc, ncols=80, unit="window"):
        start = i * s
        seg = data[:, start:start + w]
        fv = extract_features(seg, resolved_fns)
        feats.append(fv)
    
    return np.vstack(feats)


def load_joint_angles_csv(csv_path: str, verbose: bool = False) -> tuple:
    """
    Load joint angle data from CSV file.
    
    Expected format:
        timestamp,joint_0,joint_1,joint_2,...
        
    Args:
        csv_path: Path to CSV file
        verbose: Enable verbose logging
        
    Returns:
        tuple: (timestamps, angles) where angles is shape (n_samples, n_joints)
    """
    if verbose:
        logging.debug(f"Loading joint angles from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # First column is timestamp
    timestamps = df.iloc[:, 0].values
    
    # Remaining columns are joint angles
    angles = df.iloc[:, 1:].values
    
    if verbose:
        logging.debug(f"   Loaded {len(timestamps)} samples, {angles.shape[1]} joints")
    
    return timestamps, angles


def load_sync_offset(sync_dir: str, file_stem: str, verbose: bool = False) -> float:
    """
    Load synchronization offset for a recording.
    
    Args:
        sync_dir: Directory containing sync JSON files
        file_stem: Base filename without extension
        verbose: Enable debug logging
        
    Returns:
        Time offset in seconds (0.0 if no sync file found)
    """
    # Try multiple patterns
    patterns = [
        os.path.join(sync_dir, f"{file_stem}_sync.json"),
        os.path.join(sync_dir, f"{file_stem}.json"),
    ]
    
    # Also try without timestamp suffix
    base_no_timestamp = strip_timestamp_suffix(file_stem)
    if base_no_timestamp != file_stem:
        patterns.extend([
            os.path.join(sync_dir, f"{base_no_timestamp}_sync.json"),
            os.path.join(sync_dir, f"{base_no_timestamp}.json"),
        ])
    
    for pattern in patterns:
        if os.path.exists(pattern):
            with open(pattern, 'r') as f:
                sync_info = json.load(f)
                offset_sec = sync_info.get('offset_sec', 0.0)
                if verbose:
                    logging.debug(f"   Loaded sync offset: {offset_sec:.3f}s from {os.path.basename(pattern)}")
                return offset_sec
    
    if verbose:
        logging.debug(f"   No sync file found for: {file_stem}")
    return 0.0


def interpolate_angles_to_windows(
    angle_timestamps: np.ndarray,
    angles: np.ndarray,
    window_timestamps: np.ndarray,
    sync_offset_sec: float = 0.0,
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate joint angles to match EMG window timestamps.
    
    Args:
        angle_timestamps: Original timestamps from CSV (n_angle_samples,)
        angles: Joint angle values (n_angle_samples, n_joints)
        window_timestamps: Target timestamps for EMG windows (n_windows,)
        sync_offset_sec: Time offset from cross-correlation (seconds)
            Positive offset means landmark signal is DELAYED relative to EMG,
            so we SUBTRACT the offset to shift landmarks backward in time.
        method: Interpolation method ('linear', 'cubic', etc.)
        
    Returns:
        np.ndarray: Interpolated angles (n_windows, n_joints)
    """
    # Apply sync offset: subtract because positive offset means landmarks are delayed
    # (i.e., we need to shift landmark timestamps backward to align with EMG)
    angle_timestamps = angle_timestamps - sync_offset_sec
    n_joints = angles.shape[1]
    interpolated = np.zeros((len(window_timestamps), n_joints))
    
    for joint_idx in range(n_joints):
        f = interp1d(
            angle_timestamps, 
            angles[:, joint_idx], 
            kind=method, 
            fill_value='extrapolate',
            bounds_error=False
        )
        interpolated[:, joint_idx] = f(window_timestamps)
    
    return interpolated


def build_dataset(
    root_dir: str,
    file_type: str = "rhd",
    file_path: Optional[str] = None,
    file_names: Optional[List[str]] = None,
    multi_file: bool = False,
    angles_file: Optional[str] = None,
    angles_dir: Optional[str] = None,
    label: str = "",
    save_path: Optional[str] = None,
    window_ms: int = 200,
    step_ms: int = 50,
    channels: Optional[List[int]] = None,
    channel_map: Optional[str] = None,
    channel_map_file: str = "custom_channel_mappings.json",
    mapping_non_strict: bool = False,
    exclude_pattern: Optional[str] = None,
    merge_pattern: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = False,
):
    """
    Build finger kinematics regression dataset.
    
    Args:
        root_dir: Root directory containing data files
        file_type: Type of input files ('rhd', 'npz', 'csv')
        file_path: Path to single EMG file (single-file mode)
        file_names: List of specific filenames (multi-file mode)
        multi_file: Enable multi-file aggregation mode
        angles_file: Explicit path to joint angles CSV (single-file mode)
        angles_dir: Directory containing joint angle CSV files (multi-file mode)
        label: Label prefix for output filename
        save_path: Explicit output path (overrides auto-naming)
        window_ms: Feature extraction window size in ms
        step_ms: Feature extraction step size in ms
        channels: Explicit channel indices to select
        channel_map: Named channel mapping from JSON file
        channel_map_file: Path to channel mapping JSON file
        mapping_non_strict: Allow missing channels in mapping
        exclude_pattern: Pattern to exclude file stems
        merge_pattern: Pattern to filter file stems for merging
        overwrite: Overwrite existing output file
        verbose: Enable verbose logging
    """
    start_time = time()
    
    # Setup logging
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl, force=True)
    
    # Validate inputs
    if not os.path.isdir(root_dir):
        raise ValueError(f"root_dir does not exist: {root_dir}")
    
    # Determine output path
    save_path = save_path or os.path.join(
        root_dir, "dataset", f"{label}_kinematics_dataset.npz" if label else "kinematics_dataset.npz"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(save_path) and not overwrite:
        logging.info(f"[OK] Dataset exists: {save_path}")
        logging.info("  Use --overwrite to regenerate")
        return
    
    logging.info(f"[INFO] Building kinematics dataset: {os.path.basename(save_path)}")
    logging.info(f"   Root: {root_dir}")
    logging.info(f"   Type: {file_type.upper()}, Mode: {'Multi-file' if multi_file else 'Single-file'}")
    logging.info(f"   Windows: {window_ms}ms × {step_ms}ms step")
    
    # Default angles directory - check multiple possible locations
    if angles_dir is None:
        possible_dirs = [
            os.path.join(root_dir, "landmarks"),
            os.path.join(root_dir, "media", "landmarks"),
            os.path.join(root_dir, "joint_angles"),
        ]
        # Use first existing directory, or default to joint_angles
        angles_dir = None
        for d in possible_dirs:
            if os.path.exists(d):
                angles_dir = d
                logging.info(f"   Found angles directory: {os.path.relpath(d, root_dir)}")
                break
        if angles_dir is None:
            angles_dir = os.path.join(root_dir, "joint_angles")
            logging.info(f"   Using default angles directory: joint_angles/")
    
    # ===== SINGLE-FILE MODE =====
    if not multi_file:
        if not file_path:
            raise ValueError("Single-file mode requires --file_path")
        
        logging.info(f"[Loading] {os.path.basename(file_path)}")
        data = load_single_file(file_type, file_path, root_dir, verbose)
        
        # Get EMG data
        emg_data = data['amplifier_data']
        fs = data['frequency_parameters']['amplifier_sample_rate']
        
        # Select channels
        raw_names = data.get("channel_names", [f"CH{i}" for i in range(emg_data.shape[0])])
        ch_indices, ch_names = select_channels(
            raw_names, channels, channel_map, channel_map_file, mapping_non_strict
        )
        emg_data = emg_data[ch_indices, :]
        
        logging.info(f"   EMG: {emg_data.shape[0]} channels, {emg_data.shape[1]} samples @ {fs:.0f} Hz")
        
        # Preprocess EMG
        logging.info(f"   Preprocessing EMG...")
        emg_data = notch_filter(emg_data, fs=fs, f0=60)
        emg_data = bandpass_filter(emg_data, lowcut=20, highcut=450, fs=fs)
        
        # Extract features
        window_size_samples = int(window_ms * fs / 1000)
        step_size_samples = int(step_ms * fs / 1000)
        n_windows = 1 + (emg_data.shape[1] - window_size_samples) // step_size_samples
        
        logging.info(f"   Extracting {n_windows} feature windows...")
        X = extract_features_with_progress(
            emg_data, fs, window_ms, step_ms,
            desc="   Extracting features"
        )
        
        window_starts = np.arange(0, emg_data.shape[1] - window_size_samples + 1, step_size_samples)
        window_timestamps = window_starts / fs
        
        logging.info(f"   [OK] {X.shape[0]} windows × {X.shape[1]} features")
        
        # Load joint angles
        if not angles_file:
            # Try to find matching angles file with fuzzy matching
            stem = os.path.splitext(os.path.basename(file_path))[0]
            angles_file = find_matching_angle_file(stem, angles_dir, verbose)
        
        if not angles_file or not os.path.exists(angles_file):
            logging.warning(f"   [WARNING] Joint angles file not found: {angles_file}")
            logging.info(f"   Please select the joint angles CSV file for this recording.")
            angles_file = prompt_file(
                title="Select Joint Angles CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initial_dir=angles_dir
            )
            if not angles_file:
                logging.error("   [ERROR] No angles file selected. Exiting.")
                sys.exit(1)
        
        logging.info(f"   Loading joint angles from: {os.path.basename(angles_file)}")
        angle_timestamps, angles = load_joint_angles_csv(angles_file, verbose)
        
        # Load synchronization offset
        sync_dir = os.path.join(root_dir, 'sync')
        sync_offset = load_sync_offset(sync_dir, stem, verbose)
        if sync_offset != 0.0:
            logging.info(f"   Applying sync offset: {sync_offset:.3f}s")
        
        # Interpolate angles to match EMG windows
        logging.info(f"   Interpolating {angles.shape[1]} joints to {len(window_timestamps)} windows...")
        y = interpolate_angles_to_windows(angle_timestamps, angles, window_timestamps, sync_offset)
        
        # Validate shapes
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Shape mismatch: features {X.shape[0]} != angles {y.shape[0]}")
        
        logging.info(f"[OK] Dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} joints")
        
        # Save
        metadata = {
            "fs": float(fs),
            "selected_channels": ch_indices,
            "channel_names": ch_names,
            "n_joints": int(y.shape[1]),
        }
        
        np.savez(
            save_path,
            X=X,
            y=y,
            window_ms=window_ms,
            step_ms=step_ms,
            **metadata
        )
        
        elapsed = time() - start_time
        logging.info(f"[OK] Saved to: {save_path}")
        logging.info(f"[OK] Completed in {elapsed:.1f}s")
        return
    
    # ===== MULTI-FILE MODE =====
    logging.info("[Discovering files...]")
    file_groups = discover_and_group_files(root_dir, file_type, file_names, exclude_pattern, merge_pattern)
    logging.info(f"   Found {len(file_groups)} recording groups")
    
    if exclude_pattern:
        logging.info(f"   Excluding pattern: {exclude_pattern}")
    if merge_pattern:
        logging.info(f"   Merge pattern: {merge_pattern}")
    
    # Initialize accumulators
    combined_X, combined_y = [], []
    ch_indices, ch_names = None, None
    fs_values = set()
    n_joints_values = set()
    failed_files = []
    
    # Process each recording group
    logging.info(f"\n[Processing recordings...]")
    for i, (stem, files) in enumerate(file_groups.items(), 1):
        logging.info(f"[{i}/{len(file_groups)}] {stem}")
        logging.info(f"             Files: {len(files)}")
        
        try:
            # Load and merge EMG files
            data = load_files_merged(file_type, files, root_dir, verbose)
            
            emg_data = data['amplifier_data']
            fs = data['frequency_parameters']['amplifier_sample_rate']
            
            # First recording determines channel selection
            if ch_indices is None:
                raw_names = data.get("channel_names", [f"CH{j}" for j in range(emg_data.shape[0])])
                ch_indices, ch_names = select_channels(
                    raw_names, channels, channel_map, channel_map_file, mapping_non_strict
                )
                logging.info(f"             Channels: {len(ch_indices)} selected")
            
            emg_data = emg_data[ch_indices, :]
            
            # Preprocess
            if verbose:
                logging.debug(f"             Preprocessing {emg_data.shape[1]} samples...")
            emg_data = notch_filter(emg_data, fs=fs, f0=60)
            emg_data = bandpass_filter(emg_data, lowcut=20, highcut=450, fs=fs)
            
            # Extract features
            window_size_samples = int(window_ms * fs / 1000)
            step_size_samples = int(step_ms * fs / 1000)
            n_windows = 1 + (emg_data.shape[1] - window_size_samples) // step_size_samples
            
            logging.info(f"             Extracting {n_windows} feature windows...")
            X = extract_features_with_progress(
                emg_data, fs, window_ms, step_ms,
                desc=f"             Recording {i}/{len(file_groups)}"
            )
            
            window_starts = np.arange(0, emg_data.shape[1] - window_size_samples + 1, step_size_samples)
            window_timestamps = window_starts / fs
            logging.info(f"             [OK] {X.shape[0]} windows × {X.shape[1]} features")
            
            # Load joint angles - use fuzzy matching
            angles_file = find_matching_angle_file(stem, angles_dir, verbose)
            
            if not angles_file:
                base_name = strip_timestamp_suffix(stem)
                logging.warning(f"             [WARNING] No angles file found for {stem}")
                logging.info(f"             Searched in: {angles_dir}")
                logging.info(f"             Tried: {stem}_angles.csv, {base_name}_angles.csv, {stem}.csv, {base_name}.csv")
                logging.info(f"             Please select the angles CSV or cancel to skip this recording.")
                angles_file = prompt_file(
                    title=f"Select Angles CSV for {stem}",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                    initial_dir=angles_dir
                )
                if not angles_file:
                    logging.warning(f"             [SKIPPED] No file selected, skipping {stem}")
                    failed_files.append(stem)
                    continue
            
            angle_timestamps, angles = load_joint_angles_csv(angles_file, verbose)
            
            # Load synchronization offset
            sync_dir = os.path.join(root_dir, 'sync')
            sync_offset = load_sync_offset(sync_dir, stem, verbose)
            if sync_offset != 0.0:
                logging.info(f"             Sync offset: {sync_offset:.3f}s")
            
            # Interpolate
            y = interpolate_angles_to_windows(angle_timestamps, angles, window_timestamps, sync_offset)
            
            # Validate
            if X.shape[0] != y.shape[0]:
                logging.warning(f"             [WARNING] Shape mismatch: {X.shape[0]} != {y.shape[0]}")
                failed_files.append(stem)
                continue
            
            combined_X.append(X)
            combined_y.append(y)
            fs_values.add(round(fs, 6))
            n_joints_values.add(y.shape[1])
            
            logging.info(f"             [OK] Added {X.shape[0]} windows, {y.shape[1]} joints")
        
        except Exception as e:
            logging.warning(f"             [FAIL] Failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            failed_files.append(stem)
    
    # Validate results
    if not combined_X:
        logging.error("\n" + "="*60)
        logging.error("[ERROR] NO VALID DATA EXTRACTED!")
        logging.error("="*60)
        logging.error(f"   All {len(failed_files)} recording(s) failed")
        logging.error(f"\n   Check that joint angle CSV files exist in: {angles_dir}")
        logging.error(f"   Expected naming: <recording_name>_angles.csv or <recording_name>.csv")
        raise ValueError("No valid data extracted - check joint angle files")
    
    if failed_files:
        logging.warning(f"\n[WARNING] Failed to process {len(failed_files)}/{len(file_groups)} files:")
        for f in failed_files:
            logging.warning(f"   - {f}")
    
    # Check consistency
    if len(fs_values) > 1:
        logging.warning(f"[WARNING] Multiple sampling rates detected: {sorted(fs_values)} Hz")
    
    if len(n_joints_values) > 1:
        raise ValueError(f"Inconsistent joint counts across files: {sorted(n_joints_values)}")
    
    # Concatenate
    logging.info(f"\n[Merging {len(combined_X)} recordings...]")
    X = np.concatenate(combined_X, axis=0)
    y = np.concatenate(combined_y, axis=0)
    
    n_joints = list(n_joints_values)[0]
    logging.info(f"[OK] Total: {X.shape[0]} samples, {X.shape[1]} features, {n_joints} joints")
    
    # Save
    metadata = {
        "fs": float(list(fs_values)[0]) if fs_values else 0.0,
        "selected_channels": ch_indices,
        "channel_names": ch_names,
        "n_joints": int(n_joints),
    }
    
    np.savez(
        save_path,
        X=X,
        y=y,
        window_ms=window_ms,
        step_ms=step_ms,
        **metadata
    )
    
    elapsed = time() - start_time
    logging.info(f"\n[OK] Saved to: {save_path}")
    logging.info(f"[OK] Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_dir", help="Project root directory")
    parser.add_argument("--file_type", default="rhd", choices=["rhd", "npz", "csv"])
    parser.add_argument("--file_path", help="Path to single EMG file")
    parser.add_argument("--file_names", nargs="+", help="List of specific file stems to process")
    parser.add_argument("--multi_file", action="store_true", help="Process multiple files")
    parser.add_argument("--angles_file", help="Path to joint angles CSV (single-file mode)")
    parser.add_argument("--angles_dir", help="Directory with joint angle CSV files")
    parser.add_argument("--label", default="", help="Label prefix for output")
    parser.add_argument("--save_path", help="Explicit output path")
    parser.add_argument("--window_ms", type=int, default=200, help="Feature window size (ms)")
    parser.add_argument("--step_ms", type=int, default=50, help="Feature step size (ms)")
    parser.add_argument("--channels", nargs="+", help="Channel selection (e.g., 0:64)")
    parser.add_argument("--channel_map", help="Named channel mapping")
    parser.add_argument("--channel_map_file", default="custom_channel_mappings.json")
    parser.add_argument("--mapping_non_strict", action="store_true")
    parser.add_argument("--exclude_pattern", help="Pattern to exclude files")
    parser.add_argument("--merge_pattern", help="Pattern to filter files")
    parser.add_argument("--config_file", help="Load parameters from config file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Load shared config
    from pathlib import Path
    cfg = load_simple_config(Path(__file__).parent / ".kinematics_config")
    if args.config_file:
        cfg.update(load_config_file(args.config_file))
    
    # Prompt for root_dir if not provided
    if not args.root_dir:
        root_dir, was_prompted = get_or_prompt_value(
            arg_value=None,
            config=cfg,
            key='root_dir',
            prompt_func=prompt_directory,
            title="Select Project Root Directory",
            initial_dir=cfg.get('root_dir') if 'root_dir' in cfg else None
        )
        
        if not root_dir:
            print("Operation cancelled by user")
            sys.exit(0)
        
        args.root_dir = root_dir
        if was_prompted:
            cfg['root_dir'] = root_dir
            save_simple_config(cfg, Path(__file__).parent / ".kinematics_config", "Finger Kinematics Configuration")
            print("[*] Saved root_dir to .kinematics_config")
    else:
        # Save CLI-provided root_dir to config if not already there
        if 'root_dir' not in cfg or cfg.get('root_dir') != args.root_dir:
            cfg['root_dir'] = args.root_dir
            save_simple_config(cfg, Path(__file__).parent / ".kinematics_config", "Finger Kinematics Configuration")
            print("[*] Saved root_dir to .kinematics_config")
    
    # Prompt for multi_file mode if not specified and no file_path provided
    if not args.multi_file and not args.file_path and 'multi_file' not in cfg:
        from intan.io import prompt_yes_no, prompt_text
        multi_file_response = prompt_yes_no(
            "Dataset Building Mode",
            "Process multiple files from the directory?\n\n"
            "Yes: Process all matching files in directory\n"
            "No: Process a single file (you'll be prompted to select it)"
        )
        
        if multi_file_response is None:
            # User cancelled
            print("Operation cancelled by user")
            sys.exit(0)
        
        args.multi_file = multi_file_response
        # Save to config immediately
        cfg['multi_file'] = multi_file_response
        save_simple_config(cfg, Path(__file__).parent / ".kinematics_config", "Finger Kinematics Configuration")
        print(f"[*] Saved multi_file={multi_file_response} to .kinematics_config")
        
        # If multi-file, ask about excluding files
        if multi_file_response:
            wants_exclude = prompt_yes_no(
                "File Filtering",
                "Do you want to exclude any files from processing?\n\n"
                "Yes: Enter a pattern to exclude (e.g., 'test' excludes files with 'test' in name)\n"
                "No: Process all files"
            )
            
            if wants_exclude is None:
                # User cancelled
                print("Operation cancelled by user")
                sys.exit(0)
            
            if wants_exclude:
                exclude_input = prompt_text(
                    "Exclude Pattern",
                    "Enter pattern to exclude from filenames\n"
                    "(e.g., 'test' or '002' or 'bad')\n\n"
                    "Files containing this pattern will be skipped:",
                    initial_value=cfg.get('exclude_pattern', '')
                )
                
                if exclude_input is None:
                    # User cancelled
                    print("Operation cancelled by user")
                    sys.exit(0)
                
                if exclude_input:
                    args.exclude_pattern = exclude_input
                    cfg['exclude_pattern'] = exclude_input
                    save_simple_config(cfg, Path(__file__).parent / ".kinematics_config", "Finger Kinematics Configuration")
                    print(f"[*] Saved exclude_pattern='{exclude_input}' to .kinematics_config")
    
    # Prompt for file_path if single-file mode and not provided
    # Use cfg.get() to check multi_file from config if not set via CLI
    multi_file_mode = args.multi_file or cfg.get('multi_file', False)
    
    if not multi_file_mode and not args.file_path:
        file_path, was_prompted = get_or_prompt_value(
            arg_value=None,
            config=cfg,
            key='file_path',
            prompt_func=prompt_file,
            title="Select Data File",
            initial_dir=args.root_dir,
            filetypes=[
                (f"{args.file_type.upper()} files", f"*.{args.file_type}"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            # User cancelled file selection
            print("Operation cancelled by user")
            sys.exit(0)
        
        args.file_path = file_path
        # Save to config immediately
        if was_prompted:
            cfg['file_path'] = file_path
            save_simple_config(cfg, Path(__file__).parent / ".kinematics_config", "Finger Kinematics Configuration")
            print("[*] Saved file_path to .kinematics_config")
    
    channels_parsed = parse_channels_spec(args.channels) if args.channels else None
    
    params = {k: getattr(args, k) or cfg.get(k, v) for k, v in {
        "root_dir": None, "file_type": "rhd", "file_path": None, "file_names": None,
        "multi_file": False, "angles_file": None, "angles_dir": None, "label": "",
        "save_path": None, "window_ms": 200, "step_ms": 50,
        "channel_map": None, "channel_map_file": "custom_channel_mappings.json",
        "mapping_non_strict": False, "exclude_pattern": None, "merge_pattern": None,
        "overwrite": False, "verbose": False,
    }.items()}
    params["channels"] = channels_parsed if channels_parsed is not None else cfg.get("channels")
    
    try:
        build_dataset(**params)
    except Exception as e:
        logging.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
