#!/usr/bin/env python3
"""
Unified Dataset Builder for EMG Gesture Classification

Optimized script leveraging intan package utilities.
Supports RHD, NPZ, CSV formats with single/multi-file modes.

Examples:
    # Single file with channel selection
    python 1_build_dataset.py --root_dir /data --file_type rhd --file_path rec.rhd --channels 0:64 --overwrite
    
    # Multi-file with channel mapping
    python 1_build_dataset.py --root_dir /data --multi_file --channel_map 8-8-L --overwrite
    
    # CSV with IMU features
    python 1_build_dataset.py --root_dir /data --file_type csv --multi_file --modality both --imu_features rich --overwrite
    
    # Paper-style preprocessing (120Hz highpass, RMS only, 250ms windows)
    python 1_build_dataset.py --root_dir /data --multi_file --paper_style --overwrite
    
    # Orientation remapping for rotated electrodes
    python 1_build_dataset.py --root_dir /data --multi_file --channel_map 8-8-L --orientation_remap mirror --overwrite
    
    # From config file
    python 1_build_dataset.py --config_file config.json --overwrite
"""

import os
import sys
import json
import argparse
import logging
from time import time
from typing import List, Optional

import numpy as np

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
    process_recording,
    save_dataset,
    infer_grid_dimensions,
    apply_grid_permutation,
    parse_orientation_from_filename,
)


def build_dataset(
    root_dir: str, 
    file_type: str = "rhd", 
    file_path: Optional[str] = None,
    file_names: Optional[List[str]] = None, 
    multi_file: bool = False,
    events_file: Optional[str] = None, 
    label: str = "", 
    save_path: Optional[str] = None,
    window_ms: int = 200, 
    step_ms: int = 50, 
    paper_style: bool = False,
    channels: Optional[List[int]] = None, 
    channel_map: Optional[str] = None,
    channel_map_file: str = "custom_channel_mappings.json",
    mapping_non_strict: bool = False, 
    orientation: str = "auto",
    orientation_remap: str = "none", 
    modality: str = "emg",
    imu_features: str = "rich", 
    imu_norm: str = "zscore",
    ignore_labels: Optional[List[str]] = None, 
    ignore_case: bool = False,
    keep_trial_label: bool = False, 
    merge_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None, 
    overwrite: bool = False,
    verbose: bool = False,
):
    """
    Build EMG gesture classification dataset.
    
    This function orchestrates the complete dataset building pipeline:
    - Discovers and loads data files (RHD/NPZ/CSV)
    - Applies channel selection and spatial transforms
    - Preprocesses EMG signals
    - Extracts features using sliding windows
    - Loads and filters labels from event files
    - Optionally integrates IMU features
    - Saves complete dataset with metadata
    
    See module docstring for usage examples.
    
    Args:
        root_dir: Root directory containing data files
        file_type: Type of input files ('rhd', 'npz', 'csv')
        file_path: Path to single file (single-file mode)
        file_names: List of specific filenames (multi-file mode)
        multi_file: Enable multi-file aggregation mode
        events_file: Explicit path to event file (single-file mode)
        label: Label prefix for output filename
        save_path: Explicit output path (overrides auto-naming)
        window_ms: Feature extraction window size in ms
        step_ms: Feature extraction step size in ms
        paper_style: Use paper preprocessing (120Hz HP, RMS only, 250ms windows)
        channels: Explicit channel indices to select
        channel_map: Named channel mapping from JSON file
        channel_map_file: Path to channel mapping JSON file
        mapping_non_strict: Allow missing channels in mapping
        orientation: Electrode orientation ('auto' or explicit)
        orientation_remap: Spatial transform ('none', 'mirror', 'rotate90')
        modality: Feature type ('emg', 'imu', 'both')
        imu_features: IMU feature mode ('mean' or 'rich')
        imu_norm: IMU normalization ('zscore' or 'robust')
        ignore_labels: List of labels to exclude
        ignore_case: Case-insensitive label filtering
        keep_trial_label: Keep trial numbers in labels (e.g., 'fist_3')
        merge_pattern: Pattern to filter file stems for merging
        exclude_pattern: Pattern to exclude file stems
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
    
    # Default ignore labels (common non-gesture labels)
    if ignore_labels is None:
        ignore_labels = ["Start", "End", "None", "Unknown"]  # Removed "Rest" - it's a valid gesture class
        ignore_case = True  # Enable case-insensitive matching for default labels
        logging.info(f"   Default ignore labels: {ignore_labels} (case-insensitive)")
    
    if paper_style:
        window_ms, step_ms = 250, 250
        logging.info("[PAPER MODE] 120Hz highpass, RMS-only, 250ms non-overlapping windows")
    
    # Determine output path
    save_path = save_path or os.path.join(
        root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz"
    )
    
    if os.path.exists(save_path) and not overwrite:
        logging.info(f"[OK] Dataset exists: {save_path}")
        logging.info("  Use --overwrite to regenerate")
        return
    
    logging.info(f"[INFO] Building dataset: {os.path.basename(save_path)}")
    logging.info(f"   Root: {root_dir}")
    logging.info(f"   Type: {file_type.upper()}, Mode: {'Multi-file' if multi_file else 'Single-file'}")
    logging.info(f"   Windows: {window_ms}ms × {step_ms}ms step, Modality: {modality}")
    
    # ===== SINGLE-FILE MODE =====
    if not multi_file:
        if not file_path:
            raise ValueError("Single-file mode requires --file_path")
        
        logging.info(f"[Loading] {os.path.basename(file_path)}")
        data = load_single_file(file_type, file_path, root_dir, verbose)
        
        # Select channels
        raw_names = data.get("channel_names", [f"CH{i}" for i in range(data["amplifier_data"].shape[0])])
        ch_indices, ch_names = select_channels(
            raw_names, channels, channel_map, channel_map_file, mapping_non_strict
        )
        
        logging.info(f"   Channels: {len(ch_indices)} selected")
        if channel_map:
            logging.info(f"   Mapping: {channel_map}")
        
        # Process recording
        X, y, meta = process_recording(
            data, file_path, root_dir, events_file, window_ms, step_ms,
            paper_style, ch_indices, modality, imu_features, imu_norm,
            ignore_labels, ignore_case, keep_trial_label
        )
        
        # Summarize
        class_names = sorted(set(y))
        logging.info(f"[OK] Extracted {X.shape[0]} windows, {len(class_names)} classes")
        logging.info(f"   Classes: {', '.join(class_names)}")
        
        save_dataset(save_path, X, y, meta, window_ms, step_ms, channel_map, channel_map_file, modality)
        
        elapsed = time() - start_time
        logging.info(f"[OK] Completed in {elapsed:.1f}s")
        return
    
    # ===== MULTI-FILE MODE =====
    logging.info("[Discovering files...]")
    file_groups = discover_and_group_files(root_dir, file_type, file_names, exclude_pattern, merge_pattern)
    logging.info(f"   Found {len(file_groups)} recording groups")
    
    if file_names:
        logging.info(f"   Filtered to: {', '.join(file_names)}")
    if exclude_pattern:
        logging.info(f"   Excluding pattern: {exclude_pattern}")
    if merge_pattern:
        logging.info(f"   Merge pattern: {merge_pattern}")
    
    # Initialize accumulators
    combined_X, combined_y = [], []
    ch_indices, ch_names = None, None
    fs_values = set()
    failed_files = []
    
    # Pre-load channel mapping for spatial transforms
    mapping_names = grid_dims = None
    if channel_map and orientation_remap != "none":
        with open(channel_map_file) as f:
            mappings = json.load(f)
        if channel_map not in mappings:
            raise KeyError(f"Mapping '{channel_map}' not found in {channel_map_file}")
        mapping_names = list(mappings[channel_map])
        grid_dims = infer_grid_dimensions(mapping_names)
        logging.info(f"   Spatial transform: {orientation_remap} on {grid_dims[0]}×{grid_dims[1]} grid")
    
    # Process each recording group
    logging.info(f"\n[Processing recordings...]")
    for i, (stem, files) in enumerate(file_groups.items(), 1):
        logging.info(f"[{i}/{len(file_groups)}] {stem}")
        logging.info(f"             Files: {len(files)}")
        
        try:
            # Load and merge files
            data = load_files_merged(file_type, files, root_dir, verbose)
            
            # First recording determines channel selection
            if ch_indices is None:
                raw_names = data.get("channel_names", [f"CH{j}" for j in range(data["amplifier_data"].shape[0])])
                ch_indices, ch_names = select_channels(
                    raw_names, channels, channel_map, channel_map_file, mapping_non_strict
                )
                
                logging.info(f"             Channels: {len(ch_indices)} selected")
                if channel_map:
                    logging.info(f"             Mapping: {channel_map}")
                
                # Apply spatial transform if needed
                if orientation_remap != "none" and mapping_names and grid_dims != (None, None):
                    orient = parse_orientation_from_filename(files[0]) if orientation == "auto" else orientation
                    if orient and orient != "neutral":
                        logging.info(f"             Orientation: {orient} → applying {orientation_remap}")
                        ch_indices = apply_grid_permutation(ch_indices, grid_dims[0], grid_dims[1], orientation_remap)
                        ch_names = [raw_names[i] for i in ch_indices]
            
            # Process recording
            if verbose:
                logging.debug(f"             Processing: file={files[0]}")
                logging.debug(f"             Root dir: {root_dir}")
            
            X, y, meta = process_recording(
                data, files[0], root_dir, None, window_ms, step_ms,
                paper_style, ch_indices, modality, imu_features, imu_norm,
                ignore_labels, ignore_case, keep_trial_label
            )
            
            if X.shape[0] > 0:
                combined_X.append(X)
                combined_y.append(y)
                fs_values.add(round(meta["fs"], 6))
                logging.info(f"             [OK] Added {X.shape[0]} windows")
            else:
                logging.warning(f"             [WARNING] No valid windows extracted")
                failed_files.append(stem)
        
        except Exception as e:
            logging.warning(f"             [FAIL] Failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            failed_files.append(stem)
    
    # Validate results
    if not combined_X:
        from intan.io import prompt_yes_no
        import re
        
        logging.error("\n" + "="*60)
        logging.error("[ERROR] NO VALID DATA EXTRACTED!")
        logging.error("="*60)
        logging.error(f"   All {len(failed_files)} recording(s) failed to process\n")
        
        # Analyze which event files are missing
        events_dir = os.path.join(root_dir, "events")
        available_events = []
        if os.path.exists(events_dir):
            available_events = [os.path.splitext(f)[0] for f in os.listdir(events_dir) 
                              if f.endswith(('.event', '.txt'))]
        
        # Try to match failed files to event files
        missing_events = []
        for stem in failed_files:
            # Try to find matching event file
            base_stem = re.sub(r'_\d{6}_\d{6}$', '', stem)
            base_stem = re.sub(r'_\d{6}$', '', base_stem)
            base_stem = re.sub(r'_\d+$', '', base_stem)
            
            expected_event = f"{base_stem}_emg"
            if expected_event not in available_events and base_stem not in available_events:
                missing_events.append(f"{base_stem}_emg.event or {base_stem}.event")
        
        if missing_events:
            logging.error("[MISSING EVENT FILES]:")
            for me in sorted(set(missing_events)):
                logging.error(f"   [X] {me}")
        
        if available_events:
            logging.error(f"\n[AVAILABLE EVENT FILES in events/]:")
            for ef in sorted(set(available_events)):
                logging.error(f"   [+] {ef}")
        
        logging.error(f"\n[SOLUTIONS]:")
        logging.error(f"   1. Create missing event files in: {events_dir}")
        logging.error(f"   2. Use --exclude_pattern to skip recordings without events")
        logging.error(f"   3. Check event file naming matches data files")
        logging.error("="*60 + "\n")
        
        retry = prompt_yes_no(
            "Dataset Building Failed",
            f"Failed to extract data from all {len(failed_files)} files.\n\n"
            f"Missing event files or naming mismatch.\n\n"
            f"See console for details about missing files.\n\n"
            f"Exit now?"
        )
        
        if retry is None or retry:  # True means "Exit now? Yes" or None means cancelled
            logging.info("Operation cancelled")
            sys.exit(1)
        
        raise ValueError(f"No valid data extracted - check event files")
    
    if failed_files:
        logging.warning(f"\n[WARNING] Failed to process {len(failed_files)}/{len(file_groups)} files:")
        for f in failed_files:
            logging.warning(f"   - {f}")
    
    # Concatenate all data
    logging.info(f"\n[Merging {len(combined_X)} recordings...]")
    X = np.concatenate(combined_X, axis=0)
    y = np.concatenate(combined_y, axis=0)
    
    # Check sampling rate consistency
    if len(fs_values) > 1:
        logging.warning(f"[WARNING] Multiple sampling rates detected: {sorted(fs_values)} Hz")
        logging.warning("  Using first value - results may be inconsistent")
    
    class_names = sorted(set(y))
    logging.info(f"[OK] Total: {X.shape[0]} windows, {X.shape[1]} features, {len(class_names)} classes")
    logging.info(f"   Classes: {', '.join(class_names)}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    logging.info(f"   Distribution:")
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        logging.info(f"      {cls:20s}: {cnt:5d} ({100*cnt/len(y):.1f}%)")
    
    metadata = {
        "fs": float(list(fs_values)[0]) if fs_values else 0.0,
        "selected_channels": ch_indices,
        "channel_names": ch_names,
    }
    
    save_dataset(save_path, X, y, metadata, window_ms, step_ms, channel_map, channel_map_file, modality)
    
    elapsed = time() - start_time
    logging.info(f"\n[OK] Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root_dir", help="Project root directory (prompts if not provided)")
    parser.add_argument("--file_type", default="rhd", choices=["rhd", "npz", "csv"])
    parser.add_argument("--file_path")
    parser.add_argument("--file_names", nargs="+")
    parser.add_argument("--multi_file", action="store_true")
    parser.add_argument("--events_file")
    parser.add_argument("--label", default="")
    parser.add_argument("--save_path")
    parser.add_argument("--window_ms", type=int, default=200)
    parser.add_argument("--step_ms", type=int, default=50)
    parser.add_argument("--paper_style", action="store_true")
    parser.add_argument("--channels", nargs="+")
    parser.add_argument("--channel_map")
    parser.add_argument("--channel_map_file", default="custom_channel_mappings.json")
    parser.add_argument("--mapping_non_strict", action="store_true")
    parser.add_argument("--orientation", default="auto")
    parser.add_argument("--orientation_remap", default="none", choices=["none", "mirror", "rotate90"])
    parser.add_argument("--modality", default="emg", choices=["emg", "imu", "both"])
    parser.add_argument("--imu_features", default="rich", choices=["mean", "rich"])
    parser.add_argument("--imu_norm", default="zscore", choices=["zscore", "robust"])
    parser.add_argument("--ignore_labels", nargs="+")
    parser.add_argument("--ignore_case", action="store_true")
    parser.add_argument("--keep_trial_label", action="store_true")
    parser.add_argument("--merge_pattern")
    parser.add_argument("--exclude_pattern")
    parser.add_argument("--config_file")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Load shared config first, then override with specific config_file if provided
    from pathlib import Path
    cfg = load_simple_config(Path(__file__).parent / ".gesture_config")
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
            # User cancelled directory selection
            print("Operation cancelled by user")
            sys.exit(0)
        
        args.root_dir = root_dir
        # Save to config immediately
        if was_prompted:
            cfg['root_dir'] = root_dir
            save_simple_config(cfg, Path(__file__).parent / ".gesture_config", "EMG Gesture Dataset Configuration")
            print("[*] Saved root_dir to .gesture_config")
    else:
        # Save CLI-provided root_dir to config if not already there
        if 'root_dir' not in cfg or cfg.get('root_dir') != args.root_dir:
            cfg['root_dir'] = args.root_dir
            save_simple_config(cfg, Path(__file__).parent / ".gesture_config", "EMG Gesture Dataset Configuration")
            print("[*] Saved root_dir to .gesture_config")
    
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
        save_simple_config(cfg, Path(__file__).parent / ".gesture_config", "EMG Gesture Dataset Configuration")
        print(f"[*] Saved multi_file={multi_file_response} to .gesture_config")
        
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
                    save_simple_config(cfg, Path(__file__).parent / ".gesture_config", "EMG Gesture Dataset Configuration")
                    print(f"[*] Saved exclude_pattern='{exclude_input}' to .gesture_config")
    
    # Prompt for file_path if single-file mode and not provided
    if not args.multi_file and not args.file_path and 'file_path' not in cfg:
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
            save_simple_config(cfg, Path(__file__).parent / ".gesture_config", "EMG Gesture Dataset Configuration")
            print("[*] Saved file_path to .gesture_config")
    
    channels_parsed = parse_channels_spec(args.channels) if args.channels else None
    
    params = {k: getattr(args, k) or cfg.get(k, v) for k, v in {
        "root_dir": None, "file_type": "rhd", "file_path": None, "file_names": None,
        "multi_file": False, "events_file": None, "label": "", "save_path": None,
        "window_ms": 200, "step_ms": 50, "paper_style": False,
        "channel_map": None, "channel_map_file": "custom_channel_mappings.json",
        "mapping_non_strict": False, "orientation": "auto", "orientation_remap": "none",
        "modality": "emg", "imu_features": "rich", "imu_norm": "zscore",
        "ignore_labels": None, "ignore_case": False, "keep_trial_label": False,
        "merge_pattern": None, "exclude_pattern": None, "overwrite": False, "verbose": False,
    }.items()}
    params["channels"] = channels_parsed if channels_parsed is not None else cfg.get("channels")
    
    try:
        build_dataset(**params)
    except Exception as e:
        logging.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
