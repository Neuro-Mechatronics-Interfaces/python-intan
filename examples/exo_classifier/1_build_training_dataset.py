#!/usr/bin/env python3
"""
1_build_training_dataset.py

Build a training dataset following the published paper's approach:
- 120 Hz high-pass filter (4th-order Butterworth)
- RMS-only features
- 250 ms windows
- Non-overlapping windows (step = window)
- No envelope extraction

This script mimics the exact signal processing pipeline from the Journal of Neural Engineering paper.

Supports both .rhd (default) and .csv files.

Examples
--------
# RHD files (default)
python 1_build_training_dataset.py \
    --root_dir /path/to/data \
    --label paper_replication \
    --overwrite --verbose

# CSV files
python 1_build_training_dataset.py \
    --root_dir /path/to/data \
    --file_type csv \
    --label paper_style_csv \
    --overwrite

# Specific files only
python 1_build_training_dataset.py \
    --root_dir /path/to/data \
    --file_names recording1.rhd recording2.rhd \
    --label subset_data
"""

from __future__ import annotations
import os, json, glob, argparse, logging, re
from typing import List, Optional
import numpy as np

from intan.io import load_csv_files, load_rhd_file, labels_from_events
from intan.processing import EMGPreprocessor

# Import quality control helpers
try:
    from channel_quality_helpers import get_good_channels_from_recording

    QC_AVAILABLE = True
except ImportError:
    logging.warning("channel_quality_helpers not available - quality control disabled")
    QC_AVAILABLE = False


def _stem(path):
    """Extract stem from path, removing date/time suffixes."""
    if hasattr(path, "item"):
        try:
            path = path.item()
        except Exception:
            pass
    stem = os.path.splitext(os.path.basename(str(path)))[0]

    # Remove date-time pattern: _YYMMDD_HHMMSS or _YYMMDD
    # Matches: _250901_163304 or _250901
    stem = re.sub(r'_\d{6}_\d{6}$', '', stem)  # Remove _YYMMDD_HHMMSS
    stem = re.sub(r'_\d{6}$', '', stem)  # Remove _YYMMDD

    return stem


def _find_event_for(root_dir: str, rec_path: str) -> str | None:
    """Find corresponding event file for a recording using recursive search."""
    stem = _stem(rec_path)

    # Search recursively for all .event files
    all_event_files = sorted(glob.glob(
        os.path.join(root_dir, "**", "*.event"),
        recursive=True
    ))

    if not all_event_files:
        return None

    # Try to find exact matches first
    for ev_path in all_event_files:
        ev_stem = _stem(ev_path)

        # Check for exact stem match
        if ev_stem == stem:
            return ev_path

        # Check for stem_emg pattern
        if ev_stem == f"{stem}_emg":
            return ev_path

    # If only one event file exists, use it
    if len(all_event_files) == 1:
        return all_event_files[0]

    return None


def build_paper_style_dataset(
        root_dir: str,
        label: str = "paper_style",
        save_path: str | None = None,
        file_names: List[str] | None = None,
        file_type: str = "rhd",
        exclude_pattern: str | None = None,
        merge_pattern: str | None = None,
        channels: List[int] | None = None,
        enable_qc: bool = False,
        overwrite: bool = False,
        verbose: bool = False,
):
    """
    Build training dataset following the paper's exact approach:

    Signal Processing Pipeline (matches paper):
    1. High-pass filter: 120 Hz (4th-order Butterworth)
    2. RMS feature extraction over 250 ms windows
    3. Non-overlapping windows (step = 250 ms)
    4. No rectification or envelope extraction (implicit in RMS)

    Parameters
    ----------
    root_dir : str
        Root directory containing recordings and 'events/' folder
    label : str
        Label prefix for saved files
    save_path : str, optional
        Custom save path. If None, saves to <root_dir>/<label>_training_dataset.npz
    file_names : list of str, optional
        Specific file basenames to include. If None, uses all files of file_type found.
    file_type : str
        Type of files to load: 'rhd' (default) or 'csv'
    exclude_pattern : str, optional
        Pattern to exclude files (e.g., "exo_grasp" will exclude files exactly matching "exo_grasp*")
    merge_pattern : str, optional
        Pattern to group and merge multi-part files (e.g., "exo_gestures" merges exo_gestures_*.rhd)
    channels : list of int, optional
        Channel indices to use. If None, uses all channels.
    enable_qc : bool
        Enable channel quality control to automatically remove bad channels
    overwrite : bool
        Whether to overwrite existing dataset
    verbose : bool
        Enable verbose logging
    """

    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)

    # Paper-specific parameters (hardcoded to match publication)
    WINDOW_MS = 250  # N=1000 samples at 4000 Hz = 250 ms
    STEP_MS = 250  # Non-overlapping windows
    HIGHPASS_HZ = 120  # 4th-order Butterworth, removes low-freq artifacts + 60 Hz harmonics
    SAMPLING_RATE = 4000  # Expected sampling rate (will verify from data)

    logging.info("=" * 60)
    logging.info("PAPER-STYLE DATASET BUILDER")
    logging.info("=" * 60)
    logging.info(f"Replicating signal processing from Journal of Neural Engineering paper:")
    logging.info(f"  - High-pass filter: {HIGHPASS_HZ} Hz (4th-order Butterworth)")
    logging.info(f"  - Feature: RMS only")
    logging.info(f"  - Window: {WINDOW_MS} ms (non-overlapping)")
    logging.info(f"  - No envelope extraction")
    logging.info(f"  - Multi-part files: AUTO-MERGE (by base name)")
    if enable_qc:
        if QC_AVAILABLE:
            logging.info(f"  - Channel QC: ENABLED (removes bad channels)")
        else:
            logging.warning(f"  - Channel QC: REQUESTED but unavailable")
    if merge_pattern:
        logging.info(f"  - Group filter: '{merge_pattern}' only")
    logging.info("=" * 60)

    # Determine save path
    if save_path is None:
        save_path = os.path.join(root_dir, f"{label}_training_dataset.npz")

    if os.path.exists(save_path) and not overwrite:
        raise FileExistsError(f"Dataset exists at {save_path}. Use --overwrite to replace.")

    # Find recording files based on file_type (RECURSIVE SEARCH)
    file_ext = f".{file_type.lower()}"

    logging.info(f"Searching recursively for {file_ext} files in {root_dir}...")
    all_files = sorted(glob.glob(
        os.path.join(root_dir, "**", f"*{file_ext}"),
        recursive=True
    ))

    if not all_files:
        raise FileNotFoundError(
            f"No {file_ext} files found recursively in {root_dir} or its subdirectories"
        )

    logging.info(f"Found {len(all_files)} {file_ext} file(s):")
    for f in all_files[:10]:  # Show first 10
        rel_path = os.path.relpath(f, root_dir)
        logging.info(f"  {rel_path}")
    if len(all_files) > 10:
        logging.info(f"  ... and {len(all_files) - 10} more")

    # Filter by file_names if provided
    if file_names:
        file_names_set = set(n.strip() for n in file_names)
        filtered_files = []
        for p in all_files:
            basename = os.path.basename(p)
            stem = os.path.splitext(basename)[0]
            if basename in file_names_set or stem in file_names_set:
                filtered_files.append(p)
        all_files = filtered_files

    # Exclude files matching exclude_pattern
    if exclude_pattern:
        logging.info(f"Excluding files matching pattern: {exclude_pattern}")
        exclude_base = exclude_pattern.rstrip('*').rstrip('_-')

        excluded_files = []
        kept_files = []
        for p in all_files:
            basename = os.path.basename(p)
            stem = os.path.splitext(basename)[0]

            # Remove timestamps for comparison
            stem_clean = re.sub(r'_\d{6}_\d{6}$', '', stem)
            stem_clean = re.sub(r'_\d{6}$', '', stem_clean)

            # Check if this file should be excluded (exact match)
            if stem_clean == exclude_base:
                excluded_files.append(basename)
            else:
                kept_files.append(p)

        if excluded_files:
            logging.info(f"  Excluded {len(excluded_files)} file(s):")
            for ef in excluded_files[:5]:
                logging.info(f"    - {ef}")
            if len(excluded_files) > 5:
                logging.info(f"    ... and {len(excluded_files) - 5} more")

        all_files = kept_files

    if not all_files:
        raise FileNotFoundError(f"No {file_ext} files found matching criteria")

    # ========================================================================
    # GROUP FILES BY BASE NAME (always merge multi-part recordings)
    # ========================================================================
    logging.info(f"\nGrouping files by base name (auto-merging multi-part recordings)...")

    file_groups = {}
    for p in all_files:
        basename = os.path.basename(p)
        stem = os.path.splitext(basename)[0]

        # Remove timestamps to get base name: exo_gestures_251203_195153 → exo_gestures
        base_clean = re.sub(r'_\d{6}_\d{6}$', '', stem)  # Remove _YYMMDD_HHMMSS
        base_clean = re.sub(r'_\d{6}$', '', base_clean)  # Remove _YYMMDD

        if base_clean not in file_groups:
            file_groups[base_clean] = []
        file_groups[base_clean].append(p)

    # Sort files within each group by filename (chronological order)
    for group_name in file_groups:
        file_groups[group_name] = sorted(file_groups[group_name])

    # Apply merge_pattern filter if specified (only process matching groups)
    if merge_pattern:
        merge_base = merge_pattern.rstrip('*').rstrip('_-')
        filtered_groups = {k: v for k, v in file_groups.items() if k == merge_base}

        if not filtered_groups:
            logging.warning(f"No groups match merge_pattern '{merge_pattern}' - processing all groups")
        else:
            logging.info(f"Merge pattern filter: keeping only '{merge_base}' group")
            file_groups = filtered_groups

    # Log what we found
    for group_name, file_list in file_groups.items():
        if len(file_list) > 1:
            logging.info(f"  {group_name}: {len(file_list)} parts to merge")
        else:
            logging.info(f"  {group_name}: 1 file")
    # ========================================================================

    logging.info(f"\nProcessing {len(file_groups)} recording(s)")

    # Process each recording group
    combined_X = []
    combined_y = []
    fs_values = set()
    channel_names_list = []

    # For QC: store intermediate data for second pass
    intermediate_data = []  # Will store (emg_filtered, y, file_info) if QC enabled
    all_good_channels_per_file = []  # Track good channels per file

    for i, (group_name, file_paths) in enumerate(file_groups.items(), 1):
        logging.info(f"\n[{i}/{len(file_groups)}] Processing: {group_name}")
        if len(file_paths) > 1:
            logging.info(f"  Merging {len(file_paths)} parts:")
            for fp in file_paths:
                logging.info(f"    - {os.path.basename(fp)}")

        # Find corresponding event file (use first file's stem)
        ev_path = _find_event_for(root_dir, file_paths[0])
        if ev_path is None:
            logging.warning(f"  No event file found for {group_name} - skipping")
            continue
        logging.info(f"  Event file: {os.path.basename(ev_path)}")

        # Load data based on file type
        if file_type.lower() == "rhd":
            try:
                # Support merging multiple RHD files
                if len(file_paths) > 1:
                    logging.info(f"  Loading and merging {len(file_paths)} RHD files...")
                    data = load_rhd_file(
                        filepath=file_paths,
                        merge_files=True,
                        sort_files=True,
                        rebuild_time=True,
                        verbose=verbose
                    )
                else:
                    data = load_rhd_file(file_paths[0], verbose=verbose)
                emg = data["amplifier_data"]  # Shape: (channels, samples)
                emg_fs = data["frequency_parameters"]["amplifier_sample_rate"]
                emg_t = data.get("t_amplifier", None)

                # Check if file has data
                if emg is None or emg.size == 0:
                    logging.warning(f"  File contains no data - skipping")
                    continue

            except (TypeError, KeyError, ValueError) as e:
                logging.warning(f"  Error loading file: {e} - skipping")
                continue

        else:  # csv
            try:
                # CSV doesn't support merging yet - use first file
                if len(file_paths) > 1:
                    logging.warning(f"  CSV merging not supported - using first file only")

                data = load_csv_files(file_paths[0])
                emg = data["amplifier_data"]  # Shape: (channels, samples)
                emg_fs = data.get("frequency_parameters", {}).get("amplifier_sample_rate", SAMPLING_RATE)
                emg_t = data.get("t_amplifier", None)

                if emg is None or emg.size == 0:
                    logging.warning(f"  File contains no data - skipping")
                    continue

            except (TypeError, KeyError, ValueError) as e:
                logging.warning(f"  Error loading file: {e} - skipping")
                continue

        fs_values.add(emg_fs)

        if abs(emg_fs - SAMPLING_RATE) > 1.0:
            logging.warning(f"  Sample rate {emg_fs} Hz differs from expected {SAMPLING_RATE} Hz")

        # Store channel names from first recording
        if not channel_names_list:
            if "amplifier_channels" in data:
                channel_names_list = [ch["custom_channel_name"] for ch in data["amplifier_channels"]]
            elif "channel_names" in data:
                channel_names_list = list(data["channel_names"])
            else:
                channel_names_list = [f"CH{i}" for i in range(emg.shape[0])]

        # ====================================================================
        # CHANNEL QUALITY CONTROL (optional)
        # ====================================================================
        if enable_qc and QC_AVAILABLE:
            logging.info(f"  Running channel quality control...")
            good_channels_qc, bad_channels_qc = get_good_channels_from_recording(
                emg, emg_fs, window_sec=2.0, verbose=verbose
            )

            all_good_channels_per_file.append(set(good_channels_qc))
            logging.info(
                f"  QC: {len(good_channels_qc)} good channels in this file (removed {len(bad_channels_qc)} bad)")

            # Apply QC: use only good channels for THIS file
            if channels is not None:
                # User specified channels - intersect with good channels
                selected_channels = [ch for ch in channels if ch in good_channels_qc]
            else:
                # Use only good channels
                selected_channels = good_channels_qc
        else:
            # No QC
            if channels is not None:
                selected_channels = channels
            else:
                selected_channels = list(range(emg.shape[0]))
        # ====================================================================

        # Channel selection
        emg = emg[selected_channels, :]
        if channel_names_list:
            selected_channel_names = [channel_names_list[i] for i in selected_channels]
        else:
            selected_channel_names = [f"CH{i}" for i in selected_channels]

        logging.info(f"  Using {len(selected_channels)} channels")

        # ====================================================================
        # PAPER-SPECIFIC PREPROCESSING
        # ====================================================================
        # Initialize preprocessor with paper's exact parameters:
        # - High-pass at 120 Hz (removes movement artifacts and 60 Hz + harmonics)
        # - No envelope (envelope_cutoff=None)
        # - Feature: RMS only (specified in extract_emg_features)
        pre = EMGPreprocessor(
            fs=emg_fs,
            band=(HIGHPASS_HZ, emg_fs / 2.1),  # 120 Hz high-pass, near-Nyquist high cutoff
            notch_freqs=(),  # No explicit notch (covered by 120 Hz HP in paper)
            envelope_cutoff=None,  # No envelope extraction
            verbose=verbose
        )

        # Preprocess: applies the 120 Hz high-pass filter
        logging.info(f"  Applying {HIGHPASS_HZ} Hz high-pass filter (4th-order Butterworth)")
        emg_filtered = pre.preprocess(emg, rectify=False)  # No rectification (implicit in RMS)

        # Extract RMS features with paper's window parameters
        logging.info(f"  Extracting RMS features (window={WINDOW_MS}ms, step={STEP_MS}ms)")
        X_emg = pre.extract_emg_features(
            emg_filtered,
            window_ms=WINDOW_MS,
            step_ms=STEP_MS,
            feature_fns=['root_mean_square'],  # RMS ONLY (as in paper)
            progress=verbose
        )
        # ====================================================================

        # Calculate window start times for label alignment
        start_index = 0
        step_samples = int(round(STEP_MS / 1000.0 * emg_fs))
        window_starts = np.arange(X_emg.shape[0], dtype=int) * step_samples + start_index

        # Align labels from event file
        y = labels_from_events(ev_path, window_starts)

        if verbose:
            logging.debug(f"  Raw labels from event file (first 10): {y[:10]}")
            logging.debug(f"  Label types: {[type(l).__name__ for l in y[:5]]}")

        # Strip whitespace from labels (event files often have trailing spaces)
        y = np.array([label.strip() if isinstance(label, str) else label for label in y])

        if verbose:
            unique_before = np.unique(y)
            logging.debug(f"  Unique labels before filtering ({len(unique_before)}): {list(unique_before)}")
            logging.debug(f"  Label repr: {[repr(l) for l in unique_before[:5]]}")

        # Filter out Unknown/Start/None labels
        mask = ~np.isin(y, ["Unknown", "Start", "None", ""])
        X_emg = X_emg[mask]
        y = y[mask]

        if verbose:
            unique_after, counts = np.unique(y, return_counts=True)
            logging.debug(f"  Unique labels after filtering ({len(unique_after)}): {dict(zip(unique_after, counts))}")

        if X_emg.shape[0] == 0:
            logging.warning(f"  No labeled windows after filtering - skipping")
            continue

        logging.info(f"  Extracted {X_emg.shape[0]} labeled windows, {X_emg.shape[1]} features")

        # Store results
        if enable_qc and QC_AVAILABLE:
            # Store intermediate data for second pass with consistent channels
            intermediate_data.append({
                'emg_filtered': emg_filtered,
                'y': y,
                'selected_channels': selected_channels,  # The channels we used (subset)
                'file_paths': file_paths  # Store all file paths for this group
            })
        else:
            # No QC - add features directly
            combined_X.append(X_emg)
            combined_y.append(y)

    if not combined_X and not intermediate_data:
        logging.error("No usable labeled windows across any recordings!")
        return

    # ========================================================================
    # SECOND PASS: Apply consistent channel set if QC was used
    # ========================================================================
    if enable_qc and QC_AVAILABLE and all_good_channels_per_file:
        logging.info(f"\n" + "=" * 60)
        logging.info("APPLYING CONSISTENT CHANNEL SET (QC Second Pass)")
        logging.info("=" * 60)

        # Find intersection of good channels across ALL files
        consistent_channels = set.intersection(*all_good_channels_per_file)
        consistent_channels = sorted(list(consistent_channels))

        logging.info(f"Channels good in ALL files: {len(consistent_channels)}/{emg.shape[0]}")

        if len(consistent_channels) < 10:
            logging.error(f"Too few consistent good channels ({len(consistent_channels)}) - consider disabling QC")
            return

        # Reprocess each file with consistent channel set
        logging.info(f"Reprocessing {len(intermediate_data)} files with consistent channels...")

        for idx, data_dict in enumerate(intermediate_data, 1):
            emg_filtered = data_dict['emg_filtered']  # Already filtered, shape (selected_channels, samples)
            y = data_dict['y']
            file_channels = data_dict['selected_channels']  # Which channels were used for this file

            # Map consistent_channels (global indices) to file_channels (local indices)
            # Find which of the consistent channels are in this file's selected channels
            local_indices = []
            for global_ch in consistent_channels:
                if global_ch in file_channels:
                    local_idx = file_channels.index(global_ch)
                    local_indices.append(local_idx)

            if len(local_indices) != len(consistent_channels):
                logging.warning(
                    f"  [{idx}] Only {len(local_indices)}/{len(consistent_channels)} consistent channels found")
                continue

            # Select only consistent good channels using local indices
            emg_filtered_qc = emg_filtered[local_indices, :]

            # Re-extract features with consistent channels
            X_emg = pre.extract_emg_features(
                emg_filtered_qc,
                window_ms=WINDOW_MS,
                step_ms=STEP_MS,
                feature_fns=['root_mean_square'],
                progress=False
            )

            # Apply same label filtering
            start_index = 0
            step_samples = int(round(STEP_MS / 1000.0 * emg_fs))
            window_starts = np.arange(X_emg.shape[0], dtype=int) * step_samples + start_index

            # Use stored labels (already filtered)
            if len(y) != X_emg.shape[0]:
                # Need to re-align
                ev_path = _find_event_for(root_dir, data_dict['file_paths'][0])
                y_new = labels_from_events(ev_path, window_starts)

                # Strip whitespace from labels
                y_new = np.array([label.strip() if isinstance(label, str) else label for label in y_new])

                # Filter out Unknown/Start/None
                mask = ~np.isin(y_new, ["Unknown", "Start", "None", ""])
                X_emg = X_emg[mask]
                y_new = y_new[mask]
            else:
                y_new = y

            logging.info(
                f"  [{idx}/{len(intermediate_data)}] {len(consistent_channels)} channels → {X_emg.shape[0]} windows")

            combined_X.append(X_emg)
            combined_y.append(y_new)

        # Update metadata
        selected_channels = consistent_channels
        if channel_names_list:
            selected_channel_names = [channel_names_list[i] for i in selected_channels]
        else:
            selected_channel_names = [f"CH{i}" for i in selected_channels]

        logging.info(f"✓ Using {len(selected_channels)} channels consistently across all files")
        logging.info("=" * 60)
    # ========================================================================

    # Concatenate all recordings
    X = np.concatenate(combined_X, axis=0)
    y = np.concatenate(combined_y, axis=0)

    # Verify feature dimensions
    expected_features = len(selected_channels)  # RMS per channel = M features
    actual_features = X.shape[1]
    if actual_features != expected_features:
        logging.warning(f"Feature dimension mismatch: got {actual_features}, expected {expected_features}")

    logging.info(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")

    # Build feature spec metadata (RMS only, channel-major)
    feature_spec = {
        "per_channel": True,
        "order": ["root_mean_square"],  # Only RMS
        "dims_per_feature": {"root_mean_square": 1},
        "layout": "channel_major",
        "channels": "training_order",
        "n_channels": len(selected_channels),
        "n_features_per_channel": 1,  # Only RMS
        "note": "Paper-style: RMS-only features with 120 Hz HP filter"
    }
    feature_spec_json = json.dumps(feature_spec)

    # Label encoding
    class_names_sorted = sorted(set(y.tolist()))
    label_to_id = {c: i for i, c in enumerate(class_names_sorted)}

    logging.info(f"Classes detected: {class_names_sorted}")

    # Save dataset with metadata
    np.savez(
        save_path,
        X=X,
        y=y,
        emg_fs=float(list(fs_values)[0]) if len(fs_values) == 1 else SAMPLING_RATE,
        class_names=np.array(class_names_sorted, dtype=object),
        label_to_id_json=np.array(json.dumps(label_to_id), dtype=object),
        window_ms=WINDOW_MS,
        step_ms=STEP_MS,
        selected_channels=np.array(selected_channels, dtype=int),
        channel_names=np.array(selected_channel_names, dtype=object),
        feature_spec=feature_spec_json,
        # Paper-specific metadata
        processing_note=np.array("Paper-style: 120 Hz HP filter, RMS-only, 250ms non-overlapping windows",
                                 dtype=object),
        highpass_hz=np.array(HIGHPASS_HZ, dtype=np.float32),
        filter_order=np.array(4, dtype=np.int32),
    )

    logging.info(f"\n{'=' * 60}")
    logging.info(f"SUCCESS! Dataset saved to: {save_path}")
    logging.info(f"Total samples: {X.shape[0]}")
    logging.info(f"Feature dimensions: {X.shape[1]} (RMS per channel)")
    logging.info(f"Classes: {len(class_names_sorted)}")
    logging.info(f"{'=' * 60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build training dataset following the published paper's approach"
    )
    p.add_argument("--root_dir", type=str, required=True,
                   help="Root directory containing recordings and 'events/'")
    p.add_argument("--label", type=str, default="paper_style",
                   help="Label prefix for saved dataset")
    p.add_argument("--save_path", type=str, default=None,
                   help="Custom save path (default: <root_dir>/<label>_training_dataset.npz)")
    p.add_argument("--file_names", nargs="+", default=None,
                   help="Specific file basenames to include (optional)")
    p.add_argument("--file_type", type=str, default="rhd", choices=["rhd", "csv"],
                   help="Type of recording files to load (default: rhd)")
    p.add_argument("--exclude_pattern", type=str, default=None,
                   help="Pattern to exclude files (e.g., 'exo_grasp' excludes files matching exo_grasp exactly)")
    p.add_argument("--merge_pattern", type=str, default=None,
                   help="Pattern to group and merge multi-part files (e.g., 'exo_gestures' merges exo_gestures_*.rhd)")
    p.add_argument("--channels", nargs="+", type=int, default=None,
                   help="Channel indices to use (default: all channels)")
    p.add_argument("--enable_qc", action="store_true",
                   help="Enable channel quality control (removes noisy/bad channels)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing dataset")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose logging")

    args = p.parse_args()

    build_paper_style_dataset(
        root_dir=args.root_dir,
        label=args.label,
        save_path=args.save_path,
        file_names=args.file_names,
        file_type=args.file_type,
        exclude_pattern=args.exclude_pattern,
        merge_pattern=args.merge_pattern,
        channels=args.channels,
        enable_qc=args.enable_qc,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
