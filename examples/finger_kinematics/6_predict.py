#!/usr/bin/env python3
"""
6_predict.py - Finger Kinematics Prediction

STEP 5: Predict continuous joint angles from EMG data using trained regression model.

Prerequisite: Run 4_train_model.py to create trained model.

Note: Automatically applies synchronization offsets from sync/ directory when
      comparing predictions to ground truth.

Modes:
    1. file    - Predict from single RHD file with optional ground truth comparison
    2. batch   - Predict from multiple files with aggregated metrics
    3. record  - Record from device and predict
    4. stream  - Real-time streaming prediction

Examples:
    # Interactive mode selection
    python 3_predict.py
    
    # File mode with ground truth
    python 3_predict.py file --file_path recording.rhd --angles_file angles.csv
    
    # Batch mode
    python 3_predict.py batch --rhd_glob "raw/**/*.rhd" --angles_dir joint_angles/
    
    # Real-time streaming
    python 3_predict.py stream --infer_hz 20
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from intan.io import (
    load_simple_config,
    save_simple_config,
    prompt_directory,
    prompt_file,
    get_or_prompt_value,
)
from intan.ml import ModelManager
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_joint_angles_csv(csv_path: str) -> tuple:
    """Load joint angle ground truth from CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    timestamps = df.iloc[:, 0].values
    angles = df.iloc[:, 1:].values
    return timestamps, angles


def predict_file(
    root_dir: str,
    file_path: str,
    angles_file: Optional[str] = None,
    label: str = "",
    calibration_file: Optional[str] = None,
    save_predictions: bool = True,
    plot_results: bool = False,
    smooth: bool = False,
    smooth_method: str = 'savgol',
    verbose: bool = False,
):
    """
    Predict joint angles from single EMG file.
    
    Args:
        root_dir: Project root directory
        file_path: Path to EMG recording
        angles_file: Optional ground truth CSV for comparison
        label: Model label prefix
        calibration_file: Optional calibration file for normalization
        save_predictions: Save predictions to file
        plot_results: Show matplotlib plot
        smooth: Apply temporal smoothing to predictions
        smooth_method: Smoothing method ('savgol', 'lowpass', 'gaussian', 'exponential')
        verbose: Verbose logging
    """
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl, force=True)
    
    from intan.io import load_single_file
    from intan.processing import EMGPreprocessor
    from intan.ml import EMGRegressor
    from scipy.interpolate import interp1d
    
    logging.info(f"[INFO] Predicting from: {os.path.basename(file_path)}")
    
    # Load model
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGRegressor, verbose=verbose)
    manager.load_model()
    
    # Load metadata
    import json
    metadata_path = os.path.join(manager.model_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        meta = json.load(f)
    
    # Load EMG data
    data = load_single_file("rhd", file_path, root_dir, verbose)
    emg_data = data['amplifier_data']
    fs = data['frequency_parameters']['amplifier_sample_rate']
    
    # Get metadata parameters
    ch_indices = meta.get('selected_channels', list(range(emg_data.shape[0])))
    window_ms = meta.get('window_ms', 200)
    step_ms = meta.get('step_ms', 50)
    n_joints = meta.get('n_joints', 0)
    
    emg_data = emg_data[ch_indices, :]
    logging.info(f"   EMG: {emg_data.shape[0]} channels, {emg_data.shape[1]} samples @ {fs:.0f} Hz")
    
    # Apply calibration if provided
    if calibration_file and os.path.exists(calibration_file):
        logging.info(f"   Applying calibration from: {os.path.basename(calibration_file)}")
        calib = np.load(calibration_file)
        baseline = calib['baseline']
        mvc_factors = calib['mvc_factors']
        
        # Ensure calibration matches number of channels
        if len(baseline) != emg_data.shape[0]:
            logging.warning(f"   [WARNING] Calibration channels ({len(baseline)}) != EMG channels ({emg_data.shape[0]})")
            logging.warning(f"   Skipping calibration")
        else:
            # Subtract baseline and normalize by MVC
            emg_data = emg_data - baseline[:, np.newaxis]
            emg_data = emg_data / mvc_factors[:, np.newaxis]
            logging.info(f"   Baseline subtracted, MVC normalized")
    
    # Preprocess and extract features
    logging.info(f"   Preprocessing EMG (bandpass 20-450 Hz, notch @ 60 Hz, envelope)...")
    preprocessor = EMGPreprocessor(fs=fs, verbose=verbose)
    emg_preprocessed = preprocessor.preprocess(emg_data)
    
    logging.info(f"   Extracting features...")
    X = preprocessor.extract_emg_features(
        emg_preprocessed,
        window_ms=window_ms,
        step_ms=step_ms,
        progress=verbose,
        tqdm_kwargs={"desc": "Extracting features", "unit": "win", "leave": False, "ascii": True}
    )
    window_size_samples = int(window_ms * fs / 1000)
    step_size_samples = int(step_ms * fs / 1000)
    window_starts = np.arange(0, emg_data.shape[1] - window_size_samples + 1, step_size_samples)
    window_timestamps = window_starts / fs
    
    logging.info(f"   Extracted {X.shape[0]} feature windows")
    
    # Predict
    logging.info(f"   Running prediction...")
    y_pred = manager.predict(X)
    
    # Clip predictions to physiological range [0, 180] degrees
    y_pred = np.clip(y_pred, 0, 180)
    
    # Apply temporal smoothing if requested
    if smooth:
        from intan.processing import smooth_predictions
        fs_pred = 1.0 / (meta['step_ms'] / 1000.0)  # Prediction sampling rate
        
        if smooth_method == 'lowpass':
            y_pred = smooth_predictions(y_pred, method='lowpass', fs=fs_pred, cutoff=5.0)
        elif smooth_method == 'savgol':
            y_pred = smooth_predictions(y_pred, method='savgol', window_length=11, polyorder=3)
        elif smooth_method == 'gaussian':
            y_pred = smooth_predictions(y_pred, method='gaussian', sigma=2.0)
        elif smooth_method == 'exponential':
            y_pred = smooth_predictions(y_pred, method='exponential', alpha=0.3)
        else:
            logging.warning(f"   Unknown smoothing method: {smooth_method}, skipping")
        
        # Re-clip after smoothing
        y_pred = np.clip(y_pred, 0, 180)
        logging.info(f"   Applied {smooth_method} smoothing")
    
    logging.info(f"   Predicted {y_pred.shape[0]} joint angle windows ({n_joints} joints)")
    
    # Optionally compare with ground truth
    if angles_file and os.path.exists(angles_file):
        logging.info(f"   Loading ground truth: {os.path.basename(angles_file)}")
        angle_timestamps, y_true_raw = load_joint_angles_csv(angles_file)
        
        # Apply sync offset if available
        sync_dir = os.path.join(root_dir, 'sync')
        if os.path.exists(sync_dir):
            import json
            import glob
            from pathlib import Path
            
            # Try multiple filename patterns
            stem_full = Path(file_path).stem  # With timestamp
            stem_short = stem_full.split('_251110_')[0]  # Without timestamp
            
            sync_candidates = [
                f'{stem_full}_sync.json',      # test_finger_sweep_251110_130429_sync.json
                f'{stem_short}_sync.json',     # test_finger_sweep_sync.json
                f'*{stem_short}*_sync.json'    # Wildcard match
            ]
            
            for pattern in sync_candidates:
                matches = glob.glob(os.path.join(sync_dir, pattern))
                if matches:
                    sync_path = matches[0]
                    with open(sync_path, 'r') as f:
                        sync_info = json.load(f)
                    offset_sec = sync_info['offset_sec']
                    # Subtract offset (positive means landmarks delayed)
                    angle_timestamps = angle_timestamps - offset_sec
                    logging.info(f"   Applied sync offset: {offset_sec:.3f}s (from {os.path.basename(sync_path)})")
                    break
        
        # Filter NaN values from ground truth
        nan_mask = np.isnan(y_true_raw).any(axis=1)
        if nan_mask.sum() > 0:
            logging.info(f"   Removing {nan_mask.sum()} ground truth samples with NaN ({100*nan_mask.sum()/len(nan_mask):.2f}%)")
            angle_timestamps = angle_timestamps[~nan_mask]
            y_true_raw = y_true_raw[~nan_mask]
        
        # Interpolate ground truth to match prediction timestamps
        y_true = np.zeros((len(window_timestamps), y_true_raw.shape[1]))
        for joint_idx in range(y_true_raw.shape[1]):
            f = interp1d(angle_timestamps, y_true_raw[:, joint_idx], 
                        kind='linear', fill_value='extrapolate', bounds_error=False)
            y_true[:, joint_idx] = f(window_timestamps)
        
        # Filter NaN from interpolated values (extrapolation might introduce NaN)
        nan_mask_interp = np.isnan(y_true).any(axis=1)
        if nan_mask_interp.sum() > 0:
            logging.info(f"   Removing {nan_mask_interp.sum()} interpolated samples with NaN ({100*nan_mask_interp.sum()/len(nan_mask_interp):.2f}%)")
            window_timestamps = window_timestamps[~nan_mask_interp]
            y_true = y_true[~nan_mask_interp]
            y_pred = y_pred[~nan_mask_interp]
        
        # Compute metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        logging.info(f"\n   Evaluation Metrics:")
        logging.info(f"      Overall MSE: {mse:.4f}")
        logging.info(f"      Overall MAE: {mae:.4f}")
        logging.info(f"      Overall R²:  {r2:.4f}")
        
        logging.info(f"\n   Per-Joint Metrics:")
        for joint_idx in range(n_joints):
            mse_j = mean_squared_error(y_true[:, joint_idx], y_pred[:, joint_idx])
            mae_j = mean_absolute_error(y_true[:, joint_idx], y_pred[:, joint_idx])
            r2_j = r2_score(y_true[:, joint_idx], y_pred[:, joint_idx])
            logging.info(f"      Joint {joint_idx}: MSE={mse_j:.4f}, MAE={mae_j:.4f}, R²={r2_j:.4f}")
        
        # Save evaluation
        if save_predictions:
            eval_path = file_path.replace('.rhd', '_evaluation.json')
            eval_data = {
                "overall": {"mse": float(mse), "mae": float(mae), "r2": float(r2)},
                "per_joint": [
                    {
                        "joint": i,
                        "mse": float(mean_squared_error(y_true[:, i], y_pred[:, i])),
                        "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
                        "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
                    }
                    for i in range(n_joints)
                ]
            }
            import json
            with open(eval_path, 'w') as f:
                json.dump(eval_data, f, indent=2)
            logging.info(f"\n   Saved evaluation to: {eval_path}")
        
        # Plot
        if plot_results:
            fig, axes = plt.subplots(n_joints, 1, figsize=(12, 3*n_joints), sharex=True)
            if n_joints == 1:
                axes = [axes]
            
            for joint_idx in range(n_joints):
                axes[joint_idx].plot(window_timestamps, y_true[:, joint_idx], 
                                    label='Ground Truth', alpha=0.7)
                axes[joint_idx].plot(window_timestamps, y_pred[:, joint_idx], 
                                    label='Predicted', alpha=0.7)
                axes[joint_idx].set_ylabel(f'Joint {joint_idx}')
                axes[joint_idx].legend()
                axes[joint_idx].grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Time (s)')
            plt.suptitle(f'Joint Angle Prediction: {os.path.basename(file_path)}')
            plt.tight_layout()
            plt.show()
    
    # Save predictions
    if save_predictions:
        pred_path = file_path.replace('.rhd', '_predictions.txt')
        with open(pred_path, 'w') as f:
            f.write(f"# Joint angle predictions\n")
            f.write(f"# timestamp_sec," + ",".join([f"joint_{i}" for i in range(n_joints)]) + "\n")
            for t, angles in zip(window_timestamps, y_pred):
                f.write(f"{t:.6f}," + ",".join([f"{a:.6f}" for a in angles]) + "\n")
        
        logging.info(f"\n   Saved predictions to: {pred_path}")


def predict_batch(
    root_dir: str,
    rhd_glob: str,
    angles_dir: str,
    label: str = "",
    verbose: bool = False,
):
    """
    Predict from multiple files and compute aggregated metrics.
    
    Args:
        root_dir: Project root directory
        rhd_glob: Glob pattern for RHD files
        angles_dir: Directory with ground truth CSVs
        label: Model label prefix
        verbose: Verbose logging
    """
    import glob
    
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl, force=True)
    
    logging.info(f"[INFO] Batch prediction")
    
    files = sorted(glob.glob(os.path.join(root_dir, rhd_glob)))
    logging.info(f"   Found {len(files)} files")
    
    all_mse, all_mae, all_r2 = [], [], []
    
    for i, file_path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(file_path))[0]
        logging.info(f"\n[{i}/{len(files)}] {stem}")
        
        # Find matching angles file
        angles_file = os.path.join(angles_dir, f"{stem}_angles.csv")
        if not os.path.exists(angles_file):
            angles_file = os.path.join(angles_dir, f"{stem}.csv")
        
        if not os.path.exists(angles_file):
            logging.warning(f"   [WARNING] No angles file found, skipping")
            continue
        
        try:
            # Predict (without plotting)
            predict_file(root_dir, file_path, angles_file, label, 
                        save_predictions=False, plot_results=False, verbose=False)
            
            # Load saved evaluation
            eval_path = file_path.replace('.rhd', '_evaluation.json')
            if os.path.exists(eval_path):
                import json
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
                all_mse.append(eval_data['overall']['mse'])
                all_mae.append(eval_data['overall']['mae'])
                all_r2.append(eval_data['overall']['r2'])
        
        except Exception as e:
            logging.warning(f"   [FAIL] {e}")
    
    # Aggregate metrics
    if all_mse:
        logging.info(f"\n[Aggregated Metrics across {len(all_mse)} files]")
        logging.info(f"   MSE: mean={np.mean(all_mse):.4f}, std={np.std(all_mse):.4f}")
        logging.info(f"   MAE: mean={np.mean(all_mae):.4f}, std={np.std(all_mae):.4f}")
        logging.info(f"   R²:  mean={np.mean(all_r2):.4f}, std={np.std(all_r2):.4f}")


def predict_from_device(
    root_dir: str,
    seconds: float = 10.0,
    angles_file: Optional[str] = None,
    label: str = "",
    verbose: bool = False,
):
    """
    Record from Intan device and predict joint angles.
    
    Args:
        root_dir: Project root directory
        seconds: Recording duration
        angles_file: Optional ground truth CSV
        label: Model label prefix
        verbose: Verbose logging
    """
    logging.info("[INFO] Device recording mode not yet implemented")
    logging.info("   Use 'file' mode with pre-recorded data instead")


def predict_realtime_stream(
    root_dir: str,
    label: str = "",
    infer_hz: float = 20.0,
    seconds_total: Optional[float] = None,
    verbose: bool = False,
):
    """
    Real-time streaming prediction from Intan device.
    
    Args:
        root_dir: Project root directory
        label: Model label prefix
        infer_hz: Prediction rate (Hz)
        seconds_total: Total duration (None = infinite)
        verbose: Verbose logging
    """
    logging.info("[INFO] Real-time streaming mode not yet implemented")
    logging.info("   Use 'file' mode with pre-recorded data instead")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Add global arguments that work regardless of mode
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    subparsers = parser.add_subparsers(dest="mode", help="Prediction mode")
    
    # File mode
    file_parser = subparsers.add_parser("file", help="Predict from single file")
    file_parser.add_argument("--root_dir", help="Project root directory")
    file_parser.add_argument("--file_path", help="Path to EMG file (prompts if not provided)")
    file_parser.add_argument("--angles_file", help="Ground truth joint angles CSV")
    file_parser.add_argument("--calibration_file", help="Calibration .npz file for normalization")
    file_parser.add_argument("--label", default="", help="Model label prefix")
    file_parser.add_argument("--no_save", action="store_true", help="Don't save predictions")
    file_parser.add_argument("--plot", action="store_true", help="Show plot")
    file_parser.add_argument("--smooth", action="store_true", help="Apply temporal smoothing")
    file_parser.add_argument("--smooth_method", default="savgol", 
                           choices=['savgol', 'lowpass', 'gaussian', 'exponential'],
                           help="Smoothing method (default: savgol)")
    file_parser.add_argument("--verbose", action="store_true")
    
    # Batch mode
    batch_parser = subparsers.add_parser("batch", help="Predict from multiple files")
    batch_parser.add_argument("--root_dir", help="Project root directory")
    batch_parser.add_argument("--rhd_glob", required=True, help="Glob pattern for RHD files")
    batch_parser.add_argument("--angles_dir", required=True, help="Directory with ground truth CSVs")
    batch_parser.add_argument("--label", default="", help="Model label prefix")
    batch_parser.add_argument("--verbose", action="store_true")
    
    # Record mode
    record_parser = subparsers.add_parser("record", help="Record from device")
    record_parser.add_argument("--root_dir", help="Project root directory")
    record_parser.add_argument("--seconds", type=float, default=10.0)
    record_parser.add_argument("--angles_file", help="Ground truth CSV")
    record_parser.add_argument("--calibration_file", help="Calibration .npz file for normalization")
    record_parser.add_argument("--label", default="", help="Model label prefix")
    record_parser.add_argument("--verbose", action="store_true")
    
    # Stream mode
    stream_parser = subparsers.add_parser("stream", help="Real-time streaming")
    stream_parser.add_argument("--root_dir", help="Project root directory")
    stream_parser.add_argument("--label", default="", help="Model label prefix")
    stream_parser.add_argument("--infer_hz", type=float, default=20.0)
    stream_parser.add_argument("--seconds_total", type=float, help="Duration (None=infinite)")
    stream_parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # If no mode specified, prompt for interactive mode selection
    if not args.mode:
        print("\n=== Finger Kinematics Prediction ===")
        print("\nSelect prediction mode:")
        print("  1. file   - Predict from single EMG file")
        print("  2. batch  - Predict from multiple files")
        print("  3. record - Record from device and predict")
        print("  4. stream - Real-time streaming prediction")
        print("\nOr run with --help to see all options")
        
        choice = input("\nEnter mode (1-4, or 'q' to quit): ").strip()
        
        mode_map = {'1': 'file', '2': 'batch', '3': 'record', '4': 'stream'}
        if choice.lower() == 'q':
            sys.exit(0)
        elif choice not in mode_map:
            print("Invalid choice. Exiting.")
            sys.exit(1)
        
        # Set default args for interactive mode
        class InteractiveArgs:
            mode = mode_map[choice]
            root_dir = None
            label = ""
            verbose = False
            no_save = False
            plot = True  # Enable plotting by default in interactive mode
            file_path = None
            angles_file = None
        
        args = InteractiveArgs()
        
        if args.mode == 'batch' or args.mode == 'record' or args.mode == 'stream':
            print(f"\nInteractive mode for '{args.mode}' not yet implemented.")
            print(f"Please run: python 6_predict.py {args.mode} --help")
            sys.exit(0)
    
    # Load config
    cfg = load_simple_config(Path(__file__).parent / ".kinematics_config")
    
    # Prompt for root_dir if needed
    if not hasattr(args, 'root_dir') or not args.root_dir:
        root_dir, _ = get_or_prompt_value(
            arg_value=None,
            config=cfg,
            key='root_dir',
            prompt_func=prompt_directory,
            title="Select Project Root Directory",
            initial_dir=cfg.get('root_dir')
        )
        args.root_dir = root_dir
    
    try:
        if args.mode == "file":
            # Prompt for file_path if not provided
            file_path = args.file_path
            if not file_path:
                raw_dir = os.path.join(args.root_dir, 'raw')
                file_path = prompt_file(
                    title="Select EMG Recording File",
                    initial_dir=raw_dir if os.path.isdir(raw_dir) else args.root_dir,
                    filetypes=[('RHD Files', '*.rhd'), ('All Files', '*.*')]
                )
                if not file_path:
                    logging.error("No file selected. Exiting.")
                    sys.exit(1)
            
            # Optionally prompt for angles_file
            angles_file = args.angles_file
            if not angles_file and args.plot:
                # Only prompt if user wants to plot (needs ground truth)
                angles_dir = os.path.join(args.root_dir, 'media', 'landmarks')
                if not os.path.isdir(angles_dir):
                    angles_dir = os.path.join(args.root_dir, 'landmarks')
                if not os.path.isdir(angles_dir):
                    angles_dir = args.root_dir
                
                print("\nOptional: Select ground truth angles CSV for comparison (Cancel to skip)")
                angles_file = prompt_file(
                    title="Select Ground Truth Angles (Optional)",
                    initial_dir=angles_dir,
                    filetypes=[('CSV Files', '*.csv'), ('All Files', '*.*')]
                )
            
            predict_file(
                root_dir=args.root_dir,
                file_path=file_path,
                angles_file=angles_file,
                label=args.label,
                calibration_file=args.calibration_file if hasattr(args, 'calibration_file') else None,
                save_predictions=not args.no_save,
                plot_results=args.plot,
                smooth=args.smooth if hasattr(args, 'smooth') else False,
                smooth_method=args.smooth_method if hasattr(args, 'smooth_method') else 'savgol',
                verbose=args.verbose,
            )
        elif args.mode == "batch":
            predict_batch(
                root_dir=args.root_dir,
                rhd_glob=args.rhd_glob,
                angles_dir=args.angles_dir,
                label=args.label,
                verbose=args.verbose,
            )
        elif args.mode == "record":
            predict_from_device(
                root_dir=args.root_dir,
                seconds=args.seconds,
                angles_file=args.angles_file,
                label=args.label,
                verbose=args.verbose,
            )
        elif args.mode == "stream":
            predict_realtime_stream(
                root_dir=args.root_dir,
                label=args.label,
                infer_hz=args.infer_hz,
                seconds_total=args.seconds_total,
                verbose=args.verbose,
            )
    
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
