#!/usr/bin/env python3
"""
intan.ml._prediction_modes

Prediction pipeline functions for EMG gesture classification.
Provides standardized interfaces for file, batch, record, and stream prediction modes.
"""

import os
import logging
import numpy as np
from typing import Optional


def predict_file(
    root_dir: str,
    file_path: str,
    label: str = "",
    window_ms: Optional[int] = None,
    step_ms: Optional[int] = None,
    events_file: Optional[str] = None,
    save_predictions: bool = True,
    verbose: bool = False
):
    """
    Offline prediction from single RHD file with optional event comparison.
    
    Args:
        root_dir: Directory containing trained model/metadata
        file_path: Path to .rhd file to evaluate
        label: Model label/tag (e.g., '128ch')
        window_ms: Override window size (else from metadata)
        step_ms: Override step size (else from metadata)
        events_file: Optional path to events file for evaluation
        save_predictions: Save predictions and evaluation to files
        verbose: Enable verbose logging
    
    Returns:
        dict with keys: y_pred, window_starts_sec, starts_samples, eval_results (if events_file)
    """
    from intan.io import (
        load_rhd_file,
        lock_params_to_meta,
        load_metadata_json,
        select_training_channels_by_name,
    )
    from intan.processing import EMGPreprocessor
    from intan.ml import ModelManager, EMGClassifier, evaluate_against_events

    # Configure logging
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # Load metadata
    meta = load_metadata_json(root_dir, label=label)
    window_ms, step_ms, _, env_cut = lock_params_to_meta(
        meta.get('data', {}), window_ms, step_ms, None
    )
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # Load raw EMG from .rhd
    data = load_rhd_file(file_path, verbose=verbose)
    emg_fs = data['frequency_parameters']['amplifier_sample_rate']
    emg = data["amplifier_data"]
    raw_channel_names = list(data.get("channel_names", [])) or [f"CH{i}" for i in range(emg.shape[0])]

    # Timestamps
    if "t_amplifier" in data and data["t_amplifier"].size:
        emg_t = data["t_amplifier"]
    else:
        emg_t = np.arange(emg.shape[1], dtype=float) / emg_fs

    dur_s = emg.shape[1] / emg_fs
    t0 = float(emg_t[0])
    logging.info(f"RHD: fs={emg_fs:.1f} Hz, shape={emg.shape}, duration={dur_s:.2f}s, t0={t0:.3f}s")

    # Reorder channels to match training
    if "channel_names" not in meta.get("data", {}):
        raise RuntimeError("metadata missing data.channel_names (training channel order).")
    trained_names = meta["data"]["channel_names"]
    emg, sel_idx = select_training_channels_by_name(emg, raw_channel_names, trained_names)
    logging.info(f"Using {len(sel_idx)} channels locked to training order.")

    # Preprocess + features
    pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=env_cut, verbose=verbose)
    emg_pp = pre.preprocess(emg)
    X = pre.extract_emg_features(
        emg_pp, window_ms=window_ms, step_ms=step_ms,
        progress=verbose, tqdm_kwargs={"desc": "Extracting features", "leave": False}
    )

    # Compute window_starts (sample indices and seconds)
    step_samples = int((step_ms / 1000.0) * emg_fs)
    n_windows = X.shape[0]
    starts_samples = np.arange(n_windows) * step_samples
    window_starts_sec = (starts_samples / emg_fs) + t0

    # Load model and predict
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    y_pred = manager.predict(X)

    # Prepare results
    results = {
        'y_pred': y_pred,
        'window_starts_sec': window_starts_sec,
        'starts_samples': starts_samples,
        'window_ms': window_ms,
        'step_ms': step_ms,
    }

    # Save predictions if requested
    if save_predictions:
        output_dir = os.path.join(root_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(file_path))[0]
        pred_file = os.path.join(output_dir, f"{basename}_predictions.txt")
        
        with open(pred_file, 'w') as f:
            f.write(f"# Predictions for: {file_path}\n")
            f.write(f"# Window size: {window_ms} ms, Step size: {step_ms} ms\n")
            f.write(f"# Format: timestamp(s) | sample_index | prediction\n")
            f.write("#" + "="*60 + "\n")
            for t, idx, pred in zip(window_starts_sec, starts_samples, y_pred):
                f.write(f"{t:.3f}\t{idx}\t{pred}\n")
        
        logging.info(f"Predictions saved to: {pred_file}")
        results['pred_file'] = pred_file

    # Evaluate against events if provided
    if events_file:
        logging.info(f"Evaluating against events in: {events_file}")
        eval_results = evaluate_against_events(
            events_file, starts_samples, y_pred,
            return_metrics=True, return_arrays=True, verbose=True
        )
        
        if eval_results and save_predictions:
            # Save evaluation results
            eval_file = os.path.join(output_dir, f"{basename}_evaluation.json")
            import json
            with open(eval_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            logging.info(f"Evaluation results saved to: {eval_file}")
            results['eval_file'] = eval_file
        
        results['eval_results'] = eval_results
    else:
        # Quick preview
        uniq, cnt = np.unique(y_pred, return_counts=True)
        summary = ", ".join(f"{u}: {c}" for u, c in zip(uniq, cnt))
        logging.info(f"Predictions summary: {summary}")

    return results


def predict_batch(
    root_dir: str,
    rhd_glob: Optional[str] = None,
    rhd_files: Optional[list] = None,
    events_dir: Optional[str] = None,
    label: str = "",
    window_ms: Optional[int] = None,
    step_ms: Optional[int] = None,
    zero_division: int = 0,
    save_eval: bool = False,
    verbose: bool = False
):
    """
    Batch prediction across multiple RHD files with aggregated metrics.
    
    Args:
        root_dir: Directory containing trained model/metadata
        rhd_glob: Glob pattern for RHD files (for example, ``raw/**/*.rhd``)
        rhd_files: Explicit list of RHD file paths
        events_dir: Directory containing event files
        label: Model label/tag
        window_ms: Override window size
        step_ms: Override step size
        zero_division: Value for sklearn zero_division parameter
        save_eval: Save aggregated results to JSON
        verbose: Enable verbose logging
    
    Returns:
        dict with aggregated metrics
    """
    import glob
    from intan.io import (
        load_rhd_file,
        lock_params_to_meta,
        load_metadata_json,
        select_training_channels_by_name,
    )
    from intan.processing import EMGPreprocessor
    from intan.ml import (
        evaluate_against_events,
        classification_report_safe,
        ModelManager,
        EMGClassifier,
    )

    logging.basicConfig(format="[%(levelname)s] %(message)s",
                        level=(logging.DEBUG if verbose else logging.INFO))

    # Expand file list
    if rhd_glob:
        files = sorted(glob.glob(rhd_glob, recursive=True))
    else:
        files = rhd_files or []
    
    if not files:
        raise FileNotFoundError("No .rhd files matched.")
    logging.info(f"Found {len(files)} RHD files.")

    # Load metadata and model once
    meta = load_metadata_json(root_dir, label=label)
    window_ms, step_ms, _, env_cut = lock_params_to_meta(
        meta.get('data', {}), window_ms, step_ms, None
    )
    trained_names = meta["data"]["channel_names"]
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()

    y_true_all = []
    y_pred_all = []

    for i, rhd_path in enumerate(files, 1):
        # Find event file
        ev_path = None
        if events_dir:
            basename = os.path.splitext(os.path.basename(rhd_path))[0]
            ev_path = os.path.join(events_dir, basename + ".event")
            if not os.path.exists(ev_path):
                stem = basename.split('_')[0]
                candidates = [f for f in os.listdir(events_dir) if f.startswith(stem) and f.endswith('.event')]
                if candidates:
                    ev_path = os.path.join(events_dir, candidates[0])
                else:
                    ev_path = None

        if ev_path is None or not os.path.exists(ev_path):
            logging.warning(f"[{i}/{len(files)}] No event file for: {rhd_path} — skipping")
            continue

        # Process file
        try:
            data = load_rhd_file(rhd_path, verbose=False)
            emg = data["amplifier_data"]
            emg_fs = data['frequency_parameters']['amplifier_sample_rate']
            raw_channel_names = list(data.get("channel_names", [])) or [f"CH{i}" for i in range(emg.shape[0])]

            emg, sel_idx = select_training_channels_by_name(emg, raw_channel_names, trained_names)

            pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=env_cut, verbose=False)
            emg_pp = pre.preprocess(emg)
            X = pre.extract_emg_features(emg_pp, window_ms=window_ms, step_ms=step_ms, progress=False)

            step_samples = int((step_ms / 1000.0) * emg_fs)
            starts_samples = np.arange(X.shape[0]) * step_samples

            y_pred = manager.predict(X)

            y_true, y_pred_matched = evaluate_against_events(
                ev_path, starts_samples, y_pred, verbose=False, return_arrays=True
            )

            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred_matched)

            logging.info(f"[{i}/{len(files)}] Processed: {os.path.basename(rhd_path)} ({len(y_true)} windows)")

        except Exception as e:
            logging.error(f"[{i}/{len(files)}] Error processing {rhd_path}: {e}")
            continue

    # Aggregated metrics
    if not y_true_all:
        logging.warning("No valid predictions to aggregate.")
        return None

    logging.info(f"\n{'='*60}")
    logging.info(f"AGGREGATED RESULTS ({len(y_true_all)} total windows)")
    logging.info(f"{'='*60}")

    report = classification_report_safe(y_true_all, y_pred_all, zero_division=zero_division)
    logging.info(f"\n{report}")

    results = {
        "n_windows": len(y_true_all),
        "n_files": len(files),
        "classification_report": report
    }

    if save_eval:
        import json
        out_path = os.path.join(root_dir, f"batch_eval_{label}.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved evaluation to: {out_path}")
        results['eval_file'] = out_path

    return results


def predict_from_device(
    root_dir: str,
    label: str = "",
    seconds: float = 10.0,
    event_file: Optional[str] = None,
    window_ms: Optional[int] = None,
    step_ms: Optional[int] = None,
    verbose: bool = False
):
    """
    Record from device and predict gestures (fixed duration).
    
    Args:
        root_dir: Directory containing trained model/metadata
        label: Model label/tag
        seconds: Recording duration in seconds
        event_file: Optional events file for evaluation
        window_ms: Override window size
        step_ms: Override step size
        verbose: Enable verbose logging
    
    Returns:
        dict with predictions and optional evaluation results
    """
    from intan.interface import IntanRHXDevice
    from intan.io import lock_params_to_meta, load_metadata_json
    from intan.processing import EMGPreprocessor
    from intan.ml import ModelManager, EMGClassifier, evaluate_against_events

    logging.basicConfig(format='[%(levelname)s] %(message)s',
                        level=(logging.DEBUG if verbose else logging.INFO))

    # Load metadata
    meta = load_metadata_json(root_dir, label=label)
    window_ms, step_ms, _, env_cut = lock_params_to_meta(
        meta.get('data', {}), window_ms, step_ms, None
    )
    trained_names = meta["data"]["channel_names"]
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # Record from device
    logging.info(f"Recording {seconds}s from device...")
    with IntanRHXDevice(sample_rate=meta["data"].get("emg_fs", 4000), num_channels=len(trained_names)) as device:
        device.enable_wide_channel(range(len(trained_names)))
        emg, emg_fs, t0 = device.record(duration_sec=seconds)

    logging.info(f"Recorded: fs={emg_fs:.1f} Hz, shape={emg.shape}, t0={t0:.3f}s")

    # Preprocess and extract features
    pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=env_cut, verbose=verbose)
    emg_pp = pre.preprocess(emg)
    X = pre.extract_emg_features(emg_pp, window_ms=window_ms, step_ms=step_ms, progress=verbose)

    # Compute window starts
    step_samples = int((step_ms / 1000.0) * emg_fs)
    starts_samples = np.arange(X.shape[0]) * step_samples
    window_starts = (starts_samples / emg_fs) + t0

    # Predict
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    y_pred = manager.predict(X)

    results = {
        'y_pred': y_pred,
        'window_starts': window_starts,
        'starts_samples': starts_samples
    }

    # Optional evaluation
    if event_file:
        logging.info(f"Evaluating against events in: {event_file}")
        eval_results = evaluate_against_events(event_file, starts_samples, y_pred, return_metrics=True)
        results['eval_results'] = eval_results
    else:
        uniq, cnt = np.unique(y_pred, return_counts=True)
        summary = ", ".join(f"{u}: {c}" for u, c in zip(uniq, cnt))
        logging.info(f"Predictions summary: {summary}")

    return results


def predict_realtime_stream(
    root_dir: str,
    label: str = "",
    window_ms: Optional[int] = None,
    step_ms: Optional[int] = None,
    infer_period_s: Optional[float] = None,
    smooth_k: int = 1,
    seconds_total: Optional[float] = None,
    use_lsl: bool = False,
    verbose: bool = False
):
    """
    Real-time streaming prediction from device.
    
    Args:
        root_dir: Directory containing trained model/metadata
        label: Model label/tag
        window_ms: Override window size
        step_ms: Override step size
        infer_period_s: Inference period in seconds
        smooth_k: Majority vote window size for smoothing
        seconds_total: Total streaming duration (None = infinite)
        use_lsl: Enable LSL marker publishing
        verbose: Enable verbose logging
    """
    from intan.interface import IntanRHXDevice, LSLMarkerPublisher
    from intan.io import lock_params_to_meta, load_metadata_json
    from intan.ml import EMGRealTimePredictor
    import time

    logging.basicConfig(format='[%(levelname)s] %(message)s',
                        level=(logging.DEBUG if verbose else logging.INFO))

    # Load metadata
    meta = load_metadata_json(root_dir, label=label)
    window_ms, step_ms, _, env_cut = lock_params_to_meta(
        meta.get('data', {}), window_ms, step_ms, None
    )
    trained_names = meta["data"]["channel_names"]
    emg_fs = meta["data"].get("emg_fs", 4000)
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # Initialize predictor
    predictor = EMGRealTimePredictor(
        root_dir=root_dir,
        label=label,
        window_ms=window_ms,
        step_ms=step_ms,
        env_cut=env_cut,
        smooth_k=smooth_k,
        verbose=verbose
    )

    # Optional LSL publisher
    lsl_pub = None
    if use_lsl:
        lsl_pub = LSLMarkerPublisher(stream_name="EMG_Predictions")
        logging.info("LSL marker stream enabled.")

    # Stream and predict
    logging.info("Starting real-time prediction stream...")
    with IntanRHXDevice(sample_rate=emg_fs, num_channels=len(trained_names)) as device:
        device.enable_wide_channel(range(len(trained_names)))
        device.set_run_mode()

        start_time = time.time()
        infer_period_s = infer_period_s or (step_ms / 1000.0)

        try:
            while True:
                if seconds_total and (time.time() - start_time) >= seconds_total:
                    break

                emg_window = device.get_latest_data(duration_ms=window_ms)
                if emg_window is None:
                    time.sleep(0.01)
                    continue

                prediction = predictor.predict(emg_window)
                logging.info(f"Prediction: {prediction}")

                if lsl_pub:
                    lsl_pub.push_marker(prediction)

                time.sleep(infer_period_s)

        except KeyboardInterrupt:
            logging.info("\nStopped by user.")

    logging.info("Streaming complete.")
