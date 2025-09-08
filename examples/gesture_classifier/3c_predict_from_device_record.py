#!/usr/bin/env python3
"""
3_predict_from_device.py

Online/near-real-time EMG gesture prediction from an Intan RHX TCP client,
using the *exact* training settings from metadata:
- window/step/envelope cutoff locked to model metadata
- channel selection by NAME in the original training order
- optional evaluation against an events file

Example:
  python 3_predict_from_device.py \
    --root_dir "G:\\...\\2024_11_11" \
    --seconds 10 \
    --verbose
"""

import os
import argparse
import logging
import numpy as np

# --- intan package imports ---
from intan.interface import IntanRHXDevice  # your TCP client class
from intan.processing import EMGPreprocessor
from intan.ml import ModelManager, EMGClassifier, evaluate_against_events

from intan.io import (
    get_trained_channel_names,        # NEW helper you added
    load_metadata_json,
    lock_params_to_meta,
    select_training_channels_by_name,
)

def _get_device_channel_names(dev) -> list[str]:
    """
    Try to get channel names from the device; fall back to CH{i}.
    """
    names = None
    # If your RHXConfig/IntanRHXDevice exposes something like get_channel_names(), use it.
    if hasattr(dev, "get_channel_names"):
        try:
            names = list(dev.get_channel_names())
        except Exception:
            names = None
    # Fallback: CH0..CH{N-1}
    if not names:
        names = [f"CH{i}" for i in range(getattr(dev, "num_channels", 128))]
    return names

def run(
    root_dir: str,
    label: str = "",
    seconds: float = 10.0,
    event_file: str | None = None,
    window_ms: int | None = None,
    step_ms: int | None = None,
    verbose: bool = False,
):
    # Logging
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # --- Load metadata and lock params exactly as used in training ---
    meta = load_metadata_json(root_dir, label=label)
    # Prefer the 'data' block if present (same pattern as your RHD script)
    data_meta = meta.get("data", meta)

    # Lock timing & preprocessing from metadata (CLI can still override window/step if provided)
    window_ms, step_ms, _, env_cut = lock_params_to_meta(
        data_meta, window_ms, step_ms, selected_channels=None
    )
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # Trained channel names in canonical order
    trained_names = get_trained_channel_names(root_dir, label=label)
    if not trained_names:
        # Fallback for older metadata (same behavior as RHD script)
        trained_names = data_meta.get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("No training channel_names found in metadata.")

    # --- Connect to device and acquire a short buffer ---
    dev = IntanRHXDevice()  # your class already knows host/ports/sample_rate
    try:
        fs = float(getattr(dev, "sample_rate", data_meta.get("sample_rate_hz", 4000.0)))
        device_names_full = _get_device_channel_names(dev)

        # Map trained channel names -> device indices (so RHX streams only what we need)
        # We'll still re-verify order after capture.
        enable_idx = []
        name_set = set(device_names_full)
        for nm in trained_names:
            if nm not in name_set:
                raise RuntimeError(f"Trained channel '{nm}' not present in device channel list.")
            enable_idx.append(device_names_full.index(nm))

        # Configure device to stream exactly these channels
        dev.enable_wide_channel(enable_idx)

        # Record a buffer
        logging.info(f"Recording {seconds:.2f}s from device @ fs={fs:.1f} Hz on {len(enable_idx)} channels...")
        emg = dev.record(duration_sec=float(seconds), verbose=verbose)  # shape (C, N) for *enabled* channels
        if emg is None or emg.size == 0:
            raise RuntimeError("No EMG data captured from device.")

        # Make a synthetic timebase
        N = emg.shape[1]
        t = np.arange(N, dtype=float) / fs
        dur_s = N / fs
        logging.info(f"Capture done: emg shape={emg.shape}  duration≈{dur_s:.2f}s")

        # After capture, rebuild the active device channel-name list (ordered as streamed)
        # For many setups, enabled indices are streamed in ascending channel order,
        # so we reconstruct that order from the indices.
        device_active_names = [device_names_full[i] for i in enable_idx]

        # --- Reorder/select by *training* channel names (canonical order) ---
        # This ensures the feature layout matches the scaler/model (critical!)
        emg_reordered, sel_idx = select_training_channels_by_name(
            emg, device_active_names, trained_names
        )
        logging.info(f"Locked to training order: {len(sel_idx)} channels.")

        # --- Preprocess + features exactly as in training ---
        try:
            pre = EMGPreprocessor(fs=fs, envelope_cutoff=env_cut, verbose=verbose)
        except TypeError:
            pre = EMGPreprocessor(fs=fs, env_cut=env_cut)

        emg_pp = pre.preprocess(emg_reordered)
        X = pre.extract_emg_features(
            emg_pp,
            window_ms=window_ms,
            step_ms=step_ms,
            progress=True,
            tqdm_kwargs={"desc": "Extracting features", "unit": "win", "leave": False},
        )
        logging.info(f"Extracted feature matrix X with shape {X.shape}")

        # Build window start sample indices
        step_samples = int(round(step_ms / 1000.0 * fs))
        window_starts = np.arange(X.shape[0], dtype=int) * step_samples  # start_index=0 for live capture
        logging.info(
            "Alignment debug (Device):\n"
            f"  start_index=+0 samp\n"
            f"  step={step_ms} ms ({step_samples} samp)  windows={X.shape[0]}  approx_dur={dur_s:.2f}s\n"
            f"  window_starts [min..max]=[{window_starts.min()} .. {window_starts.max()}]"
        )

        # --- Load model/scaler, check feature dim, predict ---
        manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
        manager.load_model()
        n_expected = len(manager.scaler.mean_)
        if X.shape[1] != n_expected:
            raise ValueError(f"Feature dim {X.shape[1]} != scaler expectation {n_expected}")

        y_pred = manager.predict(X)

        # --- Optional offline evaluation against an events file ---
        if event_file:
            logging.info(f"Evaluating against events in: {event_file}")
            evaluate_against_events(event_file, window_starts, y_pred)
        else:
            # Quick preview of predictions
            uniq, cnt = np.unique(y_pred, return_counts=True)
            summary = ", ".join(f"{u}: {c}" for u, c in zip(uniq, cnt))
            logging.info(f"Predictions summary (first pass): {summary}")

        return y_pred

    finally:
        try:
            dev.set_run_mode("stop")
        except Exception:
            pass
        dev.close()


def main():
    p = argparse.ArgumentParser("Predict EMG gestures from Intan device (training-locked).")
    p.add_argument("--root_dir",    type=str, required=True, help="Directory that contains the trained model/metadata.")
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--seconds",     type=float, default=10.0, help="How many seconds to capture from the device.")
    p.add_argument("--event_file",  type=str, default=None, help="Optional path to an events file for evaluation.")
    p.add_argument("--window_ms",   type=int, default=None, help="Override window_ms (else from metadata).")
    p.add_argument("--step_ms",     type=int, default=None, help="Override step_ms (else from metadata).")
    p.add_argument("--verbose",     action="store_true")
    args = p.parse_args()

    run(
        root_dir=args.root_dir,
        label=args.label,
        seconds=args.seconds,
        event_file=args.event_file,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
