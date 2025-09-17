#!/usr/bin/env python3
"""
3_predict_from_rhd.py

Offline gesture prediction from an Intan .rhd file using the *exact* training settings:
- window/step/envelope cutoff from metadata
- channel selection by NAME in the original training order
- optional evaluation against an events file

Usage example:
  python 3b_predict_rhd.py \
    --root_dir "G:\\...\\2024_11_11" \
    --file_dir "G:\\...\\2024_11_11\\raw\\my_recording.rhd" \
    --events_file "G:\\...\\2024_11_11\\events\\emg.event" \
    --verbose
"""

import os
import argparse
import logging
import numpy as np
from intan.io import (
    load_rhd_file,
    load_config_file,
    lock_params_to_meta,
    load_metadata_json,
    select_training_channels_by_name,
)
from intan.processing import EMGPreprocessor
from intan.ml import ModelManager, EMGClassifier, evaluate_against_events


def run(root_dir: str, file_dir: str, label: str = "", window_ms: int | None = None, step_ms: int | None = None,
    selected_channels=None, events_file: str | None = None, verbose: bool = False):

    # Logging
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=lvl)

    # Load metadata (label-specific if available)
    meta = load_metadata_json(root_dir, label=label)

    # Lock timing & preprocessing from metadata (CLI can override window/step)
    window_ms, step_ms, _, env_cut = lock_params_to_meta(
        meta.get('data', {}), window_ms, step_ms, selected_channels
    )
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # Load raw EMG from .rhd
    data = load_rhd_file(file_dir, verbose=verbose)
    print(data.keys())
    #emg_fs = float(data["sample_rate"])
    emg_fs = data['frequency_parameters']['amplifier_sample_rate']
    emg = data["amplifier_data"]                           # (C, N)
    raw_channel_names = list(data.get("channel_names", [])) or [f"CH{i}" for i in range(emg.shape[0])]
    # timestamps may or may not be present in RHD loader; synthesize if absent
    if "t_amplifier" in data and data["t_amplifier"].size:
        emg_t = data["t_amplifier"]
    else:
        emg_t = np.arange(emg.shape[1], dtype=float) / emg_fs

    dur_s = emg.shape[1] / emg_fs
    t0 = float(emg_t[0])
    tN = float(emg_t[-1])
    logging.info(f"RHD load: fs={emg_fs:.3f} Hz  emg shape={emg.shape}  channels={len(raw_channel_names)}")
    logging.info(f"RHD time: t0={t0:.6f}s  tN={tN:.6f}s  duration≈{dur_s:.2f}s")

    # Reorder/select channels by the *training channel names* (canonical order)
    trained_names = meta.get("data", {}).get("channel_names") or meta.get("channel_names")
    if not trained_names:
        raise RuntimeError("metadata missing data.channel_names (training channel order).")
    emg, sel_idx = select_training_channels_by_name(emg, raw_channel_names, trained_names)
    logging.info(f"Using {len(sel_idx)} channels locked to training order.")

    # Preprocess + features (matching training)
    # Try both naming variants for envelope args, depending on your EMGPreprocessor version.
    try:
        pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=env_cut, verbose=verbose)
    except TypeError:
        pre = EMGPreprocessor(fs=emg_fs, env_cut=env_cut)

    emg_pp = pre.preprocess(emg)
    X = pre.extract_emg_features(emg_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
        tqdm_kwargs={"desc": "Extracting features", "unit": "win", "leave": False, "ascii": True},
    )
    logging.info(f"Extracted feature matrix X with shape {X.shape}")

    # Build window start indices in absolute sample units
    start_index = int(round(emg_t[0] * emg_fs))
    step_samples = int(round(step_ms / 1000.0 * emg_fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index

    logging.info(
        "Alignment debug (RHD):\n"
        f"  start_index={start_index:+d} samp (t0*fs)\n"
        f"  step={step_ms} ms ({step_samples} samp)  windows={X.shape[0]}  approx_dur={dur_s:.2f}s\n"
        f"  window_starts [min..max]=[{window_starts.min()} .. {window_starts.max()}]"
    )

    # Load model/scaler for this label, check feature dim, predict
    manager = ModelManager(root_dir=root_dir, label=label, model_cls=EMGClassifier, config={"verbose": verbose})
    manager.load_model()
    n_features_expected = len(manager.scaler.mean_)
    if X.shape[1] != n_features_expected:
        raise ValueError(f"Feature dim {X.shape[1]} != scaler expectation {n_features_expected}")

    y_pred = manager.predict(X)

    # Evaluate against events
    #file_path = os.path.join(root_dir, 'events', f"{os.path.basename(file_dir).split('.')[0]}_emg.event")
    #evaluate_against_events(file_path, window_starts, y_pred)

    # --- Optional offline evaluation against an events file ---
    if events_file:
        logging.info(f"Evaluating against events in: {events_file}")
        evaluate_against_events(events_file, window_starts, y_pred)
    else:
        # Quick preview of predictions
        uniq, cnt = np.unique(y_pred, return_counts=True)
        summary = ", ".join(f"{u}: {c}" for u, c in zip(uniq, cnt))
        logging.info(f"Predictions summary (first pass): {summary}")


def main():
    p = argparse.ArgumentParser("3b: Offline EMG gesture prediction from Intan RHD (training-locked)")
    p.add_argument("--config_file", type=str)
    p.add_argument("--root_dir",    type=str, required=True, help="Directory that contains the trained model/metadata.")
    p.add_argument("--file_dir",    type=str, required=True, help="Path to the .rhd file to evaluate.")
    p.add_argument("--events_file", type=str, default=None, help="If different from root_dir/events")
    p.add_argument("--label",       type=str, default="")
    p.add_argument("--window_ms",   type=int, default=None)
    p.add_argument("--step_ms",     type=int, default=None)
    p.add_argument("--verbose",     action="store_true")
    args = p.parse_args()

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)
    cfg.update({
        "root_dir": args.root_dir or cfg.get("root_dir", ""),
        "file_dir": args.file_dir,
        "events_file": args.events_file or cfg.get("events_file", None),
        "label": args.label or cfg.get("label", ""),
        "window_ms": args.window_ms or cfg.get("window_ms", None),
        "step_ms": args.step_ms or cfg.get("step_ms", None),
        "verbose": args.verbose or cfg.get("verbose", False),
    })
    run(**cfg)


if __name__ == "__main__":
    main()
