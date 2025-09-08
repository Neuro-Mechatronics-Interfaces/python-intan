#!/usr/bin/env python3
"""
3b_batch_predict_from_rhd.py

Batch-predict across MANY RHD files, align each with its .event file, and print a
SINGLE aggregated classification report + confusion matrix across all gestures.

Examples:
  python 3b_batch_predict_from_rhd.py \
    --root_dir "G:/Shared.../2024_11_12" \
    --rhd_glob "G:/Shared.../2024_11_12/raw/*/*.rhd" \
    --events_dir "G:/Shared.../2024_11_12/events" \
    --label "128ch" --verbose

  python 3b_batch_predict_from_rhd.py \
    --root_dir ... --rhd_files a.rhd b.rhd c.rhd --events_dir ... --label 128ch
"""

from __future__ import annotations
import os, glob, argparse, logging, json
from typing import List, Tuple, Optional
import numpy as np

from intan.io import load_rhd_file, lock_params_to_meta, load_metadata_json, select_training_channels_by_name
from intan.processing import EMGPreprocessor
from intan.ml import (
    evaluate_against_events,
    classification_report_safe,
    confusion_matrix_safe,
    ModelManager, EMGClassifier,
)

# ---------- helpers ----------

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _find_event_for_rhd(events_dir: Optional[str], rhd_path: str) -> Optional[str]:
    """
    Find a matching event file for <.../<Gesture>_<timestamp>.rhd>
    Tries, in order:
      1) <events_dir>/<gesture>_emg.event
      2) <events_dir>/<gesture>.event
      3) same folder as the .rhd (sibling events subdir or next to file)
    """
    if events_dir and os.path.isdir(events_dir):
        g = _stem(rhd_path).split("_")[0]  # e.g. WristFlexion_241112_170600 -> 'WristFlexion'
        cand = os.path.join(events_dir, f"{g}_emg.event")
        if os.path.isfile(cand):
            return cand
        cand = os.path.join(events_dir, f"{g}.event")
        if os.path.isfile(cand):
            return cand

    # same folder fallbacks
    rhd_dir = os.path.dirname(rhd_path)
    # events next to file
    for pat in (f"{_stem(rhd_path)}*.event", "*.event"):
        matches = sorted(glob.glob(os.path.join(rhd_dir, pat)))
        if matches:
            return matches[0]
    # events/ sibling
    evs = os.path.join(os.path.dirname(rhd_dir), "events")
    if os.path.isdir(evs):
        g = _stem(rhd_path).split("_")[0]
        for pat in (f"{g}_emg.event", f"{g}.event", "*.event"):
            matches = sorted(glob.glob(os.path.join(evs, pat)))
            if matches:
                return matches[0]
    return None


# ---------- core prediction (copy from your 3a) ----------

def predict_one_rhd(root_dir: str, rhd_path: str, label: str, window_ms: float, step_ms: float,
                    selected_channels=None, verbose: bool = False):
    """
    Returns (window_starts: np.ndarray[int], y_pred: np.ndarray[str or int])

    IMPORTANT: Copy the SAME model loading + feature extraction logic you use in 3a.
    That ensures identical preprocessing, channel order, window_ms/step_ms, etc.
    """

    # 0) Load metadata (label-specific if available)
    meta = load_metadata_json(root_dir, label=label)

    # Lock timing & preprocessing from metadata (CLI can override window/step)
    window_ms, step_ms, _, env_cut = lock_params_to_meta(meta.get('data', {}), window_ms, step_ms, selected_channels)
    logging.info(f"Training-locked params: window_ms={window_ms}  step_ms={step_ms}  env_cut={env_cut}")

    # ---- BEGIN: COPY from your 3a_predict_from_rhd.py ----
    # 1) Load RHD
    data = load_rhd_file(rhd_path, verbose=verbose)
    emg = data["amplifier_data"]         # (C, N)
    emg_fs = data['frequency_parameters']['amplifier_sample_rate']
    t  = data["t_amplifier"]

    # 2) Lock to training channel order + window/step/env_cut from root_dir (how you do in 3a)
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


    # 3) Preprocess + features (same as 3a)
    pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=env_cut, verbose=verbose)
    emg_pp = pre.preprocess(emg)
    X = pre.extract_emg_features(
        emg_pp, window_ms=window_ms, step_ms=step_ms,
        progress=verbose, tqdm_kwargs={"desc": f"Features {_stem(rhd_path)}", "leave": False}
    )

    # 4) Compute window_starts
    start_index = int(round(t[0] * emg_fs))
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

    return window_starts, y_pred


# ---------- batch driver ----------

def main():
    ap = argparse.ArgumentParser(description="Batch predict many RHD files and print one combined report.")
    ap.add_argument("--root_dir", type=str, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rhd_glob", type=str, help='Glob like "C:/data/raw/*/*.rhd"')
    g.add_argument("--rhd_files", nargs="+", help="Explicit list of .rhd files")
    ap.add_argument("--events_dir", type=str, default=None, help="Folder with *.event files (optional)")
    ap.add_argument("--label", type=str, required=True, help="Model label/tag to load (e.g., 128ch)")
    ap.add_argument("--window_ms",   type=int, default=None)
    ap.add_argument("--step_ms",     type=int, default=None)
    ap.add_argument("--zero_division", type=int, default=0)
    ap.add_argument("--save_eval", action="store_true",)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(format="[%(levelname)s] %(message)s",
                        level=(logging.DEBUG if args.verbose else logging.INFO))

    # Expand file list
    if args.rhd_glob:
        rhd_files = sorted(glob.glob(args.rhd_glob))
    else:
        rhd_files = [f for f in args.rhd_files]
    if not rhd_files:
        raise FileNotFoundError("No .rhd files matched.")
    logging.info(f"Found {len(rhd_files)} RHD files.")

    y_true_all: List[str] = []
    y_pred_all: List[str] = []

    for i, rhd_path in enumerate(rhd_files, 1):
        ev_path = _find_event_for_rhd(args.events_dir, rhd_path)
        if ev_path is None:
            logging.warning(f"[{i}/{len(rhd_files)}] No event file for: {rhd_path} — skipping")
            continue

        # Predict
        window_starts, y_pred = predict_one_rhd(
            root_dir=args.root_dir,
            rhd_path=rhd_path,
            label=args.label,
            window_ms=args.window_ms,
            step_ms=args.step_ms,
            verbose=args.verbose
        )

        # Evaluate but capture canonicalized arrays so we can aggregate
        metrics = evaluate_against_events(
            ev_path, window_starts, y_pred,
            zero_division=args.zero_division,
            return_metrics=True, return_arrays=True,
            verbose=args.verbose,
        )
        if metrics is None:
            logging.warning(f"[{i}/{len(rhd_files)}] No valid windows after filtering: {rhd_path}")
            continue

        # Accumulate
        y_true_all.extend(metrics["y_true_c"])
        y_pred_all.extend(metrics["y_pred_c"])

        # (Optional) per-file summary
        acc = metrics["accuracy"]
        logging.info(f"[{i}/{len(rhd_files)}] {os.path.basename(rhd_path)}  acc={acc:.4f}  n={len(metrics['y_true_c'])}")

    # Final aggregate report
    if not y_true_all:
        logging.warning("No data aggregated; nothing to report.")
        return

    y_true_all = np.asarray(y_true_all, dtype=object)
    y_pred_all = np.asarray(y_pred_all, dtype=object)
    labels_sorted = np.unique(np.concatenate([y_true_all, y_pred_all]))

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true_all, y_pred_all)
    print("\n================== AGGREGATED RESULTS ==================\n")
    print(f"Overall validation accuracy (canonicalized): {acc:.4f}\n")
    print("=== Classification Report (All Gestures) ===")
    rep_text = classification_report_safe(
        y_true_all, y_pred_all, labels=labels_sorted, zero_division=args.zero_division, output="text"
    )
    print(rep_text)
    print("=== Confusion Matrix (All Gestures) ===")
    cm, cm_labels = confusion_matrix_safe(y_true_all, y_pred_all, labels=labels_sorted)
    print(cm)
    print(f"Label order: {list(cm_labels)}")

    # --- Save aggregated results for later plotting ---
    if not args.save_eval:
        return

    out_dir = os.path.join(args.root_dir, "model")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.label}_batch_eval.json")

    # Also keep a dict form of the classification report for tables later
    rep_dict = classification_report_safe(
        y_true_all, y_pred_all, labels=labels_sorted, zero_division=args.zero_division, output="dict"
    )

    payload = {
        "labels": list(labels_sorted),  # order used in cm
        "confusion_matrix": cm.tolist(),  # integer counts
        "accuracy": float(acc),
        "classification_report": rep_dict,  # per-class P/R/F1 etc
        "y_true_all": y_true_all.tolist(),  # canonicalized strings
        "y_pred_all": y_pred_all.tolist(),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved aggregated evaluation → {out_path}\n")


if __name__ == "__main__":
    main()
