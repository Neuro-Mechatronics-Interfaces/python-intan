#!/usr/bin/env python3
"""
1d_build_training_dataset_npz_multi.py (refactored)

Aggregate MANY NPZ recordings (each exported from RHD via intan.io.save_as_npz)
into a single (X, y) training dataset using per-recording .event files for labels.

Expected layout:

<root_dir>/
  emg/
    recA.npz
    recB.npz
  events/
    recA_emg.event
    recB_emg.event

Usage:
  python 1d_build_training_dataset_npz_multi.py --root_dir /path/to/root \
        --window_ms 200 --step_ms 50 \
        --channel_map 8-8-L --channel_map_file custom_channel_mappings.json \
        --overwrite --verbose
"""

from __future__ import annotations
import os
import json
import glob
import argparse
import logging
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional, Callable

from intan.io import (
    load_npz_files,             # legacy single-file, now supports allow_many
    labels_from_events,
    build_indices_from_mapping,
)
from intan.processing import EMGPreprocessor, FEATURE_REGISTRY


def _stem(path):
    if hasattr(path, "item"):  # handles 0-D numpy arrays
        try:
            path = path.item()
        except Exception:
            pass
    # Want to parse our the label at the beginning of teh file name, like getting 'IndexExtension' from 'IndexExtension_241112_170737'
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    if "_" in stem:
        stem = stem.split("_")[0]
    return stem
    #return os.path.splitext(os.path.basename(str(path)))[0]



def _first(path_glob: str | None) -> str | None:
    matches = sorted(glob.glob(path_glob)) if path_glob else []
    return matches[0] if matches else None


def _find_event_for_npz(root_dir: str, npz_path: str) -> str | None:
    """Try several sensible locations/filenames for the matching .event file."""
    stem = _stem(npz_path)
    rd_events = os.path.join(root_dir, "events")
    npz_dir = os.path.dirname(npz_path)

    print(f"rd event dir {rd_events}, npz dir {npz_dir}, stem {stem}")

    # 1) <root>/events/<stem>_emg.event
    cand = os.path.join(rd_events, f"{stem}_emg.event")
    if os.path.isfile(cand):
        return cand

    # 2) <root>/events/<stem>.event
    cand = os.path.join(rd_events, f"{stem}.event")
    if os.path.isfile(cand):
        return cand

    # 3) <npz_parent>/../events/<stem>*.event
    cand = _first(os.path.join(os.path.dirname(npz_dir), "events", f"{stem}*.event"))
    if cand:
        return cand

    # 4) If only one *.event under <root>/events, use it
    evs = sorted(glob.glob(os.path.join(rd_events, "*.event"))) if os.path.isdir(rd_events) else []
    if len(evs) == 1:
        return evs[0]

    # 5) Next to the NPZ
    cand = _first(os.path.join(npz_dir, f"{stem}*.event"))
    if cand:
        return cand

    return None


def _compute_selected_channels(raw_names, channels, channel_map, channel_map_file, mapping_non_strict):
    """Decide channel indices once (by mapping names→indices or explicit list)."""
    if channel_map:
        if not os.path.isfile(channel_map_file):
            raise FileNotFoundError(f"Mapping file not found: {channel_map_file}")
        with open(channel_map_file, "r", encoding="utf-8") as f:
            mapping_json = json.load(f)
        if channel_map not in mapping_json:
            raise KeyError(f"Mapping '{channel_map}' not in {channel_map_file}")
        mapping_names = list(mapping_json[channel_map])
        sel = build_indices_from_mapping(raw_names, mapping_names, strict=(not mapping_non_strict))
        sel_names = [raw_names[i] for i in sel]
        return sel, sel_names

    if channels is not None:
        sel_names = [raw_names[i] for i in channels]
        return channels, sel_names

    return None, raw_names


def _load_all_npz(root_dir: str, verbose: bool):
    """
    Load all NPZs under <root_dir>/emg and return a list of (data_dict, npz_path).
    We **always** pair each loaded dict with the actual NPZ filename on disk,
    ignoring any 'file_path' stored inside the NPZ (which may reference the RHD).
    """
    emg_dir = os.path.join(root_dir, "emg")
    if not os.path.isdir(emg_dir):
        raise FileNotFoundError(f"Expected NPZs under {emg_dir}")

    npz_paths = sorted(glob.glob(os.path.join(emg_dir, "*.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No NPZ files found under {emg_dir}")

    # Your renamed API: expects a list/iterable of paths and returns list[dict]
    dict_list = load_npz_files(npz_paths, verbose=verbose)

    if len(dict_list) != len(npz_paths):
        logging.warning(
            f"Loaded {len(dict_list)} NPZ dicts but found {len(npz_paths)} files. "
            f"Proceeding by zipping the shortest length."
        )

    # Pair each dict with its real NPZ path (string)
    return list(zip(dict_list, npz_paths))

def _load_all_csv(root_dir: str, verbose: bool):
    """
    Load all CSVs under <root_dir>/csv and return a list of (data_dict, csv_path)
    where data_dict mimics the NPZ structure your pipeline expects:
      {
        'amplifier_data': np.ndarray shape (C, N),
        't_amplifier':    np.ndarray shape (N,),
        'sample_rate':    float,
        'channel_names':  list[str]
      }
    """
    csv_dir = os.path.join(root_dir, "csv")
    if not os.path.isdir(csv_dir):
        raise FileNotFoundError(f"Expected CSVs under {csv_dir}")

    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {csv_dir}")

    items = []
    for path in csv_paths:
        # read with flexible sep
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=";")

        # normalize headers
        cols = [c.strip() for c in df.columns]
        df.columns = cols

        # pick EMG columns
        emg_cols = [c for c in cols if re.match(r"^EMG_\d+$", c)]
        if not emg_cols:
            raise ValueError(f"{os.path.basename(path)} has no EMG_* columns")

        emg = df[emg_cols].to_numpy(dtype=float).T  # (C, N)
        N = emg.shape[1]

        # time base
        if "t_host" in df.columns:
            t = df["t_host"].astype(float).to_numpy()
            # ensure strictly increasing if there are duplicates
            # (optional, only if your logger could repeat t)
            # t = np.maximum.accumulate(t)
            dt = np.median(np.diff(t)) if len(t) > 1 else 0.0
            fs = (1.0 / dt) if dt > 0 else 0.0
        else:
            # last resort—assume 1000 Hz if t_host missing (you can add a CLI arg)
            fs = 1000.0
            t = (np.arange(N, dtype=float) / fs)

        # optional: clip IMU columns list (if present)
        imu_cols = ["IMU_seq","roll","pitch","yaw","ax","ay","az","gx","gy","gz"]
        present_imu = [c for c in imu_cols if c in df.columns]
        imu = df[present_imu].to_numpy(dtype=float) if present_imu else None

        data = {
            "amplifier_data": emg,
            "t_amplifier": t,
            "sample_rate": float(fs),
            "channel_names": emg_cols,
            # pass IMU through so we can aggregate per window
            "_imu_frame": imu,
            "_imu_cols": present_imu,
            "file_path": path,
        }
        items.append((data, path))
    return items

def _aggregate_imu_over_windows(
    imu_frame: Optional[np.ndarray],
    imu_cols: List[str],
    start_indices: np.ndarray,
    window_samples: int,
    reducer: Callable[[np.ndarray, int], np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    For each window [s, s+window_samples), compute an aggregate over IMU rows.
    Assumes IMU rows align 1:1 with EMG samples (your CSV logger’s behavior).

    Returns (imu_win, imu_cols):
      - imu_win shape = (n_windows, n_imu_features) or None if no IMU present
    """
    if imu_frame is None or not imu_cols:
        return None, []

    if reducer is None:
        # default reducer: mean over each window (nan-safe)
        def reducer(seg, axis): return np.nanmean(seg, axis=axis)

    out = []
    L = imu_frame.shape[0]
    for s in start_indices:
        e = s + window_samples
        s0 = int(max(0, s)); e0 = int(min(L, max(s0, e)))
        if e0 <= s0:
            out.append([float("nan")] * len(imu_cols))
        else:
            seg = imu_frame[s0:e0, :]
            out.append(reducer(seg, axis=0).tolist())
    return np.asarray(out, dtype=float), imu_cols


def _append_imu_features(
    X: np.ndarray,
    y: np.ndarray,
    imu_win: Optional[np.ndarray],
    imu_cols: List[str],
    label_mask: np.ndarray,
    imu_norm: str = "zscore",
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply label mask; normalize IMU block (train-set stats); append to X.
    Returns (X_aug, meta) where meta contains imu_cols, counts, and norm stats.
    """
    meta = {
        "imu_cols": np.array(imu_cols or [], dtype=object),
        "imu_feature_count": np.array(len(imu_cols or []), dtype=np.int32),
        "imu_norm_kind": np.array(imu_norm, dtype=object),
        "imu_norm_mean": np.array([], dtype=np.float32),
        "imu_norm_std":  np.array([], dtype=np.float32),
    }

    if imu_win is None or len(imu_cols) == 0:
        return X, meta

    # Apply the SAME mask you used on (X, y)
    imu_win = imu_win[label_mask]
    if imu_win.shape[0] != X.shape[0]:
        raise RuntimeError("IMU rows after mask do not match X rows.")

    # Normalize IMU columns (train-set stats) if requested
    if imu_norm == "zscore":
        mu = np.nanmean(imu_win, axis=0).astype(np.float32)
        sd = np.nanstd(imu_win, axis=0).astype(np.float32)
        sd[sd < eps] = 1.0  # avoid blow-ups on constant columns
        imu_win = (imu_win - mu) / sd
        meta["imu_norm_mean"] = mu
        meta["imu_norm_std"]  = sd
    elif imu_norm == "robust":
        # median/IQR (optional alternative)
        med = np.nanmedian(imu_win, axis=0).astype(np.float32)
        q75 = np.nanpercentile(imu_win, 75, axis=0).astype(np.float32)
        q25 = np.nanpercentile(imu_win, 25, axis=0).astype(np.float32)
        iqr = q75 - q25
        iqr[iqr < eps] = 1.0
        imu_win = (imu_win - med) / iqr
        meta["imu_norm_kind"] = np.array("robust", dtype=object)
        meta["imu_norm_mean"] = med
        meta["imu_norm_std"]  = iqr
    else:
        # "none": pass through as-is
        pass

    X_aug = np.hstack([X, imu_win])
    return X_aug, meta


def build_training_dataset_multi_npz(
    root_dir: str,
    label: str = "",
    save_path: str | None = None,
    window_ms: int = 200,
    step_ms: int = 50,
    imu_norm: str = "zscore",
    channels: list[int] | None = None,
    channel_map: str | None = None,
    channel_map_file: str = "custom_channel_mappings.json",
    mapping_non_strict: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
):
    # Load all NPZs at once
    csv_items = _load_all_csv(root_dir, verbose=verbose)  # list of (data_dict, file_path)

    if save_path is None:
        save_path = os.path.join(root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz")
    if os.path.exists(save_path) and not overwrite:
        logging.info(f"Dataset already exists: {save_path}. Use --overwrite to remake.")
        return

    combined_X = []
    combined_y = []
    combined_IMU = []
    imu_header = None

    selected_channels = None
    selected_channel_names = None

    # Optional: basic sample-rate consistency check
    fs_values = set()

    for i, (data, csv_path) in enumerate(csv_items, 1):
        emg = data["amplifier_data"]     # (C, N)
        emg_t = data["t_amplifier"]      # (N,)
        emg_fs = float(data["sample_rate"])
        fs_values.add(round(emg_fs, 6))
        raw_names = list(data.get("channel_names", [])) or [f"CH{j}" for j in range(emg.shape[0])]

        # Decide channels once
        if selected_channels is None:
            selected_channels, selected_channel_names = _compute_selected_channels(
                raw_names, channels, channel_map, channel_map_file, mapping_non_strict
            )
            if verbose:
                logging.info(f"[prime] selected channels: {selected_channel_names}")

        # Apply selection
        if selected_channels is not None:
            emg = emg[selected_channels, :]

        # Preprocess & features
        pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=5.0, verbose=verbose)
        emg_pp = pre.preprocess(emg)
        X = pre.extract_emg_features(
            emg_pp, window_ms=window_ms, step_ms=step_ms,
            progress=verbose, tqdm_kwargs={"desc": f"Features {i}/{len(csv_items)}", "leave": False}
        )

        # Window starts
        start_index = int(round(emg_t[0] * emg_fs))
        step_samples = int(round(step_ms / 1000.0 * emg_fs))
        window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index
        window_samples = int(round(window_ms / 1000.0 * emg_fs))

        # Labels (resolve .event using the file path we just loaded)
        ev_path = _find_event_for_npz(root_dir, csv_path)
        if ev_path is None:
            logging.warning(f"[skip] No events found for {os.path.basename(csv_path)}")
            continue
        y = labels_from_events(ev_path, window_starts)

        # Filter out Unknown/Start
        mask = ~np.isin(y, ["Unknown", "Start"])
        X, y = X[mask], y[mask]
        if X.shape[0] == 0:
            logging.warning(f"[skip] No labeled windows for {os.path.basename(csv_path)}")
            continue

        # Compute IMU aggregates for these windows (before masking)
        imu_win, imu_cols = _aggregate_imu_over_windows(
            data.get("_imu_frame"), data.get("_imu_cols", []), window_starts, window_samples
        )

        # Append (normalized) IMU to X in a polished, single call
        X, imu_meta = _append_imu_features(
            X=X, y=y,
            imu_win=imu_win,
            imu_cols=imu_cols,
            label_mask=mask,
            imu_norm="zscore",  # or "none" / "robust"
        )

        # Track IMU header once
        if imu_header is None and imu_meta["imu_feature_count"] > 0:
            imu_header = imu_meta["imu_cols"].tolist()

        combined_X.append(X)
        combined_y.append(y)

    # Warn if sample rates differed
    if len(fs_values) > 1:
        logging.warning(f"Multiple sample rates detected across NPZs: {sorted(fs_values)}. "
                        f"Consider resampling upstream for strict consistency.")

    X = np.concatenate(combined_X, axis=0)
    y = np.concatenate(combined_y, axis=0)

    # Feature spec
    try:
        n_ch = len(selected_channel_names) if selected_channel_names is not None else X.shape[-1]
        pre_tmp = EMGPreprocessor(fs=1000.0)
        feature_spec = pre_tmp.feature_spec(n_channels=n_ch)
    except Exception:
        names = list(FEATURE_REGISTRY.keys())
        feature_spec = {
            "per_channel": True,
            "order": names,
            "dims_per_feature": {n: 1 for n in names},
            "layout": "channel_major",
            "channels": "training_order",
            "n_channels": int(len(selected_channel_names) if selected_channel_names else 0),
            "n_features_per_channel": len(names),
        }
    feature_spec_json = json.dumps(feature_spec)

    # Save
    np.savez(
        save_path,
        X=X, y=y,
        emg_fs=float(fs_values.pop()) if len(fs_values) == 1 else 0.0,
        class_names=np.array(sorted(set(y)), dtype=object),
        label_to_id_json=np.array(json.dumps({c: i for i, c in enumerate(sorted(set(y))) }), dtype=object),
        window_ms=window_ms, step_ms=step_ms,
        selected_channels=np.array(selected_channels if selected_channels else [], dtype=int),
        channel_names=np.array(selected_channel_names if selected_channel_names else [], dtype=object),
        feature_spec=feature_spec_json,
        channel_mapping_name=np.array(channel_map or "", dtype=object),
        channel_mapping_file=np.array(channel_map_file or "", dtype=object),
        imu_cols=np.array(imu_header or [], dtype=object),
        imu_feature_count=np.array(len(imu_header or []), dtype=np.int32),
        # Optional: if you choose to normalize on the concatenated matrix instead,
        # save the single global stats here. If you normalized per the helper above,
        # either collect/merge stats or set empty arrays; up to your eval pipeline.
        # imu_norm_kind=np.array("zscore", dtype=object),
        # imu_norm_mean=imu_norm_mean,   # if computed globally
        # imu_norm_std=imu_norm_std,     # if computed globally
        modality=np.array("emg+imu" if len(imu_header or []) > 0 else "emg", dtype=object),
    )
    logging.info(f"Saved aggregated dataset: {save_path}  (X={X.shape}, y={y.shape})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Aggregate MANY NPZ recordings into one training dataset.")
    p.add_argument("--root_dir", type=str, required=True, help="Root containing 'emg/' and optionally 'events/'.")
    p.add_argument("--label", type=str, default="")
    p.add_argument("--channels", nargs="+", type=int, default=None)
    p.add_argument("--channel_map", type=str, default=None)
    p.add_argument("--channel_map_file", type=str, default="custom_channel_mappings.json")
    p.add_argument("--mapping_non_strict", action="store_true")
    p.add_argument("--window_ms", type=int, default=200)
    p.add_argument("--step_ms", type=int, default=50)
    p.add_argument("--imu_norm", type=str, choices=["zscore", "none"], default="zscore",
                   help="Normalize IMU columns in X using train-set stats (zscore) or leave as-is.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)

    build_training_dataset_multi_npz(
        root_dir=args.root_dir,
        label=args.label,
        save_path=args.save_path,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        imu_norm=args.imu_norm,
        channels=args.channels,
        channel_map=args.channel_map,
        channel_map_file=args.channel_map_file,
        mapping_non_strict=args.mapping_non_strict,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
