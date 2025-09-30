#!/usr/bin/env python3
"""
Aggregate MANY RHD recordings (each a gesture) into one (X, y) dataset.

Expected layout:
<root_dir>/
  raw/
    GestureA_2025-09-01_12-30-00.rhd
    GestureB_2025-09-01_12-33-10.rhd
  events/
    GestureA_emg.event
    GestureB_emg.event
(or any matching <stem>*.event that _find_event_for_file() can resolve)

Usage:
  python 1c_build_training_dataset_multi_rhd.py --root_dir /path/to/root \
        --window_ms 200 --step_ms 50 \
        --channel_map 8-8-L --channel_map_file custom_channel_mappings.json \
        --overwrite --verbose
"""
from __future__ import annotations

import os
import glob
import json
import pathlib
import argparse
import logging
import numpy as np
from typing import List

from intan.io import (
    load_rhd_file,
    labels_from_events,
    build_indices_from_mapping,
)
from intan.processing import EMGPreprocessor, FEATURE_REGISTRY


def _stem(path):
    base = os.path.splitext(os.path.basename(str(path)))[0]
    return base.split("_")[0] if "_" in base else base


def _first(pattern: str | None) -> str | None:
    matches = sorted(glob.glob(pattern)) if pattern else []
    return matches[0] if matches else None


def _find_event_for_file(root_dir: str, data_path: str) -> str | None:
    stem = _stem(data_path)
    rd_events = os.path.join(root_dir, "events")
    data_dir = os.path.dirname(data_path)

    cand = os.path.join(rd_events, f"{stem}_emg.event")
    if os.path.isfile(cand):
        return cand

    cand = os.path.join(rd_events, f"{stem}.event")
    if os.path.isfile(cand):
        return cand

    cand = _first(os.path.join(os.path.dirname(data_dir), "events", f"{stem}*.event"))
    if cand:
        return cand

    evs = sorted(glob.glob(os.path.join(rd_events, "*.event"))) if os.path.isdir(rd_events) else []
    if len(evs) == 1:
        return evs[0]

    cand = _first(os.path.join(data_dir, f"{stem}*.event"))
    if cand:
        return cand

    return None


def _compute_selected_channels(raw_names, channels, channel_map, channel_map_file, mapping_non_strict):
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

def _discover_rhd_files(root_dir: str) -> List[str]:
    """
    Discover .rhd files with support for these layouts (searched in order):
      1) <root>/raw/*.rhd
      2) <root>/raw/*/*.rhd            # e.g., raw/GestureA/GestureA.rhd
      3) <root>/*.rhd
      4) <root>/*/*.rhd
    If multiple .rhd exist inside a gesture folder, prefer one that matches the
    folder name (e.g., raw/GestureA/GestureA.rhd) but include others as well.
    """
    candidates = []

    # 1) raw/*.rhd
    candidates += glob.glob(os.path.join(root_dir, "raw", "*.rhd"))

    # 2) raw/*/*.rhd  (one level of gesture subdirs)
    deep = glob.glob(os.path.join(root_dir, "raw", "*", "*.rhd"))
    if deep:
        # Prefer same-name file within each folder, but keep others too
        by_dir = {}
        for p in deep:
            d = os.path.dirname(p)
            by_dir.setdefault(d, []).append(p)

        preferred_then_rest = []
        for d, files in by_dir.items():
            base = os.path.basename(d)
            # exact match first if present
            exact = [p for p in files if os.path.splitext(os.path.basename(p))[0] == base]
            others = sorted(set(files) - set(exact))
            preferred_then_rest.extend(exact + others)
        candidates += preferred_then_rest

    # 3) <root>/*.rhd (fallback)
    candidates += glob.glob(os.path.join(root_dir, "*.rhd"))

    # 4) <root>/*/*.rhd (one level deep fallback)
    candidates += glob.glob(os.path.join(root_dir, "*", "*.rhd"))

    # Deduplicate, normalize, and sort
    norm = sorted({str(pathlib.Path(p).resolve()) for p in candidates})
    return norm


def _load_one_rhd(rhd_path: str, verbose: bool):
    if os.path.isdir(rhd_path):
        rhds = sorted(glob.glob(os.path.join(rhd_path, "*.rhd")))
        if not rhds:
            raise FileNotFoundError(f"No .rhd found under directory: {rhd_path}")
        rhd_path = rhds[0]
    data = load_rhd_file(rhd_path, verbose=verbose)
    return data, rhd_path


def build_training_dataset_multi_rhd(
    root_dir: str,
    label: str = "",
    save_path: str | None = None,
    window_ms: int = 200,
    step_ms: int = 50,
    channels: list[int] | None = None,
    channel_map: str | None = None,
    channel_map_file: str = "custom_channel_mappings.json",
    mapping_non_strict: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
):
    rhd_paths = _discover_rhd_files(root_dir)
    if not rhd_paths:
        raise FileNotFoundError(f"No .rhd files found under {root_dir} or {os.path.join(root_dir, 'raw')}")

    if save_path is None:
        save_path = os.path.join(root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz")
    if os.path.exists(save_path) and not overwrite:
        logging.info(f"Dataset already exists: {save_path}. Use --overwrite to remake.")
        return

    combined_X = []
    combined_y = []
    selected_channels = None
    selected_channel_names = None

    fs_values = set()

    for i, rhd_path in enumerate(rhd_paths, 1):
        if verbose:
            logging.info(f"[{i}/{len(rhd_paths)}] Loading {os.path.basename(rhd_path)}")

        data, true_path = _load_one_rhd(rhd_path, verbose=verbose)

        emg_fs = float(data["frequency_parameters"]["amplifier_sample_rate"])
        emg = data["amplifier_data"]
        emg_t = data["t_amplifier"]
        raw_names = list(data.get("channel_names", [])) or [f"CH{j}" for j in range(emg.shape[0])]
        fs_values.add(round(emg_fs, 6))

        if selected_channels is None:
            selected_channels, selected_channel_names = _compute_selected_channels(
                raw_names, channels, channel_map, channel_map_file, mapping_non_strict
            )
            if verbose:
                logging.info(f"[prime] selected channels: {selected_channel_names}")

        if selected_channels is not None:
            emg = emg[selected_channels, :]

        pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=5.0, verbose=verbose)
        emg_pp = pre.preprocess(emg)
        X = pre.extract_emg_features(
            emg_pp, window_ms=window_ms, step_ms=step_ms,
            progress=verbose, tqdm_kwargs={"desc": f"Features {i}/{len(rhd_paths)}", "leave": False}
        )

        start_index = int(round(emg_t[0] * emg_fs))
        step_samples = int(round(step_ms / 1000.0 * emg_fs))
        window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index

        ev_path = _find_event_for_file(root_dir, true_path)
        if ev_path is None:
            logging.warning(f"[skip] No events found for {os.path.basename(true_path)}")
            continue
        y = labels_from_events(ev_path, window_starts)

        mask = ~np.isin(y, ["Unknown", "Start"])
        X, y = X[mask], y[mask]
        if X.shape[0] == 0:
            logging.warning(f"[skip] No labeled windows for {os.path.basename(true_path)}")
            continue

        combined_X.append(X)
        combined_y.append(y)

    if len(combined_X) == 0:
        raise RuntimeError("No usable RHD+event pairs produced data.")

    if len(fs_values) > 1:
        logging.warning(f"Multiple sample rates detected across RHD files: {sorted(fs_values)}. "
                        f"Consider resampling upstream for strict consistency.")

    X = np.concatenate(combined_X, axis=0)
    y = np.concatenate(combined_y, axis=0)

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

    np.savez(
        save_path,
        X=X, y=y,
        emg_fs=float(fs_values.pop()) if len(fs_values) == 1 else 0.0,
        class_names=np.array(sorted(set(y)), dtype=object),
        label_to_id_json=np.array(json.dumps({c: i for i, c in enumerate(sorted(set(y)))}), dtype=object),
        window_ms=window_ms, step_ms=step_ms,
        selected_channels=np.array(selected_channels if selected_channels else [], dtype=int),
        channel_names=np.array(selected_channel_names if selected_channel_names else [], dtype=object),
        feature_spec=feature_spec_json,
        channel_mapping_name=np.array(channel_map or "", dtype=object),
        channel_mapping_file=np.array(channel_map_file or "", dtype=object),
    )
    logging.info(f"Saved aggregated dataset: {save_path}  (X={X.shape}, y={y.shape})")


def _parse_channels_arg(arg: str | None, total: int | None = None) -> list[int] | None:
    if not arg:
        return None
    s = arg.strip().lower()
    if s == "all":
        return list(range(int(total))) if total is not None else None
    if ":" in s:
        a, b = s.split(":", 1)
        return list(range(int(a), int(b)))
    parts = [tok for piece in s.split(",") for tok in piece.split()]
    return [int(x) for x in parts if x != ""]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Aggregate MANY RHD recordings into one training dataset.")
    p.add_argument("--root_dir", type=str, required=True, help="Root containing 'raw/' and 'events/' folders.")
    p.add_argument("--label", type=str, default="")
    p.add_argument("--channels", type=str, default=None,
                   help="Channel list: 'all', '0:64', '0 1 2', or '0,1,2'")
    p.add_argument("--channel_map", type=str, default=None)
    p.add_argument("--channel_map_file", type=str, default="custom_channel_mappings.json")
    p.add_argument("--mapping_non_strict", action="store_true")
    p.add_argument("--window_ms", type=int, default=200)
    p.add_argument("--step_ms", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    parsed_channels = _parse_channels_arg(args.channels)

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)

    build_training_dataset_multi_rhd(
        root_dir=args.root_dir,
        label=args.label,
        save_path=args.save_path,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        channels=parsed_channels,
        channel_map=args.channel_map,
        channel_map_file=args.channel_map_file,
        mapping_non_strict=args.mapping_non_strict,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )