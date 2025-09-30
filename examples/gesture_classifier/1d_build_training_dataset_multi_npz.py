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
from typing import List, Dict, Tuple

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




def build_training_dataset_multi_npz(
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
    # Load all NPZs at once
    npz_items = _load_all_npz(root_dir, verbose=verbose)  # list of (data_dict, file_path)

    if save_path is None:
        save_path = os.path.join(root_dir, f"{label}_training_dataset.npz" if label else "training_dataset.npz")
    if os.path.exists(save_path) and not overwrite:
        logging.info(f"Dataset already exists: {save_path}. Use --overwrite to remake.")
        return

    combined_X = []
    combined_y = []
    selected_channels = None
    selected_channel_names = None

    # Optional: basic sample-rate consistency check
    fs_values = set()

    for i, (data, npz_path) in enumerate(npz_items, 1):
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
            progress=verbose, tqdm_kwargs={"desc": f"Features {i}/{len(npz_items)}", "leave": False}
        )

        # Window starts
        start_index = int(round(emg_t[0] * emg_fs))
        step_samples = int(round(step_ms / 1000.0 * emg_fs))
        window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index

        # Labels (resolve .event using the file path we just loaded)
        ev_path = _find_event_for_npz(root_dir, npz_path)
        if ev_path is None:
            logging.warning(f"[skip] No events found for {os.path.basename(npz_path)}")
            continue
        y = labels_from_events(ev_path, window_starts)

        # Filter out Unknown/Start
        mask = ~np.isin(y, ["Unknown", "Start"])
        X, y = X[mask], y[mask]
        if X.shape[0] == 0:
            logging.warning(f"[skip] No labeled windows for {os.path.basename(npz_path)}")
            continue

        combined_X.append(X)
        combined_y.append(y)

    if len(combined_X) == 0:
        raise RuntimeError("No usable NPZ+event pairs produced data.")

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
        channels=args.channels,
        channel_map=args.channel_map,
        channel_map_file=args.channel_map_file,
        mapping_non_strict=args.mapping_non_strict,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
