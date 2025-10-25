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
  python 1c_build_training_dataset_npz_multi.py --root_dir /path/to/root \
        --window_ms 200 --step_ms 50 \
        --channel_map 8-8-L --channel_map_file custom_channel_mappings.json \
        --overwrite --verbose
"""

from __future__ import annotations
import os
import re
import json
import glob
import argparse
import logging
import numpy as np

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


# def _first(path_glob: str | None) -> str | None:
#     matches = sorted(glob.glob(path_glob)) if path_glob else []
#     return matches[0] if matches else None

def parse_channels_spec(specs) -> list[int] | None:
    """
    Parse channels spec from CLI.
    Accepts:
      - single indices: 5  12
      - python-slice style: 0:128        (end-exclusive)
                            0:128:2      (step)
                            :64          (start defaults to 0)
      - dash ranges:       1-8           (inclusive)
      - comma separated within a token:  0:64,70,75-80
    Returns a sorted list of unique non-negative ints.
    """
    if specs is None:
        return None
    # Normalize to a comma-joined string, then split on commas
    if isinstance(specs, (list, tuple)):
        joined = ",".join(str(s) for s in specs)
    else:
        joined = str(specs)

    out: set[int] = set()

    def add_range_inclusive(a: int, b: int, step: int = 1):
        if step == 0:
            raise ValueError("step cannot be 0")
        if a <= b and step > 0:
            for i in range(a, b + 1, step):
                if i < 0: continue
                out.add(i)
        elif a >= b and step < 0:
            for i in range(a, b - 1, step):
                if i < 0: continue
                out.add(i)
        else:
            # empty range; ignore
            pass

    for token in filter(None, (t.strip() for t in joined.split(","))):
        # integer?
        if token.isdigit():
            out.add(int(token))
            continue

        # dash-form a-b (inclusive)
        if "-" in token and ":" not in token:
            parts = token.split("-", 1)
            if len(parts) == 2 and parts[0].strip().lstrip("-").isdigit() and parts[1].strip().lstrip("-").isdigit():
                a = int(parts[0]); b = int(parts[1])
                add_range_inclusive(a, b, 1 if a <= b else -1)
                continue
            else:
                raise ValueError(f"Invalid dash range: {token}")

        # slice-form a:b[:step] (end-exclusive, like Python)
        if ":" in token:
            s_parts = token.split(":")
            if not (1 <= len(s_parts) <= 3):
                raise ValueError(f"Invalid slice: {token}")
            a_str, b_str, *step_str = s_parts + [None] * (3 - len(s_parts))
            a = int(a_str) if (a_str is not None and a_str != "") else 0
            if b_str in (None, ""):
                raise ValueError(f"Slice end missing in: {token}")
            b = int(b_str)
            step = int(step_str[0]) if (step_str and step_str[0]) else 1
            # emulate python range(a, b, step)
            if step == 0:
                raise ValueError("step cannot be 0")
            # convert end-exclusive slice to inclusive range logic
            if step > 0:
                last = b - 1
                if last >= a:
                    add_range_inclusive(a, last, step)
            else:
                last = b + 1
                if last <= a:
                    add_range_inclusive(a, last, step)
            continue

        raise ValueError(f"Unrecognized channels token: '{token}'")

    return sorted(out)


def _find_event_for_npz(root_dir: str, npz_path: str) -> str | None:
    """Try several sensible locations/filenames for the matching .event or .txt file (prefers .event)."""
    stem = _stem(npz_path)
    rd_events = os.path.join(root_dir, "events")
    npz_dir = os.path.dirname(npz_path)

    exts = [".event", ".txt"]  # preference order

    def _first_with_exts(pattern_no_ext: str) -> str | None:
        """Given a pattern without extension (may include *), try with allowed extensions in order."""
        for ext in exts:
            # If the pattern already has a *, keep it; otherwise append the extension
            if "*" in pattern_no_ext:
                patt = pattern_no_ext + ext
            else:
                patt = pattern_no_ext + ext
            hits = sorted(glob.glob(patt))
            if hits:
                return hits[0]
        return None

    print(f"rd event dir {rd_events}, npz dir {npz_dir}, stem {stem}")

    # 1) <root>/events/<stem>_emg.(event|txt)
    cand = _first_with_exts(os.path.join(rd_events, f"{stem}_emg"))
    if cand:
        return cand

    # 2) <root>/events/<stem>.(event|txt)
    cand = _first_with_exts(os.path.join(rd_events, f"{stem}"))
    if cand:
        return cand

    # 3) <npz_parent>/../events/<stem>*.{event,txt}
    cand = _first_with_exts(os.path.join(os.path.dirname(npz_dir), "events", f"{stem}*"))
    if cand:
        return cand

    # 4) If only one *.(event|txt) under <root>/events, use it
    evs = []
    if os.path.isdir(rd_events):
        for ext in exts:
            evs.extend(glob.glob(os.path.join(rd_events, f"*{ext}")))
    evs = sorted(evs)
    if len(evs) == 1:
        return evs[0]

    # 5) Next to the NPZ: <npz_dir>/<stem>*.{event,txt}
    cand = _first_with_exts(os.path.join(npz_dir, f"{stem}*"))
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
    ignore_labels: list[str] | None = None,
    ignore_case: bool = False,
    overwrite: bool = False,
    keep_trial_label: bool = False,
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
        emg = data["amplifier_data"]    # (C, N)
        emg_t = data["t_amplifier"]     # (N,)
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

        # Labels
        #print(f"Finding events for NPZ: {npz_path}")
        #print("Looking in root dir:", root_dir)

        ev_path = _find_event_for_npz(root_dir, npz_path)
        if ev_path is None:
            logging.warning(f"[skip] No events found for {os.path.basename(npz_path)}")
            continue
        y = labels_from_events(ev_path, window_starts)

        if not keep_trial_label:
            y = np.asarray(
                [re.sub(r"_(\d+)$", "", lab) if isinstance(lab, str) else lab for lab in y],
                dtype=object,
            )

        if ignore_labels:
            if ignore_case:
                ignore_set = {s.lower() for s in ignore_labels}
                mask_ignore = np.array(
                    [((str(lab).lower() in ignore_set) if isinstance(lab, str) else False) for lab in y])
            else:
                ignore_set = set(ignore_labels)
                mask_ignore = np.array([((lab in ignore_set) if isinstance(lab, str) else False) for lab in y])
        else:
            mask_ignore = np.zeros_like(y, dtype=bool)

        # Filter out Unknown/Start and the user-ignored labels
        mask_builtin = ~np.isin(y, ["Unknown", "Start"])
        mask = mask_builtin & ~mask_ignore
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
    p.add_argument("--channels", nargs="+", type=str, default=None,
                   help='Channel indices/ranges, e.g. "0:128" or "0:128,130,200-215"')
    p.add_argument("--channel_map", type=str, default=None)
    p.add_argument("--channel_map_file", type=str, default="custom_channel_mappings.json")
    p.add_argument("--mapping_non_strict", action="store_true")
    p.add_argument("--ignore_labels", nargs="+", type=str, default=None, help='Labels to ignore (repeat or comma-separate). Example: --ignore_labels "Rest,Noise" Idle')
    p.add_argument("--ignore_case", action="store_true", help="Treat --ignore_labels case-insensitively.")
    p.add_argument("--window_ms", type=int, default=200)
    p.add_argument("--step_ms", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep_trial_label", action="store_true", help="Keep numeric trial suffix in labels (e.g., WristFlexion_14). Default: strip it.")
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    channels_parsed = parse_channels_spec(args.channels) if args.channels is not None else None

    def _split_labels(tokens):
        if tokens is None:
            return None
        out = []
        for t in tokens:
            out.extend([s.strip() for s in t.split(",") if s.strip() != ""])
        return out or None


    ignore_labels = _split_labels(args.ignore_labels)

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)

    build_training_dataset_multi_npz(
        root_dir=args.root_dir,
        label=args.label,
        save_path=args.save_path,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        channels=channels_parsed,
        channel_map=args.channel_map,
        channel_map_file=args.channel_map_file,
        mapping_non_strict=args.mapping_non_strict,
        ignore_labels=ignore_labels,
        ignore_case=args.ignore_case,
        overwrite=args.overwrite,
        keep_trial_label=args.keep_trial_label,
        verbose=args.verbose,
    )
