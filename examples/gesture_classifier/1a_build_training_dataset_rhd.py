#!/usr/bin/env python3
"""
Build a training dataset (X, y) from an OEBin recording + emg.event labels.

This script mirrors your current pipeline but with tighter structure & comments.
"""
import os
import json
import argparse
import logging
import numpy as np
from intan.io import load_rhd_file, load_config_file, labels_from_events, build_indices_from_mapping
from intan.processing import EMGPreprocessor, FEATURE_REGISTRY


def _parse_channels_arg(arg: str | None, total: int | None = None) -> list[int] | None:
    if not arg:
        return None
    s = arg.strip().lower()
    if s == "all":
        # fall back to total if known; otherwise None means “use all”
        return list(range(int(total))) if total is not None else None
    if ":" in s:
        a, b = s.split(":", 1)
        return list(range(int(a), int(b)))
    # allow comma or whitespace
    parts = [tok for piece in s.split(",") for tok in piece.split()]
    return [int(x) for x in parts if x != ""]

def build_training_dataset(root_dir: str, events_file: str | None = None, file_dir: str | None = None, label: str = "", save_path: str | None = None, window_ms: int = 200,
    step_ms: int = 50, channels: list[int] | None = None, channel_map: str | None = None,
    channel_map_file: str = "custom_channel_mappings.json", mapping_non_strict: bool = False, overwrite: bool = False,
    verbose: bool = False):

    # Default save path
    if save_path is None:
        name = f"{label}_training_dataset.npz" if label else "training_dataset.npz"
        save_path = os.path.join(root_dir, name)

    if os.path.exists(save_path) and not overwrite:
        logging.info(f"Dataset exists at {save_path}. Use --overwrite to regenerate.")
        return


    # Load raw EMG
    data = load_rhd_file(file_dir, verbose=verbose)
    emg_fs = float(data["frequency_parameters"]['amplifier_sample_rate'])
    emg = data["amplifier_data"]         # (C, N)
    emg_t = data["t_amplifier"]          # (N,)
    raw_channel_names = list(data.get("channel_names", [])) or [f"CH{i}" for i in range(emg.shape[0])]

    # Channel selection (by name mapping or by explicit indices)
    selected_channels = channels

    # If a channel_map is provided, it takes precedence over --channels
    map_name = channel_map
    if map_name:
        print(f"Using custom channel mapping for {map_name}...")
        map_path = channel_map_file
        if not os.path.isfile(map_path):
            raise FileNotFoundError(f"Mapping file not found: {map_path}")
        with open(map_path, "r", encoding="utf-8") as f:
            mapping_json = json.load(f)
        if map_name not in mapping_json:
            raise KeyError(
                f"Mapping '{map_name}' not found in {map_path}. Available: {list(mapping_json.keys())[:8]}...")
        mapping_names = list(mapping_json[map_name])
        print(f"Mapping names: {mapping_names[:8]}{'...' if len(mapping_names) > 8 else ''}")
        # Build indices in the exact mapped order
        selected_channels = build_indices_from_mapping(raw_channel_names, mapping_names, strict=(not mapping_non_strict))

    if selected_channels is not None:
        logging.info(f"Using {len(selected_channels)} selected channels.")
        emg = emg[selected_channels, :]
        selected_channel_names = [raw_channel_names[i] for i in selected_channels]
    else:
        selected_channel_names = raw_channel_names

    if verbose:
        print(f"Using selected channels: {selected_channel_names}")

    # Preprocess (matching your training defaults)
    pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=5.0, verbose=verbose)
    emg_pp = pre.preprocess(emg)

    # Feature extraction
    X = pre.extract_emg_features(emg_pp, window_ms=window_ms, step_ms=step_ms, progress=True,
        tqdm_kwargs={"desc": "Building dataset", "leave": False, "ascii": True},
    )

    # Compute window start indices (left edges in absolute sample index)
    start_index = int(round(emg_t[0] * emg_fs))
    step_samples = int(round(step_ms / 1000.0 * emg_fs))
    window_starts = np.arange(X.shape[0], dtype=int) * step_samples + start_index

    # Labels from events
    if events_file is None:
        events_name = f"{label}_emg.event" if label else "emg.event"
        print(f"No events_file provided, using: {events_name}")
        events_file = os.path.join(root_dir, "events", events_name)

    print(f"Loading events from: {events_file}")
    y = labels_from_events(events_file, window_starts)

    # Filter out Unknown/Start
    mask = ~np.isin(y, ["Unknown", "Start"])
    X, y, window_starts = X[mask], y[mask], window_starts[mask]

    if X.shape[0] != len(y):
        raise ValueError(f"Mismatch windows ({X.shape[0]}) vs labels ({len(y)})")

    # Extract feature information
    try:
        feature_spec = pre.feature_spec(n_channels=emg.shape[0])
    except Exception:
        names = FEATURE_REGISTRY.keys()
        feature_spec = {
            "per_channel": True,
            "order": names,
            "dims_per_feature": {n: 1 for n in names},
            "layout": "channel_major",
            "channels": "training_order",
            "n_channels": int(emg.shape[0]),
            "n_features_per_channel": len(names),
        }
    feature_spec_json = json.dumps(feature_spec)

    # Save compact npz with metadata
    np.savez(save_path, X=X,  y=y, emg_fs=emg_fs, window_ms=window_ms, step_ms=step_ms,
        selected_channels=selected_channels,
        channel_names=selected_channel_names,
        feature_spec=feature_spec_json,
        channel_mapping_name=np.array(map_name or "", dtype=object),
        channel_mapping_file=np.array(channel_map_file or "", dtype=object),
    )
    logging.info(f"Saved dataset: {save_path}")


if __name__ == "__main__":

    # Example terminal call:
    #
    # python 1_build_training_dataset.py \
    #   --root_dir "G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31" \
    #   --label sleeve_halfcount --channel_map sleeve_halfcount \
    #   --channel_map_file "G:\Shared drives\NML_shared\DataShare\HDEMG Human Healthy\HD-EMG_Cuff\Jonathan\2025_07_31\custom_channel_mappings.json" \
    #   --overwrite --verbose
    #

    p = argparse.ArgumentParser(description="Build EMG training dataset from OEBin + events.")
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--events_file", type=str, default=None, help="If different from root_dir/events")
    p.add_argument("--file_dir", type=str, required=True, help="If different from root_dir/raw")
    p.add_argument("--config_file", type=str, default=None)
    p.add_argument("--label", type=str, default="")
    #p.add_argument("--channels", nargs="+", type=int, default=None)
    p.add_argument("--channels", type=str, default=None, help="Channel list: 'all', '0:64', '0 1 2', or '0,1,2'")
    p.add_argument("--channel_map", type=str, default=None, help="Name inside the mapping JSON (e.g., sleeve_halfcount)")
    p.add_argument("--channel_map_file", type=str, default="custom_channel_mappings.json", help="Path to mapping JSON")
    p.add_argument("--mapping_non_strict", action="store_true", help="Allow missing names in mapping (skip them)")
    p.add_argument("--window_ms", type=int, default=200)
    p.add_argument("--step_ms", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_path", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    parsed_channels = _parse_channels_arg(args.channels)

    cfg = {}
    if args.config_file:
        cfg = load_config_file(args.config_file)
    cfg.update({
        "root_dir": args.root_dir or cfg.get("root_dir", ""),
        "events_file": args.events_file or cfg.get("events_file", None),
        "file_dir": args.file_dir or cfg.get("file_dir", None),
        "label": args.label or cfg.get("label", ""),
        "channels": parsed_channels if parsed_channels is not None else cfg.get("channels", None),
        "channel_map": args.channel_map or cfg.get("channel_map", None),
        "channel_map_file": args.channel_map_file or cfg.get("channel_map_file", "custom_channel_mappings.json"),
        "mapping_non_strict": args.mapping_non_strict or cfg.get("mapping_non_strict", False),
        "window_ms": args.window_ms or cfg.get("window_ms", 200),
        "step_ms": args.step_ms or cfg.get("step_ms", 50),
        "overwrite": args.overwrite or cfg.get("overwrite", False),
        "save_path": args.save_path or cfg.get("save_path", None),
        "verbose": args.verbose or cfg.get("verbose", False),
    })
    lvl = logging.DEBUG if cfg["verbose"] else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)
    build_training_dataset(**cfg)
