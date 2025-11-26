#!/usr/bin/env python3
"""
1e_build_training_dataset_multi_csv.py

Load CSV files from multiple recordings, extract features, align labels from event files,
and aggregate into a single training dataset NPZ.

Expected layout:

<root_dir>/
  raw/ or csv/       # CSV recordings
    recA.csv
    recB.csv
  events/
    recA_emg.event
    recB_emg.event

Examples
--------
# Export EMG+IMU for neutral recordings only
python 1e_build_training_dataset_multi_csv.py --root_dir /path/to/root \
  --csv_names all_gestures_wrist_neutral_emg_imu \
  --out_name orient_neutral --modality both --overwrite --verbose

# Export IMU-only features (rich) for pronated
python 1e_build_training_dataset_multi_csv.py --root_dir /path/to/root \
  --csv_names all_gestures_wrist_pronated_emg_imu.csv \
  --out_name orient_pronated --modality imu --imu_features rich --overwrite

# Export EMG-only with orientation remap (mirror) for supinated
python 1e_build_training_dataset_multi_csv.py --root_dir /path/to/root \
  --csv_names all_gestures_wrist_supinated_emg_imu \
  --out_name orient_supinated --modality emg \
  --channel_map 8-8-L --orientation auto --orientation_remap mirror --overwrite
"""

from __future__ import annotations
import os, json, glob, argparse, logging
from typing import List, Dict, Tuple, Optional
import numpy as np

# Your package APIs
from intan.io import load_csv_files, labels_from_events, build_indices_from_mapping
from intan.processing import EMGPreprocessor, FEATURE_REGISTRY


# -------------------------
# Helpers: orientation/grid
# -------------------------
def infer_grid_from_mapping(mapping_names: list[str]) -> tuple[int, int] | tuple[None, None]:
    """Infer grid size from mapping names. Defaults to 8x8 if there are 64 channels."""
    n = len(mapping_names)
    if n == 64:
        return 8, 8
    return None, None


def apply_grid_permutation(idx: list[int], nrows: int, ncols: int, mode: str) -> list[int]:
    """Return new index order after a spatial transform on an (nrows x ncols) grid."""
    if mode == "none" or nrows is None:
        return idx
    grid = np.array(idx, dtype=int).reshape(nrows, ncols)
    if mode == "mirror":         # left↔right
        grid2 = grid[:, ::-1]
    elif mode == "rotate90":     # 90° clockwise
        grid2 = np.rot90(grid, k=3)
    else:
        grid2 = grid
    return grid2.reshape(-1).tolist()


def orientation_from_filename(path: str) -> str | None:
    s = os.path.basename(path).lower()
    for o in ("neutral", "pronated", "supinated"):
        if o in s:
            return o
    return None


# -------------------------
# Misc helpers
# -------------------------
def _basename(p: str) -> str:
    return os.path.basename(os.fspath(p))


def _matches_any_name(rec_path: str, names: Optional[List[str]]) -> bool:
    """True if rec_path's basename matches any item in names (supports stem-only)."""
    if not names:
        return True
    b = _basename(rec_path)
    stem = os.path.splitext(b)[0]
    targets = set(n.strip() for n in names if n and n.strip())
    return (b in targets) or (stem in targets)


def _stem(path):
    if hasattr(path, "item"):
        try:
            path = path.item()
        except Exception:
            pass
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    if "_" in stem:
        stem = stem.split("_")[0]
    return stem


def _first(path_glob: str | None) -> str | None:
    matches = sorted(glob.glob(path_glob)) if path_glob else []
    return matches[0] if matches else None


def _find_event_for(root_dir: str, rec_path: str) -> str | None:
    stem = _stem(rec_path)
    rd_events = os.path.join(root_dir, "events")
    rec_dir = os.path.dirname(rec_path)

    # 1) <root>/events/<stem>_emg.event
    cand = os.path.join(rd_events, f"{stem}_emg.event")
    if os.path.isfile(cand):
        return cand

    # 2) <root>/events/<stem>.event
    cand = os.path.join(rd_events, f"{stem}.event")
    if os.path.isfile(cand):
        return cand

    # 3) <rec_parent>/../events/<stem>*.event
    cand = _first(os.path.join(os.path.dirname(rec_dir), "events", f"{stem}*.event"))
    if cand:
        return cand

    # 4) Unique event in <root>/events
    evs = sorted(glob.glob(os.path.join(rd_events, "*.event"))) if os.path.isdir(rd_events) else []
    if len(evs) == 1:
        return evs[0]

    # 5) Next to the recording
    cand = _first(os.path.join(rec_dir, f"{stem}*.event"))
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


# -------------------------
# IMU feature aggregation
# -------------------------
def _aggregate_imu_over_windows(
    board_adc_data: Optional[np.ndarray],
    board_adc_channels: Optional[List[Dict[str, str]]],
    start_indices: np.ndarray,
    window_samples: int,
    mode: str = "rich",  # "mean" (legacy) or "rich"
) -> Tuple[Optional[np.ndarray], List[str]]:
    if board_adc_data is None or board_adc_channels is None or len(board_adc_channels) == 0:
        return None, []

    imu_cols = [c.get("channel_name", f"ADC{i}") for i, c in enumerate(board_adc_channels)]
    L = board_adc_data.shape[1]
    feats = []
    headers = []

    # reducers
    def feat_mean(seg):   return np.nanmean(seg, axis=1)
    if mode == "mean":
        reducers = [("mean", feat_mean)]
    else:
        def feat_std(seg):    return np.nanstd(seg, axis=1)
        def feat_energy(seg): return np.nanmean(seg**2, axis=1)
        def feat_diffm(seg):
            d = np.diff(seg, axis=1)
            return np.nanmean(np.abs(d), axis=1)
        reducers = [("mean", feat_mean), ("std", feat_std), ("energy", feat_energy), ("diff", feat_diffm)]

    for s in start_indices:
        e = s + window_samples
        s0 = int(max(0, s))
        e0 = int(min(L, max(s0, e)))
        if e0 <= s0:
            feats.append([float("nan")] * (board_adc_data.shape[0] * len(reducers)))
        else:
            seg = board_adc_data[:, s0:e0]
            fwin = [r(seg) for _, r in reducers]
            feats.append(np.concatenate(fwin, axis=0))

    for stat_name, _ in reducers:
        headers.extend([f"{col}_{stat_name}" for col in imu_cols])

    return np.asarray(feats, dtype=float), headers


def _append_imu_features(
    X: np.ndarray,
    y: np.ndarray,
    imu_win: Optional[np.ndarray],
    imu_cols: List[str],
    label_mask: np.ndarray,
    imu_norm: str = "zscore",
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    meta = {
        "imu_cols": np.array(imu_cols or [], dtype=object),
        "imu_feature_count": np.array(len(imu_cols or []), dtype=np.int32),
        "imu_norm_kind": np.array(imu_norm, dtype=object),
        "imu_norm_mean": np.array([], dtype=np.float32),
        "imu_norm_std":  np.array([], dtype=np.float32),
    }

    if imu_win is None or len(imu_cols) == 0:
        return X, meta

    imu_win = imu_win[label_mask]
    if imu_win.shape[0] != X.shape[0]:
        raise RuntimeError("IMU rows after mask do not match X rows.")

    if imu_norm == "zscore":
        mu = np.nanmean(imu_win, axis=0).astype(np.float32)
        sd = np.nanstd(imu_win, axis=0).astype(np.float32)
        sd[sd < eps] = 1.0
        imu_win = (imu_win - mu) / sd
        meta["imu_norm_mean"] = mu
        meta["imu_norm_std"]  = sd
    elif imu_norm == "robust":
        med = np.nanmedian(imu_win, axis=0).astype(np.float32)
        q75 = np.nanpercentile(imu_win, 75, axis=0).astype(np.float32)
        q25 = np.nanpercentile(imu_win, 25, axis=0).astype(np.float32)
        iqr = q75 - q25
        iqr[iqr < eps] = 1.0
        imu_win = (imu_win - med) / iqr
        meta["imu_norm_kind"] = np.array("robust", dtype=object)

    X_aug = np.hstack([X, imu_win])
    return X_aug, meta


# -------------------------
# Main builder
# -------------------------
def build_training_dataset_multi_any(
    root_dir: str,
    label: str = "",
    save_path: str | None = None,
    csv_names: Optional[List[str]] = None,
    out_name: Optional[str] = None,
    window_ms: int = 200,
    step_ms: int = 50,
    imu_norm: str = "zscore",
    channels: list[int] | None = None,
    channel_map: str | None = None,
    channel_map_file: str = "custom_channel_mappings.json",
    mapping_non_strict: bool = False,
    overwrite: bool = False,
    modality: str = "both",
    orientation: str = "auto",                 # NEW
    orientation_remap: str = "none",           # NEW
    imu_features: str = "rich",                # NEW
    verbose: bool = False,
):
    # Load CSVs
    rec_items = load_csv_files(root_dir, verbose=verbose)
    if not rec_items:
        raise FileNotFoundError(f"No recordings found under {root_dir}/emg or {root_dir}/csv (or raw).")

    # Filter by explicit CSV list
    if csv_names:
        before = len(rec_items)
        rec_items = [(d, p) for (d, p) in rec_items if _matches_any_name(p, csv_names)]
        after = len(rec_items)
        if verbose:
            logging.info(f"[filter] Using csv_names={csv_names} → kept {after}/{before} recordings")
        if after == 0:
            raise FileNotFoundError(f"No recordings matched csv_names={csv_names} under {root_dir}.")

    # Output path
    if save_path is None:
        base = []
        if label:
            base.append(label)
        if out_name:
            base.append(out_name)
        base_str = "_".join(base) if base else "training_dataset"
        if not out_name or modality not in out_name:
            base_str = f"{base_str}_{modality}"
        save_path = os.path.join(root_dir, f"{base_str}_training_dataset.npz")
    if os.path.exists(save_path) and not overwrite:
        logging.info(f"Dataset already exists: {save_path}. Use --overwrite to remake.")
        return

    combined_X = []
    combined_y = []
    imu_header = None
    selected_channels = None
    selected_channel_names = None
    fs_values = set()

    # If we will remap, load mapping names once
    mapping_names = None
    nrows = ncols = None
    if channel_map:
        if not os.path.isfile(channel_map_file):
            raise FileNotFoundError(f"Mapping file not found: {channel_map_file}")
        with open(channel_map_file, "r", encoding="utf-8") as f:
            mapping_json = json.load(f)
        if channel_map not in mapping_json:
            raise KeyError(f"Mapping '{channel_map}' not in {channel_map_file}")
        mapping_names = list(mapping_json[channel_map])
        nrows, ncols = infer_grid_from_mapping(mapping_names)

    for i, (data, rec_path) in enumerate(rec_items, 1):
        if verbose:
            logging.info(f"[{i}/{len(rec_items)}] using {os.path.basename(rec_path)}")

        # Orientation for this recording
        ori = orientation if orientation != "auto" else (orientation_from_filename(rec_path) or "neutral")

        emg = data["amplifier_data"]     # (C, N)
        emg_t = data.get("t_amplifier")  # (N,)
        emg_fs = float(data.get("frequency_parameters", {}).get("amplifier_sample_rate", data.get("sample_rate", 0.0)))
        fs_values.add(round(emg_fs, 6))
        raw_names = list(data.get("channel_names", [])) or [f"CH{j}" for j in range(emg.shape[0])]

        # Events (label source)
        ev_path = _find_event_for(root_dir, rec_path)
        if ev_path is None:
            logging.warning(f"[skip] No events found for {os.path.basename(rec_path)}")
            continue

        # Select channels once
        if selected_channels is None:
            selected_channels, selected_channel_names = _compute_selected_channels(
                raw_names, channels, channel_map, channel_map_file, mapping_non_strict
            )
            if verbose:
                logging.info(f" Selected EMG channels: {selected_channel_names}")

        # Apply selection
        if selected_channels is not None:
            emg = emg[selected_channels, :]

        # Orientation remap (align non-neutral to neutral reference)
        if orientation_remap != "none" and (ori in ("pronated", "supinated")) and (nrows is not None):
            base_idx = list(range(emg.shape[0]))  # after selection
            perm_idx = apply_grid_permutation(base_idx, nrows, ncols, orientation_remap)
            emg = emg[perm_idx, :]
            if selected_channel_names:
                # Reorder names to match the remap
                selected_channel_names = [selected_channel_names[j] for j in perm_idx]

        # Preprocess & EMG features
        pre = EMGPreprocessor(fs=emg_fs, envelope_cutoff=5.0, verbose=verbose)
        emg_pp = pre.preprocess(emg)
        X_emg = pre.extract_emg_features(
            emg_pp, window_ms=window_ms, step_ms=step_ms,
            progress=verbose, tqdm_kwargs={"desc": f"Features {i}/{len(rec_items)}", "leave": False}
        )

        # Windowing indices
        start_index = int(round((emg_t[0] if emg_t is not None else 0.0) * emg_fs))
        step_samples = int(round(step_ms / 1000.0 * emg_fs))
        window_starts = np.arange(X_emg.shape[0], dtype=int) * step_samples + start_index
        window_samples = int(round(window_ms / 1000.0 * emg_fs))

        # Labels per window
        y = labels_from_events(ev_path, window_starts)

        # Filter out Unknown/Start
        mask = ~np.isin(y, ["Unknown", "Start"])
        X_emg, y = X_emg[mask], y[mask]
        if X_emg.shape[0] == 0:
            logging.warning(f"[skip] No labeled windows for {os.path.basename(rec_path)}")
            continue

        # IMU features (optional)
        board_adc_data = data.get("board_adc_data")
        board_adc_channels = data.get("board_adc_channels")
        X_out = None

        if modality == "imu":
            imu_win, imu_cols = _aggregate_imu_over_windows(
                board_adc_data=board_adc_data,
                board_adc_channels=board_adc_channels,
                start_indices=window_starts,
                window_samples=window_samples,
                mode=imu_features,
            )
            if imu_win is None or len(imu_cols) == 0:
                raise RuntimeError(f"Requested modality=imu but no IMU columns found in {os.path.basename(rec_path)}.")
            X_dummy = np.zeros((mask.sum(), 0), dtype=float)
            X_out, imu_meta = _append_imu_features(
                X=X_dummy, y=y, imu_win=imu_win, imu_cols=imu_cols, label_mask=mask, imu_norm=imu_norm
            )

        elif modality == "both":
            imu_win, imu_cols = _aggregate_imu_over_windows(
                board_adc_data=board_adc_data,
                board_adc_channels=board_adc_channels,
                start_indices=window_starts,
                window_samples=window_samples,
                mode=imu_features,
            )
            X_out, imu_meta = _append_imu_features(
                X=X_emg, y=y, imu_win=imu_win, imu_cols=imu_cols, label_mask=mask, imu_norm=imu_norm
            )
        else:  # emg
            X_out, imu_meta = _append_imu_features(
                X=X_emg, y=y, imu_win=None, imu_cols=[], label_mask=mask, imu_norm=imu_norm
            )

        if imu_header is None and imu_meta["imu_feature_count"] > 0:
            imu_header = imu_meta["imu_cols"].tolist()

        combined_X.append(X_out)
        combined_y.append(y)

    if not combined_X:
        print("No usable labeled windows across recordings. Exiting")
        return

    # Warn if sample rates differed
    if len(fs_values) > 1:
        logging.warning(f"Multiple sample rates detected across recordings: {sorted(fs_values)}. "
                        f"Consider resampling upstream for strict consistency.")

    X = np.concatenate(combined_X, axis=0)
    y = np.concatenate(combined_y, axis=0)

    # Feature spec (best effort)
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

    # Labels
    class_names_sorted = sorted(set(y.tolist()))
    label_to_id = {c: i for i, c in enumerate(class_names_sorted)}

    # Save
    np.savez(
        save_path,
        X=X, y=y,
        emg_fs=float(fs_values.pop()) if len(fs_values) == 1 else 0.0,
        class_names=np.array(class_names_sorted, dtype=object),
        label_to_id_json=np.array(json.dumps(label_to_id), dtype=object),
        window_ms=window_ms, step_ms=step_ms,
        selected_channels=np.array(selected_channels if selected_channels else [], dtype=int),
        channel_names=np.array(selected_channel_names if selected_channel_names else [], dtype=object),
        feature_spec=feature_spec_json,
        channel_mapping_name=np.array(channel_map or "", dtype=object),
        channel_mapping_file=np.array(channel_map_file or "", dtype=object),
        imu_cols=np.array(imu_header or [], dtype=object),
        imu_feature_count=np.array(len(imu_header or []), dtype=np.int32),
        modality=np.array(modality, dtype=object),
        orientation=np.array(orientation, dtype=object),
        orientation_remap=np.array(orientation_remap, dtype=object),
        imu_feature_mode=np.array(imu_features, dtype=object),
    )
    logging.info(f"Saved aggregated dataset: {save_path}  (X={X.shape}, y={y.shape})")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Aggregate MANY CSV recordings into one training dataset.")
    p.add_argument("--root_dir", type=str, required=True,
                   help="Root containing 'raw/' or 'csv/' recordings, and 'events/'.")
    p.add_argument("--label", type=str, default="", help="Unique label to assign to dataset")
    p.add_argument("--channels", nargs="+", type=int, default=None, help="Channel list: indices, e.g. 0 1 2")
    p.add_argument("--channel_map", type=str, default=None, help="Mapping name in JSON (e.g., '8-8-L')")
    p.add_argument("--channel_map_file", type=str, default="custom_channel_mappings.json")
    p.add_argument("--mapping_non_strict", action="store_true", help="Allow missing names in mapping (skip them)")
    p.add_argument("--window_ms", type=int, default=200, help="Feature window size in ms.")
    p.add_argument("--step_ms", type=int, default=50, help="Feature step size in ms.")
    p.add_argument("--imu_norm", type=str, choices=["zscore", "robust", "none"], default="zscore",
                   help="Normalize IMU columns in X (zscore/robust) or pass through (none).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset at save_path.")
    p.add_argument("--save_path", type=str, default=None, help="Path to save the aggregated dataset NPZ.")

    # NEW: I/O selection
    p.add_argument("--csv_names", nargs="+", default=None,
                   help="One or more CSV basenames (with or without .csv) to include. "
                        "Example: --csv_names all_gestures_wrist_neutral_emg_imu.csv")
    p.add_argument("--out_name", type=str, default=None,
                   help="Suffix for the output dataset filename, e.g., 'orient_neutral'.")

    # NEW: variants
    p.add_argument("--modality", type=str, choices=["both", "emg", "imu"], default="both",
                   help="Feature set to export: EMG only, IMU only, or EMG+IMU.")
    p.add_argument("--orientation", type=str, choices=["neutral", "pronated", "supinated", "auto"], default="auto",
                   help="Orientation tag for this recording (or 'auto' to infer from filename).")
    p.add_argument("--orientation_remap", type=str, choices=["none", "mirror", "rotate90"], default="none",
                   help="Spatial remap applied to EMG channels so orientations align (applied to non-neutral).")
    p.add_argument("--imu_features", type=str, choices=["mean", "rich"], default="rich",
                   help="IMU feature mode: 'mean' (legacy) or 'rich' (stats + deltas).")

    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = p.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=lvl)

    build_training_dataset_multi_any(
        root_dir=args.root_dir,
        label=args.label,
        save_path=args.save_path,
        csv_names=args.csv_names,
        out_name=args.out_name,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        imu_norm=args.imu_norm,
        channels=args.channels,
        channel_map=args.channel_map,
        channel_map_file=args.channel_map_file,
        mapping_non_strict=args.mapping_non_strict,
        overwrite=args.overwrite,
        modality=args.modality,
        orientation=args.orientation,
        orientation_remap=args.orientation_remap,
        imu_features=args.imu_features,
        verbose=args.verbose,
    )
