import os
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any, Sequence, Callable


def parse_event_file(event_files, verbose=False):
    """
    Extract all events from event file(s) and return a combined DataFrame.

    Args:
        event_files (str or list): Path(s) to the event file(s).
        verbose (bool): If True, print debug information.

    Returns:
        pd.DataFrame: DataFrame containing all events with columns:
            - 'sample_index': Sample index of the event (int)
            - 'timestamp': Timestamp string (str or None if missing)
            - 'label': Cleaned label string (str)
    """
    if isinstance(event_files, str):
        event_files = [event_files]

    all_events = []

    for file_path in event_files:
        if verbose:
            print(f"> Parsing event file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Check for header
            first_line = lines[0].strip().lower()
            has_header = any(h in first_line for h in ['timestamp', 'label', 'sample'])

            if has_header:
                lines = lines[1:]

            for line_num, line in enumerate(lines, start=2 if has_header else 1):
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                try:
                    parts = [p.strip() for p in line.split(',')]

                    if len(parts) < 2:
                        if verbose:
                            print(f"  Skipping malformed line {line_num}: {line}")
                        continue

                    sample_index = int(parts[0])
                    if len(parts) == 2:
                        timestamp = None
                        label = parts[1].split('#')[0].strip()
                    else:
                        timestamp = parts[1]
                        label = parts[2].split('#')[0].strip()

                    all_events.append({
                        'sample_index': sample_index,
                        'timestamp': timestamp,
                        'label': label
                    })

                except Exception as e:
                    if verbose:
                        print(f"  Error parsing line {line_num} in {file_path}: {e}")

        except Exception as e:
            if verbose:
                print(f"Failed to read {file_path}: {e}")

    return pd.DataFrame(all_events)


def parse_numeric_args(numeric_args, default_channels=[0, 1, 2, 3]):
    """
    Parses the numeric argument from the command line.
    Allows for:
      string "all"
      in list form: 0 1 2 3, 0:64, etc.
    """
    print(f"Received argument: {numeric_args}")
    if numeric_args is None:
        print("[Warning] No channels specified. Using default:", default_channels)
        return default_channels
    if len(numeric_args) == 1 and numeric_args[0].lower() == "all":
        return "all"
    elif len(numeric_args) == 1 and ":" in numeric_args[0]:
        # Support slice format, e.g. --channels 0:64
        start, end = map(int, numeric_args[0].split(":"))
        return list(range(start, end))
    else:
        try:
            return list(map(int, numeric_args))
        except ValueError:
            print("[Warning] Invalid argument. Using default:", default_channels)
            return default_channels


def convert_events_to_list(ev_path, window_starts, verbose=False):
    """
    Converts event file to a list of labels corresponding to the provided window starts.
    """
    events = parse_event_file(ev_path, verbose=verbose)
    events = events.sort_values('sample_index').reset_index(drop=True)
    y = []
    idx = 0
    for ws in window_starts:
        while idx + 1 < len(events) and events.loc[idx + 1, 'sample_index'] <= ws:
            idx += 1
        if len(events) == 0 or ws < events.loc[0, 'sample_index']:
            y.append('Unknown')
        else:
            new_label = events.loc[idx, 'label']
            #print(f"Window start {ws} assigned label '{new_label}' from event at sample {events.loc[idx, 'sample_index']}")
            y.append(new_label)
    y = np.array(y, dtype=str)
    if verbose:
        print(f"Converted {len(events)} events to {len(y)} labels.")
    return y


def lock_params_to_meta(meta: Dict, window_ms: Optional[int], step_ms: Optional[int],
                        selected_channels: Optional[List[int]]) -> Tuple[int, int, Optional[List[int]], float]:
    """Return (window_ms, step_ms, selected_channels, envelope_cut_hz) locked to training meta, if present."""
    win = int(meta.get("window_ms", window_ms or 200))
    stp = int(meta.get("step_ms",   step_ms   or 50))
    env = float(meta.get("envelope_cutoff_hz", 5.0))
    selected_channels = meta.get("selected_channels", None)
    return win, stp, selected_channels, env


def load_metadata_json(root_dir: str, label: str = "") -> dict:

    if root_dir.endswith(".json"):
        # If given a file path, load it directly
        with open(root_dir, "r", encoding="utf-8") as f:
            return json.load(f)

    model_dir = os.path.join(root_dir, "model")
    cand = os.path.join(model_dir, f"{label}_metadata.json") if label else None
    path = cand if (cand and os.path.isfile(cand)) else os.path.join(model_dir, "metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def normalize_name(s: str) -> str:
    # match CH1 vs ch_1 vs Ch01, etc.
    s = s.strip().upper()
    s = s.replace("_", "").replace("-", "")
    if s.startswith("CH") and s[2:].isdigit():
        return f"CH{int(s[2:])}"
    return s


def build_indices_from_mapping(raw_channel_names: list[str], mapping_names: list[str], *, strict: bool = True) -> list[int]:
    lookup = {normalize_name(n): i for i, n in enumerate(raw_channel_names)}
    indices = []
    missing = []
    for nm in mapping_names:
        key = normalize_name(nm)
        if key in lookup:
            indices.append(lookup[key])
        else:
            missing.append(nm)
    if strict and missing:
        raise ValueError(f"Channel mapping references names not present in recording: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return indices

def align_channels_by_name(
    emg: np.ndarray,
    source_names: Sequence[str],
    target_names: Sequence[str],
    *,
    normalizer: Callable[[str], str] = None,
    missing: str = "error",          # {"error","zero","nan"}
    duplicates: str = "first",       # {"error","first","last"}
    return_report: bool = True,
) -> Tuple[np.ndarray, List[int], Optional[Dict[str, Any]]]:
    """
    Reorder (C, N) EMG rows to match a target channel-name order.

    Parameters
    ----------
    emg : np.ndarray
        Array shaped (C, N) (channels x samples).
    source_names : Sequence[str]
        Names for rows of `emg` in their current order.
    target_names : Sequence[str]
        Desired channel-name order (e.g., training order).
    normalizer : Callable[[str], str], optional
        Function to normalize names before matching (e.g., strip, upper, remove punctuation).
        Defaults to pyoephys.io.normalize_name if available, else identity.
    missing : {"error","zero","nan"}, optional
        What to do when a target channel is not found in source:
          - "error": raise RuntimeError (strict).
          - "zero":  synthesize a zero-filled row.
          - "nan":   synthesize a NaN-filled row.
    duplicates : {"error","first","last"}, optional
        What to do when a source name appears more than once:
          - "error": raise RuntimeError.
          - "first": use the first occurrence.
          - "last":  use the last occurrence.
    return_report : bool, optional
        If True, return a dict with details about mapping/missing/duplicates.

    Returns
    -------
    aligned : np.ndarray
        EMG reordered to (len(target_names), N). If missing!="error", rows may be synthesized.
    indices : List[int]
        Source row indices used for each target (=-1 for synthesized rows).
    report : dict or None
        Keys: {"missing", "extras", "duplicates", "index_map", "used_indices"} (when return_report=True).

    Raises
    ------
    RuntimeError
        On missing channels (when missing="error") or duplicates (when duplicates="error").
    ValueError
        If shapes/lengths are inconsistent.
    """
    if emg.ndim != 2:
        raise ValueError(f"`emg` must be 2D (C,N); got shape {emg.shape}")
    C, N = emg.shape

    if len(source_names) != C:
        raise ValueError(f"len(source_names)={len(source_names)} != C={C}")

    if normalizer is None:
        # Fall back to identity if normalize_name isn't in scope.
        try:
            normalizer = normalize_name
        except Exception:
            normalizer = lambda s: s

    # Build normalized map from source names -> indices (handling duplicates per policy)
    norm_src = [normalize_name(s) for s in source_names]
    name_to_indices: Dict[str, List[int]] = {}
    for i, n in enumerate(norm_src):
        name_to_indices.setdefault(n, []).append(i)

    # Detect duplicates
    dupes = {n: idxs for n, idxs in name_to_indices.items() if len(idxs) > 1}
    if dupes and duplicates == "error":
        raise RuntimeError(f"Duplicate source channel names detected: { {k: v[:5] for k,v in dupes.items()} }")
    # Collapse duplicates based on policy
    idx_map: Dict[str, int] = {}
    for n, idxs in name_to_indices.items():
        if len(idxs) == 1:
            idx_map[n] = idxs[0]
        else:
            idx_map[n] = idxs[0] if duplicates == "first" else idxs[-1]

    # Align in target order
    norm_tgt = [normalizer(t) for t in target_names]
    aligned = np.empty((len(target_names), N), dtype=emg.dtype)
    indices: List[int] = []
    missing_list: List[str] = []

    for ti, (orig_name, nname) in enumerate(zip(target_names, norm_tgt)):
        if nname in idx_map:
            si = idx_map[nname]
            aligned[ti, :] = emg[si, :]
            indices.append(si)
        else:
            # handle missing
            if missing == "error":
                missing_list.append(orig_name)
            elif missing == "zero":
                aligned[ti, :] = 0
                indices.append(-1)
            elif missing == "nan":
                aligned[ti, :] = np.nan
                indices.append(-1)
            else:
                raise ValueError(f"Unknown `missing` policy: {missing}")

    if missing_list and missing == "error":
        raise RuntimeError(f"Recording is missing channels required by model: {missing_list}")

    extras = [source_names[i] for i, n in enumerate(norm_src) if n not in set(norm_tgt)]

    report = None
    if return_report:
        report = {
            "missing": missing_list,
            "extras": extras,
            "duplicates": dupes,
            "index_map": idx_map,        # normalized source name -> chosen source index
            "used_indices": indices,     # -1 for synthesized rows
        }

    return aligned, indices, report


# --- Back-compat wrapper mirroring your current helper's strict behavior ---
def select_training_channels_by_name(
    emg: np.ndarray,
    raw_names: Sequence[str],
    trained_names: Sequence[str],
) -> Tuple[np.ndarray, List[int]]:
    """
    Strict selection: reorder by name, missing/duplicates => errors.
    Matches old `_select_training_channels_by_name` semantics.
    """
    aligned, indices, _ = align_channels_by_name(
        emg,
        source_names=raw_names,
        target_names=trained_names,
        normalizer=None,     # use normalize_name if defined in this module; else identity
        missing="error",
        duplicates="error",
        return_report=False,
    )
    return aligned, indices


def trained_channel_names_from_meta(meta: dict) -> list[str]:
    """
    Pull training channel names out of metadata. We accept both:
      - meta["data"]["channel_names"] (preferred)
      - meta["channel_names"] (legacy)
    Returns [] if not found.
    """
    return list(meta.get("data", {}).get("channel_names") or meta.get("channel_names") or [])


def trained_channel_names_from_dataset_npz(root_dir: str, label: str | None = "") -> list[str]:
    """
    Fallback: look inside the training dataset NPZ for channel names.
    Tries label-specific first, then common defaults.
    """
    candidates = []
    if label:
        candidates.append(os.path.join(root_dir, f"{label}_training_dataset.npz"))
    candidates += [
        os.path.join(root_dir, "training_dataset.npz"),
        os.path.join(root_dir, "dataset_emg_windows.npz"),
        os.path.join(root_dir, "emg", "dataset_emg_windows.npz"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                with np.load(p, allow_pickle=True) as d:
                    if "channel_names" in d.files:
                        return list(d["channel_names"].tolist())
            except Exception:
                pass
    return []


def get_trained_channel_names(root_dir: str, label: str = "") -> list[str]:
    """
    High-level: load metadata then fallback to dataset NPZ.
    """
    meta = load_metadata_json(root_dir, label=label)
    names = trained_channel_names_from_meta(meta)
    if names:
        return names
    return trained_channel_names_from_dataset_npz(root_dir, label=label)