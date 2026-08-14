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
    Parse a channel argument from the command line.

    Accepts the string ``"all"``, integer lists, or a single slice-like
    value such as ``"0:64"``.
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
    Pull training channel names from metadata.

    The nested ``meta["data"]["channel_names"]`` location is preferred;
    ``meta["channel_names"]`` is retained for compatibility. Returns an
    empty list if neither location is present.
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


# =============================================================================
# Channel Parsing Utilities
# =============================================================================

def parse_channel_spec(spec, total: int = None) -> List[int]:
    """
    Parse a flexible channel specification into a list of indices.
    
    This function handles multiple common formats for specifying channels
    from command line arguments or configuration files.
    
    Supported formats:
        - None or ""           → None (use all channels)
        - "all"                → list(range(total)) if total provided, else None
        - "0:64"               → [0, 1, ..., 63] (end-exclusive, like Python slice)
        - "0:128:2"            → [0, 2, 4, ..., 126] (with step)
        - "0 1 2 3"            → [0, 1, 2, 3] (space-separated)
        - "0,1,2,3"            → [0, 1, 2, 3] (comma-separated)
        - "0-7"                → [0, 1, ..., 7] (inclusive dash range)
        - "0:32,64,70-75"      → mixed formats combined
        - ["0:32", "64"]       → list input also accepted
    
    Args:
        spec: Channel specification string, list of strings, or None
        total: Total channel count (used for "all" keyword)
    
    Returns:
        Sorted list of unique channel indices, or None if no spec given
    
    Examples:
        >>> parse_channel_spec("0:8")
        [0, 1, 2, 3, 4, 5, 6, 7]
        
        >>> parse_channel_spec("1-4")
        [1, 2, 3, 4]
        
        >>> parse_channel_spec("0:8,16,20-22")
        [0, 1, 2, 3, 4, 5, 6, 7, 16, 20, 21, 22]
        
        >>> parse_channel_spec("all", total=128)
        [0, 1, 2, ..., 127]
    """
    import re
    
    # Handle None or empty
    if spec is None:
        return None
    
    # Convert list input to comma-separated string
    if isinstance(spec, (list, tuple)):
        spec = ",".join(str(s) for s in spec)
    
    spec = str(spec).strip()
    if not spec:
        return None
    
    # Handle "all" keyword
    if spec.lower() == "all":
        return list(range(int(total))) if total is not None else None
    
    # Normalize: replace whitespace with commas
    normalized = re.sub(r'\s+', ',', spec)
    
    # Parse tokens
    out: set = set()
    
    for token in normalized.split(','):
        token = token.strip()
        if not token:
            continue
        
        # Pure integer
        if token.lstrip('-').isdigit():
            out.add(int(token))
            continue
        
        # Slice notation "a:b" or "a:b:step" (end-exclusive)
        if ":" in token:
            parts = token.split(":")
            try:
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if len(parts) > 1 and parts[1] else None
                step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
                
                if end is not None and step != 0:
                    out.update(range(start, end, step))
            except ValueError:
                pass  # Skip malformed
            continue
        
        # Dash range "a-b" (inclusive)
        if "-" in token and token[0] != '-':
            parts = token.split("-", 1)
            if len(parts) == 2:
                try:
                    a, b = int(parts[0]), int(parts[1])
                    step = 1 if a <= b else -1
                    out.update(range(a, b + step, step))
                except ValueError:
                    pass
            continue
    
    return sorted(out) if out else None


def load_channel_mapping(
    mapping_name: str,
    mapping_file: str,
) -> List[str]:
    """
    Load a named channel mapping from a JSON file.
    
    The JSON file should contain a dict of named mappings, each being
    a list of channel names in the desired order.
    
    Args:
        mapping_name: Key in the mapping JSON (e.g., "sleeve_halfcount")
        mapping_file: Path to the JSON file
    
    Returns:
        List of channel names in mapped order
    
    Raises:
        FileNotFoundError: If mapping file doesn't exist
        KeyError: If mapping name not found in file
    
    Example JSON file::

        {
          "sleeve_halfcount": ["A-001", "A-002", "A-003", "B-001"]
        }
    """
    if not os.path.isfile(mapping_file):
        raise FileNotFoundError(f"Channel mapping file not found: {mapping_file}")
    
    with open(mapping_file, "r", encoding="utf-8") as f:
        mapping_json = json.load(f)
    
    if mapping_name not in mapping_json:
        available = list(mapping_json.keys())[:8]
        suffix = "..." if len(mapping_json) > 8 else ""
        raise KeyError(
            f"Mapping '{mapping_name}' not found in {mapping_file}. "
            f"Available: {available}{suffix}"
        )
    
    return list(mapping_json[mapping_name])


def resolve_channel_selection(
    raw_channel_names: List[str],
    channels: List[int] = None,
    channel_map: str = None,
    channel_map_file: str = "custom_channel_mappings.json",
    strict: bool = True,
) -> Tuple[Optional[List[int]], List[str]]:
    """
    Resolve channel selection from either explicit indices or named mapping.
    
    Priority: channel_map > channels > all channels
    
    Args:
        raw_channel_names: All channel names from the data source
        channels: Explicit channel indices, or None
        channel_map: Name of mapping in JSON file, or None
        channel_map_file: Path to mapping JSON
        strict: If True, raise error for missing channels; else skip them
    
    Returns:
        Tuple of (selected_indices, selected_names)
        If no selection specified, returns (None, raw_channel_names)
    
    Example:
        >>> names = ["A-001", "A-002", "A-003", "B-001", "B-002"]
        >>> resolve_channel_selection(names, channels=[0, 2, 4])
        ([0, 2, 4], ['A-001', 'A-003', 'B-002'])
    """
    # Channel mapping takes priority
    if channel_map:
        mapping_names = load_channel_mapping(channel_map, channel_map_file)
        selected_indices = build_indices_from_mapping(
            raw_channel_names, 
            mapping_names, 
            strict=strict
        )
        selected_names = [raw_channel_names[i] for i in selected_indices]
        return selected_indices, selected_names
    
    # Explicit channel indices
    if channels is not None:
        # Validate indices
        max_idx = len(raw_channel_names) - 1
        valid = [i for i in channels if 0 <= i <= max_idx]
        if strict and len(valid) < len(channels):
            invalid = set(channels) - set(valid)
            raise ValueError(f"Channel indices out of range: {invalid}")
        selected_names = [raw_channel_names[i] for i in valid]
        return valid, selected_names
    
    # No selection = use all
    return None, list(raw_channel_names)


def normalize_channel_name_1based(name: str) -> str:
    """
    Normalize Intan channel names to 1-based format (A-001, B-002, etc.).
    
    Handles various input formats:
        - "A-000" → "A-001" (0-based to 1-based)
        - "a_0" → "A-001"
        - "B-17" → "B-018" (zero-padded, assumes 0-based input)
        - "b 5" → "B-006"
    
    Args:
        name: Channel name string
    
    Returns:
        Normalized channel name in "X-NNN" format (1-based)
    
    Example:
        >>> normalize_channel_name_1based("a-000")
        'A-001'
    """
    import re
    
    # Pattern to match port letter and number
    pattern = re.compile(r'^([A-Da-d])\s*[-_ ]?\s*(\d{1,3})$')
    match = pattern.match(str(name).strip())
    
    if not match:
        return str(name)  # Return unchanged if doesn't match expected format
    
    port = match.group(1).upper()
    num = int(match.group(2))
    
    # Always add 1 to ensure 1-based output (assuming 0-based input)
    return f"{port}-{num + 1:03d}"


def parse_channels_spec(specs) -> list[int] | None:
    """
    Parse channel specification from CLI arguments.
    
    Accepts:
    - Single indices: 5 12
    - Python slice: 0:128, 0:128:2, :64
    - Dash ranges: 1-8 (inclusive)
    - Comma-separated: 0:64,70,75-80
    
    Returns sorted list of unique channel indices, or None if specs is None.
    
    Examples
    --------
    >>> parse_channels_spec("0:64")
    [0, 1, 2, ..., 63]
    >>> parse_channels_spec("5,10,15-20")
    [5, 10, 15, 16, 17, 18, 19, 20]
    >>> parse_channels_spec(["0:64", "100-110"])
    [0, 1, ..., 63, 100, 101, ..., 110]
    """
    if specs is None:
        return None
    
    # Normalize to comma-joined string
    if isinstance(specs, (list, tuple)):
        joined = ",".join(str(s) for s in specs)
    else:
        joined = str(specs)
    
    out: set[int] = set()
    
    def add_range_inclusive(a: int, b: int, step: int = 1):
        """Add inclusive range [a, b] with optional step."""
        if step > 0:
            for i in range(a, b + 1, step):
                out.add(i)
        else:
            for i in range(a, b - 1, -1):
                out.add(i)
    
    for token in filter(None, (t.strip() for t in joined.split(","))):
        token = token.strip()
        
        # Python slice notation: a:b or a:b:step
        if ":" in token:
            parts = token.split(":")
            if len(parts) == 2:
                a_str, b_str = parts
                a = int(a_str) if a_str else 0
                b = int(b_str)
                # Python convention: end-exclusive
                for i in range(a, b):
                    out.add(i)
            elif len(parts) == 3:
                a_str, b_str, step_str = parts
                a = int(a_str) if a_str else 0
                b = int(b_str)
                step = int(step_str) if step_str else 1
                for i in range(a, b, step):
                    out.add(i)
        
        # Dash range: a-b (inclusive)
        elif "-" in token and not token.startswith("-"):
            parts = token.split("-", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                a, b = int(parts[0]), int(parts[1])
                add_range_inclusive(a, b)
        
        # Single index
        else:
            out.add(int(token))
    
    return sorted(out) if out else None


def find_event_for_file(root_dir: str, data_path: str) -> str | None:
    """
    Find corresponding event file for a data file using multiple search strategies.
    
    Search order:
    1. <root>/events/<stem>_emg.event or .txt
    2. <root>/events/<stem>.event or .txt
    3. Recursive search in events/ for <stem>*
    4. If only one event file in events/, use it
    5. Next to the data file
    
    Parameters
    ----------
    root_dir : str
        Root directory containing events/ folder
    data_path : str
        Path to data file
    
    Returns
    -------
    str or None
        Path to event file if found, None otherwise
    """
    import re
    import glob
    
    # Extract stem, removing date/time patterns
    stem = os.path.splitext(os.path.basename(str(data_path)))[0]
    stem = re.sub(r'_\d{6}_\d{6}$', '', stem)  # Remove _YYMMDD_HHMMSS
    stem = re.sub(r'_\d{6}$', '', stem)         # Remove _YYMMDD
    stem = re.sub(r'_\d+$', '', stem)           # Remove _14, _02, etc.
    
    rd_events = os.path.join(root_dir, "events")
    data_dir = os.path.dirname(data_path)
    
    exts = [".event", ".txt"]
    
    def _first_with_exts(pattern_no_ext: str) -> str | None:
        for ext in exts:
            if os.path.isfile(pattern_no_ext + ext):
                return pattern_no_ext + ext
        # Try glob for wildcard patterns
        for ext in exts:
            matches = sorted(glob.glob(pattern_no_ext + ext))
            if matches:
                return matches[0]
        return None
    
    # 1) <root>/events/<stem>_emg.event
    cand = _first_with_exts(os.path.join(rd_events, f"{stem}_emg"))
    if cand:
        return cand
    
    # 2) <root>/events/<stem>.event
    cand = _first_with_exts(os.path.join(rd_events, f"{stem}"))
    if cand:
        return cand
    
    # 3) Recursive search in events/
    if os.path.isdir(rd_events):
        for ext in exts:
            matches = sorted(glob.glob(
                os.path.join(rd_events, "**", f"{stem}*{ext}"),
                recursive=True
            ))
            if matches:
                return matches[0]
    
    # 4) If only one event file under events/, use it
    evs = []
    if os.path.isdir(rd_events):
        for ext in exts:
            evs.extend(glob.glob(os.path.join(rd_events, f"*{ext}")))
    if len(evs) == 1:
        return evs[0]
    
    # 5) Next to the data file
    cand = _first_with_exts(os.path.join(data_dir, f"{stem}*"))
    if cand:
        return cand
    
    return None


def file_stem(path: str) -> str:
    """
    Extract base name from file, removing date/time/trial suffixes.
    
    Removes common suffixes like:
    - _YYMMDD_HHMMSS (timestamp)
    - _YYMMDD (date)
    - _N (trial number)
    
    Args:
        path: File path to extract stem from
        
    Returns:
        str: Base filename without extension or suffixes
        
    Examples:
        >>> file_stem("data_230615_143022.rhd")
        'data'
        >>> file_stem("gesture_5.rhd")
        'gesture'
    """
    import re
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    stem = re.sub(r'_\d{6}_\d{6}$', '', stem)  # _YYMMDD_HHMMSS
    stem = re.sub(r'_\d{6}$', '', stem)         # _YYMMDD
    stem = re.sub(r'_\d+$', '', stem)           # _14, _02
    return stem


def discover_and_group_files(
    root_dir: str,
    file_type: str,
    file_names: List[str] | None = None,
    exclude_pattern: str | None = None,
    merge_pattern: str | None = None,
) -> dict[str, List[str]]:
    """
    Discover files and group by stem for multi-part recordings.
    
    Args:
        root_dir: Root directory to search
        file_type: Type of files to search for ('rhd', 'npz', 'csv')
        file_names: Optional list of specific filenames to filter
        exclude_pattern: Pattern to exclude from file stems
        merge_pattern: Pattern that must be in file stems to include
        
    Returns:
        dict: Mapping of file stems to lists of file paths
        
    Example:
        >>> groups = discover_and_group_files("/data", "rhd")
        >>> groups
        {'recording_1': ['recording_1_part1.rhd', 'recording_1_part2.rhd'],
         'recording_2': ['recording_2.rhd']}
    """
    import glob
    
    ext = {"rhd": ".rhd", "npz": ".npz", "csv": ".csv"}[file_type]
    search_dirs = {"rhd": ["raw", ""], "npz": ["emg", ""], "csv": ["csv", "raw", ""]}[file_type]
    
    all_files = []
    for sdir in search_dirs:
        all_files.extend(glob.glob(os.path.join(root_dir, sdir, "**", f"*{ext}"), recursive=True))
    all_files = sorted(set(all_files))
    
    if not all_files:
        raise FileNotFoundError(f"No {ext} files in {root_dir}")
    
    if file_names:
        all_files = [f for f in all_files if os.path.basename(f) in file_names or 
                     os.path.splitext(os.path.basename(f))[0] in file_names]
    
    if exclude_pattern:
        all_files = [f for f in all_files if exclude_pattern not in file_stem(f)]
    
    groups = {}
    for f in all_files:
        stem = file_stem(f)
        groups.setdefault(stem, []).append(f)
    
    for stem in groups:
        groups[stem] = sorted(groups[stem])
    
    if merge_pattern:
        groups = {k: v for k, v in groups.items() if merge_pattern in k}
    
    return groups


def load_single_file(file_type: str, file_path: str, root_dir: str, verbose: bool = False):
    """
    Load single file based on type.
    
    Args:
        file_type: Type of file ('rhd', 'npz', 'csv')
        file_path: Path to the file
        root_dir: Root directory (used for CSV loading)
        verbose: Print verbose output
        
    Returns:
        dict: Loaded data dictionary
    """
    from intan.io import load_rhd_file, load_npz_file, load_csv_files
    
    if file_type == "rhd":
        return load_rhd_file(file_path, verbose=verbose)
    elif file_type == "npz":
        return load_npz_file(file_path, verbose=verbose)
    elif file_type == "csv":
        items = load_csv_files(root_dir, verbose=verbose)
        for data, path in items:
            if os.path.basename(path) == os.path.basename(file_path):
                return data
        raise FileNotFoundError(f"CSV not found: {file_path}")
    raise ValueError(f"Unknown file_type: {file_type}")


def load_files_merged(file_type: str, files: List[str], root_dir: str, verbose: bool = False):
    """
    Load and merge multiple files.
    
    Args:
        file_type: Type of files ('rhd', 'npz', 'csv')
        files: List of file paths to merge
        root_dir: Root directory (used for CSV loading)
        verbose: Print verbose output
        
    Returns:
        dict: Merged data dictionary
    """
    from intan.io import load_rhd_file, load_npz_file, load_npz_files, load_csv_files
    
    if file_type == "rhd":
        # Pass ALL files to load_rhd_file for proper merging
        if len(files) == 1:
            return load_rhd_file(files[0], merge_files=False, verbose=verbose)
        else:
            return load_rhd_file(files, merge_files=True, verbose=verbose)
    elif file_type == "npz":
        if len(files) == 1:
            return load_npz_file(files[0], verbose=verbose)
        dicts = load_npz_files(files, verbose=verbose)
        merged = dicts[0].copy()
        merged["amplifier_data"] = np.concatenate([d["amplifier_data"] for d in dicts], axis=1)
        return merged
    elif file_type == "csv":
        items = load_csv_files(root_dir, verbose=verbose)
        for data, path in items:
            if path in files:
                return data
        raise FileNotFoundError(f"CSV not found in {files}")


def file_stem(path: str) -> str:
    """
    Get file name without extension.
    
    Args:
        path: File path
        
    Returns:
        Filename without extension
    """
    return os.path.splitext(os.path.basename(path))[0]


def find_event_for_file(
    events_dir: Optional[str],
    data_path: str,
    pattern: Optional[str] = None
) -> Optional[str]:
    """
    Find matching event file for a data file.
    
    Tries in order:
      1) <events_dir>/<gesture>_emg.event
      2) <events_dir>/<gesture>.event
      3) Same folder as data file
      4) events/ sibling directory
    
    Args:
        events_dir: Directory containing event files
        data_path: Path to data file (.rhd, .npz, etc)
        pattern: Optional pattern to match (e.g., "emg", "imu")
        
    Returns:
        Path to event file or None if not found
    """
    import glob
    import re
    
    stem = file_stem(data_path)
    
    # Strip timestamp patterns to get base gesture name
    gesture = stem
    gesture = re.sub(r'_\d{6}_\d{6}$', '', gesture)  # Remove _YYMMDD_HHMMSS
    gesture = re.sub(r'_\d{6}$', '', gesture)         # Remove _YYMMDD
    gesture = re.sub(r'_\d+$', '', gesture)           # Remove trailing _digits
    
    # Try events_dir first
    if events_dir and os.path.isdir(events_dir):
        suffix = f"_{pattern}" if pattern else "_emg"
        candidates = [
            os.path.join(events_dir, f"{gesture}{suffix}.event"),
            os.path.join(events_dir, f"{gesture}.event"),
            os.path.join(events_dir, f"{stem}.event"),
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
    
    # Try same folder
    data_dir = os.path.dirname(data_path)
    for pat in (f"{stem}*.event", "*.event"):
        matches = sorted(glob.glob(os.path.join(data_dir, pat)))
        if matches:
            return matches[0]
    
    # Try events/ sibling
    evs = os.path.join(os.path.dirname(data_dir), "events")
    if os.path.isdir(evs):
        for pat in (f"{gesture}_emg.event", f"{gesture}.event", "*.event"):
            matches = sorted(glob.glob(os.path.join(evs, pat)))
            if matches:
                return matches[0]
    
    return None
