#!/usr/bin/env python3
"""
Segment EMG from .poly5 recordings using .event files with
columns:  Sample Index, Timestamp, Label

Output: one .npz per segment with keys:
  - emg: np.ndarray (channels, samples)
  - label: str
  - fs_hz: float or None (if available)
  - raw_file: str
  - start_sample: int
  - end_sample: int
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# ---------- IO helpers ----------

def _normalize_token(s: str) -> str:
    """lowercase, strip non-letters, collapse to help filename matching."""
    return re.sub(r"[^a-z]", "", s.lower())


def _gesture_from_raw_filename(raw_path: Path) -> str:
    """
    Convert e.g. 'key_grip_close_3_reps.poly5' -> 'KeyGripClose'
    """
    base = raw_path.stem
    parts = [p for p in re.split(r"[_\W]+", base) if p and not p.isdigit() and p.lower() not in {"rep", "reps"}]
    return "".join(w.capitalize() for w in parts)


def _guess_event_path_for_raw(raw_path: Path, events_dir: Path) -> Optional[Path]:
    """
    Try to find the matching .event file by comparing normalized names.
    Expects files like 'KeyGripClose_emg.event'.
    """
    gesture = _gesture_from_raw_filename(raw_path)              # KeyGripClose
    want_norm = _normalize_token(gesture)                       # keygripclose
    print(f"🔍 Looking for .event matching '{gesture}' in {events_dir}")
    print(f"   (normalized: '{want_norm}')")

    candidates = sorted(events_dir.glob("*.event"))
    print(f"   Found {len(candidates)} .event files to check.")
    print("   Candidates:", [p.name for p in candidates])
    best = None
    for ev in candidates:
        ev_base = ev.stem.replace("_emg", "")
        ev_norm = _normalize_token(ev_base)
        if ev_norm in want_norm or want_norm in ev_norm:
            best = ev
            break
    return best


def load_events_table(event_path: Path) -> pd.DataFrame:
    """
    Read 'Sample Index,Timestamp,Label' CSV/TSV and return
    DataFrame with ['Sample','Time','Label'] (sorted by Sample, int).
    """
    # Try comma first, then tab
    try:
        df = pd.read_csv(event_path)
    except Exception:
        df = pd.read_csv(event_path, sep="\t")

    # Standardize columns
    colmap = {c.lower(): c for c in df.columns}
    # Flexible matching
    sample_col = next((c for c in df.columns if _normalize_token(c) in {"sampleindex", "sample"}), None)
    time_col   = next((c for c in df.columns if _normalize_token(c) in {"timestamp", "time"}), None)
    label_col  = next((c for c in df.columns if _normalize_token(c) == "label"), None)

    if sample_col is None or label_col is None:
        raise ValueError(f"Unexpected columns in {event_path}: {df.columns.tolist()}")

    out = pd.DataFrame({
        "Sample": df[sample_col].astype(int),
        "Label":  df[label_col].astype(str).str.strip(),
    })
    if time_col is not None:
        out["Time"] = df[time_col].astype(str)
    else:
        out["Time"] = ""

    out = out.sort_values("Sample").reset_index(drop=True)
    return out


def read_poly5_emg(poly5_path: Path) -> Tuple[np.ndarray, Optional[float]]:
    """
    Return (emg[ch, samples], fs_hz) from a .poly5 recording.
    Tries a few common readers; adjust this to your environment if needed.
    """
    # Try tmsi-python-interface
    try:
        from tmsi.io import Poly5Reader  # type: ignore
        reader = Poly5Reader(str(poly5_path))
        data, hdr = reader.read_data()   # most builds return (samples x channels), header
        emg = np.asarray(data).T         # -> (channels, samples)
        fs = getattr(hdr, "sample_rate", None)
        return emg, fs
    except Exception as e:
        last_err = f"tmsi.io.Poly5Reader failed: {e}"

    # Try poly5 package
    try:
        from poly5 import Poly5Reader  # type: ignore
        reader = Poly5Reader(str(poly5_path))
        data = reader.read_data()      # some versions return dict-like
        # Heuristics:
        if isinstance(data, dict):
            # common keys: 'samples', 'header', 'sample_rate'
            if "samples" in data:
                X = np.asarray(data["samples"]).T
            else:
                # best effort: convert first array-like
                X = np.asarray(next(iter(data.values())))
                if X.shape[0] < X.shape[1]:
                    X = X.T
                X = X
            fs = data.get("sample_rate", None)
        else:
            X = np.asarray(data)
            if X.shape[0] < X.shape[1]:
                X = X.T
            fs = None
        return np.asarray(X), fs
    except Exception as e:
        last_err += f" | poly5.Poly5Reader failed: {e}"

    raise RuntimeError(f"Could not read {poly5_path} as .poly5. {last_err}")


# ---------- segmentation ----------

def variable_length_segments(
    emg: np.ndarray,
    cues: pd.DataFrame,
    include_rest: bool = True,
    include_last: bool = True
) -> Tuple[List[np.ndarray], List[str], List[Tuple[int, int]]]:
    """
    Build segments [Sample_i : Sample_{i+1}) with label at i.
    Optionally include 'Rest' labels and final tail to end-of-file.

    Returns: segments, labels, (start,end) pairs
    """
    starts = cues["Sample"].to_numpy().astype(int)
    labels = cues["Label"].astype(str).tolist()

    seg_arrays: List[np.ndarray] = []
    seg_labels: List[str] = []
    seg_bounds: List[Tuple[int, int]] = []

    n = emg.shape[1]

    for i in range(len(starts) - 1):
        start, end = starts[i], starts[i + 1]
        lab = labels[i].strip()
        if not include_rest and lab.lower() == "rest":
            continue
        if not (0 <= start < end <= n):
            continue
        seg_arrays.append(emg[:, start:end])
        seg_labels.append(lab)
        seg_bounds.append((start, end))

    # tail: last cue → end
    if include_last and len(starts) > 0 and starts[-1] < n:
        lab = labels[-1].strip()
        if include_rest or lab.lower() != "rest":
            start, end = int(starts[-1]), n
            if end - start > 0:
                seg_arrays.append(emg[:, start:end])
                seg_labels.append(lab)
                seg_bounds.append((start, end))

    return seg_arrays, seg_labels, seg_bounds


# ---------- main pipeline ----------

def process_one_recording(
    poly5_path: Path,
    events_dir: Path,
    out_dir: Path,
    include_rest: bool,
    include_last: bool
) -> int:
    """Process a single .poly5 with its matching .event; returns #segments written."""
    event_path = _guess_event_path_for_raw(poly5_path, events_dir)
    if event_path is None or not event_path.exists():
        print(f"⚠️  No matching .event found for {poly5_path.name}")
        return 0

    cues = load_events_table(event_path)
    emg, fs = read_poly5_emg(poly5_path)

    seg_arrays, seg_labels, seg_bounds = variable_length_segments(
        emg, cues, include_rest=include_rest, include_last=include_last
    )

    base = poly5_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for k, (arr, lab, (s, e)) in enumerate(zip(seg_arrays, seg_labels, seg_bounds), start=1):
        out_name = f"{base}__{lab}__seg{k:03d}.npz"
        out_path = out_dir / out_name
        np.savez_compressed(
            out_path,
            emg=arr.astype(np.float32),
            label=str(lab),
            fs_hz=(None if fs is None else float(fs)),
            raw_file=str(poly5_path),
            start_sample=int(s),
            end_sample=int(e),
        )
        n_written += 1

    print(f"✓ {poly5_path.name} → wrote {n_written} segments to {out_dir}")
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Folder with .poly5 files")
    ap.add_argument("--events", required=True, help="Folder with .event files")
    ap.add_argument("--out", required=True, help="Output folder for .npz")
    ap.add_argument("--include-rest", action="store_true", help="Keep 'Rest' segments")
    ap.add_argument("--include-last", action="store_true", help="Keep tail (last cue → end)")
    args = ap.parse_args()

    raw_dir   = Path(args.raw)
    events_dir= Path(args.events)
    out_dir   = Path(args.out)

    poly5_files = sorted(raw_dir.glob("*.poly5"))
    if not poly5_files:
        raise SystemExit(f"No .poly5 files found in {raw_dir}")

    total = 0
    for poly5 in poly5_files:
        total += process_one_recording(
            poly5, events_dir, out_dir,
            include_rest=args.include_rest,
            include_last=args.include_last
        )

    print(f"\nDone. Total segments written: {total}")


if __name__ == "__main__":
    main()
