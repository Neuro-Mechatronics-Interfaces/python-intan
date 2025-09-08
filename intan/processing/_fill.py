# emg_fill.py
from __future__ import annotations
import os, math
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class FillStats:
    total_missing: int
    pct_missing: float
    per_channel_pct: np.ndarray     # (C,)
    per_channel_longest_run: np.ndarray  # (C,)
    method: str
    filled_remaining_missing: int


def _naninf_mask(X: np.ndarray) -> np.ndarray:
    return ~np.isfinite(X)


def _longest_run_len(mask_1d: np.ndarray) -> int:
    if mask_1d.size == 0:
        return 0
    # run-length encode on booleans
    # convert to 1,0 and find consecutive 1s
    a = mask_1d.astype(np.int8)
    if a.max() == 0:
        return 0
    # indices where a[i] == 1
    idx = np.flatnonzero(a)
    # if all ones, longest run is len
    if idx.size == a.size:
        return a.size
    # compute gaps where sequence breaks
    runs = 1
    longest = 1
    for k in range(1, idx.size):
        if idx[k] == idx[k-1] + 1:
            runs += 1
        else:
            longest = max(longest, runs)
            runs = 1
    longest = max(longest, runs)
    return int(longest)


def _fill_repeat_last_run(x: np.ndarray) -> np.ndarray:
    """Fill NaN/Inf runs by repeating the last L valid samples (wrapping/tiling if needed)."""
    x = np.asarray(x, dtype=np.float32)
    bad = ~np.isfinite(x)
    if not np.any(bad):
        return x

    y = x.copy()
    N = y.size
    i = 0
    while i < N:
        if not bad[i]:
            i += 1
            continue
        # find end of the bad run
        j = i
        while j < N and bad[j]:
            j += 1
        L = j - i  # run length

        # source: last L finite samples *before* i
        left0 = max(0, i - L)
        src = y[left0:i]
        src = src[np.isfinite(src)]

        if src.size == 0:
            # fallback: try next L finite samples *after* j
            right1 = min(N, j + L)
            src = y[j:right1]
            src = src[np.isfinite(src)]

        if src.size == 0:
            # total fallback: constant 0
            y[i:j] = 0.0
        else:
            # tile/repeat to length L
            reps = int(math.ceil(L / src.size))
            tiled = np.tile(src, reps)[:L]
            y[i:j] = tiled

        i = j
    return y


def _fill_interp(x: np.ndarray) -> np.ndarray:
    """Linear interpolation across NaN/Inf; edges are held to nearest finite."""
    x = np.asarray(x, dtype=np.float32)
    good = np.isfinite(x)
    if not np.any(good):
        return np.zeros_like(x)
    if np.all(good):
        return x.copy()
    xi = np.arange(x.size, dtype=np.float64)
    xp = xi[good]
    fp = x[good].astype(np.float64)
    out = np.interp(xi, xp, fp, left=fp[0], right=fp[-1]).astype(np.float32)
    return out


def _fill_ffill(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = x.copy()
    bad = ~np.isfinite(y)
    if not np.any(bad):
        return y
    # forward fill: replace bad with last good
    last = np.nan
    for k in range(y.size):
        if np.isfinite(y[k]):
            last = y[k]
        else:
            if np.isfinite(last):
                y[k] = last
    # backfill leading NaNs with first finite value
    if not np.isfinite(y[0]):
        first_idx = np.flatnonzero(np.isfinite(y))
        if first_idx.size:
            y[:first_idx[0]] = y[first_idx[0]]
        else:
            y[:] = 0.0
    return y


def _fill_bfill(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = x.copy()
    bad = ~np.isfinite(y)
    if not np.any(bad):
        return y
    # backward fill: replace bad with next good
    nextv = np.nan
    for k in range(y.size - 1, -1, -1):
        if np.isfinite(y[k]):
            nextv = y[k]
        else:
            if np.isfinite(nextv):
                y[k] = nextv
    # forward fill trailing NaNs with last finite
    if not np.isfinite(y[-1]):
        last_idx = np.flatnonzero(np.isfinite(y))
        if last_idx.size:
            y[last_idx[-1]:] = y[last_idx[-1]]
        else:
            y[:] = 0.0
    return y


def _fill_constant(x: np.ndarray, value: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    y[~np.isfinite(y)] = float(value)
    return y


def _fill_median(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32).copy()
    good = np.isfinite(y)
    if not np.any(good):
        return np.zeros_like(y)
    med = float(np.median(y[good]))
    y[~good] = med
    return y


def _plot_nan_map(bad: np.ndarray, fs: float | None, out_path: Path, title: str):
    # bad is (C, S) bool
    C, S = bad.shape
    # downsample columns for faster plotting if very long
    max_cols = 6000
    if S > max_cols:
        factor = int(np.ceil(S / max_cols))
        # OR downsample: any bad in block -> bad
        bad_ds = bad.reshape(C, -1, factor).any(axis=2)
    else:
        bad_ds = bad
        factor = 1
    plt.figure(figsize=(16, min(10, 0.12*C + 2)))
    plt.imshow(bad_ds, aspect='auto', interpolation='nearest', cmap='gray_r')
    plt.xlabel("Samples" + (f" (/{factor})" if factor > 1 else ""))
    plt.ylabel("Channels")
    plt.title(title)
    plt.colorbar(label="1=NaN/Inf, 0=finite")
    if fs and fs > 0:
        xt = plt.gca().get_xticks()
        secs = (xt * factor) / float(fs)
        plt.gca().set_xticklabels([f"{t:.1f}s" for t in secs])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_overlays(raw: np.ndarray, fixed: np.ndarray, fs: float | None, out_path: Path, channels=(0,1,2)):
    t = np.arange(raw.shape[1], dtype=np.float64)
    if fs and fs > 0:
        t = t / float(fs)
        xlabel = "Time (s)"
    else:
        xlabel = "Samples"
    rows = len(channels)
    plt.figure(figsize=(16, 2.4*rows + 1.5))
    for r, ch in enumerate(channels, start=1):
        if ch < 0 or ch >= raw.shape[0]:
            continue
        ax = plt.subplot(rows, 1, r)
        ax.plot(t, raw[ch], lw=0.7, alpha=0.7, label="raw")
        ax.plot(t, fixed[ch], lw=0.8, label="filled")
        ax.set_title(f"Channel {ch}")
        ax.set_xlabel(xlabel)
        ax.legend(loc="upper right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def fix_missing_emg(
    emg: np.ndarray,
    method: str = "repeat",
    constant_value: float = 0.0,
    make_figures: bool = False,
    out_dir: str | os.PathLike | None = None,
    fs: float | None = None,
    overlay_channels: tuple[int, ...] = (0, 1, 2),
    verbose: bool = True,
) -> tuple[np.ndarray, FillStats]:
    """
    Fill NaN/Inf in EMG (C,S) and return (filled_emg, stats).

    method: "repeat" | "interp" | "ffill" | "bfill" | "median" | "constant"
      - "repeat": for each bad run of length L, copy the previous L finite samples (tiled if needed).
      - "interp": linear interpolation across finite points; edges held.
      - "ffill": forward fill; leading bads get the first finite value (or 0 if none).
      - "bfill": backward fill; trailing bads get the last finite value (or 0 if none).
      - "median": per-channel median for bad samples.
      - "constant": fill with constant_value.

    make_figures: if True, writes a NaN map and small before/after overlays to out_dir.
    """
    X = np.asarray(emg)
    if X.ndim != 2:
        raise ValueError("EMG must be 2D (C, S).")
    C, S = X.shape
    bad = _naninf_mask(X)
    total_missing = int(bad.sum())
    per_ch_pct = bad.mean(axis=1) * 100.0
    per_ch_long = np.array([_longest_run_len(bad[c]) for c in range(C)], dtype=int)

    if verbose:
        miss_pct = (total_missing / max(1, C*S)) * 100.0
        print(f"[nanfix] C={C} S={S} missing={total_missing} ({miss_pct:.2f}%)")
        print(f"[nanfix] per-channel missing % (first 16): {np.round(per_ch_pct[:16], 2)}")
        print(f"[nanfix] per-channel longest run (first 16): {per_ch_long[:16]}")

    if method == "repeat":
        filled = np.vstack([_fill_repeat_last_run(X[c]) for c in range(C)])
    elif method == "interp":
        filled = np.vstack([_fill_interp(X[c]) for c in range(C)])
    elif method == "ffill":
        filled = np.vstack([_fill_ffill(X[c]) for c in range(C)])
    elif method == "bfill":
        filled = np.vstack([_fill_bfill(X[c]) for c in range(C)])
    elif method == "median":
        filled = np.vstack([_fill_median(X[c]) for c in range(C)])
    elif method == "constant":
        filled = np.vstack([_fill_constant(X[c], constant_value) for c in range(C)])
    else:
        raise ValueError(f"Unknown method: {method}")

    remaining = int((~np.isfinite(filled)).sum())
    if verbose:
        print(f"[nanfix] remaining NaN/Inf after fill: {remaining}")

    stats = FillStats(
        total_missing=total_missing,
        pct_missing=(total_missing / max(1, C*S)) * 100.0,
        per_channel_pct=per_ch_pct,
        per_channel_longest_run=per_ch_long,
        method=method,
        filled_remaining_missing=remaining,
    )

    if make_figures and out_dir is not None:
        out_dir = Path(out_dir)
        _plot_nan_map(bad, fs, out_dir / "nan_map_before.png", "NaN/Inf map (before)")
        _plot_nan_map(~np.isfinite(filled), fs, out_dir / "nan_map_after.png", "NaN/Inf map (after)")
        # small overlay of a few channels
        _plot_overlays(X, filled, fs, out_dir / "overlay_before_after.png", channels=overlay_channels)

        # also drop a quick CSV of basic stats
        try:
            import csv
            with open(out_dir / "nan_stats.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["channel", "pct_missing", "longest_run_samples"])
                for c in range(C):
                    w.writerow([c, f"{per_ch_pct[c]:.4f}", int(per_ch_long[c])])
        except Exception:
            pass

    return filled.astype(np.float32, copy=False), stats
