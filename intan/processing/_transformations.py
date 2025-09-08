import numpy as np


def _z(x, eps=1e-8):
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    return x / (x.std() + eps)

def _decimate(x, q):
    if q <= 1: return x
    n = (len(x) // q) * q
    x = x[:n].reshape(-1, q).mean(axis=1)  # simple box decimation
    return x

def normxcorr_offset_full(a, b, max_lag=None, min_overlap=256):
    """Return (lag, score); lag>0 means b shifted +lag aligns to a."""
    a = _z(a); b = _z(b)
    n = min(len(a), len(b))
    a = a[:n]; b = b[:n]
    if max_lag is None: max_lag = n - 1
    max_lag = int(max(0, max_lag))
    # full linear cross-corr via numpy (fast enough for typical EMG windows)
    corr = np.correlate(a, b, mode='full')
    lags = np.arange(-n + 1, n, dtype=int)
    keep = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr[keep]; lags = lags[keep]
    overlaps = n - np.abs(lags)
    valid = overlaps >= max(1, int(min_overlap))
    if not np.any(valid): valid = np.ones_like(lags, bool)
    corr = corr[valid] / overlaps[valid]
    lags = lags[valid]
    i = int(np.argmax(corr))
    return int(lags[i]), float(corr[i])


def estimate_lag_coarse_to_fine(a, b, fs, max_lag="auto", decim=8, refine_s=1.0, check_polarity=True):
    """
    Coarse search at low rate across the *entire* recording, then refine
    around that estimate at full rate. Optionally check inverted polarity.
    """
    n = min(len(a), len(b))
    if max_lag == "auto":
        max_lag = n - 1                      # allow any shift within the recording
    # --- coarse ---
    a_c = _decimate(a, decim)
    b_c = _decimate(b, decim)
    lag_c, score_c = normxcorr_offset_full(a_c, b_c, max_lag=max_lag//decim,
                                           min_overlap=max(64, int(0.25*fs/decim)))
    lag0 = lag_c * decim

    # --- refine around lag0 at full rate ---
    half = int(max(1, refine_s * fs))
    a_ref = a; b_ref = b
    # shift b by lag0 and search ±half around it
    # we do that by trimming so that the residual search is centered
    # then add lag0 back to the final estimate
    lag_ref, score_ref = normxcorr_offset_full(a_ref, b_ref, max_lag=half,
                                               min_overlap=max(256, int(0.25*fs)))
    lag_best, score_best = lag0 + lag_ref, score_c  # seed with coarse

    # Try polarity flip if requested
    if check_polarity:
        lag_c2, score_c2 = normxcorr_offset_full(a_c, -b_c, max_lag=max_lag//decim,
                                                 min_overlap=max(64, int(0.25*fs/decim)))
        lag0b = lag_c2 * decim
        lag_ref2, score_ref2 = normxcorr_offset_full(a_ref, -b_ref, max_lag=half,
                                                     min_overlap=max(256, int(0.25*fs)))
        lag2 = lag0b + lag_ref2
        if score_ref2 > score_ref:
            lag_best, score_best = lag2, score_ref2
            return lag_best, score_best, True  # flipped
    return lag0 + lag_ref, score_ref, False


def normxcorr_offset(a: np.ndarray, b: np.ndarray, max_lag: int | None = None, min_overlap: int = 256,
                     eps: float = 1e-8) -> tuple[int, float]:
    """
    Return (best_lag, best_score) where 'best_lag' (samples) means:
        b shifted by +best_lag aligns to a.
    Both a and b may be different length; only the first min(len(a), len(b)) is used.
    Normalization: z-score each vector globally, then divide the dot product at
    each lag by the overlap length so scores are comparable across lags.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = int(min(a.size, b.size))
    if n < 2:
        return 0, 0.0
    a = a[:n]; b = b[:n]

    # Global z-scoring (stationary assumption)
    a = (a - a.mean()) / (a.std() + eps)
    b = (b - b.mean()) / (b.std() + eps)

    # Full linear cross-correlation (length 2n-1), lags in [-(n-1), ..., +(n-1)]
    corr = np.correlate(a, b, mode='full')
    lags = np.arange(-n + 1, n, dtype=int)

    if max_lag is None:
        max_lag = n - 1
    max_lag = int(max(0, max_lag))

    # Keep only desired lag window and require minimum overlap
    keep = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr[keep]
    lags = lags[keep]
    overlaps = n - np.abs(lags)
    valid = overlaps >= max(1, int(min_overlap))
    if not np.any(valid):
        # Fall back to the best within the window if everything is too short
        valid = np.ones_like(lags, dtype=bool)

    corr = corr[valid] / (overlaps[valid].astype(np.float64) + eps)
    lags = lags[valid]

    idx = int(np.argmax(corr))
    return int(lags[idx]), float(corr[idx])


def align_by_lag(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Crop a and b to their overlapping region implied by 'lag', without padding.
    Positive 'lag': drop first 'lag' samples from b and last 'lag' from a.
    Negative 'lag': drop first '-lag' from a and last '-lag' from b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = min(a.size, b.size)
    a = a[:n]; b = b[:n]

    if lag > 0:
        # shift b forward
        L = n - lag
        if L <= 0: return np.array([]), np.array([])
        return a[:L], b[lag:lag+L]
    elif lag < 0:
        k = -lag
        L = n - k
        if L <= 0: return np.array([]), np.array([])
        return a[k:k+L], b[:L]
    else:
        return a, b


def align_multichannel_by_lag(A: np.ndarray, B: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the same lag cropping to (C×N) arrays A and B.
    Returns A_aligned (C×L), B_aligned (C×L) with identical L.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    assert A.ndim == 2 and B.ndim == 2, "A and B must be (C×N)"
    n = min(A.shape[1], B.shape[1])
    A = A[:, :n]; B = B[:, :n]

    if lag > 0:
        L = n - lag
        if L <= 0: return A[:, :0], B[:, :0]
        return A[:, :L], B[:, lag:lag+L]
    elif lag < 0:
        k = -lag
        L = n - k
        if L <= 0: return A[:, :0], B[:, :0]
        return A[:, k:k+L], B[:, :L]
    else:
        return A, B
