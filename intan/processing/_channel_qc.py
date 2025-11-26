from dataclasses import dataclass
import numpy as np
from collections import deque
from scipy.signal import iirnotch, butter, lfilter

@dataclass
class QCParams:
    # Robust RMS thresholds
    robust_z_warn: float = 2.0
    robust_z_bad: float = 3.0
    # Powerline ratio threshold (e.g., 60 Hz band / 20–450 Hz)
    pl_ratio_thresh: float = 0.30
    # Flatline / saturation
    flat_std_min: float = 1.0      # after filtering, tune to your uV scale
    zc_min_hz: float = 3.0         # min zero-crossings per second
    # Hysteresis
    consec_bad_needed: int = 3
    consec_good_needed: int = 5
    # PSD bands
    psd_min_hz: float = 20.0
    psd_max_hz: float = 450.0
    pl_low: float = 58.0
    pl_high: float = 62.0
    # Filtering
    bp_low_hz: float = 10.0
    bp_high_hz: float = 500.0
    notch_hz: float = 60.0
    notch_Q: float = 30.0
    # Performance
    psd_every_n_evals: int = 2     # compute PSD every N evals

class ChannelQC:
    """
    Reusable EMG channel quality checker with adaptive thresholds.
    Usage:
        qc = ChannelQC(fs=2000, n_channels=128, window_sec=0.2)
        qc.update(chunk)        # chunk: (samples, n_channels)
        out = qc.evaluate()     # dict with 'bad', 'watch', 'excluded', 'metrics', ...
        excluded = out['excluded']  # set of channel indices to ignore downstream
    """
    def __init__(self, fs: int, n_channels: int, window_sec: float = 0.2, params: QCParams = None):
        self.fs = int(fs)
        self.n_channels = int(n_channels)
        self.window_sec = float(window_sec)
        self.window_samples = max(8, int(self.fs * self.window_sec))
        self.params = params or QCParams(   )

        # Sliding buffers (per channel)
        self.buffers = [deque(maxlen=self.window_samples) for _ in range(self.n_channels)]

        # Precomputed filters
        nyq = 0.5 * self.fs
        bp_low_hz = max(0.1, self.params.bp_low_hz)
        bp_high_hz = min(self.params.bp_high_hz, nyq * 0.98)
        if bp_low_hz >= bp_high_hz:
            bp_high_hz = min(nyq * 0.98, bp_low_hz + 0.1 * nyq)
        Wn = [bp_low_hz / nyq, bp_high_hz / nyq]

        self._notch_ba = iirnotch(self.params.notch_hz, self.params.notch_Q, self.fs)
        self._bp_ba = butter(4, Wn, btype='band')

        # Hysteresis state
        self._bad_counts = np.zeros(self.n_channels, dtype=int)
        self._good_counts = np.zeros(self.n_channels, dtype=int)
        self._is_bad = np.zeros(self.n_channels, dtype=bool)

        # PSD cache and cadence
        self._pl_ratio_cache = np.zeros(self.n_channels, dtype=float)
        self._eval_idx = 0

        # Optional calibration (freeze robust center/scale across session)
        self._calibrated = False
        self._robust_center = None
        self._robust_scale = None

    # ---------- Public API ----------
    def update(self, chunk: np.ndarray) -> None:
        """Append a data chunk shaped (samples, n_channels) to buffers."""
        if chunk is None:
            return
        arr = np.asarray(chunk)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < self.n_channels:
            # pad missing channels with zeros
            pad = np.zeros((arr.shape[0], self.n_channels - arr.shape[1]), dtype=arr.dtype)
            arr = np.hstack([arr, pad])
        for row in arr:
            for ch in range(self.n_channels):
                self.buffers[ch].append(float(row[ch]))

    def evaluate(self, compute_psd: bool = True) -> dict:
        """
        Compute QC metrics and update hysteresis.
        Returns:
            {
              'bad': np.bool_(n_channels),
              'watch': np.bool_(n_channels),
              'excluded': set(int),
              'metrics': {
                 'rms': float[n],
                 'std': float[n],
                 'zc_hz': float[n],
                 'robust_z': float[n],
                 'pl_ratio': float[n],
                 'median_rms': float,
                 'mad_scale': float
              }
            }
        """
        p = self.params
        rms = np.zeros(self.n_channels, dtype=float)
        sdev = np.zeros(self.n_channels, dtype=float)
        zc_hz = np.zeros(self.n_channels, dtype=float)
        ready = np.zeros(self.n_channels, dtype=bool)

        # Per-channel filter + metrics
        for ch in range(self.n_channels):
            buf = np.asarray(self.buffers[ch], dtype=float)
            if buf.size < self.window_samples:
                continue
            try:
                sig = self._apply_filters(buf)
                rms[ch] = self._rms(sig)
                sdev[ch] = sig.std()
                zc_hz[ch] = self._zc_per_sec(sig)
                ready[ch] = True
            except Exception:
                pass

        # Robust center/scale across channels
        if np.any(ready):
            if not self._calibrated or self._robust_center is None:
                med, scale = self._robust_center_scale(rms[ready])
            else:
                med, scale = self._robust_center, self._robust_scale
            robust_z = (rms - med) / (scale + 1e-12)
        else:
            med, scale = 0.0, 1.0
            robust_z = np.zeros_like(rms)

        # PSD-based powerline ratio (cadenced)
        do_psd = compute_psd and (self._eval_idx % p.psd_every_n_evals == 0)
        if do_psd and np.any(ready):
            for ch in np.where(ready)[0]:
                buf = np.asarray(self.buffers[ch], dtype=float)
                try:
                    self._pl_ratio_cache[ch] = self._bandpower_ratio_fft(
                        buf,
                        num_band=(p.pl_low, p.pl_high),
                        den_band=(max(p.psd_min_hz, 0.1), min(p.psd_max_hz, self.fs/2 - 1))
                    )
                except Exception:
                    pass

        # Criteria
        crit_z_watch = robust_z > p.robust_z_warn
        crit_z_bad = robust_z > p.robust_z_bad
        crit_pl_bad = self._pl_ratio_cache > p.pl_ratio_thresh
        crit_flat = (sdev < p.flat_std_min) | (zc_hz < p.zc_min_hz)

        watch = ready & (crit_z_watch | crit_pl_bad)
        bad = ready & (crit_z_bad | crit_pl_bad | crit_flat)

        # Hysteresis
        for ch in range(self.n_channels):
            if bad[ch]:
                self._bad_counts[ch] += 1
                self._good_counts[ch] = 0
            else:
                self._good_counts[ch] += 1
                if self._bad_counts[ch] > 0 and self._good_counts[ch] >= p.consec_good_needed:
                    self._bad_counts[ch] = 0

            # transitions
            if not self._is_bad[ch] and self._bad_counts[ch] >= p.consec_bad_needed:
                self._is_bad[ch] = True
            elif self._is_bad[ch] and self._good_counts[ch] >= p.consec_good_needed and not bad[ch]:
                self._is_bad[ch] = False

        excluded = {int(i) for i in np.where(self._is_bad)[0]}
        self._eval_idx += 1

        return {
            'bad': bad,
            'watch': watch,
            'excluded': excluded,
            'metrics': {
                'rms': rms,
                'std': sdev,
                'zc_hz': zc_hz,
                'robust_z': robust_z,
                'pl_ratio': self._pl_ratio_cache.copy(),
                'median_rms': float(med),
                'mad_scale': float(scale)
            }
        }

    def begin_calibration(self):
        """Freeze the robust center/scale from the NEXT evaluate() call."""
        self._calibrated = False
        self._robust_center = None
        self._robust_scale = None

    def finalize_calibration(self):
        """
        Call after a stable recording period:
        locks in current robust center/scale so thresholds don't drift.
        """
        out = self.evaluate(compute_psd=False)
        self._robust_center = out['metrics']['median_rms']
        self._robust_scale = out['metrics']['mad_scale']
        self._calibrated = True

    # ---------- Helpers ----------
    def _apply_filters(self, x: np.ndarray) -> np.ndarray:
        b, a = self._notch_ba
        x = lfilter(b, a, x)
        b, a = self._bp_ba
        x = lfilter(b, a, x)
        return x

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x * x)))

    def _zc_per_sec(self, x: np.ndarray) -> float:
        s = np.sign(x)
        s[s == 0] = 1
        zc = np.sum(s[:-1] != s[1:])
        dur = len(x) / self.fs
        return float((zc / 2.0) / max(dur, 1e-9))

    def _bandpower_ratio_fft(self, x: np.ndarray, num_band, den_band) -> float:
        N = len(x)
        if N < 8:
            return 0.0
        X = np.fft.rfft(x * np.hanning(N))
        freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
        nb_lo, nb_hi = num_band
        db_lo, db_hi = den_band
        nb = (freqs >= nb_lo) & (freqs <= nb_hi)
        db = (freqs >= db_lo) & (freqs <= db_hi)
        num = np.sum(np.abs(X[nb]) ** 2)
        den = np.sum(np.abs(X[db]) ** 2) + 1e-12
        return float(num / den)

    @staticmethod
    def _robust_center_scale(x: np.ndarray) -> tuple[float, float]:
        m = np.median(x)
        mad = np.median(np.abs(x - m))
        scale = 1.4826 * mad + 1e-12
        return float(m), float(scale)
