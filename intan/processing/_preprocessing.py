import numpy as np
from typing import Callable, List, Union, Dict, Optional
from tqdm import tqdm
from ._features import rectify, extract_features, feature_spec_from_registry, FEATURE_REGISTRY
from ._realtime_filter import RealtimeFilter
from scipy.signal import butter, sosfilt, sosfilt_zi


class EMGPreprocessor:
    """
    Streaming-safe EMG pipeline:

      1) Band-pass (default 20–498 Hz) + 60 Hz notch (and harmonics if configured)
      2) Rectify (|x|)
      3) Low-pass envelope (default off, e.g. 5–10 Hz for gestures)

    Shapes are (C, N) throughout. Keeps filter state across calls for realtime use,
    but also works offline on full arrays in one call.
    """

    def __init__(self, fs: float = 5000.0, band: tuple[float, float] = (20.0, 498.0),
        notch_freqs: tuple[float, ...] = (60.0,), notch_q: float = 30.0, envelope_cutoff: float | None = None,
        envelope_order: int = 4, feature_fns: List[Union[str, Callable]] | None = None, verbose: bool = False,
    ):
        self.fs = float(fs)
        self.band = (float(band[0]), float(band[1]))
        self.notch_freqs = tuple(float(f) for f in notch_freqs)
        self.notch_q = float(notch_q)
        self.verbose = bool(verbose)

        # front-end filter: band-pass + notch (no low-pass here)
        self.frontend: RealtimeFilter | None = None  # lazy init when we see channel count

        # envelope LP stage (after rectification)
        self.env_cut = envelope_cutoff
        self.env_order = int(envelope_order)
        self._env_sos = None
        self._env_zi = None   # shape (n_sections, C, 2)
        self._C = None        # channel count we’re initialized for

        # Features
        self.feature_fns = feature_fns
        if self.feature_fns is None:
            self._feature_names = list(FEATURE_REGISTRY.keys())
        else:
            self._feature_names = [fn if isinstance(fn, str) else fn.__name__ for fn in self.feature_fns]
            # ensure all feature names are registered
            for fn in self.feature_fns:
                if isinstance(fn, str) and fn not in FEATURE_REGISTRY:
                    raise ValueError(f"Feature function '{fn}' is not registered in FEATURE_REGISTRY.")

    def _ensure_initialized(self, C: int):
        if self._C == C and self.frontend is not None:
            return
        self._C = int(C)

        # build / reset the front-end
        self.frontend = RealtimeFilter(
            fs=self.fs,
            n_channels=self._C,
            bp_low=self.band[0],
            bp_high=self.band[1],
            bp_order=4,
            enable_bandpass=True,
            notch_freqs=self.notch_freqs,
            notch_q=self.notch_q,
            enable_notch=True,
            enable_lowpass=False,  # low-pass happens AFTER rectify
        )

        # (re)build envelope LP
        if self.env_cut is not None:
            self._env_sos = butter(self.env_order, self.env_cut, btype="low", fs=self.fs, output="sos")
            # zi per section per channel
            base_zi = sosfilt_zi(self._env_sos)[:, None, :]  # (nsec, 1, 2)
            self._env_zi = np.tile(base_zi, (1, self._C, 1)).astype(np.float32)
        else:
            self._env_sos = None
            self._env_zi = None

    def reset_states(self):
        """Zero internal state for both front-end and envelope LP."""
        if self.frontend is not None:
            self.frontend.reset()
        if self._env_sos is not None and self._env_zi is not None:
            base_zi = sosfilt_zi(self._env_sos)[:, None, :]
            self._env_zi[:] = np.tile(base_zi, (1, self._C, 1))

    # ---------- public API ----------

    def preprocess(self, emg: np.ndarray, *, rectify: bool = True, envelope_cutoff: float | None = None) -> np.ndarray:
        """
        Offline OR streaming-compatible preprocess of a (C, N) block:

          bandpass+notch -> (rectify) -> (low-pass envelope)

        If `envelope_cutoff` is provided, it overrides the constructor’s default for this call.
        """
        emg = np.asarray(emg, dtype=np.float32)
        if emg.ndim != 2:
            raise ValueError(f"Expected (C, N) array; got {emg.shape}")
        C = emg.shape[0]
        self._ensure_initialized(C)

        # 1) front-end (stateful)
        y = self.frontend.process(emg)

        # 2) rectify
        if rectify:
            y = np.abs(y)

        # 3) envelope LP (stateful)
        cut = self.env_cut if envelope_cutoff is None else envelope_cutoff
        if cut is not None:
            if self._env_sos is None or cut != self.env_cut or self._env_zi is None or self._env_zi.shape[1] != C:
                # rebuild LP if cutoff changed or channels changed
                self.env_cut = float(cut)
                self._C = None  # force rebuild path to set env with new cut
                self._ensure_initialized(C)
            # per-channel
            for ch in range(C):
                y[ch, :], self._env_zi[:, ch, :] = sosfilt(self._env_sos, y[ch, :], zi=self._env_zi[:, ch, :])
        return y

    def extract_emg_features(self, emg: np.ndarray, window_ms: int = 200, step_ms: int = 50, feature_fns=None,
                             return_windows: bool = False, *, progress: bool = False, tqdm_kwargs: dict | None = None):
        """
        Sliding-window feature extraction over a preprocessed EMG trace.
        Returns (n_windows, n_features) (and window start indices if return_windows=True).

        Set progress=True to show a tqdm progress bar (ETA, rate, etc.).
        Pass extra bar options via tqdm_kwargs (e.g., {'desc': 'EMG', 'leave': False}).

        """
        fs = self.fs
        w = int(window_ms * fs / 1000)
        s = int(step_ms * fs / 1000)
        n_samps = emg.shape[1]
        if n_samps < w:
            return (np.zeros((0,)), np.array([], dtype=int)) if return_windows else np.zeros((0,))

        starts = np.arange(0, n_samps - w + 1, s, dtype=int)
        n_win = starts.size
        if n_win == 0:
            return (np.zeros((0,)), np.array([], dtype=int)) if return_windows else np.zeros((0,))

        # Compute first window to determine feature length & dtype, then preallocate
        f0 = extract_features(emg[:, starts[0]: starts[0] + w], feature_fns)
        f0 = np.asarray(f0)
        X = np.empty((n_win, f0.size), dtype=f0.dtype)
        X[0] = f0

        it = range(1, n_win)
        if progress and tqdm is not None:
            kw = {'total': n_win - 1, 'desc': 'EMG features', 'unit': 'win'}
            if tqdm_kwargs:
                kw.update(tqdm_kwargs)
            it = tqdm(it, **kw)

        for idx in it:
            i = starts[idx]
            X[idx] = extract_features(emg[:, i:i + w], feature_fns)

        if return_windows:
            return X, starts
        return X


    def feature_spec(self, n_channels: Optional[int] = None) -> Dict:
        spec = feature_spec_from_registry(
            FEATURE_REGISTRY,
            self._feature_names,  # use the exact names we enabled
            per_channel=True,
            layout="channel_major",
            channels="training_order",
        )
        # add convenience counts (handy for validation)
        if n_channels is not None:
            spec["n_channels"] = int(n_channels)
            spec["n_features_per_channel"] = len(self._feature_names)
        return spec