from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos



@dataclass
class _FilterSpec:
    fs: float
    bp: Optional[Tuple[float, float]]
    notch: Optional[Iterable[float]]
    order: int = 4


class RealtimeEMGFilter:
    """
    Stateful, realtime EMG filter: bandpass + multi-notch, with per-channel state.

    Parameters
    ----------
    fs : float
        Sampling rate.
    bp : (low, high) or None
        Bandpass edges in Hz. Use None to disable.
    notch_hz : list[float] | tuple | None
        One or more notch center frequencies in Hz (e.g., [50] or [60, 120]).
    order : int
        Butterworth order for bandpass (>=2 recommended).
    n_channels : int
        Number of channels. If None, states are allocated on first call to process().
    """

    def __init__(
            self,
            fs: float,
            bp: Optional[Tuple[float, float]] = (30.0, 400.0),
            notch_hz: Optional[Iterable[float]] = (60.0,),
            order: int = 4,
            n_channels: Optional[int] = None,
    ) -> None:
        self.spec = _FilterSpec(fs=float(fs), bp=bp, notch=tuple(notch_hz) if notch_hz else None, order=int(order))
        self._sos = self._design_sos(self.spec)
        self._zi = None  # initialized on first process if n_channels is None
        if n_channels is not None:
            self._allocate_state(int(n_channels))

    # ---------- public API ----------

    def reset_state(self) -> None:
        """Zero the internal filter state."""
        if self._zi is not None:
            self._zi[:] = 0.0

    def reconfigure(
            self,
            *,
            fs: Optional[float] = None,
            bp: Optional[Tuple[float, float] | None] = None,
            notch_hz: Optional[Iterable[float] | None] = None,
            order: Optional[int] = None,
            preserve_state: bool = False,
            n_channels: Optional[int] = None,
    ) -> None:
        """
        Redesign filters and (optionally) keep state shape.

        If `preserve_state=True` and the number of SOS sections is unchanged,
        existing zi is reused (safer to reset_state() after big changes).
        """
        spec = _FilterSpec(
            fs=float(fs) if fs is not None else self.spec.fs,
            bp=bp if bp is not None else self.spec.bp,
            notch=tuple(notch_hz) if notch_hz is not None else self.spec.notch,
            order=int(order) if order is not None else self.spec.order,
        )
        new_sos = self._design_sos(spec)
        self.spec = spec
        if preserve_state and self._zi is not None and new_sos.shape[0] == self._sos.shape[0]:
            self._sos = new_sos
        else:
            self._sos = new_sos
            if self._zi is not None:
                # resize state
                n_ch = self._zi.shape[0]
                self._allocate_state(n_ch)
        if n_channels is not None and (self._zi is None or self._zi.shape[0] != int(n_channels)):
            self._allocate_state(int(n_channels))

    def process(self, x: np.ndarray) -> np.ndarray:
        """
        Filter a chunk of data.

        Input:
            x: shape (C, N)
        Output:
            y: shape (C, N)
        """
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2, "Expected (C, N) input array."
        C, N = x.shape
        if self._zi is None:
            self._allocate_state(C)
        y = np.empty_like(x)
        # Apply SOS per channel with persistent state
        for c in range(C):
            y[c], self._zi[c] = sosfilt(self._sos, x[c], zi=self._zi[c])
        return y

    # Backwards-compatible alias
    __call__ = process

    # ---------- internals ----------

    def _allocate_state(self, n_channels: int) -> None:
        zi_single = sosfilt_zi(self._sos)  # (n_sections, 2)
        self._zi = np.tile(zi_single[None, :, :], (n_channels, 1, 1)).astype(np.float32)

    def _design_sos(self, spec: _FilterSpec) -> np.ndarray:
        fs = spec.fs
        if fs <= 0:
            raise ValueError("Sampling rate fs must be > 0.")
        sos_list = []

        # Bandpass
        if spec.bp is not None:
            lo, hi = spec.bp
            if not (0 < lo < hi < fs / 2):
                # clamp with a warning-like behavior
                lo = max(1.0, min(lo, fs / 2 - 1.0))
                hi = max(lo + 1.0, min(hi, fs / 2 - 0.1))
            sos_list.append(butter(spec.order, [lo, hi], btype="bandpass", fs=fs, output="sos"))

        # Notches
        if spec.notch:
            for f0 in spec.notch:
                if f0 <= 0 or f0 >= fs / 2:
                    continue
                # Q ~ 30 is a decent default for mains; adjust as needed
                bw = 1.0
                Q = max(5.0, f0 / bw)
                sos_list.append(iirnotch(f0, Q, fs=fs))

        if not sos_list:
            # identity
            return np.array([[1, 0, 0, 1, 0, 0]], dtype=np.float64)

        return np.vstack(sos_list)


# Backwards-compatible alias
#RealtimeFilter = RealtimeEMGFilter


class RealtimeFilter:
    """
    Stateful realtime filter cascade for multi-channel streams.

    Order of stages (if enabled): BANDPASS -> NOTCH (one or more) -> LOWPASS

    Input/Output blocks are shaped (C, N): C = channels, N = samples.
    """

    def __init__(
        self,
        fs: float,
        n_channels: int,
        # bandpass
        bp_low: float = 20.0,
        bp_high: float = 498.0,
        bp_order: int = 4,
        enable_bandpass: bool = True,
        # notch
        notch_freqs=(60.0,),        # can be (60., 120., 180., ...) if desired
        notch_q: float = 30.0,
        enable_notch: bool = True,
        # optional post low-pass
        lp_cut: float | None = None,
        lp_order: int = 4,
        enable_lowpass: bool = False,
    ):
        self.fs = float(fs)
        self.C = int(n_channels)

        # config
        self.enable_bandpass = bool(enable_bandpass)
        self.bp_low = float(bp_low)
        self.bp_high = float(bp_high)
        self.bp_order = int(bp_order)

        self.enable_notch = bool(enable_notch)
        self.notch_freqs = tuple(float(f) for f in notch_freqs)
        self.notch_q = float(notch_q)

        self.enable_lowpass = bool(enable_lowpass)
        self.lp_cut = None if lp_cut is None else float(lp_cut)
        self.lp_order = int(lp_order)

        # designed filter sections and per-channel states
        self._stages = []          # list of dicts: {name, sos, zi} ; zi shape (n_sections, C, 2)
        self._design_filters()

    # ------------- public API -------------

    def reconfigure(self, **kwargs):
        """
        Update parameters (e.g., bp_low=30, notch_freqs=(60,120)) and redesign filters.
        Resets filter state (safer for realtime); set preserve_state=True if you really
        want to keep old states (not recommended when topology changes).
        """
        preserve_state = kwargs.pop("preserve_state", False)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown config: {k}")
            setattr(self, k, v)
        self._design_filters(preserve_state=preserve_state)

    def reset(self):
        """Zero the internal filter state (zi) for all stages/channels."""
        for st in self._stages:
            st["zi"][:] = sosfilt_zi(st["sos"])[:, None, :]

    def process(self, block: np.ndarray) -> np.ndarray:
        """
        Filter a block of shape (C, N). Returns filtered block of same shape.
        Maintains internal state across calls.
        """
        if block.ndim != 2 or block.shape[0] != self.C:
            raise ValueError(f"Expected block shape (C,N)=({self.C}, N); got {block.shape}")
        y = block.astype(np.float32, copy=True)

        # Apply each stage in sequence, maintaining per-channel zi
        for st in self._stages:
            sos = st["sos"]
            zi = st["zi"]  # (n_sections, C, 2)
            # process each channel with its own zi
            for ch in range(self.C):
                y[ch, :], zi[:, ch, :] = sosfilt(sos, y[ch, :], zi=zi[:, ch, :])
        return y

    # ------------- internals -------------

    def _design_filters(self, preserve_state: bool = False):
        """(Re)build SOS stages and initialize per-channel zi."""
        old_states = None
        if preserve_state and self._stages:
            # shallow capture of previous zi to try to keep continuity if topology identical
            old_states = [(st["name"], st["sos"].copy(), st["zi"].copy()) for st in self._stages]

        self._stages = []
        nyq = 0.5 * self.fs

        # helper to clamp and validate edges
        def _bounded(f):
            return max(1e-6, min(f, nyq * 0.999))

        # BANDPASS
        if self.enable_bandpass:
            lo = _bounded(self.bp_low)
            hi = _bounded(self.bp_high)
            if hi <= lo:
                hi = lo + 1.0
            sos_bp = butter(self.bp_order, [lo, hi], btype="band", fs=self.fs, output="sos")
            self._stages.append(self._mk_stage("bandpass", sos_bp))

        # NOTCH (one or more)
        if self.enable_notch and len(self.notch_freqs) > 0:
            for f0 in self.notch_freqs:
                f0b = _bounded(f0)
                # design second-order notch and convert to sos
                b, a = iirnotch(w0=f0b, Q=self.notch_q, fs=self.fs)  # returns (b, a)
                sos_n = tf2sos(b, a)
                self._stages.append(self._mk_stage(f"notch_{int(round(f0))}", sos_n))

        # LOWPASS (optional)
        if self.enable_lowpass and self.lp_cut is not None:
            cut = _bounded(self.lp_cut)
            sos_lp = butter(self.lp_order, cut, btype="low", fs=self.fs, output="sos")
            self._stages.append(self._mk_stage("lowpass", sos_lp))

        # initialize zi arrays (n_sections, C, 2)
        for st in self._stages:
            nsec = st["sos"].shape[0]
            st["zi"] = np.tile(sosfilt_zi(st["sos"])[:, None, :], (1, self.C, 1)).astype(np.float32)

        # attempt to preserve state if topology identical
        if old_states is not None:
            for name, sos_old, zi_old in old_states:
                for st in self._stages:
                    if st["name"] == name and st["sos"].shape == sos_old.shape and np.allclose(st["sos"], sos_old):
                        st["zi"] = zi_old  # keep prior state

    def _mk_stage(self, name, sos):
        return {"name": name, "sos": np.asarray(sos, dtype=np.float64), "zi": None}

