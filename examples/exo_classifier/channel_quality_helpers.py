#!/usr/bin/env python3
"""
channel_quality_helpers.py

Helper functions for channel quality control in dataset building and realtime prediction.
Uses the ChannelQC class from intan.processing to identify bad channels.
"""

import numpy as np
import logging
from typing import List, Set

try:
    from intan.processing import ChannelQC, QCParams
except ImportError:
    logging.warning("Could not import ChannelQC from intan.processing - quality control disabled")
    ChannelQC = None
    QCParams = None


def get_good_channels_from_recording(
        emg_data: np.ndarray,
        fs: float,
        window_sec: float = 2.0,
        verbose: bool = False
) -> tuple[List[int], Set[int]]:
    """
    Analyze a full EMG recording and return good channel indices.

    Parameters
    ----------
    emg_data : np.ndarray
        EMG data, shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    window_sec : float
        Window size for QC analysis (default: 2 seconds)
    verbose : bool
        Print detailed QC results

    Returns
    -------
    good_channels : list of int
        Indices of channels that passed QC
    bad_channels : set of int
        Indices of channels that failed QC
    """

    if ChannelQC is None:
        logging.warning("ChannelQC not available - using all channels")
        return list(range(emg_data.shape[0])), set()

    n_channels = emg_data.shape[0]
    n_samples = emg_data.shape[1]

    if verbose:
        logging.info(f"Running channel QC: {n_channels} channels, {n_samples} samples ({n_samples / fs:.1f}s)")

    # Initialize QC
    params = QCParams(
        robust_z_bad=3.0,  # Flag channels with RMS > 3 std from median
        robust_z_warn=2.0,  # Warn at 2 std
        pl_ratio_thresh=0.30,  # Flag if 60Hz power > 30% of total
        flat_std_min=1.0,  # Flag if std < 1 µV after filtering
        zc_min_hz=3.0,  # Flag if < 3 zero crossings per second
        consec_bad_needed=3,  # Need 3 consecutive bad evaluations
        consec_good_needed=5,  # Need 5 consecutive good to recover
    )

    qc = ChannelQC(fs=int(fs), n_channels=n_channels, window_sec=window_sec, params=params)

    # Feed data in chunks
    chunk_samples = int(window_sec * fs)
    n_chunks = n_samples // chunk_samples

    for i in range(n_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = emg_data[:, start:end].T  # QC expects (samples, channels)
        qc.update(chunk)

        # Evaluate periodically
        if i % 5 == 0 or i == n_chunks - 1:
            qc.evaluate()

    # Final evaluation
    result = qc.evaluate()
    bad_channels = result['excluded']
    good_channels = [i for i in range(n_channels) if i not in bad_channels]

    if verbose:
        metrics = result['metrics']
        logging.info(f"\nChannel QC Results:")
        logging.info(f"  Good channels: {len(good_channels)}/{n_channels}")
        logging.info(f"  Bad channels: {len(bad_channels)}/{n_channels}")

        if bad_channels and verbose:
            logging.info(f"  Bad channel indices: {sorted(bad_channels)}")

            # Show why they're bad
            for ch in sorted(list(bad_channels)[:10]):  # Show first 10
                reasons = []
                if metrics['robust_z'][ch] > params.robust_z_bad:
                    reasons.append(f"high RMS (z={metrics['robust_z'][ch]:.1f})")
                if metrics['pl_ratio'][ch] > params.pl_ratio_thresh:
                    reasons.append(f"60Hz noise ({metrics['pl_ratio'][ch] * 100:.1f}%)")
                if metrics['std'][ch] < params.flat_std_min:
                    reasons.append(f"flatline (std={metrics['std'][ch]:.2f})")
                if metrics['zc_hz'][ch] < params.zc_min_hz:
                    reasons.append(f"low activity (zc={metrics['zc_hz'][ch]:.1f}Hz)")

                logging.info(f"    CH{ch}: {', '.join(reasons)}")

    return good_channels, bad_channels


def get_good_channels_realtime(
        qc: 'ChannelQC',
        emg_chunk: np.ndarray,
        verbose: bool = False
) -> tuple[List[int], Set[int]]:
    """
    Update realtime QC with new chunk and return current good channels.

    Parameters
    ----------
    qc : ChannelQC
        Active ChannelQC instance
    emg_chunk : np.ndarray
        New EMG data chunk, shape (n_channels, n_samples)
    verbose : bool
        Print QC updates

    Returns
    -------
    good_channels : list of int
        Currently good channel indices
    bad_channels : set of int
        Currently bad channel indices
    """

    if qc is None:
        n_channels = emg_chunk.shape[0]
        return list(range(n_channels)), set()

    # Update QC with new data
    chunk_transposed = emg_chunk.T  # QC expects (samples, channels)
    qc.update(chunk_transposed)

    # Evaluate
    result = qc.evaluate()
    bad_channels = result['excluded']
    n_channels = emg_chunk.shape[0]
    good_channels = [i for i in range(n_channels) if i not in bad_channels]

    if verbose and bad_channels:
        logging.debug(f"QC: {len(bad_channels)} bad channels: {sorted(list(bad_channels)[:5])}...")

    return good_channels, bad_channels


def initialize_realtime_qc(fs: float, n_channels: int, window_sec: float = 0.5) -> 'ChannelQC':
    """
    Initialize ChannelQC for realtime use.

    Parameters
    ----------
    fs : float
        Sampling frequency
    n_channels : int
        Number of channels
    window_sec : float
        Window size for analysis (default: 0.5 seconds)

    Returns
    -------
    qc : ChannelQC or None
        Initialized QC instance, or None if unavailable
    """

    if ChannelQC is None:
        logging.warning("ChannelQC not available - quality control disabled")
        return None

    params = QCParams(
        robust_z_bad=3.0,
        robust_z_warn=2.0,
        pl_ratio_thresh=0.30,
        flat_std_min=1.0,
        zc_min_hz=3.0,
        consec_bad_needed=3,
        consec_good_needed=5,
        psd_every_n_evals=5,  # Check PSD every 5 evaluations (not every window)
    )

    qc = ChannelQC(
        fs=int(fs),
        n_channels=n_channels,
        window_sec=window_sec,
        params=params
    )

    return qc