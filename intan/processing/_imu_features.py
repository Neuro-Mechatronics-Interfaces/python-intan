"""
intan.processing._imu_features

Utilities for extracting and normalizing IMU features from board ADC channels.

Used primarily in multi-modal EMG+IMU datasets for gesture classification.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def aggregate_imu_features(
    board_adc_data: Optional[np.ndarray],
    board_adc_channels: Optional[List[Dict[str, str]]],
    window_starts: np.ndarray,
    window_samples: int,
    mode: str = "rich",
) -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Compute IMU features over sliding windows from board ADC data.
    
    Parameters
    ----------
    board_adc_data : np.ndarray, optional
        Board ADC data array, shape (n_adc_channels, n_samples)
    board_adc_channels : list of dict, optional
        Channel metadata with 'channel_name' keys
    window_starts : np.ndarray
        Start indices for each window (1D array of integers)
    window_samples : int
        Window size in samples
    mode : str
        Feature mode:
        - 'mean': Simple mean per window
        - 'rich': Mean, std, min, max, RMS per window
    
    Returns
    -------
    features : np.ndarray or None
        Feature matrix, shape (n_windows, n_features), or None if no ADC data
    feature_names : list of str
        Feature column names (e.g., ['AccX_mean', 'AccX_std', ...])
    
    Examples
    --------
    >>> window_starts = np.array([0, 100, 200, 300])
    >>> features, names = aggregate_imu_features(
    ...     board_adc_data, board_adc_channels, window_starts, 100, mode='rich'
    ... )
    >>> features.shape
    (4, 30)  # 4 windows, 6 IMU channels * 5 stats = 30 features
    """
    if board_adc_data is None or board_adc_channels is None or len(board_adc_channels) == 0:
        return None, []
    
    imu_cols = [c.get("channel_name", f"ADC{i}") for i, c in enumerate(board_adc_channels)]
    L = board_adc_data.shape[1]
    feats = []
    
    # Define feature extractors
    def feat_mean(seg): return np.mean(seg, axis=0)
    def feat_std(seg): return np.std(seg, axis=0)
    def feat_min(seg): return np.min(seg, axis=0)
    def feat_max(seg): return np.max(seg, axis=0)
    def feat_rms(seg): return np.sqrt(np.mean(seg**2, axis=0))
    
    if mode == "mean":
        reducers = [("mean", feat_mean)]
    elif mode == "rich":
        reducers = [
            ("mean", feat_mean),
            ("std", feat_std),
            ("min", feat_min),
            ("max", feat_max),
            ("rms", feat_rms),
        ]
    else:
        raise ValueError(f"Unknown IMU feature mode: {mode}. Use 'mean' or 'rich'.")
    
    # Compute per window
    for s in window_starts:
        e = min(s + window_samples, L)
        seg = board_adc_data[:, s:e]
        if seg.shape[1] == 0:
            # Empty window - use zeros
            row = np.zeros(len(imu_cols) * len(reducers), dtype=np.float32)
        else:
            row = np.concatenate([fn(seg) for _, fn in reducers])
        feats.append(row)
    
    # Build feature names
    feature_names = []
    for stat_name, _ in reducers:
        for col in imu_cols:
            feature_names.append(f"{col}_{stat_name}")
    
    return np.array(feats, dtype=np.float32), feature_names


def append_imu_features(
    X: np.ndarray,
    y: np.ndarray,
    imu_features: Optional[np.ndarray],
    imu_feature_names: List[str],
    label_mask: np.ndarray,
    norm_mode: str = "zscore",
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Append IMU features to EMG feature matrix with normalization.
    
    Parameters
    ----------
    X : np.ndarray
        EMG feature matrix, shape (n_windows, n_emg_features)
    y : np.ndarray
        Label array (for metadata only, not modified)
    imu_features : np.ndarray, optional
        IMU feature matrix, shape (n_windows_total, n_imu_features)
    imu_feature_names : list of str
        Names of IMU features
    label_mask : np.ndarray
        Boolean mask indicating which windows to keep (after label filtering)
    norm_mode : str
        Normalization method: 'zscore' or 'robust'
    eps : float
        Small constant to avoid division by zero
    
    Returns
    -------
    X_augmented : np.ndarray
        Combined EMG+IMU features, shape (n_windows, n_emg_features + n_imu_features)
    metadata : dict
        Dictionary with normalization statistics:
        - 'imu_cols': Feature names
        - 'imu_feature_count': Number of IMU features
        - 'imu_norm_kind': Normalization method used
        - 'imu_norm_mean': Mean values (for zscore) or median (for robust)
        - 'imu_norm_std': Std values (for zscore) or MAD (for robust)
    
    Examples
    --------
    >>> X_emg = np.random.randn(100, 64)  # 100 windows, 64 EMG features
    >>> imu_feats = np.random.randn(120, 30)  # 120 windows (before filtering), 30 IMU features
    >>> mask = np.ones(120, dtype=bool)
    >>> mask[50:70] = False  # Remove some windows
    >>> X_combined, meta = append_imu_features(
    ...     X_emg, y, imu_feats, imu_names, mask, norm_mode='zscore'
    ... )
    >>> X_combined.shape
    (100, 94)  # 64 EMG + 30 IMU features
    """
    metadata = {
        "imu_cols": np.array(imu_feature_names or [], dtype=object),
        "imu_feature_count": np.array(len(imu_feature_names or []), dtype=np.int32),
        "imu_norm_kind": np.array(norm_mode, dtype=object),
        "imu_norm_mean": np.array([], dtype=np.float32),
        "imu_norm_std": np.array([], dtype=np.float32),
    }
    
    if imu_features is None or len(imu_feature_names) == 0:
        return X, metadata
    
    # Apply mask to IMU features
    imu_filtered = imu_features[label_mask]
    
    if imu_filtered.shape[0] != X.shape[0]:
        raise ValueError(
            f"IMU window count ({imu_filtered.shape[0]}) doesn't match "
            f"EMG window count ({X.shape[0]}) after filtering"
        )
    
    # Normalize IMU features
    if norm_mode == "zscore":
        mu = np.mean(imu_filtered, axis=0)
        sigma = np.std(imu_filtered, axis=0) + eps
        imu_normalized = (imu_filtered - mu) / sigma
        metadata["imu_norm_mean"] = mu.astype(np.float32)
        metadata["imu_norm_std"] = sigma.astype(np.float32)
    
    elif norm_mode == "robust":
        med = np.median(imu_filtered, axis=0)
        mad = np.median(np.abs(imu_filtered - med), axis=0) + eps
        imu_normalized = (imu_filtered - med) / mad
        metadata["imu_norm_mean"] = med.astype(np.float32)
        metadata["imu_norm_std"] = mad.astype(np.float32)
    
    else:
        raise ValueError(f"Unknown normalization mode: {norm_mode}. Use 'zscore' or 'robust'.")
    
    # Concatenate
    X_augmented = np.hstack([X, imu_normalized])
    
    return X_augmented, metadata
