"""
Temporal filtering utilities for smoothing time-series predictions.

Provides various filters for reducing noise in continuous angle predictions
while preserving dynamic movement characteristics.
"""

import numpy as np
from typing import Optional, Literal
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.ndimage import gaussian_filter1d


def moving_average(
    signal: np.ndarray,
    window_size: int = 5,
    axis: int = 0
) -> np.ndarray:
    """
    Apply moving average filter to smooth predictions.
    
    Args:
        signal: Input signal (n_samples, n_outputs) or (n_samples,)
        window_size: Window size for averaging (must be odd)
        axis: Axis along which to apply filter (0 for time)
        
    Returns:
        Smoothed signal with same shape as input
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd
    
    pad_size = window_size // 2
    
    if signal.ndim == 1:
        # 1D signal
        padded = np.pad(signal, pad_size, mode='edge')
        smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
    else:
        # Multi-dimensional - apply along axis
        padded = np.pad(signal, [(pad_size, pad_size) if i == axis else (0, 0) 
                                 for i in range(signal.ndim)], mode='edge')
        kernel = np.ones(window_size) / window_size
        smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), 
                                       axis, padded)
    
    return smoothed


def lowpass_filter(
    signal: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
    axis: int = 0
) -> np.ndarray:
    """
    Apply Butterworth lowpass filter to smooth predictions.
    
    Args:
        signal: Input signal (n_samples, n_outputs) or (n_samples,)
        cutoff: Cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
        axis: Axis along which to apply filter
        
    Returns:
        Smoothed signal with same shape as input
    """
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    return filtfilt(b, a, signal, axis=axis)


def savitzky_golay(
    signal: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    axis: int = 0
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for smoothing with edge preservation.
    
    Better than moving average for preserving peaks/valleys.
    
    Args:
        signal: Input signal (n_samples, n_outputs) or (n_samples,)
        window_length: Window size (must be odd, ≥ polyorder+2)
        polyorder: Polynomial order
        axis: Axis along which to apply filter
        
    Returns:
        Smoothed signal with same shape as input
    """
    if window_length % 2 == 0:
        window_length += 1
    
    return savgol_filter(signal, window_length, polyorder, axis=axis)


def gaussian_smooth(
    signal: np.ndarray,
    sigma: float = 2.0,
    axis: int = 0
) -> np.ndarray:
    """
    Apply Gaussian smoothing filter.
    
    Args:
        signal: Input signal (n_samples, n_outputs) or (n_samples,)
        sigma: Standard deviation of Gaussian kernel
        axis: Axis along which to apply filter
        
    Returns:
        Smoothed signal with same shape as input
    """
    return gaussian_filter1d(signal, sigma, axis=axis)


def exponential_smooth(
    signal: np.ndarray,
    alpha: float = 0.3,
    axis: int = 0
) -> np.ndarray:
    """
    Apply exponential moving average for causal smoothing.
    
    Suitable for real-time applications (causal, no look-ahead).
    
    Args:
        signal: Input signal (n_samples, n_outputs) or (n_samples,)
        alpha: Smoothing factor (0-1), smaller = more smoothing
        axis: Axis along which to apply filter
        
    Returns:
        Smoothed signal with same shape as input
    """
    if signal.ndim == 1:
        smoothed = np.zeros_like(signal)
        smoothed[0] = signal[0]
        for t in range(1, len(signal)):
            smoothed[t] = alpha * signal[t] + (1 - alpha) * smoothed[t-1]
    else:
        # Apply along specified axis
        smoothed = np.zeros_like(signal)
        if axis == 0:
            smoothed[0] = signal[0]
            for t in range(1, signal.shape[0]):
                smoothed[t] = alpha * signal[t] + (1 - alpha) * smoothed[t-1]
        else:
            smoothed[:, 0] = signal[:, 0]
            for t in range(1, signal.shape[1]):
                smoothed[:, t] = alpha * signal[:, t] + (1 - alpha) * smoothed[:, t-1]
    
    return smoothed


def adaptive_smooth(
    signal: np.ndarray,
    velocity_threshold: float = 10.0,
    slow_alpha: float = 0.1,
    fast_alpha: float = 0.7,
    fs: Optional[float] = None,
    axis: int = 0
) -> np.ndarray:
    """
    Apply adaptive smoothing based on movement velocity.
    
    Uses heavy smoothing during slow movements, light smoothing during fast.
    
    Args:
        signal: Input signal (n_samples, n_outputs) or (n_samples,)
        velocity_threshold: Threshold for switching smoothing (degrees/sec)
        slow_alpha: Smoothing factor for slow movements (0-1)
        fast_alpha: Smoothing factor for fast movements (0-1)
        fs: Sampling frequency (Hz), required if using velocity threshold
        axis: Axis along which to apply filter
        
    Returns:
        Smoothed signal with same shape as input
    """
    if fs is None:
        raise ValueError("Sampling frequency (fs) required for adaptive smoothing")
    
    # Compute velocity
    velocity = np.diff(signal, axis=axis) * fs
    velocity = np.pad(velocity, [(1, 0) if i == axis else (0, 0) 
                                 for i in range(signal.ndim)], mode='edge')
    
    # Get velocity magnitude
    if signal.ndim == 1:
        vel_mag = np.abs(velocity)
    else:
        vel_mag = np.linalg.norm(velocity, axis=1 if axis == 0 else 0)
        if axis == 1:
            vel_mag = vel_mag[:, np.newaxis]
    
    # Adaptive alpha
    alpha = np.where(vel_mag > velocity_threshold, fast_alpha, slow_alpha)
    
    # Apply exponential smoothing with adaptive alpha
    if signal.ndim == 1:
        smoothed = np.zeros_like(signal)
        smoothed[0] = signal[0]
        for t in range(1, len(signal)):
            smoothed[t] = alpha[t] * signal[t] + (1 - alpha[t]) * smoothed[t-1]
    else:
        smoothed = np.zeros_like(signal)
        if axis == 0:
            smoothed[0] = signal[0]
            for t in range(1, signal.shape[0]):
                a = alpha[t] if alpha.ndim == 1 else alpha[t, :]
                smoothed[t] = a * signal[t] + (1 - a) * smoothed[t-1]
        else:
            raise NotImplementedError("Adaptive smoothing only supports axis=0")
    
    return smoothed


def smooth_predictions(
    predictions: np.ndarray,
    method: Literal['moving_average', 'lowpass', 'savgol', 'gaussian', 
                    'exponential', 'adaptive'] = 'savgol',
    fs: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """
    Smooth prediction time-series using specified method.
    
    Args:
        predictions: Predictions to smooth (n_samples, n_joints)
        method: Smoothing method to use
        fs: Sampling frequency (Hz), required for some methods
        **kwargs: Additional parameters for specific methods
            moving_average: window_size (default: 5)
            lowpass: cutoff (default: 5.0), order (default: 4)
            savgol: window_length (default: 11), polyorder (default: 3)
            gaussian: sigma (default: 2.0)
            exponential: alpha (default: 0.3)
            adaptive: velocity_threshold, slow_alpha, fast_alpha
            
    Returns:
        Smoothed predictions with same shape
        
    Examples:
        >>> # Savitzky-Golay (good default)
        >>> smooth = smooth_predictions(preds, method='savgol', window_length=15)
        
        >>> # Lowpass filter
        >>> smooth = smooth_predictions(preds, method='lowpass', fs=20, cutoff=5.0)
        
        >>> # Adaptive based on movement speed
        >>> smooth = smooth_predictions(preds, method='adaptive', fs=20, 
        ...                             velocity_threshold=15.0)
    """
    if method == 'moving_average':
        window_size = kwargs.get('window_size', 5)
        return moving_average(predictions, window_size=window_size)
    
    elif method == 'lowpass':
        if fs is None:
            raise ValueError("Sampling frequency (fs) required for lowpass filter")
        cutoff = kwargs.get('cutoff', 5.0)
        order = kwargs.get('order', 4)
        return lowpass_filter(predictions, cutoff=cutoff, fs=fs, order=order)
    
    elif method == 'savgol':
        window_length = kwargs.get('window_length', 11)
        polyorder = kwargs.get('polyorder', 3)
        return savitzky_golay(predictions, window_length=window_length, 
                             polyorder=polyorder)
    
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 2.0)
        return gaussian_smooth(predictions, sigma=sigma)
    
    elif method == 'exponential':
        alpha = kwargs.get('alpha', 0.3)
        return exponential_smooth(predictions, alpha=alpha)
    
    elif method == 'adaptive':
        if fs is None:
            raise ValueError("Sampling frequency (fs) required for adaptive smoothing")
        return adaptive_smooth(predictions, fs=fs, **kwargs)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
