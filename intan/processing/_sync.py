"""
Synchronization utilities for aligning multi-modal recordings.

This module provides tools for temporal alignment of data streams with
different sampling rates and start times (e.g., EMG and video).
"""

from typing import Dict, Tuple
import numpy as np
from scipy.signal import correlate, butter, filtfilt
from scipy.interpolate import interp1d


def compute_landmark_movement_signal(
    landmarks: np.ndarray,
    time_vector: np.ndarray,
    method: str = 'velocity'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute movement signal from hand landmarks.
    
    Uses fingertip positions (landmarks 4, 8, 12, 16, 20) to capture
    overall hand movement for synchronization with EMG.
    
    Args:
        landmarks: (n_frames, 21, 3) landmark positions (MediaPipe hand format)
        time_vector: (n_frames,) timestamps in seconds
        method: 'velocity', 'acceleration', or 'displacement'
        
    Returns:
        signal: (n_frames,) movement signal (scalar at each frame)
        timestamps: (n_frames,) time values (same as input)
        
    Examples:
        >>> landmarks = np.load('hand_landmarks.npz')['landmarks']
        >>> time_vec = np.load('hand_landmarks.npz')['time_vector']
        >>> velocity_signal, times = compute_landmark_movement_signal(
        ...     landmarks, time_vec, method='velocity'
        ... )
    """
    from scipy.ndimage import gaussian_filter1d
    
    # Fingertip indices (thumb, index, middle, ring, pinky)
    fingertip_indices = [4, 8, 12, 16, 20]
    
    # Get fingertip positions
    fingertips = landmarks[:, fingertip_indices, :]  # (frames, 5, 3)
    
    # Compute centroid of fingertips
    centroid = np.mean(fingertips, axis=1)  # (frames, 3)
    
    if method == 'velocity':
        # Compute velocity magnitude
        dt = np.diff(time_vector)
        dt = np.concatenate([[dt[0]], dt])  # Pad to match length
        
        velocity = np.diff(centroid, axis=0, prepend=centroid[0:1])
        velocity_mag = np.linalg.norm(velocity, axis=1) / (dt + 1e-9)
        signal = velocity_mag
        
    elif method == 'acceleration':
        # Compute acceleration magnitude
        dt = np.diff(time_vector)
        dt = np.concatenate([[dt[0]], dt])
        
        velocity = np.diff(centroid, axis=0, prepend=centroid[0:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        accel_mag = np.linalg.norm(acceleration, axis=1) / (dt**2 + 1e-9)
        signal = accel_mag
        
    elif method == 'displacement':
        # Total displacement from first frame
        displacement = np.linalg.norm(centroid - centroid[0], axis=1)
        signal = displacement
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'velocity', 'acceleration', or 'displacement'")
    
    # Smooth the signal to reduce noise
    signal = gaussian_filter1d(signal, sigma=2)
    
    return signal, time_vector


def compute_emg_envelope_signal(
    emg_data: np.ndarray,
    fs: float,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    notch_freq: float = 60.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute aggregate EMG envelope signal for synchronization.
    
    Processes multi-channel EMG to create a single time-series signal
    representing overall muscle activity.
    
    Args:
        emg_data: (channels, samples) raw EMG data
        fs: Sampling rate in Hz
        lowcut: Bandpass filter low cutoff (Hz)
        highcut: Bandpass filter high cutoff (Hz)
        notch_freq: Power line frequency to remove (Hz, 0 to disable)
        
    Returns:
        signal: (samples,) normalized envelope signal
        timestamps: (samples,) time values in seconds
        
    Examples:
        >>> from intan.io import load_rhd_file
        >>> data = load_rhd_file('recording.rhd')
        >>> emg = data['amplifier_data']
        >>> fs = data['frequency_parameters']['amplifier_sample_rate']
        >>> envelope, times = compute_emg_envelope_signal(emg, fs)
    """
    from ._filters import bandpass_filter, notch_filter
    
    # Preprocess: remove line noise and bandpass filter
    if notch_freq > 0:
        emg_filt = notch_filter(emg_data, fs=fs, f0=notch_freq)
    else:
        emg_filt = emg_data.copy()
    
    emg_filt = bandpass_filter(emg_filt, lowcut=lowcut, highcut=highcut, fs=fs)
    
    # Compute envelope (rectify + smooth)
    emg_rect = np.abs(emg_filt)
    
    # Average across channels
    emg_avg = np.mean(emg_rect, axis=0)
    
    # Low-pass filter for envelope (10 Hz cutoff)
    b, a = butter(4, 10.0 / (fs / 2), btype='low')
    envelope = filtfilt(b, a, emg_avg)
    
    # Normalize (zero mean, unit std)
    envelope = (envelope - np.mean(envelope)) / (np.std(envelope) + 1e-9)
    
    timestamps = np.arange(len(envelope)) / fs
    
    return envelope, timestamps


def find_sync_offset(
    emg_signal: np.ndarray,
    emg_times: np.ndarray,
    landmark_signal: np.ndarray,
    landmark_times: np.ndarray,
    max_offset_sec: float = 10.0
) -> Dict:
    """
    Find time offset between EMG and landmark data using cross-correlation.
    
    Interpolates both signals to a common timeline and performs cross-correlation
    to find the time shift that maximizes alignment.
    
    Args:
        emg_signal: EMG envelope signal (1D array)
        emg_times: EMG timestamps in seconds
        landmark_signal: Landmark movement signal (1D array)
        landmark_times: Landmark timestamps in seconds
        max_offset_sec: Maximum expected offset in seconds (search window)
        
    Returns:
        Dictionary containing:
            - offset_sec: Time offset in seconds (positive = landmarks delayed)
            - offset_samples_emg: Offset in EMG samples
            - offset_samples_landmarks: Offset in landmark frames
            - correlation_peak: Peak cross-correlation value
            - confidence: Confidence score (0-1)
            - emg_fs: EMG sampling rate
            - landmark_fs: Landmark sampling rate
            
    Note:
        A positive offset means the landmark signal is DELAYED relative to EMG.
        To align data, SUBTRACT the offset from landmark timestamps.
        
    Examples:
        >>> emg_env, emg_t = compute_emg_envelope_signal(emg_data, fs=1000)
        >>> lm_vel, lm_t = compute_landmark_movement_signal(landmarks, times)
        >>> sync = find_sync_offset(emg_env, emg_t, lm_vel, lm_t)
        >>> print(f"Offset: {sync['offset_sec']:.3f}s, Confidence: {sync['confidence']:.2f}")
    """
    # Determine common sampling rate (use EMG rate)
    common_fs = 1.0 / (emg_times[1] - emg_times[0])
    
    # Create common time vector spanning overlap with margin
    t_start = max(emg_times[0], landmark_times[0]) - max_offset_sec
    t_end = min(emg_times[-1], landmark_times[-1]) + max_offset_sec
    common_times = np.arange(t_start, t_end, 1.0 / common_fs)
    
    # Interpolate both signals to common timeline
    f_emg = interp1d(emg_times, emg_signal, kind='linear', 
                     fill_value=0, bounds_error=False)
    f_landmark = interp1d(landmark_times, landmark_signal, kind='linear',
                          fill_value=0, bounds_error=False)
    
    emg_resampled = f_emg(common_times)
    landmark_resampled = f_landmark(common_times)
    
    # Normalize signals
    emg_resampled = (emg_resampled - np.mean(emg_resampled)) / (np.std(emg_resampled) + 1e-9)
    landmark_resampled = (landmark_resampled - np.mean(landmark_resampled)) / (np.std(landmark_resampled) + 1e-9)
    
    # Cross-correlate
    correlation = correlate(emg_resampled, landmark_resampled, mode='same')
    
    # Find peak within search window
    center_idx = len(correlation) // 2
    max_offset_samples = int(max_offset_sec * common_fs)
    search_start = max(0, center_idx - max_offset_samples)
    search_end = min(len(correlation), center_idx + max_offset_samples)
    
    search_region = correlation[search_start:search_end]
    peak_idx_rel = np.argmax(search_region)
    peak_idx = search_start + peak_idx_rel
    
    # Compute offset
    offset_samples = peak_idx - center_idx
    offset_sec = offset_samples / common_fs
    
    # Compute confidence metric (peak prominence)
    peak_value = correlation[peak_idx]
    mean_corr = np.mean(np.abs(correlation[search_start:search_end]))
    std_corr = np.std(correlation[search_start:search_end])
    confidence = min(1.0, abs(peak_value - mean_corr) / (3 * std_corr + 1e-9))
    
    # Convert offset to original sample rates
    emg_fs = 1.0 / (emg_times[1] - emg_times[0])
    landmark_fs = 1.0 / (landmark_times[1] - landmark_times[0])
    
    offset_samples_emg = int(offset_sec * emg_fs)
    offset_samples_landmarks = int(offset_sec * landmark_fs)
    
    return {
        'offset_sec': float(offset_sec),
        'offset_samples_emg': int(offset_samples_emg),
        'offset_samples_landmarks': int(offset_samples_landmarks),
        'correlation_peak': float(peak_value),
        'confidence': float(confidence),
        'emg_fs': float(emg_fs),
        'landmark_fs': float(landmark_fs)
    }


def load_sync_offset(sync_file: str) -> float:
    """
    Load synchronization offset from JSON file.
    
    Args:
        sync_file: Path to sync JSON file
        
    Returns:
        Time offset in seconds (0.0 if file not found or invalid)
        
    Examples:
        >>> offset = load_sync_offset('sync/recording_sync.json')
        >>> aligned_times = landmark_times - offset
    """
    import json
    import os
    
    if not os.path.exists(sync_file):
        return 0.0
    
    try:
        with open(sync_file, 'r') as f:
            sync_info = json.load(f)
        return float(sync_info.get('offset_sec', 0.0))
    except (json.JSONDecodeError, KeyError, ValueError):
        return 0.0


def save_sync_offset(sync_info: Dict, output_file: str):
    """
    Save synchronization info to JSON file.
    
    Args:
        sync_info: Dictionary from find_sync_offset()
        output_file: Path to output JSON file
        
    Examples:
        >>> sync_info = find_sync_offset(emg_env, emg_t, lm_vel, lm_t)
        >>> save_sync_offset(sync_info, 'sync/recording_sync.json')
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(sync_info, f, indent=2)
