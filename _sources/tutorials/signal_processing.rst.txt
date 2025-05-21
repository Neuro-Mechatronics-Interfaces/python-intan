Signal Processing and Feature Extraction
===========================================

This tutorial explains the main EMG signal processing utilities in ``intan.processing``, including filtering, rectification, normalization, feature extraction, and dimensionality reduction.
All functions are compatible with multi-channel EMG arrays loaded from `.rhd` or `.dat` files.

Filtering and Preprocessing
----------------------------

The `intan.processing._filters` module provides robust EMG filtering routines, including bandpass, lowpass, notch filtering, and rectification.

**Example: Bandpass and Notch Filtering**

.. code-block:: python

    from intan.processing import bandpass_filter, notch_filter, rectify

    # Assume emg_data shape is (channels, samples), fs is the sampling rate in Hz
    filtered = bandpass_filter(emg_data, lowcut=20, highcut=450, fs=fs)
    filtered = notch_filter(filtered, fs=fs, f0=60)  # Remove 60 Hz noise

    # Rectify EMG (absolute value)
    rectified = rectify(filtered)

----

Windowed and Smoothed RMS
--------------------------

RMS (Root Mean Square) is often used to measure the envelope of EMG signals.

.. code-block:: python

    from intan.processing import window_rms, calculate_rms

    # Windowed RMS across each channel (e.g., 400-sample window)
    windowed_rms = window_rms(emg_data, window_size=400)

    # Non-overlapping RMS for feature extraction (returns (channels, windows))
    rms_features = calculate_rms(emg_data, window_size=400)

----

Envelope Extraction
---------------------

Use the analytic Hilbert envelope for precise EMG envelope estimation:

.. code-block:: python

    from intan.processing import envelope_extraction

    # Returns same shape as input
    envelope = envelope_extraction(emg_data, method='hilbert')

----

Normalization and Referencing
--------------------------------

**Z-score normalization:** Standardizes each channel to zero mean/unit variance.

.. code-block:: python

    from intan.processing import z_score_norm

    normalized = z_score_norm(emg_data)

----

**Common Average Referencing (CAR):** Remove global artifacts across all channels.

.. code-block:: python

    from intan.processing import common_average_reference

    car_data = common_average_reference(emg_data)

----

Dimensionality Reduction (PCA)
----------------------------------

Reduce channel count and noise with Principal Component Analysis (PCA):

.. code-block:: python

    from intan.processing import apply_pca

    # Reduce to 8 principal components
    pca_data, explained_var = apply_pca(emg_data, num_components=8)
    print("Explained variance ratio:", explained_var)

----

Sliding Windows for Segmentation
-----------------------------------

Split continuous EMG into overlapping windows for further analysis:

.. code-block:: python

    from intan.processing import sliding_window

    # Example: 250 ms windows with 50 ms step size
    window_size = int(0.25 * fs)
    step_size = int(0.05 * fs)
    windows = sliding_window(emg_data, window_size, step_size)
    # Each entry is (channels, window_size)

----

Feature Extraction
----------------------

The `intan.processing._features` module provides standard time-domain EMG features:

- Mean Absolute Value (MAV)
- Zero Crossings (ZC)
- Slope Sign Changes (SSC)
- Waveform Length (WL)
- Root Mean Square (RMS)

Extract all features for each channel in a window:

.. code-block:: python

    from intan.processing import extract_features

    # Example: compute all features for every channel in a segment
    feats = extract_features(emg_window)
    print("Feature vector shape:", feats.shape)

----

You can also specify which features to extract:

.. code-block:: python

    feature_list = ['mean_absolute_value', 'zero_crossings', 'waveform_length']
    feats = extract_features(emg_window, feature_fns=feature_list)

----

Full Preprocessing Pipeline Example
----------------------------------------

The following combines filtering, rectification, and RMS envelope smoothing in a single pipeline:

.. code-block:: python

    from intan.processing import bandpass_filter, notch_filter, rectify, window_rms

    # Bandpass filter
    emg_filtered = bandpass_filter(emg_data, lowcut=20, highcut=450, fs=fs)
    # Notch filter for line noise
    emg_notched = notch_filter(emg_filtered, fs=fs, f0=60)
    # Rectify
    emg_rectified = rectify(emg_notched)
    # Windowed RMS smoothing
    emg_smooth = window_rms(emg_rectified, window_size=400)

----

Noise Characterization Example
-----------------------------------

Compare baseline RMS noise between two conditions (e.g., dry vs. wet electrodes):

.. code-block:: python

    from intan.processing import bandpass_filter, notch_filter
    import numpy as np

    # Load EMG for each condition
    emg_dry = ...   # (channels, samples)
    emg_wet = ...   # (channels, samples)
    fs = 4000

    def compute_rms(data):
        return np.sqrt(np.mean(data ** 2, axis=1))

    emg_dry_filt = notch_filter(bandpass_filter(emg_dry, fs=fs), fs=fs)
    emg_wet_filt = notch_filter(bandpass_filter(emg_wet, fs=fs), fs=fs)
    rms_dry = compute_rms(emg_dry_filt)
    rms_wet = compute_rms(emg_wet_filt)

    print("Dry mean RMS:", np.mean(rms_dry))
    print("Wet mean RMS:", np.mean(rms_wet))

----

Additional Tools
--------------------

- **Downsampling:** Reduce sampling rate for long recordings.

.. code-block:: python

    from intan.processing import downsample

    downsampled = downsample(emg_data, sampling_rate=fs, target_fs=1000)

----

- **Grid Averaging:** Compute averages across spatially grouped channels.

.. code-block:: python

    from intan.processing import compute_grid_average

    grid_avg = compute_grid_average(emg_data, grid_spacing=8)

----

Summary
-----------

These processing tools form the foundation for all subsequent feature extraction and classification tasks using EMG data collected with Intan systems.
For details on each function, see the API Reference.

