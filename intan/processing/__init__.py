from ._metrics_utils import load_metrics_data, get_metrics_file
from ._filters import (
    notch_filter,
    butter_bandpass,
    butter_lowpass,
    butter_lowpass_filter,
    butter_bandpass_filter,
    filter_emg,
    rectify,
    window_rms,
    window_rms_1D,
    compute_rms,
    downsample,
    common_average_reference,
    compute_grid_average,
    z_score_norm,
    apply_pca,
    orthogonalize,
    normalize
)
from ._emg_trial_selector import EMGViewerApp, launch_emg_selector