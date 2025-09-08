from ._metrics_utils import load_metrics_data, get_metrics_file
from ._filters import (
    notch_filter,
    bandpass_filter,
    lowpass_filter,
)
from ._realtime_filter import RealtimeFilter
from ._features import (
    FEATURE_REGISTRY,
    rectify,
    window_rms,
    window_rms_1D,
    compute_rms,
    compute_rolling_rms,
    downsample,
    common_average_reference,
    compute_grid_average,
    z_score_norm,
    #apply_pca,
    orthogonalize,
    normalize,
    extract_features,
    extract_features_sliding_window,
    feature_spec_from_registry,
    envelope_extraction,
)
from ._preprocessing import EMGPreprocessor
from ._transformations import (
    estimate_lag_coarse_to_fine,
    normxcorr_offset,
    align_by_lag,
    align_multichannel_by_lag,
)
from ._fill import FillStats, fix_missing_emg
