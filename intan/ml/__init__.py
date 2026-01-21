"""Top-level `intan.ml` package.

This module uses lazy imports so importing `intan.ml` does not immediately
load heavy ML libraries (PyTorch/TensorFlow). Accessing symbols (e.g.
`EMGRealTimePredictor`, `ModelManager`) will import their defining modules on
first use.
"""
from importlib import import_module
from typing import Dict, Tuple

# Map exported names to (module_name, attr_name)
_EXPORTS: Dict[str, Tuple[str, str]] = {
    'EMGRealTimePredictor': ('._ml_utilities', 'EMGRealTimePredictor'),
    'EMGRegressor': ('._models', 'EMGRegressor'),
    'EMGClassifier': ('._models', 'EMGClassifier'),
    'EMGClassifierCNNLSTM': ('._models', 'EMGClassifierCNNLSTM'),
    'ModelManager': ('._model_manager', 'ModelManager'),
    'write_training_metadata': ('._model_manager', 'write_training_metadata'),
    'load_training_metadata': ('._model_manager', 'load_training_metadata'),
    'evaluate_against_events': ('._evaluation', 'evaluate_against_events'),
    'classification_report_safe': ('._evaluation', 'classification_report_safe'),
    'confusion_matrix_safe': ('._evaluation', 'confusion_matrix_safe'),
    'extract_features_from_emg': ('._prediction_utils', 'extract_features_from_emg'),
    'compute_window_starts': ('._prediction_utils', 'compute_window_starts'),
    'predict_with_model': ('._prediction_utils', 'predict_with_model'),
    'predict_rhd_file': ('._prediction_utils', 'predict_rhd_file'),
    'predict_file': ('._prediction_modes', 'predict_file'),
    'predict_batch': ('._prediction_modes', 'predict_batch'),
    'predict_from_device': ('._prediction_modes', 'predict_from_device'),
    'predict_realtime_stream': ('._prediction_modes', 'predict_realtime_stream'),
}


def __getattr__(name: str):
    """Lazily import symbols from submodules on first attribute access."""
    if name in _EXPORTS:
        mod_name, attr = _EXPORTS[name]
        mod = import_module(__name__ + mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))


__all__ = list(_EXPORTS.keys())
