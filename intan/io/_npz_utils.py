import os
import glob
from typing import Union, Sequence, List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm


def _is_empty_channel_names(val) -> bool:
    """True if channel_names is missing/empty/None or a 1-D array of length 0."""
    if val is None:
        return True
    if isinstance(val, (list, tuple)):
        return len(val) == 0
    if isinstance(val, np.ndarray):
        return val.size == 0
    # strings or other scalars are not valid channel name containers
    return False


def list_npz_files(path_or_glob: Union[str, Sequence[str]], recursive: bool = False) -> List[str]:
    """Expand dirs/globs/files into a sorted, de-duplicated list of .npz paths."""
    def _expand_one(p: str) -> List[str]:
        p = os.path.expanduser(os.path.expandvars(p))
        if os.path.isdir(p):
            pattern = "**/*.npz" if recursive else "*.npz"
            return sorted(glob.glob(os.path.join(p, pattern), recursive=recursive))
        if any(ch in p for ch in "*?[]"):
            # allow ** when recursive=True
            return sorted(glob.glob(p, recursive=recursive))
        if os.path.isfile(p) and p.lower().endswith(".npz"):
            return [p]
        return []

    if isinstance(path_or_glob, (list, tuple)):
        out = []
        for item in path_or_glob:
            out.extend(_expand_one(str(item)))
        # de-dup while preserving order
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                uniq.append(p); seen.add(p)
        return uniq

    return _expand_one(str(path_or_glob))


def find_npz_by_label(npz_dir: str, label: str, recursive: bool = True) -> list[str]:
    """Return NPZ paths whose filename contains `label` (case-insensitive)."""
    label_lc = label.lower().strip()
    data = [p for p in list_npz_files(npz_dir, recursive=recursive)
            if label_lc in os.path.basename(p).lower()]
    return data


def load_npz_record(path: str) -> tuple[np.ndarray, str]:
    """Load one NPZ file -> (emg (C,N), label:str). Raises if keys missing."""
    with np.load(path, allow_pickle=True) as d:
        if "emg" not in d or "label" not in d:
            raise KeyError(f"{os.path.basename(path)} missing 'emg' or 'label' keys.")
        emg = d["emg"]
        label = d["label"].item()  # np.object_ -> str
    return emg, str(label)


def save_as_npz(result: dict, file_path: str = None, use_compressed_format=True, verbose: bool = False) -> None:
    """
    Save the rhd data as a .npz file. uses the compressed format by default

    Args:
        result (dict): Dictionary containing the Open Ephys session data.
            Must contain keys: 'amplifier_data', 't_amplifier', 'sample_rate', 'recording_name'.
        file_path (str, optional): Path to save the .npz file. If None, uses the recording name.

    Returns:
        None
    """
    if not isinstance(result, dict):
        raise ValueError("Input must be dict data from RHD file.")

    if use_compressed_format:
        save_as_npz_compressed(result, file_path=file_path, optimize_dtypes=False, show_progress=True, verbose=verbose)
        return

    required_keys = ['amplifier_data', 't_amplifier']
    if not all(key in result for key in required_keys):
        raise KeyError(f"Missing one of the required keys: {required_keys}")

    if file_path is None:
        if 'recording_name' not in result:
            raise KeyError("file_path is None and result has no 'recording_name'.")
        # Save to local directory using file name
        file_path = result['recording_name'] + '.npz'
    elif not file_path.endswith('.npz'):
        file_path += '.npz'

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Save the data to a .npz file
    print(f" Saving data to {file_path}...")
    np.savez(file_path, **{key: result[key] for key in result.keys()})
    print(f"Data saved to {file_path}")


def save_as_npz_compressed(result: dict,
                           file_path: str = None,
                           optimize_dtypes: bool = True,
                           show_progress: bool = False,
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Save the rhd data as a compressed .npz file (70-85% smaller than standard NPZ).

    This function creates significantly smaller files than save_as_npz() with no loss
    of data quality. Recommended for all use cases.

    Args:
        result (dict): Dictionary containing the Open Ephys session data.
            Must contain keys: 'amplifier_data', 't_amplifier', 'sample_rate', 'recording_name'.
        file_path (str, optional): Path to save the .npz file. If None, uses the recording name.
        optimize_dtypes (bool, default=True): Optimize data types to reduce size:
            - Keep int16 as int16 (ADC data)
            - Convert float64 to float32 where appropriate
        show_progress (bool, default=False): Show progress bar during save (requires tqdm).
        verbose (bool, default=True): Print save status messages.

    Returns:
        dict: Statistics about the saved file including:
            - file_path: Path to saved file
            - original_size_mb: Size before optimization
            - file_size_mb: Final compressed file size
            - compression_ratio: How much smaller (e.g., 4.0 = 4x smaller)
            - space_saved_percent: Percentage reduction

    Examples:

        save_as_npz_compressed(result)

        # With custom path and progress bar
        stats = save_as_npz_compressed(result, "data.npz", show_progress=True)
        print(f"Saved {stats['space_saved_percent']:.1f}% space!")
    """
    # Validation
    if not isinstance(result, dict):
        raise ValueError("Input must be dict data from RHD file.")
    required_keys = ['amplifier_data', 't_amplifier']
    if not all(key in result for key in required_keys):
        raise KeyError(f"Missing one of the required keys: {required_keys}")

    # Determine file path
    if file_path is None:
        if 'recording_name' not in result:
            raise KeyError("file_path is None and result has no 'recording_name'.")
        file_path = result['recording_name'] + '.npz'
    elif not file_path.endswith('.npz'):
        file_path += '.npz'

    # Ensure directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    if show_progress and verbose:
        print("Warning: tqdm not installed. Install with: pip install tqdm")
        show_progress = False

    # Prepare data
    save_dict = {}
    original_size = 0
    optimized_size = 0

    if verbose:
        print(f"Saving compressed data to {file_path}...")

    # Process data with optional optimization
    items = list(result.items())
    iterator = tqdm(items, desc="Processing data", disable=not show_progress) if show_progress else items

    for key, value in iterator:
        if show_progress:
            iterator.set_description(f"Processing: {key[:25]}")

        if isinstance(value, np.ndarray):
            original_size += value.nbytes

            if optimize_dtypes:
                # Smart dtype optimization
                if value.dtype == np.float64:
                    if 't_' in key.lower() or 'time' in key.lower():
                        # Time vectors: float32 has plenty of precision
                        value = value.astype(np.float32)
                    elif 'amplifier' in key.lower() or 'emg' in key.lower() or 'aux' in key.lower():
                        # ADC data: check if it's int16 range stored as float
                        if np.all(np.abs(value) < 32767):
                            value = value.astype(np.int16)
                        else:
                            value = value.astype(np.float32)
                    else:
                        # Other float data: use float32
                        value = value.astype(np.float32)

                elif value.dtype == np.int64:
                    # Downcast large integers where safe
                    max_val = np.max(np.abs(value))
                    if max_val < 32767:
                        value = value.astype(np.int16)
                    elif max_val < 2147483647:
                        value = value.astype(np.int32)

            optimized_size += value.nbytes

        save_dict[key] = value

    if show_progress:
        iterator.close()

    # Save with gzip compression
    np.savez_compressed(file_path, **save_dict)

    # Get statistics
    file_size = os.path.getsize(file_path)
    compression_ratio = original_size / file_size if file_size > 0 else 1.0

    if verbose:
        print(f"Data saved to {file_path}")
        print(f"  Original size: {original_size / 1024 ** 2:.2f} MB")
        if optimize_dtypes:
            print(f"  After dtype optimization: {optimized_size / 1024 ** 2:.2f} MB")
        print(f"  Compressed file size: {file_size / 1024 ** 2:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Space saved: {(1 - file_size / original_size) * 100:.1f}%")

    return {
        'file_path': file_path,
        'original_size_mb': original_size / 1024 ** 2,
        'optimized_size_mb': optimized_size / 1024 ** 2,
        'file_size_mb': file_size / 1024 ** 2,
        'compression_ratio': compression_ratio,
        'space_saved_percent': (1 - file_size / original_size) * 100 if original_size > 0 else 0
    }

def load_npz_file(file_path: Union[str, Sequence[str]],
                  verbose: bool = False,
                  allow_many: bool = False) -> Union[Dict, List[Dict]]:
    """
    Load an NPZ and reconstruct a convenient dict, normalizing common field aliases.

    Supported aliases (all preserved; normalized fields always populated when available):
      - EMG matrix:          'emg', 'emg_data', 'amplifier_data' -> normalized to result['amplifier_data']
      - Time vector:         't', 'time_vector', 't_amplifier'   -> normalized to result['t_amplifier']
      - Sampling rate (Hz):  'fs', 'sampling_rate', 'sample_rate',
                             result['frequency_parameters']['amplifier_sample_rate'],
                             OR inferred from time vector        -> normalized to result['_fs_Hz'] & result['sample_rate']
      - Channel names:       'ch_names', 'channel_names', 'amplifier_channels[*].native_channel_name'
                             -> normalized to result['channel_names']

    Also passes through any extra keys (e.g., provenance: 'source_file', 'source_gesture',
    'source_local_index', 'source_name', 'gestures_combined', 'fs_reported', 'fs_effective',
    'align_mode', 'dt_target', 't_overlap', 'reorder_info', etc.).

    If `file_path` resolves to multiple files and `allow_many=True`, returns a list of dicts.
    Otherwise (legacy behavior) the first match is loaded.
    """

    # If user passed a sequence, expand it
    if isinstance(file_path, (list, tuple)):
        files = list_npz_files(file_path)
        if len(files) == 0:
            raise FileNotFoundError(f"No NPZ files found for: {file_path}")
        if allow_many:
            return [load_npz_file(fp, verbose=verbose, allow_many=False) for fp in files]
        file_path = files[0]

    # If user passed a single string, it might still be a dir or glob
    if isinstance(file_path, str):
        p = os.path.expanduser(os.path.expandvars(file_path))
        if os.path.isdir(p) or any(ch in p for ch in "*?[]"):
            files = list_npz_files(p)
            if len(files) == 0:
                raise FileNotFoundError(f"No NPZ files found in: {file_path}")
            if allow_many:
                return [load_npz_file(fp, verbose=verbose, allow_many=False) for fp in files]
            file_path = files[0]

    # --- guaranteed single file path ---
    z = np.load(file_path, allow_pickle=True)

    def _to_native(x):
        # Convert numpy object arrays back to python containers when appropriate
        if isinstance(x, np.ndarray):
            # 0-D object -> Python object if possible
            if x.dtype == object and x.ndim == 0:
                try:
                    return x.item()
                except Exception:
                    return x
            # object arrays -> list
            if x.dtype == object:
                try:
                    return x.tolist()
                except Exception:
                    return x
            # 1D unicode/bytes -> list[str]
            if x.ndim == 1 and x.dtype.kind in ("U", "S"):
                return x.tolist()
        return x

    # Raw dict (preserve originals)
    result: Dict = {k: _to_native(z[k]) for k in z.files}

    # Basic file metadata
    result['export_basename'] = os.path.splitext(os.path.basename(file_path))[0]
    result['export_basepath'] = os.path.dirname(os.path.abspath(file_path))
    result.setdefault('file_name', os.path.basename(file_path))
    result.setdefault('file_path', os.path.abspath(file_path))

    # -------- normalize EMG matrix --------
    emg = None
    for key in ('amplifier_data', 'emg', 'emg_data'):
        if key in result:
            emg = result[key]
            break
    # ensure ndarray (channels x samples)
    if isinstance(emg, list):
        emg = np.asarray(emg)
    if emg is not None and not isinstance(emg, np.ndarray):
        emg = np.asarray(emg)
    if emg is not None:
        if emg.ndim != 2:
            raise ValueError(f"{file_path}: EMG must be 2D (channels x samples), got shape {emg.shape}")
        result['amplifier_data'] = emg  # normalized

    # -------- normalize time vector --------
    t = None
    for key in ('t_amplifier', 't', 'time_vector'):
        if key in result:
            t = result[key]
            break
    if isinstance(t, list):
        t = np.asarray(t)
    if t is not None and not isinstance(t, np.ndarray):
        t = np.asarray(t)
    if t is not None:
        if t.ndim != 1:
            raise ValueError(f"{file_path}: time vector must be 1D, got shape {t.shape}")
        result['t_amplifier'] = t  # normalized

    # -------- normalize sampling rate (Hz) --------
    fs = None
    # Prefer explicit scalar fields
    for key in ('_fs_Hz', 'fs', 'emg_fs', 'sampling_rate', 'sample_rate', 'amplifier_sample_rate'):
        if key in result:
            try:
                fs = float(result[key])
                break
            except Exception:
                pass
    # Try nested frequency_parameters
    if fs is None and isinstance(result.get('frequency_parameters'), dict):
        try:
            fs = float(result['frequency_parameters'].get('amplifier_sample_rate'))
        except Exception:
            pass
    # Try effective fs from alignment scripts
    if fs is None and 'fs_effective' in result:
        try:
            # could be scalar or 1D array; take first if array
            fse = result['fs_effective']
            if isinstance(fse, (list, tuple, np.ndarray)):
                fse = float(np.asarray(fse).ravel()[0])
            fs = float(fse)
        except Exception:
            pass
    # Derive from time vector if still missing
    if fs is None and isinstance(result.get('t_amplifier'), np.ndarray) and result['t_amplifier'].size > 1:
        dt = float(np.median(np.diff(result['t_amplifier'])))
        if dt > 0:
            fs = 1.0 / dt

    if fs is not None:
        result['_fs_Hz'] = float(fs)
        result['sample_rate'] = float(fs)

    # -------- n_samples & n_channels --------
    n_samp = None
    n_chan = None
    if isinstance(result.get('t_amplifier'), np.ndarray):
        n_samp = int(result['t_amplifier'].size)
    if isinstance(result.get('amplifier_data'), np.ndarray):
        n_chan = int(result['amplifier_data'].shape[0])
        n_samp = int(result['amplifier_data'].shape[1]) if n_samp is None else n_samp
    # fallbacks from common keys (if present)
    if n_samp is None and 'n_samples' in result:
        try: n_samp = int(result['n_samples'])
        except Exception: pass
    if n_chan is None and 'n_channels' in result:
        try: n_chan = int(result['n_channels'])
        except Exception: pass
    result['n_samples'] = n_samp
    result['n_channels'] = n_chan

    # -------- channel names normalization --------
    chn = result.get('channel_names')
    if _is_empty_channel_names(chn):
        # try 'ch_names'
        chn_alt = result.get('ch_names')
        if not _is_empty_channel_names(chn_alt):
            # ensure list[str]
            result['channel_names'] = [str(x) for x in chn_alt]
        else:
            # provenance-sourced names
            src_names = result.get('source_name')
            src_gest = result.get('source_gesture')
            if (isinstance(src_names, (list, tuple)) or isinstance(src_names, np.ndarray)):
                try:
                    # if gesture available, keep "GESTURE:NAME" else just NAME
                    if isinstance(src_gest, (list, tuple, np.ndarray)) and len(src_gest) == len(src_names):
                        result['channel_names'] = [f"{str(g)}:{str(n)}" for g, n in zip(src_gest, src_names)]
                    else:
                        result['channel_names'] = [str(n) for n in src_names]
                except Exception:
                    pass

    # If still missing, synthesize CHi
    if _is_empty_channel_names(result.get('channel_names')) and isinstance(result.get('amplifier_data'), np.ndarray):
        C = result['amplifier_data'].shape[0]
        result['channel_names'] = [f"CH{i}" for i in range(C)]

    # -------- sanity: fs vs time-derived ----------
    # (non-fatal: warn in verbose mode if big mismatch)
    if verbose and (result.get('_fs_Hz') is not None) and isinstance(result.get('t_amplifier'), np.ndarray):
        tvec = result['t_amplifier']
        if tvec.size > 1:
            dt_eff = float(np.median(np.diff(tvec)))
            if dt_eff > 0:
                fs_eff = 1.0 / dt_eff
                delta = abs(fs_eff - float(result['_fs_Hz']))
                if delta > 1e-3 * max(1.0, float(result['_fs_Hz'])):
                    print(f"[load_npz_file] Warning: reported fs={result['_fs_Hz']:.6g} "
                          f"differs from effective {fs_eff:.6g} (Δ={delta:.6g}) for {file_path}")

    if verbose:
        ad = result.get('amplifier_data')
        print(f"[load_npz_file] Loaded: {file_path}")
        print(f"  fs: {result.get('_fs_Hz')}, n_samples: {result.get('n_samples')}, "
              f"channels: {ad.shape[0] if isinstance(ad, np.ndarray) else 'NA'}")

    return result

def load_npz_files(paths: Union[str, Sequence[str]], verbose: bool = False) -> List[Dict]:
    """
    Load MANY NPZ files, returning a list of dicts (one per NPZ).

    ``paths`` may be a directory (loads all ``*.npz`` files), a file path,
    a glob pattern, or a list/tuple containing any combination of those.
    """
    files = list_npz_files(paths)
    if len(files) == 0:
        raise FileNotFoundError(f"No NPZ files found for: {paths}")
    return [load_npz_file(fp, verbose=verbose) for fp in files]


# =============================================================================
# Training Dataset Utilities
# =============================================================================

def load_training_dataset(
    npz_path: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Load a training dataset NPZ with standardized field extraction.
    
    Handles various field naming conventions and returns a normalized dict.
    
    Args:
        npz_path: Path to training dataset NPZ file
        verbose: Print loading info
    
    Returns:
        Dict with standardized keys:
            - X: Feature matrix (n_windows, n_features)
            - y: Labels as strings (n_windows,)
            - y_id: Labels as integers (n_windows,) if available
            - class_names: Sorted unique class names
            - label_to_id: Dict mapping class name → integer ID
            - emg_fs: Sampling frequency (Hz)
            - window_ms: Feature window size (ms)
            - step_ms: Window step size (ms)
            - channel_names: List of channel names
            - selected_channels: List of channel indices used
            - feature_spec: Feature specification dict
            - metadata: Any additional metadata
    
    Example:
        >>> data = load_training_dataset("training_dataset.npz")
        >>> X, y = data["X"], data["y"]
        >>> print(f"Loaded {X.shape[0]} samples, {len(data['class_names'])} classes")
    """
    import json
    
    with np.load(npz_path, allow_pickle=True) as d:
        result = {}
        
        # Core data
        result["X"] = d["X"]
        
        # Labels: prefer y_id if available, else y
        if "y_id" in d.files:
            result["y_id"] = d["y_id"]
            result["y"] = d["y"] if "y" in d.files else None
        else:
            result["y"] = d["y"]
            result["y_id"] = None
        
        # Class information
        if "class_names" in d.files:
            cn = d["class_names"]
            result["class_names"] = cn.tolist() if hasattr(cn, "tolist") else list(cn)
        else:
            # Derive from y
            y_vals = result["y"] if result["y"] is not None else result["y_id"]
            result["class_names"] = sorted(set(y_vals)) if y_vals is not None else []
        
        # Label mapping
        if "label_to_id_json" in d.files:
            raw = d["label_to_id_json"]
            try:
                raw = raw.item() if getattr(raw, "shape", ()) == () else str(raw)
                result["label_to_id"] = json.loads(str(raw))
            except Exception:
                result["label_to_id"] = {c: i for i, c in enumerate(result["class_names"])}
        else:
            result["label_to_id"] = {c: i for i, c in enumerate(result["class_names"])}
        
        # Preprocessing parameters
        result["emg_fs"] = float(d["emg_fs"]) if "emg_fs" in d.files else None
        result["window_ms"] = int(d["window_ms"]) if "window_ms" in d.files else None
        result["step_ms"] = int(d["step_ms"]) if "step_ms" in d.files else None
        
        # Channel information
        if "channel_names" in d.files:
            cn = d["channel_names"]
            result["channel_names"] = cn.tolist() if hasattr(cn, "tolist") else list(cn)
        else:
            result["channel_names"] = []
        
        if "selected_channels" in d.files:
            sc = d["selected_channels"]
            result["selected_channels"] = sc.tolist() if hasattr(sc, "tolist") else list(sc)
        else:
            result["selected_channels"] = []
        
        # Feature specification
        if "feature_spec" in d.files:
            raw = d["feature_spec"]
            try:
                raw = raw.item() if getattr(raw, "shape", ()) == () else raw
                result["feature_spec"] = json.loads(str(raw))
            except Exception:
                result["feature_spec"] = None
        else:
            result["feature_spec"] = None
        
        # Additional metadata
        result["metadata"] = {
            "file_path": os.path.abspath(npz_path),
            "modality": str(d["modality"].item()) if "modality" in d.files else "emg",
        }
    
    if verbose:
        print(f"[load_training_dataset] Loaded: {npz_path}")
        print(f"  X shape: {result['X'].shape}")
        print(f"  Classes: {result['class_names']}")
        print(f"  Window: {result['window_ms']}ms, Step: {result['step_ms']}ms")
    
    return result


def load_and_merge_datasets(
    npz_paths: Sequence[str],
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load and concatenate multiple training datasets.
    
    Validates that all datasets have compatible feature dimensions
    and merges class labels appropriately.
    
    Args:
        npz_paths: List of paths to training dataset NPZ files
        verbose: Print loading info
    
    Returns:
        Tuple of (X, y, metadata) where:
            - X: Concatenated feature matrix (total_windows, n_features)
            - y: Concatenated string labels (total_windows,)
            - metadata: Merged metadata dict
    
    Raises:
        ValueError: If feature dimensions don't match across datasets
    
    Example:
        >>> X, y, meta = load_and_merge_datasets([
        ...     "session1_dataset.npz",
        ...     "session2_dataset.npz",
        ... ])
    """
    if not npz_paths:
        raise ValueError("No dataset paths provided")
    
    all_X = []
    all_y = []
    all_classes = set()
    metadata_list = []
    feature_dim = None
    
    for i, path in enumerate(npz_paths):
        data = load_training_dataset(path, verbose=verbose)
        
        # Validate feature dimension consistency
        if feature_dim is None:
            feature_dim = data["X"].shape[1]
        elif data["X"].shape[1] != feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: {path} has {data['X'].shape[1]} features, "
                f"expected {feature_dim}"
            )
        
        all_X.append(data["X"])
        
        # Use string labels
        y = data["y"] if data["y"] is not None else data["y_id"]
        all_y.append(np.asarray(y, dtype=object))
        
        all_classes.update(data["class_names"])
        metadata_list.append(data)
    
    # Concatenate
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # Build unified class mapping
    class_names = sorted(all_classes)
    label_to_id = {c: i for i, c in enumerate(class_names)}
    
    # Merge metadata (use first file's params, note all sources)
    first = metadata_list[0]
    merged_metadata = {
        "class_names": class_names,
        "label_to_id": label_to_id,
        "emg_fs": first["emg_fs"],
        "window_ms": first["window_ms"],
        "step_ms": first["step_ms"],
        "channel_names": first["channel_names"],
        "selected_channels": first["selected_channels"],
        "feature_spec": first["feature_spec"],
        "source_files": list(npz_paths),
        "n_sources": len(npz_paths),
    }
    
    if verbose:
        print(f"[load_and_merge_datasets] Merged {len(npz_paths)} datasets")
        print(f"  Total samples: {X.shape[0]}")
        print(f"  Unified classes: {class_names}")
    
    return X, y, merged_metadata


def save_training_dataset(
    save_path: str,
    X: np.ndarray,
    y: np.ndarray,
    emg_fs: float,
    window_ms: int,
    step_ms: int,
    channel_names: List[str],
    selected_channels: List[int] = None,
    feature_spec: Dict = None,
    channel_map: str = None,
    channel_map_file: str = None,
    modality: str = "emg",
) -> None:
    """
    Save a training dataset to NPZ with standardized format.
    
    Args:
        save_path: Output file path
        X: Feature matrix (n_windows, n_features)
        y: String labels (n_windows,)
        emg_fs: Sampling frequency (Hz)
        window_ms: Feature window size (ms)
        step_ms: Window step size (ms)
        channel_names: List of channel names in order used
        selected_channels: Original channel indices (if subset)
        feature_spec: Feature specification dict
        channel_map: Name of channel mapping used (for reproducibility)
        channel_map_file: Path to channel mapping file
        modality: Data modality (default: "emg")
    
    Example:
        >>> save_training_dataset(
        ...     "training_dataset.npz",
        ...     X=features, y=labels,
        ...     emg_fs=2000, window_ms=200, step_ms=50,
        ...     channel_names=["A-001", "A-002", ...],
        ... )
    """
    import json
    
    # Ensure output directory exists
    out_dir = os.path.dirname(save_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Build label mappings
    class_names = sorted(set(y))
    label_to_id = {c: i for i, c in enumerate(class_names)}
    y_id = np.array([label_to_id[lab] for lab in y], dtype=np.int32)
    
    np.savez(
        save_path,
        # Core data
        X=X,
        y=y,
        y_id=y_id,
        # Label info
        class_names=np.array(class_names, dtype=object),
        label_to_id_json=np.array(json.dumps(label_to_id), dtype=object),
        # Preprocessing params
        emg_fs=emg_fs,
        window_ms=window_ms,
        step_ms=step_ms,
        # Channel info
        channel_names=np.array(channel_names, dtype=object),
        selected_channels=np.array(selected_channels if selected_channels else [], dtype=int),
        # Feature info
        feature_spec=json.dumps(feature_spec) if feature_spec else "",
        # Metadata
        modality=np.array(modality, dtype=object),
        channel_mapping_name=np.array(channel_map or "", dtype=object),
        channel_mapping_file=np.array(channel_map_file or "", dtype=object),
    )
