import os
import glob
from typing import Union, Sequence, List, Dict
import numpy as np


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


def list_npz_files(path_or_glob):
    def _expand_one(p):
        p = os.path.expanduser(os.path.expandvars(p))
        if os.path.isdir(p):
            return sorted(glob.glob(os.path.join(p, "*.npz")))
        if any(ch in p for ch in "*?[]"):
            return sorted(glob.glob(p))
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

    # original single-string behavior
    return _expand_one(str(path_or_glob))

# def list_npz_files(npz_dir: str, recursive: bool = True) -> list[str]:
#     npz_dir = os.path.abspath(npz_dir)
#     out = []
#     if recursive:
#         for root, _, files in os.walk(npz_dir):
#             for f in files:
#                 if f.lower().endswith(".npz"):
#                     out.append(os.path.join(root, f))
#     else:
#         for f in os.listdir(npz_dir):
#             if f.lower().endswith(".npz"):
#                 out.append(os.path.join(npz_dir, f))
#     return sorted(out)


def find_npz_by_label(npz_dir: str, label: str, recursive: bool = True) -> list[str]:
    """Return NPZ paths whose FILENAME contains `label` (case-insensitive)."""
    label_lc = label.lower().strip()
    data = [p for p in list_npz_files(npz_dir, recursive=recursive)
            if label_lc in os.path.basename(p).lower()]

    # Keep list output for now
    return data

def load_npz_record(path: str) -> tuple[np.ndarray, str]:
    """Load one NPZ file -> (emg (C,N), label:str). Raises if keys missing."""
    with np.load(path, allow_pickle=True) as d:
        if "emg" not in d or "label" not in d:
            raise KeyError(f"{os.path.basename(path)} missing 'emg' or 'label' keys.")
        emg = d["emg"]
        label = d["label"].item()  # np.object_ -> str
    return emg, str(label)

def save_as_npz(result: dict, file_path: str = None):
    """
    Save the rhd data as a .npz file.

    Args:
        result (dict): Dictionary containing the Open Ephys session data.
            Must contain keys: 'amplifier_data', 't_amplifier', 'sample_rate', 'recording_name'.
        file_path (str, optional): Path to save the .npz file. If None, uses the recording name.

    Returns:
        None
    """
    if not isinstance(result, dict):
        raise ValueError("Input must be dict data from RHD file.")

    required_keys = ['amplifier_data', 't_amplifier']
    if not all(key in result for key in required_keys):
        raise KeyError(f"Missing one of the required keys: {required_keys}")

    if file_path is None:
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

    # For every key in result, save it to the .npz file
    np.savez(file_path, **{key: result[key] for key in result.keys()})
    print(f"Data saved to {file_path}")

def load_npz_files(paths: Union[str, Sequence[str]],
                  verbose: bool = False) -> List[Dict]:
    """
    Load MANY NPZ files, returning a list of dicts (one per NPZ).

    `paths` may be:
      - a directory (loads all *.npz inside),
      - a file path,
      - a glob pattern (e.g., '/data/emg/*.npz'),
      - a list/tuple of any mix of the above.
    """
    files = list_npz_files(paths)
    if len(files) == 0:
        raise FileNotFoundError(f"No NPZ files found for: {paths}")
    return [load_npz_file(fp, verbose=verbose) for fp in files]

def load_npz_file(file_path: Union[str, Sequence[str]],
                  verbose: bool = False,
                  allow_many: bool = False) -> Union[Dict, List[Dict]]:
    """
    Load an NPZ and reconstruct a convenient dict.

    Now supports multi-file input:
      - If `file_path` is a list/tuple OR expands to multiple matches and
        `allow_many=True`, returns a **list of dicts** (one per NPZ).
      - Otherwise (default), preserves legacy behavior and loads only the first.

    Returns
    -------
    dict OR list[dict]
        Single dict (legacy) or list of dicts if multiple and allow_many=True.
    """
    # If user passed a sequence, expand it
    if isinstance(file_path, (list, tuple)):
        files = list_npz_files(file_path)
        if len(files) == 0:
            raise FileNotFoundError(f"No NPZ files found for: {file_path}")
        if allow_many:
            return [load_npz_file(fp, verbose=verbose, allow_many=False) for fp in files]
        # legacy behavior: take the first
        file_path = files[0]

    # If user passed a single string, it might still be a dir or glob
    if isinstance(file_path, str):
        p = os.path.expanduser(os.path.expandvars(file_path))
        # Directory or glob expansion
        if os.path.isdir(p) or any(ch in p for ch in "*?[]"):
            files = list_npz_files(p)
            if len(files) == 0:
                raise FileNotFoundError(f"No NPZ files found in: {file_path}")
            if allow_many:
                return [load_npz_file(fp, verbose=verbose, allow_many=False) for fp in files]
            # legacy behavior: default to first
            file_path = files[0]

    # --- from here on: guaranteed single file path ---
    # allow_pickle is required if you saved dicts/lists as object arrays
    z = np.load(file_path, allow_pickle=True)

    def _to_native(x):
        # Convert numpy object arrays back to python containers when appropriate
        if isinstance(x, np.ndarray):
            # 1) object arrays -> Python objects/lists
            if x.dtype == object:
                if x.ndim == 0:
                    try:
                        return x.item()
                    except Exception:
                        return x
                try:
                    return x.tolist()
                except Exception:
                    return x
            # 2) 1-D string arrays (unicode/bytes) -> list[str]
            if x.ndim == 1 and x.dtype.kind in ("U", "S"):
                return x.tolist()
        return x

    result: Dict = {k: _to_native(z[k]) for k in z.files}

    # Basic file metadata
    result['export_basename'] = os.path.splitext(os.path.basename(file_path))[0]
    result['export_basepath'] = os.path.dirname(os.path.abspath(file_path))
    result.setdefault('file_name', os.path.basename(file_path))
    result.setdefault('file_path', os.path.abspath(file_path))

    # Infer sampling rate (Hz)
    fs = None
    for key in ('_fs_Hz', 'fs', 'emg_fs', 'sample_rate', 'amplifier_sample_rate'):
        if key in result:
            try:
                fs = float(result[key]); break
            except Exception:
                pass
    if fs is None and isinstance(result.get('frequency_parameters'), dict):
        try:
            fs = float(result['frequency_parameters'].get('amplifier_sample_rate'))
        except Exception:
            pass
    if fs is None and isinstance(result.get('t_amplifier'), np.ndarray) and result['t_amplifier'].size > 1:
        dt = float(np.median(np.diff(result['t_amplifier'])))
        if dt > 0:
            fs = 1.0 / dt
    if fs is not None:
        result['_fs_Hz'] = float(fs)
        result['sample_rate'] = float(fs)

    # n_samples
    n_samp = None
    if isinstance(result.get('t_amplifier'), np.ndarray):
        n_samp = int(result['t_amplifier'].size)
    elif isinstance(result.get('amplifier_data'), np.ndarray):
        n_samp = int(result['amplifier_data'].shape[-1])
    result['n_samples'] = n_samp

    # Channel names convenience
    chn = result.get('channel_names')
    if _is_empty_channel_names(chn):
        ch_names = []
        amps = result.get('amplifier_channels')
        if isinstance(amps, (list, tuple)):
            for ch in amps:
                try:
                    ch_names.append(str(ch.get('native_channel_name')))
                except Exception:
                    ch_names.append(None)
        if ch_names:
            result['channel_names'] = ch_names
        else:
            if isinstance(result.get('amplifier_data'), np.ndarray):
                C = result['amplifier_data'].shape[0]
                result['channel_names'] = [f"CH{i}" for i in range(C)]
            else:
                result['channel_names'] = []

    if verbose:
        chs = result.get('amplifier_data')
        print(f"[load_npz_file] Loaded: {file_path}")
        print(f"  fs: {result.get('_fs_Hz')}, n_samples: {result.get('n_samples')}, "
              f"channels: {chs.shape[0] if isinstance(chs, np.ndarray) else 'NA'}")

    return result