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
