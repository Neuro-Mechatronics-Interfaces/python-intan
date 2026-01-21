Sync EMG ↔ Video Demo
=====================

Purpose
-------
This demo shows how to estimate the temporal offset between an EMG recording and video landmark traces (e.g., hand/finger landmarks). It calls the `intan.processing` sync utilities to compute an offset, save a JSON result, and optionally plot the alignment.

Requirements
------------
- `python` (use the project's virtualenv where `python-intan` is installed editable)
- Typical extras: `opencv-python`, `mediapipe` (only if you want to extract landmarks from video)

Quick usage
-----------
Run the demo non-interactively by providing data paths. Example (PowerShell / CMD):

```powershell
python examples/synchronization/sync_emg_video_demo.py \
  --root_dir C:\path\to\data \
  --emg_file emg.npy \
  --landmarks_file landmarks.json \
  --out sync_result.json \
  --plot
```

- `--root_dir` : folder where `--emg_file` and `--landmarks_file` live (optional if full paths given)
- `--emg_file` : EMG data file (supported formats depend on demo; e.g., `.npy`, `.npz`, or `.rhd`)
- `--landmarks_file` : JSON or NumPy file with landmark positions and timestamps
- `--out` : path to save the resulting sync JSON (default: `sync_result.json` in working dir)
- `--plot` : enable saving/displaying an alignment plot (if demo supports plotting)

What the demo writes
---------------------
- A JSON file containing `offset_sec`, `offset_samples_emg`, `offset_samples_landmarks`, `confidence`, and metadata about sampling rates.
- Optionally a PNG/figure showing the aligned signals when `--plot` is used.

Next steps
----------
- If you have raw video but no landmarks, use an example landmark extraction script (e.g., with MediaPipe) to produce the landmarks JSON first.
- If you want, run the provided synthetic test in `examples/finger_kinematics` to validate the pipeline locally.

Support
-------
If the demo errors on missing dependencies, install them into the repo venv:

```powershell
& .venv\Scripts\Activate.ps1
pip install opencv-python mediapipe
```

For more details, see the `examples/synchronization/sync_emg_video_demo.py` script.
