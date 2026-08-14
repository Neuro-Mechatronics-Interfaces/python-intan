# GUI applications

Run these examples from the repository root after installing the GUI extra:

```bash
python -m pip install -e '.[gui]'
```

## Packaged applications

The supported EMG viewer and trial selector are available through installed console commands:

```bash
intan-emg-viewer --help
intan-trial-selector --help
intan-emg-viewer
intan-trial-selector
```

The equivalent repository launchers are:

```bash
python examples/applications/run_emg_viewer.py
python examples/applications/run_trial_selector.py
```

Both applications are interactive and require a desktop display. Model training and prediction are handled by the maintained scripts in `examples/gesture_classifier/`, not by the viewer.

## Workflow GUIs

`dataset_builder_gui.py` exposes the dataset-building options in a PyQt interface. `gesture_pipeline_gui.py` launches the maintained scripts in `examples/gesture_classifier/` as child processes.

```bash
python examples/applications/dataset_builder_gui.py
python examples/applications/gesture_pipeline_gui.py
```

The pipeline reads `gesture_pipeline_profile.json`. Its script paths are relative to this directory by default; data and model locations remain blank until the user selects them. Use the Scripts tab to override a path.

The other `emg_file_viewer*.py` files are standalone UI experiments retained for compatibility and are not installed as console applications.

## Requirements and limitations

- Dataset, model, and prediction actions need the same inputs as their corresponding command-line scripts.
- Channel and live views need an active LSL stream or RHX TCP server.
- The GUIs do not bundle datasets, trained models, or hardware drivers.
- Run from the repository root so editable source and relative example paths resolve consistently.
