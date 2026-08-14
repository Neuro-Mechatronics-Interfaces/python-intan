# File I/O examples

Run these scripts from the repository root. The package currently supports RHD files, Intan per-signal DAT directories, CSV files, and NPZ archives. RHS parsing is not implemented.

## RHD files

```bash
python examples/Read_Files/load_rhd_demo.py --help
python examples/Read_Files/load_rhd_demo.py path/to/recording.rhd
```

Omit the path to open a file picker. Add `--save_npz` to save the loaded result beside the source recording.

The returned mapping includes `amplifier_data` in `(channels, samples)` order, `t_amplifier`, channel metadata, and `frequency_parameters`.

## Per-signal DAT directories

```bash
python examples/Read_Files/load_dat_demo.py --help
python examples/Read_Files/load_dat_demo.py path/to/recording_directory --save-npz
```

A recording directory normally contains `info.rhd`, `time.dat`, `amplifier.dat`, and optional auxiliary/analog/digital files.

## CSV files

```bash
python examples/Read_Files/load_csv_demo.py --help
python examples/Read_Files/load_csv_demo.py path/to/emg.csv --fs 2000
```

CSV EMG columns should use names such as `EMG_0`, `EMG_1`, and so on. A valid time column can override the fallback sample rate.

## Lower-level and segmentation examples

`parse_rhd_data.py` and `parse_dat_data.py` show lower-level format access. `segment_emg_from_events.py --help` extracts labeled intervals and also contains optional Poly5 adapters; those third-party Poly5 libraries are not package dependencies.

These examples require user-supplied recordings. No data files are bundled with the distribution.
