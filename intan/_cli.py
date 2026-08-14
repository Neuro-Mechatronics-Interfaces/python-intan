"""Console entry points for the optional GUI applications."""

import argparse


def _parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description)


def emg_viewer_main() -> None:
    _parser("Launch the python-intan EMG viewer.").parse_args()
    from intan.applications import launch_emg_viewer

    launch_emg_viewer()


def trial_selector_main() -> None:
    _parser("Launch the python-intan EMG trial selector.").parse_args()
    from intan.applications import launch_emg_trial_selector

    launch_emg_trial_selector()
