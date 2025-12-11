#!/usr/bin/env python3
"""
EMG Trial Selector Application Launcher

Launch the interactive trial segmentation and labeling tool.

Features:
- Load and visualize EMG recordings
- Manual trial boundary marking
- Label assignment per trial
- Export segmented data and event files

Usage:
    python run_trial_selector.py

Requirements:
    pip install python-intan[gui]
"""

from intan.applications import launch_emg_trial_selector

if __name__ == "__main__":
    launch_emg_trial_selector()