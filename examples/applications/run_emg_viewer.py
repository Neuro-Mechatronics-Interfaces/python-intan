#!/usr/bin/env python3
"""
EMG Viewer Application Launcher

Launch the real-time EMG signal visualization GUI.

Features:
- Multi-channel waveform display
- Real-time updates from Intan RHX devices or LSL streams
- Configurable time windows and scaling
- Export capabilities

Usage:
    python run_emg_viewer.py

Requirements:
    pip install python-intan[gui]
"""

from intan.applications import launch_emg_viewer

if __name__ == "__main__":
    launch_emg_viewer()