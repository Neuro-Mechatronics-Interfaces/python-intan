"""Regression tests for EMG viewer recording normalization."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
import unittest

import numpy as np

from intan.applications import _emg_viewer


class EMGViewerLoadingTests(unittest.TestCase):
    def test_loads_sample_major_npz_and_preserves_metadata(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample_major.npz"
            data = np.arange(400, dtype=float).reshape(100, 4)
            np.savez(
                path,
                emg_data=data,
                time_vector=np.arange(100) / 2000.0,
                ch_names=np.array(["A", "B", "C", "D"]),
            )

            loaded, time_vector, sampling_rate, labels = (
                _emg_viewer._load_emg_recording(path)
            )

        self.assertEqual(loaded.shape, (4, 100))
        self.assertEqual(time_vector.size, 100)
        self.assertAlmostEqual(sampling_rate, 2000.0)
        self.assertEqual(labels, ["A", "B", "C", "D"])

    def test_loads_minimal_channel_major_npz(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "minimal.npz"
            np.savez(path, amplifier_data=np.ones((8, 40)), sample_rate=4000)

            loaded, time_vector, sampling_rate, labels = (
                _emg_viewer._load_emg_recording(path)
            )

        self.assertEqual(loaded.shape, (8, 40))
        self.assertEqual(time_vector.size, 40)
        self.assertEqual(sampling_rate, 4000.0)
        self.assertEqual(len(labels), 8)

    def test_loads_rhd_arrays_without_ambiguous_truth_testing(self):
        result = {
            "amplifier_data": np.ones((2, 10)),
            "t_amplifier": np.arange(10) / 1000.0,
            "frequency_parameters": {"amplifier_sample_rate": 1000.0},
            "channel_names": ["A", "B"],
        }
        with mock.patch.object(_emg_viewer, "load_rhd_file", return_value=result):
            loaded, time_vector, sampling_rate, labels = (
                _emg_viewer._load_emg_recording("recording.rhd")
            )

        self.assertEqual(loaded.shape, (2, 10))
        self.assertEqual(time_vector.size, 10)
        self.assertEqual(sampling_rate, 1000.0)
        self.assertEqual(labels, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
