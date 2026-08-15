"""Regression tests for truncated or unavailable RHD data."""

from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import mock
import unittest

from intan.io._exceptions import FileSizeError
from intan.io._rhd_block_parser import _read_exact
from intan.io._rhd_header_parsing import calculate_data_size


class RHDShortReadTests(unittest.TestCase):
    def test_read_exact_reports_offset_and_expected_size(self):
        stream = BytesIO(b"012345")
        stream.seek(2)

        with self.assertRaisesRegex(FileSizeError, "expected 8 bytes, received 4"):
            _read_exact(stream, 8, "test samples")

    def test_cloud_size_probe_rejects_unreadable_reported_tail(self):
        header = {
            "num_samples_per_data_block": 128,
            "num_amplifier_channels": 1,
            "num_aux_input_channels": 0,
            "num_supply_voltage_channels": 0,
            "num_temp_sensor_channels": 0,
            "num_board_adc_channels": 0,
            "num_board_dig_in_channels": 0,
            "num_board_dig_out_channels": 0,
            "sample_rate": 2000.0,
        }
        with NamedTemporaryFile(delete=False) as temporary:
            temporary.write(b"header" + b"x" * 768)
            path = Path(temporary.name)

        try:
            with path.open("rb") as stream:
                stream.seek(6)
                with mock.patch("os.path.getsize", return_value=1542):
                    with self.assertRaisesRegex(
                        FileSizeError, "cloud-backed contents are not fully available"
                    ):
                        calculate_data_size(header, str(path), stream, verbose=False)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
