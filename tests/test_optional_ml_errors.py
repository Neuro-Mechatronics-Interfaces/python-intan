"""Regression tests for actionable optional-ML import errors."""

from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class OptionalMLErrorTests(unittest.TestCase):
    def test_missing_torch_error_includes_install_hint(self):
        code = textwrap.dedent(
            """
            import sys

            sys.modules["torch"] = None
            import intan.ml

            try:
                intan.ml.EMGClassifier
            except ModuleNotFoundError as exc:
                assert "python-intan[ml]" in str(exc), str(exc)
                assert "torch" in str(exc), str(exc)
            else:
                raise AssertionError("EMGClassifier unexpectedly loaded without torch")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
