"""Core workflows must work without the analysis and GUI stacks."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest


def test_core_workflows_without_optional_imports():
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        import tempfile
        from pathlib import Path
        import numpy as np

        blocked = {'pandas', 'matplotlib', 'sklearn', 'torch', 'joblib', 'PyQt5', 'pyqtgraph'}
        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split('.')[0]
                if root in blocked:
                    raise ModuleNotFoundError(f'No module named {root!r}', name=root)
        sys.meta_path.insert(0, BlockOptional())

        from intan.io import load_rhd_file, read_digital_edges, load_csv_file, load_csv_files, parse_event_file
        from intan.processing import fix_missing_emg, bandpass_filter, load_metrics_data
        from intan.decomposition import pca, whitening_pca
        from intan.interface import IntanRHXDevice
        import intan.ml
        fixed, stats = fix_missing_emg(np.array([[1., np.nan, 3.]]), method='interp', verbose=False)
        np.testing.assert_array_equal(fixed, [[1., 2., 3.]])
        assert stats.total_missing == 1
        filtered = bandpass_filter(np.ones((2, 1000)), lowcut=20, highcut=450, fs=1000)
        assert filtered.shape == (2, 1000) and np.isfinite(filtered).all()
        assert not blocked.intersection(sys.modules)

        def needs_extra(call, extra):
            try:
                call()
            except ImportError as exc:
                assert f'python-intan[{extra}]' in str(exc), str(exc)
            else:
                raise AssertionError(f'Expected installation guidance for {extra}')

        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / 'csv' / 'recording.csv'
            csv.parent.mkdir()
            csv.write_text('EMG_0\\n1\\n2\\n')
            needs_extra(lambda: load_csv_file(str(csv)), 'data')
            needs_extra(lambda: load_csv_files(directory), 'data')
        needs_extra(lambda: parse_event_file([]), 'data')
        needs_extra(lambda: load_metrics_data('missing.csv'), 'data')
        needs_extra(lambda: pca(np.ones((20, 2)), n_components=2), 'analysis')
        needs_extra(lambda: __import__('intan.plotting'), 'plot')
        needs_extra(lambda: intan.ml.ModelManager, 'ml')
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_optional_import_preserves_transitive_errors(monkeypatch):
    from intan import _optional

    failure = ModuleNotFoundError("broken dependency", name="pandas_internal_dependency")
    def broken_import(name):
        raise failure
    monkeypatch.setattr(_optional, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError) as exc:
        _optional.require_optional("pandas", "data")
    assert exc.value is failure


def test_csv_and_event_tables_with_data_extra(tmp_path):
    pytest.importorskip("pandas")
    from intan.io import load_csv_file, load_csv_files, parse_event_file

    recording = tmp_path / "csv" / "recording.csv"
    recording.parent.mkdir()
    recording.write_text("time,EMG_0,EMG_1\n" + "".join(
        f"{i / 1000},{i},{i + 10}\n" for i in range(8)
    ))
    result = load_csv_file(str(recording), return_dataframe=True)
    np.testing.assert_array_equal(result["amplifier_data"], [np.arange(8), np.arange(8) + 10])
    assert result["frequency_parameters"]["amplifier_sample_rate"] == pytest.approx(1000)
    assert result["dataframe"].shape == (8, 3)
    batch = load_csv_files(str(tmp_path))
    assert len(batch) == 1
    np.testing.assert_array_equal(batch[0][0]["amplifier_data"], result["amplifier_data"])
    events = tmp_path / "events.csv"
    events.write_text("sample,label\n0,start\n5,stop\n")
    table = parse_event_file(str(events))
    assert list(table["sample_index"]) == [0, 5]
    assert list(table["label"]) == ["start", "stop"]


def test_pca_with_analysis_extra():
    pytest.importorskip("sklearn")
    from intan.decomposition import pca

    data = np.random.default_rng(0).normal(size=(32, 4))
    reconstructed, components = pca(data, n_components=4, variance_threshold=1.0)
    assert 1 <= components <= 4
    assert reconstructed.shape == (4, 32)
    assert np.isfinite(reconstructed).all()


def test_plotting_with_plot_extra(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from intan.processing import fix_missing_emg

    fixed, stats = fix_missing_emg(
        np.array([[1., np.nan, 3.]]), method="interp", make_figures=True,
        out_dir=tmp_path, overlay_channels=(0,), verbose=False,
    )
    assert stats.filled_remaining_missing == 0
    assert (tmp_path / "overlay_before_after.png").stat().st_size > 0
    from intan.plotting import waterfall
    assert callable(waterfall)
