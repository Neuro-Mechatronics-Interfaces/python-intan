import ast
import json
import pathlib
import re
import sys

import pytest

import intan


ROOT = pathlib.Path(__file__).parents[1]


def _project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return re.search(r'^version = "([^"]+)"$', text, re.MULTILINE).group(1)


def test_version_is_consistent():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert intan.__version__ == _project_version()
    assert f"version: {intan.__version__}" in citation


def test_readme_logo_uses_an_absolute_url_for_pypi():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    expected = (
        'src="https://raw.githubusercontent.com/'
        'Neuro-Mechatronics-Interfaces/python-intan/main/docs/figs/logo.png"'
    )
    assert expected in readme


def test_documented_subpackages_are_lazy_attributes():
    assert intan.processing.__name__ == "intan.processing"
    assert intan.io.__name__ == "intan.io"


@pytest.mark.parametrize("entry", ["emg_viewer_main", "trial_selector_main"])
def test_console_entry_points_support_help(entry, monkeypatch):
    from intan import _cli

    monkeypatch.setattr(sys, "argv", [entry, "--help"])
    with pytest.raises(SystemExit) as exc:
        getattr(_cli, entry)()
    assert exc.value.code == 0


def test_all_python_examples_parse():
    failures = []
    for path in (ROOT / "examples").rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    assert not failures, "\n".join(failures)


def test_pipeline_profile_references_existing_scripts():
    profile_path = ROOT / "examples" / "applications" / "gesture_pipeline_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    for relative in profile["scripts"].values():
        assert (profile_path.parent / relative).resolve().is_file(), relative


def test_no_private_drive_defaults_or_merge_markers():
    failures = []
    for base in (ROOT / "intan", ROOT / "examples", ROOT / "docs"):
        for path in base.rglob("*"):
            if path.suffix.lower() not in {".py", ".md", ".rst", ".json", ".toml"}:
                continue
            text = path.read_text(encoding="utf-8-sig")
            if "G:\\Shared drives" in text or "<<<<<<< " in text or ">>>>>>> " in text:
                failures.append(str(path.relative_to(ROOT)))
    assert not failures, failures


def test_tensorflow_is_not_part_of_the_supported_codebase():
    failures = []
    for path in [ROOT / "pyproject.toml", ROOT / "README.md"]:
        if "tensorflow" in path.read_text(encoding="utf-8").lower():
            failures.append(str(path.relative_to(ROOT)))
    for base in (ROOT / "intan", ROOT / "examples", ROOT / "docs"):
        for path in base.rglob("*"):
            if path.suffix.lower() not in {".py", ".md", ".rst", ".toml"}:
                continue
            if "tensorflow" in path.read_text(encoding="utf-8-sig").lower():
                failures.append(str(path.relative_to(ROOT)))
    assert not failures, failures
