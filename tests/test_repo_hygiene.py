from __future__ import annotations

from pathlib import Path
import subprocess

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import pytest


def test_gitignore_covers_generated_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gitignore = (repo_root / ".gitignore").read_text()
    required_patterns = [
        ".DS_Store",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
        ".venv/",
        "build/",
        "dist/",
        "*.egg-info/",
        "work/",
        "*.h5ad",
        "*.pt",
        "*.npz",
    ]
    for pattern in required_patterns:
        assert pattern in gitignore


def test_no_generated_files_tracked_when_git_metadata_is_available() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("git metadata is unavailable in this workspace")

    tracked = subprocess.check_output(
        ["git", "-C", str(repo_root), "ls-files"], text=True
    ).splitlines()
    forbidden_names = {".DS_Store"}
    forbidden_parts = {"__pycache__", "build", "dist"}
    forbidden_suffixes = {".pyc", ".pyo"}

    for path in tracked:
        parts = set(Path(path).parts)
        assert Path(path).name not in forbidden_names
        assert not (parts & forbidden_parts)
        assert not any(part.endswith(".egg-info") for part in parts)
        assert not any(path.endswith(suffix) for suffix in forbidden_suffixes)


def test_package_version_matches_pyproject() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)

    import spatial_ot

    assert spatial_ot.__version__ == pyproject["project"]["version"]


def test_removed_boundary_generation_workflow_is_not_packaged() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    removed_paths = [
        repo_root / "spatial_ot" / ("multi" + "level"),
        repo_root / "spatial_ot" / "deep",
        repo_root / "spatial_ot" / ("optimal" + "_search.py"),
        repo_root / "spatial_ot" / "doctor.py",
        repo_root / "configs",
        repo_root / "scripts" / "run.sh",
        repo_root / "scripts" / "run_prepared_cohort_gpu.sh",
        repo_root / "scripts" / "run_visium_hd_cohort_gpu.sh",
        repo_root / "scripts" / "run_xenium_cohort_gpu.sh",
    ]
    for path in removed_paths:
        assert not path.exists(), f"legacy boundary workflow path still exists: {path}"


def test_cli_exposes_only_current_workflow() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cli_text = (repo_root / "spatial_ot" / "cli.py").read_text()
    assert "pairwise-niche" in cli_text
    assert "cell-niche" in cli_text
    assert ("multi" + "level-ot") not in cli_text
    assert ("optimal" + "-search") not in cli_text
    assert "plot-sample-niches" not in cli_text
