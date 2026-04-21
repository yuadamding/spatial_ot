from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

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

    tracked = subprocess.check_output(["git", "-C", str(repo_root), "ls-files"], text=True).splitlines()
    forbidden_names = {".DS_Store"}
    forbidden_parts = {"__pycache__"}
    forbidden_suffixes = {".pyc", ".pyo"}

    for path in tracked:
        parts = set(Path(path).parts)
        assert Path(path).name not in forbidden_names
        assert not (parts & forbidden_parts)
        assert not any(path.endswith(suffix) for suffix in forbidden_suffixes)


def test_packaged_helpers_use_relative_spatial_ot_inputs() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    run_sh = (repo_root / "run.sh").read_text()
    install_sh = (repo_root / "install_env.sh").read_text()
    pyproject_toml = (repo_root / "pyproject.toml").read_text()
    helper_sh = (repo_root / "run_spatial_ot_input.sh").read_text()
    p2_sh = (repo_root / "run_p2_crc_multilevel_ot.sh").read_text()
    exploratory_sh = (repo_root / "run_p2_crc_multilevel_ot_exploratory_umap.sh").read_text()
    config_toml = (repo_root / "configs" / "multilevel_deep_example.toml").read_text()

    assert "../spatial_ot_input" in run_sh
    assert "../outputs/" in run_sh
    assert "../.venv" in run_sh
    assert 'FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-X}"' in run_sh
    assert 'COMPUTE_DEVICE="${COMPUTE_DEVICE:-cuda}"' in run_sh
    assert 'MIN_CELLS="${MIN_CELLS:-1}"' in run_sh
    assert 'MAX_SUBREGIONS="${MAX_SUBREGIONS:-0}"' in run_sh
    assert 'ALLOW_UMAP_AS_FEATURE="${ALLOW_UMAP_AS_FEATURE:-0}"' in run_sh
    assert 'CPU_THREADS="${CPU_THREADS:-28}"' in run_sh
    assert 'CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-all}"' in run_sh
    assert 'PARALLEL_RESTARTS="${PARALLEL_RESTARTS:-auto}"' in run_sh
    assert 'CUDA_TARGET_VRAM_GB="${CUDA_TARGET_VRAM_GB:-50}"' in run_sh
    assert 'X_FEATURE_COMPONENTS="${X_FEATURE_COMPONENTS:-512}"' in run_sh
    assert 'X_TARGET_SUM="${X_TARGET_SUM:-10000}"' in run_sh
    assert 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"' in run_sh
    assert 'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"' in run_sh
    assert 'export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"' in run_sh
    assert 'export SPATIAL_OT_TORCH_NUM_THREADS="${SPATIAL_OT_TORCH_NUM_THREADS:-$TORCH_INTRAOP_THREADS}"' in run_sh
    assert 'export SPATIAL_OT_TORCH_NUM_INTEROP_THREADS="${SPATIAL_OT_TORCH_NUM_INTEROP_THREADS:-$TORCH_INTEROP_THREADS}"' in run_sh
    assert 'export SPATIAL_OT_CUDA_DEVICE_LIST="${SPATIAL_OT_CUDA_DEVICE_LIST:-$CUDA_DEVICE_LIST}"' in run_sh
    assert 'export SPATIAL_OT_PARALLEL_RESTARTS="${SPATIAL_OT_PARALLEL_RESTARTS:-$PARALLEL_RESTARTS}"' in run_sh
    assert 'export SPATIAL_OT_CUDA_TARGET_VRAM_GB="${SPATIAL_OT_CUDA_TARGET_VRAM_GB:-$CUDA_TARGET_VRAM_GB}"' in run_sh
    assert 'export SPATIAL_OT_X_SVD_COMPONENTS="${SPATIAL_OT_X_SVD_COMPONENTS:-$X_FEATURE_COMPONENTS}"' in run_sh
    assert 'export SPATIAL_OT_X_TARGET_SUM="${SPATIAL_OT_X_TARGET_SUM:-$X_TARGET_SUM}"' in run_sh
    assert "pool-inputs" in run_sh
    assert "plot-sample-niches" in run_sh
    assert "sample_niche_plots" in run_sh
    assert "pooled_cell_x" in run_sh
    assert "pooled_cell_y" in run_sh
    assert '--min-cells "$MIN_CELLS"' in run_sh
    assert '--max-subregions "$MAX_SUBREGIONS"' in run_sh
    assert "--allow-umap-as-feature" in run_sh
    assert "../.venv" in install_sh
    assert "python3" in install_sh
    assert "ensurepip" in install_sh
    assert "python3.10" in install_sh
    assert 'setuptools<82' in install_sh
    assert 'requires-python = ">=3.10"' in pyproject_toml
    assert "anndata>=0.11.4,<0.12; python_version < '3.11'" in pyproject_toml
    assert "anndata>=0.12; python_version >= '3.11'" in pyproject_toml
    assert "tomli>=2.0; python_version < '3.11'" in pyproject_toml
    assert "exec bash \"$SCRIPT_DIR/run.sh\"" in helper_sh
    assert "POOL_ALL_INPUTS" in p2_sh
    assert "POOL_ALL_INPUTS" in exploratory_sh
    assert "FEATURE_OBSM_KEY" in exploratory_sh
    assert "exec bash \"$SCRIPT_DIR/run.sh\"" in p2_sh
    assert "exec bash \"$SCRIPT_DIR/run.sh\"" in exploratory_sh
    assert "../spatial_ot_input/" in config_toml
    assert "../outputs/" in config_toml
    assert 'feature_obsm_key = "X"' in config_toml
    assert "allow_umap_as_feature = false" in config_toml
    assert 'output_embedding = "context"' in config_toml
    assert "min_cells = 1" in config_toml
    assert "max_subregions = 0" in config_toml
    assert "allow_convex_hull_fallback = true" in config_toml


def test_module_and_console_entrypoints_resolve_active_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    module_help = subprocess.run(
        [sys.executable, "-m", "spatial_ot", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert module_help.returncode == 0
    assert "multilevel-ot" in module_help.stdout

    console = shutil.which("spatial-ot")
    if console is None:
        pytest.skip("console entrypoint is unavailable in this test environment")
    console_help = subprocess.run(
        [console, "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert console_help.returncode == 0
    assert "multilevel-ot" in console_help.stdout
