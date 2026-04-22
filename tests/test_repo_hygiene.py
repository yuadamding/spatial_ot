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
    pool_helper_sh = (repo_root / "pool_spatial_ot_input.sh").read_text()
    prepare_helper_sh = (repo_root / "prepare_spatial_ot_input.sh").read_text()
    prepare_all_helper_sh = (repo_root / "prepare_all_spatial_ot_input.sh").read_text()
    p2_sh = (repo_root / "run_p2_crc_multilevel_ot.sh").read_text()
    exploratory_sh = (repo_root / "run_p2_crc_multilevel_ot_exploratory_umap.sh").read_text()
    config_toml = (repo_root / "configs" / "multilevel_deep_example.toml").read_text()

    assert "../spatial_ot_input" in run_sh
    assert "../outputs/" in run_sh
    assert "../.venv" in run_sh
    assert 'REFRESH_POOLED_INPUT="${REFRESH_POOLED_INPUT:-0}"' in run_sh
    assert 'POOLED_INPUT_NAME="${POOLED_INPUT_NAME:-spatial_ot_input_pooled.h5ad}"' in run_sh
    assert 'PREPARE_INPUTS_AHEAD="${PREPARE_INPUTS_AHEAD:-1}"' in run_sh
    assert 'REFRESH_PREPARED_FEATURES="${REFRESH_PREPARED_FEATURES:-0}"' in run_sh
    assert 'PREPARED_FEATURE_OBSM_KEY="${PREPARED_FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}}"' in run_sh
    assert 'FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-}"' in run_sh
    assert 'COMPUTE_DEVICE="${COMPUTE_DEVICE:-cuda}"' in run_sh
    assert 'RADIUS_UM="${RADIUS_UM:-100}"' in run_sh
    assert 'STRIDE_UM="${STRIDE_UM:-$RADIUS_UM}"' in run_sh
    assert 'BASIC_NICHE_SIZE_UM="${BASIC_NICHE_SIZE_UM:-50}"' in run_sh
    assert 'MIN_CELLS="${MIN_CELLS:-25}"' in run_sh
    assert 'MAX_SUBREGIONS="${MAX_SUBREGIONS:-1500}"' in run_sh
    assert 'REQUIRE_FULL_CELL_COVERAGE="${REQUIRE_FULL_CELL_COVERAGE:-1}"' in run_sh
    assert 'ALLOW_UMAP_AS_FEATURE="${ALLOW_UMAP_AS_FEATURE:-0}"' in run_sh
    assert 'ALLOW_OBSERVED_HULL_GEOMETRY="${ALLOW_OBSERVED_HULL_GEOMETRY:-0}"' in run_sh
    assert 'CPU_THREADS="${CPU_THREADS:-28}"' in run_sh
    assert 'CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-all}"' in run_sh
    assert 'PARALLEL_RESTARTS="${PARALLEL_RESTARTS:-auto}"' in run_sh
    assert 'CUDA_TARGET_VRAM_GB="${CUDA_TARGET_VRAM_GB:-50}"' in run_sh
    assert 'X_FEATURE_COMPONENTS="${X_FEATURE_COMPONENTS:-512}"' in run_sh
    assert 'X_TARGET_SUM="${X_TARGET_SUM:-10000}"' in run_sh
    assert 'DEEP_FEATURE_METHOD="${DEEP_FEATURE_METHOD:-autoencoder}"' in run_sh
    assert 'DEEP_OUTPUT_EMBEDDING="${DEEP_OUTPUT_EMBEDDING:-context}"' in run_sh
    assert 'DEEP_DEVICE="${DEEP_DEVICE:-cuda}"' in run_sh
    assert 'DEEP_BATCH_SIZE="${DEEP_BATCH_SIZE:-32768}"' in run_sh
    assert 'LAMBDA_X="${LAMBDA_X:-0.5}"' in run_sh
    assert 'LAMBDA_Y="${LAMBDA_Y:-1.0}"' in run_sh
    assert 'OT_EPS="${OT_EPS:-0.03}"' in run_sh
    assert 'OVERLAP_CONSISTENCY_WEIGHT="${OVERLAP_CONSISTENCY_WEIGHT:-0.05}"' in run_sh
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
    assert "prepare_spatial_ot_input.sh" in run_sh
    assert "plot-sample-niches" in run_sh
    assert "sample_niche_plots" in run_sh
    assert '${INPUT_DIR}/${POOLED_INPUT_NAME}' in run_sh
    assert "pooled_cell_x" in run_sh
    assert "pooled_cell_y" in run_sh
    assert 'REQUIRE_FULL_CELL_COVERAGE=1 requires STRIDE_UM <= RADIUS_UM' in run_sh
    assert 'cell_subregion_coverage_fraction' in run_sh
    assert 'uncovered_cell_count' in run_sh
    assert '--min-cells "$MIN_CELLS"' in run_sh
    assert '--max-subregions "$MAX_SUBREGIONS"' in run_sh
    assert '--radius-um "$RADIUS_UM"' in run_sh
    assert '--stride-um "$STRIDE_UM"' in run_sh
    assert '--lambda-x "$LAMBDA_X"' in run_sh
    assert '--lambda-y "$LAMBDA_Y"' in run_sh
    assert '--overlap-consistency-weight "$OVERLAP_CONSISTENCY_WEIGHT"' in run_sh
    assert '--deep-feature-method "$DEEP_FEATURE_METHOD"' in run_sh
    assert '--deep-output-embedding "$DEEP_OUTPUT_EMBEDDING"' in run_sh
    assert '--deep-device "$DEEP_DEVICE"' in run_sh
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
    assert "../spatial_ot_input" in pool_helper_sh
    assert "../.venv" in pool_helper_sh
    assert 'OUTPUT_H5AD="${OUTPUT_H5AD:-${INPUT_DIR}/spatial_ot_input_pooled.h5ad}"' in pool_helper_sh
    assert "pool-inputs" in pool_helper_sh
    assert "../spatial_ot_input" in prepare_helper_sh
    assert "../.venv" in prepare_helper_sh
    assert 'PREPARED_FEATURE_OBSM_KEY="${PREPARED_FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}}"' in prepare_helper_sh
    assert "pooled_has_prepared_key()" in prepare_helper_sh
    assert 'if [[ "$REFRESH_PREPARED_FEATURES" == "1" ]] || ! pooled_has_prepared_key; then' in prepare_helper_sh
    assert "pool-inputs" in prepare_helper_sh
    assert "prepare-inputs" in prepare_helper_sh
    assert "../spatial_ot_input" in prepare_all_helper_sh
    assert "../.venv" in prepare_all_helper_sh
    assert 'PREPARE_POOLED_INPUT="${PREPARE_POOLED_INPUT:-1}"' in prepare_all_helper_sh
    assert 'PREPARED_FEATURE_OBSM_KEY="${PREPARED_FEATURE_OBSM_KEY:-X_spatial_ot_x_svd_${X_FEATURE_COMPONENTS}}"' in prepare_all_helper_sh
    assert 'VERIFY_PREPARED_FEATURES="${VERIFY_PREPARED_FEATURES:-1}"' in prepare_all_helper_sh
    assert 'WRITE_BACK_TO_SOURCE_INPUTS="${WRITE_BACK_TO_SOURCE_INPUTS:-0}"' in prepare_all_helper_sh
    assert "Missing prepared feature cache" in prepare_all_helper_sh
    assert "prepare_spatial_ot_input.sh" in prepare_all_helper_sh
    assert "distribute-prepared-inputs" in prepare_all_helper_sh
    assert "POOL_ALL_INPUTS" in p2_sh
    assert "POOL_ALL_INPUTS" in exploratory_sh
    assert "FEATURE_OBSM_KEY" in exploratory_sh
    assert 'DEEP_FEATURE_METHOD="${DEEP_FEATURE_METHOD:-none}"' in exploratory_sh
    assert "exec bash \"$SCRIPT_DIR/run.sh\"" in p2_sh
    assert "exec bash \"$SCRIPT_DIR/run.sh\"" in exploratory_sh
    optimal_search_sh = (repo_root / "run_optimal_setting_search.sh").read_text()
    assert "../spatial_ot_input/spatial_ot_input_pooled.h5ad" in optimal_search_sh
    assert "../work/spatial_ot_runs/cohort_optimal_search" in optimal_search_sh
    assert "optimal-search" in optimal_search_sh
    assert 'BASIC_NICHE_SIZE_UM="${BASIC_NICHE_SIZE_UM:-50}"' in optimal_search_sh
    assert 'TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-20}"' in optimal_search_sh
    assert "../spatial_ot_input/" in config_toml
    assert "../outputs/" in config_toml
    assert 'feature_obsm_key = "X"' in config_toml
    assert "allow_umap_as_feature = false" in config_toml
    assert 'output_embedding = "context"' in config_toml
    assert "basic_niche_size_um = 50.0" in config_toml
    assert "min_cells = 25" in config_toml
    assert "max_subregions = 1500" in config_toml
    assert "allow_convex_hull_fallback = false" in config_toml

    legacy_training_py = (repo_root / "spatial_ot" / "legacy" / "training.py").read_text()
    legacy_training_facade_py = (repo_root / "spatial_ot" / "training.py").read_text()
    deep_io_py = (repo_root / "spatial_ot" / "deep" / "io.py").read_text()
    multilevel_io_py = (repo_root / "spatial_ot" / "multilevel" / "io.py").read_text()
    assert '"method_family": "legacy_teacher_student"' in legacy_training_py
    assert '"communication_source": "legacy"' in legacy_training_py
    assert "from .legacy.training import *" in legacy_training_facade_py
    assert '"method_family": "deep_feature_adapter"' in deep_io_py
    assert '"communication_source": "none"' in deep_io_py
    assert '"method_family": "multilevel_ot"' in multilevel_io_py
    assert '"communication_source": "none"' in multilevel_io_py


def test_legacy_namespace_is_canonical_and_root_modules_are_facades() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    compatibility_modules = {
        "communication": "from .legacy.communication import CommunicationResult, _masked_sinkhorn, fit_communication_flows",
        "nn": "from .legacy.nn import *",
        "ot": "from .legacy.ot import *",
        "preprocessing": "from .legacy.preprocessing import *",
        "programs": "from .legacy.programs import *",
        "training": "from .legacy.training import *",
        "visualization": "from .legacy.visualization import *",
    }
    for module_name, expected_import in compatibility_modules.items():
        canonical_path = repo_root / "spatial_ot" / "legacy" / f"{module_name}.py"
        facade_path = repo_root / "spatial_ot" / f"{module_name}.py"
        assert canonical_path.exists(), canonical_path
        facade_text = facade_path.read_text()
        assert expected_import in facade_text


def test_root_package_exports_active_helpers_without_eager_import_failures() -> None:
    import spatial_ot

    expected_exports = {
        "fit_deep_features_on_h5ad",
        "fit_multilevel_ot",
        "plot_sample_niche_maps_from_run_dir",
        "pool_h5ad_files",
        "pool_h5ads_in_directory",
        "prepare_h5ad_feature_cache",
        "distribute_pooled_feature_cache_to_inputs",
        "run_multilevel_optimal_search",
        "run_multilevel_ot_with_config",
    }

    assert expected_exports.issubset(set(spatial_ot.__all__))
    for name in expected_exports:
        assert hasattr(spatial_ot, name), name


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
