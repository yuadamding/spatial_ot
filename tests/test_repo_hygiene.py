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


def test_package_version_matches_0_1_13_state() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject_toml = (repo_root / "pyproject.toml").read_text()
    package_init = (repo_root / "spatial_ot" / "__init__.py").read_text()
    assert 'version = "0.1.14"' in pyproject_toml
    assert '__version__ = "0.1.14"' in package_init


def test_packaged_helpers_use_relative_spatial_ot_inputs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = repo_root / "scripts"

    assert not list(repo_root.glob("*.sh"))

    run_sh = (script_dir / "run.sh").read_text()
    install_sh = (script_dir / "install_env.sh").read_text()
    pyproject_toml = (repo_root / "pyproject.toml").read_text()
    pool_helper_sh = (script_dir / "pool_spatial_ot_input.sh").read_text()
    prepare_helper_sh = (script_dir / "prepare_spatial_ot_input.sh").read_text()
    prepare_all_helper_sh = (script_dir / "prepare_all_spatial_ot_input.sh").read_text()
    prepared_gpu_sh = (script_dir / "run_prepared_cohort_gpu.sh").read_text()
    deep_segmentation_sh = (script_dir / "run_deep_segmentation_cohort_gpu.sh").read_text()
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
    assert 'MAX_SUBREGIONS="${MAX_SUBREGIONS:-5000}"' in run_sh
    assert 'AUTO_N_CLUSTERS="${AUTO_N_CLUSTERS:-0}"' in run_sh
    assert 'CANDIDATE_N_CLUSTERS="${CANDIDATE_N_CLUSTERS:-15-25}"' in run_sh
    assert 'SEED="${SEED:-1337}"' in run_sh
    assert 'MIN_SUBREGIONS_PER_CLUSTER="${MIN_SUBREGIONS_PER_CLUSTER:-50}"' in run_sh
    assert 'AUTO_K_MAX_SCORE_SUBREGIONS="${AUTO_K_MAX_SCORE_SUBREGIONS:-2500}"' in run_sh
    assert 'AUTO_K_GAP_REFERENCES="${AUTO_K_GAP_REFERENCES:-8}"' in run_sh
    assert 'AUTO_K_MDS_COMPONENTS="${AUTO_K_MDS_COMPONENTS:-8}"' in run_sh
    assert 'AUTO_K_PILOT_N_INIT="${AUTO_K_PILOT_N_INIT:-1}"' in run_sh
    assert 'AUTO_K_PILOT_MAX_ITER="${AUTO_K_PILOT_MAX_ITER:-3}"' in run_sh
    assert 'REQUIRE_FULL_CELL_COVERAGE="${REQUIRE_FULL_CELL_COVERAGE:-0}"' in run_sh
    assert 'ALLOW_UMAP_AS_FEATURE="${ALLOW_UMAP_AS_FEATURE:-0}"' in run_sh
    assert 'ALLOW_OBSERVED_HULL_GEOMETRY="${ALLOW_OBSERVED_HULL_GEOMETRY:-0}"' in run_sh
    assert 'SHAPE_LEAKAGE_PERMUTATIONS="${SHAPE_LEAKAGE_PERMUTATIONS:-16}"' in run_sh
    assert 'CPU_THREADS="${CPU_THREADS:-$(default_cpu_threads)}"' in run_sh
    assert 'CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-all}"' in run_sh
    assert 'PARALLEL_RESTARTS="${PARALLEL_RESTARTS:-auto}"' in run_sh
    assert 'CUDA_TARGET_VRAM_GB="${CUDA_TARGET_VRAM_GB:-50}"' in run_sh
    assert 'CUDA_MAX_TARGET_FRACTION="${CUDA_MAX_TARGET_FRACTION:-0.9}"' in run_sh
    assert 'X_FEATURE_COMPONENTS="${X_FEATURE_COMPONENTS:-512}"' in run_sh
    assert 'X_TARGET_SUM="${X_TARGET_SUM:-10000}"' in run_sh
    assert 'DEEP_FEATURE_METHOD="${DEEP_FEATURE_METHOD:-none}"' in run_sh
    assert 'DEEP_OUTPUT_EMBEDDING="${DEEP_OUTPUT_EMBEDDING:-context}"' in run_sh
    assert 'DEEP_DEVICE="${DEEP_DEVICE:-cuda}"' in run_sh
    assert 'DEEP_BATCH_SIZE="${DEEP_BATCH_SIZE:-32768}"' in run_sh
    assert 'DEEP_PRETRAINED_MODEL="${DEEP_PRETRAINED_MODEL:-}"' in run_sh
    assert 'DEEP_SEGMENTATION_REFINEMENT_ITERS="${DEEP_SEGMENTATION_REFINEMENT_ITERS:-6}"' in run_sh
    assert 'SUBREGION_FEATURE_WEIGHT="${SUBREGION_FEATURE_WEIGHT:-0}"' in run_sh
    assert 'SUBREGION_FEATURE_DIMS="${SUBREGION_FEATURE_DIMS:-16}"' in run_sh
    assert 'LAMBDA_X="${LAMBDA_X:-0.5}"' in run_sh
    assert 'LAMBDA_Y="${LAMBDA_Y:-1.0}"' in run_sh
    assert 'OT_EPS="${OT_EPS:-0.03}"' in run_sh
    assert 'GEOMETRY_SAMPLES="${GEOMETRY_SAMPLES:-64}"' in run_sh
    assert 'COMPRESSED_SUPPORT_SIZE="${COMPRESSED_SUPPORT_SIZE:-48}"' in run_sh
    assert 'ALIGN_ITERS="${ALIGN_ITERS:-2}"' in run_sh
    assert 'N_INIT="${N_INIT:-2}"' in run_sh
    assert 'MAX_ITER="${MAX_ITER:-5}"' in run_sh
    assert 'SINKHORN_MAX_ITER="${SINKHORN_MAX_ITER:-256}"' in run_sh
    assert 'SINKHORN_TOL="${SINKHORN_TOL:-1e-4}"' in run_sh
    assert 'LIGHT_CELL_H5AD="${LIGHT_CELL_H5AD:-1}"' in run_sh
    assert 'H5AD_COMPRESSION="${H5AD_COMPRESSION:-lzf}"' in run_sh
    assert 'WRITE_SAMPLE_SPATIAL_MAPS="${WRITE_SAMPLE_SPATIAL_MAPS:-0}"' in run_sh
    assert 'PROGRESS_LOG="${PROGRESS_LOG:-1}"' in run_sh
    assert 'WRITE_CONCERN_REPORT="${WRITE_CONCERN_REPORT:-1}"' in run_sh
    assert 'STRICT_CONCERN_REPORT="${STRICT_CONCERN_REPORT:-0}"' in run_sh
    assert 'CONCERN_COORDINATE_BASELINE_RUN_DIR="${CONCERN_COORDINATE_BASELINE_RUN_DIR:-}"' in run_sh
    assert 'CONCERN_STABILITY_RUN_DIRS="${CONCERN_STABILITY_RUN_DIRS:-}"' in run_sh
    assert 'CONCERN_LEAKAGE_ABLATION_RUN_DIRS="${CONCERN_LEAKAGE_ABLATION_RUN_DIRS:-}"' in run_sh
    assert 'OVERLAP_CONSISTENCY_WEIGHT="${OVERLAP_CONSISTENCY_WEIGHT:-0.05}"' in run_sh
    assert 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"' in run_sh
    assert 'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"' in run_sh
    assert 'export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"' in run_sh
    assert 'export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$CPU_THREADS}"' in run_sh
    assert 'export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"' in run_sh
    assert 'export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"' in run_sh
    assert 'export SPATIAL_OT_TORCH_NUM_THREADS="${SPATIAL_OT_TORCH_NUM_THREADS:-$TORCH_INTRAOP_THREADS}"' in run_sh
    assert 'export SPATIAL_OT_TORCH_NUM_INTEROP_THREADS="${SPATIAL_OT_TORCH_NUM_INTEROP_THREADS:-$TORCH_INTEROP_THREADS}"' in run_sh
    assert 'export SPATIAL_OT_CPU_THREADS="${SPATIAL_OT_CPU_THREADS:-$CPU_THREADS}"' in run_sh
    assert 'export SPATIAL_OT_CUDA_DEVICE_LIST="${SPATIAL_OT_CUDA_DEVICE_LIST:-$CUDA_DEVICE_LIST}"' in run_sh
    assert 'export SPATIAL_OT_PARALLEL_RESTARTS="${SPATIAL_OT_PARALLEL_RESTARTS:-$PARALLEL_RESTARTS}"' in run_sh
    assert 'export SPATIAL_OT_CUDA_TARGET_VRAM_GB="${SPATIAL_OT_CUDA_TARGET_VRAM_GB:-$CUDA_TARGET_VRAM_GB}"' in run_sh
    assert 'export SPATIAL_OT_CUDA_MAX_TARGET_FRACTION="${SPATIAL_OT_CUDA_MAX_TARGET_FRACTION:-$CUDA_MAX_TARGET_FRACTION}"' in run_sh
    assert 'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"' in run_sh
    assert 'export SPATIAL_OT_X_SVD_COMPONENTS="${SPATIAL_OT_X_SVD_COMPONENTS:-$X_FEATURE_COMPONENTS}"' in run_sh
    assert 'export SPATIAL_OT_X_TARGET_SUM="${SPATIAL_OT_X_TARGET_SUM:-$X_TARGET_SUM}"' in run_sh
    assert 'export SPATIAL_OT_SINKHORN_MAX_ITER="${SPATIAL_OT_SINKHORN_MAX_ITER:-$SINKHORN_MAX_ITER}"' in run_sh
    assert 'export SPATIAL_OT_SINKHORN_TOL="${SPATIAL_OT_SINKHORN_TOL:-$SINKHORN_TOL}"' in run_sh
    assert 'export SPATIAL_OT_LIGHT_CELL_H5AD="${SPATIAL_OT_LIGHT_CELL_H5AD:-$LIGHT_CELL_H5AD}"' in run_sh
    assert 'export SPATIAL_OT_H5AD_COMPRESSION="${SPATIAL_OT_H5AD_COMPRESSION:-$H5AD_COMPRESSION}"' in run_sh
    assert 'export SPATIAL_OT_WRITE_SAMPLE_SPATIAL_MAPS="${SPATIAL_OT_WRITE_SAMPLE_SPATIAL_MAPS:-$WRITE_SAMPLE_SPATIAL_MAPS}"' in run_sh
    assert 'export SPATIAL_OT_PROGRESS="${SPATIAL_OT_PROGRESS:-$PROGRESS_LOG}"' in run_sh
    assert "pool-inputs" in run_sh
    assert "prepare_spatial_ot_input.sh" in run_sh
    assert "plot-sample-niches" in run_sh
    assert "plot-sample-spot-latent" in run_sh
    assert "validate-run-concerns" in run_sh
    assert '--coordinate-baseline-run-dir "$CONCERN_COORDINATE_BASELINE_RUN_DIR"' in run_sh
    assert '--stability-run-dir "$concern_dir"' in run_sh
    assert '--leakage-ablation-run-dir "$concern_dir"' in run_sh
    assert "CONCERN_FLAGS+=(--strict)" in run_sh
    assert "sample_niche_plots" in run_sh
    assert "sample_spot_latent_plots" in run_sh
    assert "COMPUTE_SPOT_LATENT" in run_sh
    assert "--auto-n-clusters" in run_sh
    assert "--candidate-n-clusters" in run_sh
    assert '--min-subregions-per-cluster "$MIN_SUBREGIONS_PER_CLUSTER"' in run_sh
    assert '--seed "$SEED"' in run_sh
    assert '${INPUT_DIR}/${POOLED_INPUT_NAME}' in run_sh
    assert "pooled_cell_x" in run_sh
    assert "pooled_cell_y" in run_sh
    assert 'REQUIRE_FULL_CELL_COVERAGE=1 requires STRIDE_UM <= RADIUS_UM' in run_sh
    assert 'cell_subregion_coverage_fraction' in run_sh
    assert 'uncovered_cell_count' in run_sh
    assert "Run completed with incomplete analyzed-subregion coverage" in run_sh
    assert "runtime_memory_qc" in run_sh
    assert "observed_peak_reserved_gb" in run_sh
    assert '--min-cells "$MIN_CELLS"' in run_sh
    assert '--max-subregions "$MAX_SUBREGIONS"' in run_sh
    assert '--radius-um "$RADIUS_UM"' in run_sh
    assert '--stride-um "$STRIDE_UM"' in run_sh
    assert '--subregion-construction-method "$SUBREGION_CONSTRUCTION_METHOD"' in run_sh
    assert '--subregion-feature-weight "$SUBREGION_FEATURE_WEIGHT"' in run_sh
    assert '--subregion-feature-dims "$SUBREGION_FEATURE_DIMS"' in run_sh
    assert '--deep-segmentation-knn "$DEEP_SEGMENTATION_KNN"' in run_sh
    assert '--lambda-x "$LAMBDA_X"' in run_sh
    assert '--lambda-y "$LAMBDA_Y"' in run_sh
    assert '--overlap-consistency-weight "$OVERLAP_CONSISTENCY_WEIGHT"' in run_sh
    assert '--deep-feature-method "$DEEP_FEATURE_METHOD"' in run_sh
    assert '--deep-output-embedding "$DEEP_OUTPUT_EMBEDDING"' in run_sh
    assert '--deep-device "$DEEP_DEVICE"' in run_sh
    assert "--pretrained-deep-model" in run_sh
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
    assert "../spatial_ot_input" in prepared_gpu_sh
    assert "../outputs/spatial_ot/cohort_multilevel_ot_prepared_gpu" in prepared_gpu_sh
    assert "../.venv" in prepared_gpu_sh
    assert 'PREPARE_INPUTS_AHEAD="${PREPARE_INPUTS_AHEAD:-0}"' in prepared_gpu_sh
    assert 'REFRESH_POOLED_INPUT="${REFRESH_POOLED_INPUT:-0}"' in prepared_gpu_sh
    assert 'REFRESH_PREPARED_FEATURES="${REFRESH_PREPARED_FEATURES:-0}"' in prepared_gpu_sh
    assert 'COMPUTE_DEVICE="${COMPUTE_DEVICE:-cuda}"' in prepared_gpu_sh
    assert 'AUTO_N_CLUSTERS="${AUTO_N_CLUSTERS:-1}"' in prepared_gpu_sh
    assert 'CANDIDATE_N_CLUSTERS="${CANDIDATE_N_CLUSTERS:-15-25}"' in prepared_gpu_sh
    assert 'MIN_SUBREGIONS_PER_CLUSTER="${MIN_SUBREGIONS_PER_CLUSTER:-50}"' in prepared_gpu_sh
    assert 'CPU_THREADS="${CPU_THREADS:-$(default_cpu_threads)}"' in prepared_gpu_sh
    assert 'export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-$CPU_THREADS}"' in prepared_gpu_sh
    assert 'export SPATIAL_OT_CPU_THREADS="${SPATIAL_OT_CPU_THREADS:-$CPU_THREADS}"' in prepared_gpu_sh
    assert 'exec bash "$SCRIPT_DIR/run.sh"' in prepared_gpu_sh
    assert "/storage/" not in prepared_gpu_sh
    assert "cohort_multilevel_ot_deep_segmentation_vram9_" in deep_segmentation_sh
    assert 'SUBREGION_CONSTRUCTION_METHOD="${SUBREGION_CONSTRUCTION_METHOD:-deep_segmentation}"' in deep_segmentation_sh
    assert 'DEEP_FEATURE_METHOD="${DEEP_FEATURE_METHOD:-autoencoder}"' in deep_segmentation_sh
    assert 'DEEP_LATENT_DIM="${DEEP_LATENT_DIM:-64}"' in deep_segmentation_sh
    assert 'DEEP_HIDDEN_DIM="${DEEP_HIDDEN_DIM:-1024}"' in deep_segmentation_sh
    assert 'DEEP_LAYERS="${DEEP_LAYERS:-3}"' in deep_segmentation_sh
    assert 'DEEP_BATCH_SIZE="${DEEP_BATCH_SIZE:-81920}"' in deep_segmentation_sh
    assert 'CUDA_TARGET_VRAM_GB="${CUDA_TARGET_VRAM_GB:-9}"' in deep_segmentation_sh
    assert 'SPATIAL_OT_CUDA_MAX_TARGET_FRACTION="${SPATIAL_OT_CUDA_MAX_TARGET_FRACTION:-$CUDA_MAX_TARGET_FRACTION}"' in deep_segmentation_sh
    assert 'DEEP_SEGMENTATION_REFINEMENT_ITERS="${DEEP_SEGMENTATION_REFINEMENT_ITERS:-6}"' in deep_segmentation_sh
    assert "run_prepared_cohort_gpu.sh" in deep_segmentation_sh
    assert "/storage/" not in deep_segmentation_sh
    optimal_search_sh = (script_dir / "run_optimal_setting_search.sh").read_text()
    assert "../spatial_ot_input/spatial_ot_input_pooled.h5ad" in optimal_search_sh
    assert "../work/spatial_ot_runs/cohort_optimal_search" in optimal_search_sh
    assert "optimal-search" in optimal_search_sh
    assert 'BASIC_NICHE_SIZE_UM="${BASIC_NICHE_SIZE_UM:-50}"' in optimal_search_sh
    assert 'MIN_SUBREGIONS_PER_CLUSTER="${MIN_SUBREGIONS_PER_CLUSTER:-50}"' in optimal_search_sh
    assert 'TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-20}"' in optimal_search_sh
    assert 'ALLOW_OBSERVED_HULL_GEOMETRY="${ALLOW_OBSERVED_HULL_GEOMETRY:-0}"' in optimal_search_sh
    assert 'SUBREGION_CONSTRUCTION_METHOD="${SUBREGION_CONSTRUCTION_METHOD:-data_driven}"' in optimal_search_sh
    assert 'SUBREGION_CLUSTERING_METHOD="${SUBREGION_CLUSTERING_METHOD:-pooled_subregion_latent}"' in optimal_search_sh
    assert 'SUBREGION_CLUSTERING_METHOD="${SUBREGION_CLUSTERING_METHOD:-pooled_subregion_latent}"' in run_sh
    assert 'SUBREGION_LATENT_EMBEDDING_MODE="${SUBREGION_LATENT_EMBEDDING_MODE:-mean_std_shrunk}"' in run_sh
    assert 'SUBREGION_LATENT_HETEROGENEITY_WEIGHT="${SUBREGION_LATENT_HETEROGENEITY_WEIGHT:-0.5}"' in run_sh
    assert 'SUBREGION_LATENT_SAMPLE_PRIOR_WEIGHT="${SUBREGION_LATENT_SAMPLE_PRIOR_WEIGHT:-0.5}"' in run_sh
    assert '--subregion-latent-embedding-mode "$SUBREGION_LATENT_EMBEDDING_MODE"' in run_sh
    assert 'SUBREGION_FEATURE_WEIGHT="${SUBREGION_FEATURE_WEIGHT:-0}"' in optimal_search_sh
    assert '--subregion-feature-weight "$SUBREGION_FEATURE_WEIGHT"' in optimal_search_sh
    assert 'DEEP_FEATURE_METHOD="${DEEP_FEATURE_METHOD:-autoencoder}"' in optimal_search_sh
    assert 'DEEP_OUTPUT_EMBEDDING="${DEEP_OUTPUT_EMBEDDING:-context}"' in optimal_search_sh
    assert '--min-subregions-per-cluster "$MIN_SUBREGIONS_PER_CLUSTER"' in optimal_search_sh
    assert "../spatial_ot_input/" in config_toml
    assert "../outputs/" in config_toml
    assert 'feature_obsm_key = "X"' in config_toml
    assert "allow_umap_as_feature = false" in config_toml
    assert 'output_embedding = "context"' in config_toml
    assert "basic_niche_size_um = 50.0" in config_toml
    assert "min_cells = 25" in config_toml
    assert "max_subregions = 5000" in config_toml
    assert "allow_convex_hull_fallback = false" in config_toml
    assert 'subregion_construction_method = "deep_segmentation"' in config_toml
    assert 'subregion_clustering_method = "pooled_subregion_latent"' in config_toml
    assert 'subregion_latent_embedding_mode = "mean_std_shrunk"' in config_toml
    assert 'subregion_latent_heterogeneity_weight = 0.5' in config_toml
    assert 'subregion_latent_sample_prior_weight = 0.5' in config_toml
    assert "auto_n_clusters = false" in config_toml
    assert "candidate_n_clusters = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]" in config_toml
    assert "min_subregions_per_cluster = 50" in config_toml

    deep_io_py = (repo_root / "spatial_ot" / "deep" / "io.py").read_text()
    multilevel_io_py = (repo_root / "spatial_ot" / "multilevel" / "io.py").read_text()
    assert '"method_family": "deep_feature_adapter"' in deep_io_py
    assert '"communication_source": "none"' in deep_io_py
    assert '"method_family": "multilevel_ot"' in multilevel_io_py
    assert '"communication_source": "none"' in multilevel_io_py


def test_package_tree_excludes_removed_scaffold_and_root_facades() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert not (repo_root / "spatial_ot" / "legacy").exists()
    for module_name in [
        "communication",
        "nn",
        "ot",
        "preprocessing",
        "programs",
        "training",
        "visualization",
    ]:
        assert not (repo_root / "spatial_ot" / f"{module_name}.py").exists()
    for config_name in ["crc_demo_programs.json", "p2_crc_smoke.toml", "p2_crc_pilot.toml"]:
        assert not (repo_root / "configs" / config_name).exists()


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
