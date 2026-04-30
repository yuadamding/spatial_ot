from __future__ import annotations

from pathlib import Path
import tempfile

from spatial_ot.config import (
    DeepFeatureConfig,
    MultilevelExperimentConfig,
    load_multilevel_config,
    validate_multilevel_config,
)
from spatial_ot.cli import build_parser, _resolve_multilevel_config_from_args
from spatial_ot.feature_source import default_precomputed_x_feature_key


def test_multilevel_config_rejects_unknown_sections() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.toml"
        path.write_text(
            """
            [paths]
            input_h5ad = "cells.h5ad"
            output_dir = "runs/test"
            feature_obsm_key = "X"

            [data]
            subset_strategy = "spatial_knn"
            """
        )
        try:
            load_multilevel_config(path)
        except KeyError:
            pass
        else:
            raise AssertionError("Expected inactive config sections to be rejected")


def test_auto_device_defaults_are_explicit() -> None:
    assert DeepFeatureConfig().device == "auto"
    assert MultilevelExperimentConfig().ot.compute_device == "auto"


def test_multilevel_config_rejects_zero_geometry_and_feature_weights() -> None:
    config = MultilevelExperimentConfig()
    config.paths.input_h5ad = "input.h5ad"
    config.paths.output_dir = "out"
    config.paths.feature_obsm_key = "X"
    config.ot.lambda_x = 0.0
    config.ot.lambda_y = 0.0
    try:
        validate_multilevel_config(config)
    except ValueError as exc:
        assert "lambda_x or lambda_y" in str(exc)
    else:
        raise AssertionError("Expected zero lambda_x/lambda_y to be rejected")


def test_multilevel_config_requires_region_obs_key_for_region_geometry_json() -> None:
    config = MultilevelExperimentConfig()
    config.paths.input_h5ad = "input.h5ad"
    config.paths.output_dir = "out"
    config.paths.feature_obsm_key = "X"
    config.paths.region_geometry_json = "regions.json"
    try:
        validate_multilevel_config(config)
    except ValueError as exc:
        assert "region_obs_key" in str(exc)
    else:
        raise AssertionError(
            "Expected region_geometry_json without region_obs_key to be rejected"
        )


def test_multilevel_config_rejects_negative_joint_refinement_margin() -> None:
    config = MultilevelExperimentConfig()
    config.paths.input_h5ad = "input.h5ad"
    config.paths.output_dir = "out"
    config.paths.feature_obsm_key = "X"
    config.ot.joint_refinement_acceptance_margin = -1e-3
    try:
        validate_multilevel_config(config)
    except ValueError as exc:
        assert "joint_refinement_acceptance_margin" in str(exc)
    else:
        raise AssertionError(
            "Expected negative joint_refinement_acceptance_margin to be rejected"
        )


def test_optimal_search_cli_preserves_config_sample_key_when_not_overridden() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.toml"
        path.write_text(
            """
            [paths]
            input_h5ad = "cells.h5ad"
            output_dir = "runs/search"
            feature_obsm_key = "X_pca"
            sample_obs_key = "patient_id"
            """
        )
        args = build_parser().parse_args(["optimal-search", "--config", str(path)])

        config = _resolve_multilevel_config_from_args(args)

    assert config.paths.sample_obs_key == "patient_id"


def test_optimal_search_cli_defaults_to_prepared_feature_cache() -> None:
    args = build_parser().parse_args(
        [
            "optimal-search",
            "--input-h5ad",
            "cells.h5ad",
            "--output-dir",
            "runs/search",
        ]
    )

    config = _resolve_multilevel_config_from_args(args)

    assert config.paths.feature_obsm_key == default_precomputed_x_feature_key()
