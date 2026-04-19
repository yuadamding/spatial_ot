from __future__ import annotations

from .config import ExperimentConfig, load_config

__all__ = [
    "DeepFeatureConfig",
    "DeepFeatureResult",
    "ExperimentConfig",
    "MultilevelExperimentConfig",
    "MultilevelOTConfig",
    "MultilevelPathConfig",
    "RegionGeometry",
    "SpatialOTFeatureEncoder",
    "fit_deep_features",
    "fit_multilevel_ot",
    "fit_ot_shape_normalizer",
    "load_config",
    "load_multilevel_config",
    "make_reference_points_unit_disk",
    "run_multilevel_ot_on_h5ad",
    "run_multilevel_ot_with_config",
]


def __getattr__(name: str):
    if name in {
        "DeepFeatureConfig",
        "ExperimentConfig",
        "MultilevelExperimentConfig",
        "MultilevelOTConfig",
        "MultilevelPathConfig",
        "load_config",
        "load_multilevel_config",
    }:
        from . import config

        return getattr(config, name)
    if name in {
        "DeepFeatureResult",
        "RegionGeometry",
        "SpatialOTFeatureEncoder",
        "fit_deep_features",
        "fit_multilevel_ot",
        "fit_ot_shape_normalizer",
        "make_reference_points_unit_disk",
        "run_multilevel_ot_on_h5ad",
        "run_multilevel_ot_with_config",
    }:
        if name in {"DeepFeatureResult", "SpatialOTFeatureEncoder", "fit_deep_features"}:
            from . import deep_features

            return getattr(deep_features, name)
        from . import multilevel

        return getattr(multilevel, name)
    raise AttributeError(f"module 'spatial_ot' has no attribute {name!r}")
