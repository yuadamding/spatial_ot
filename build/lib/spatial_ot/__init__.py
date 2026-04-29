from __future__ import annotations

from importlib import import_module

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DeepFeatureConfig": ("config", "DeepFeatureConfig"),
    "DeepFeatureResult": ("deep", "DeepFeatureResult"),
    "MultilevelExperimentConfig": ("config", "MultilevelExperimentConfig"),
    "MultilevelOTConfig": ("config", "MultilevelOTConfig"),
    "MultilevelPathConfig": ("config", "MultilevelPathConfig"),
    "RegionGeometry": ("multilevel", "RegionGeometry"),
    "SpatialOTFeatureEncoder": ("deep", "SpatialOTFeatureEncoder"),
    "build_concern_resolution_report": ("multilevel", "build_concern_resolution_report"),
    "distribute_pooled_feature_cache_to_inputs": ("pooling", "distribute_pooled_feature_cache_to_inputs"),
    "fit_deep_features": ("deep", "fit_deep_features"),
    "fit_deep_features_on_h5ad": ("deep", "fit_deep_features_on_h5ad"),
    "fit_multilevel_ot": ("multilevel", "fit_multilevel_ot"),
    "fit_ot_shape_normalizer": ("multilevel", "fit_ot_shape_normalizer"),
    "load_multilevel_config": ("config", "load_multilevel_config"),
    "make_reference_points_unit_disk": ("multilevel", "make_reference_points_unit_disk"),
    "plot_sample_niche_maps": ("multilevel", "plot_sample_niche_maps"),
    "plot_sample_niche_maps_from_run_dir": ("multilevel", "plot_sample_niche_maps_from_run_dir"),
    "plot_sample_spatial_maps": ("multilevel", "plot_sample_spatial_maps"),
    "plot_sample_spatial_maps_from_run_dir": ("multilevel", "plot_sample_spatial_maps_from_run_dir"),
    "pool_h5ad_files": ("pooling", "pool_h5ad_files"),
    "pool_h5ads_in_directory": ("pooling", "pool_h5ads_in_directory"),
    "prepare_h5ad_feature_cache": ("feature_source", "prepare_h5ad_feature_cache"),
    "run_multilevel_optimal_search": ("optimal_search", "run_multilevel_optimal_search"),
    "run_multilevel_ot_on_h5ad": ("multilevel", "run_multilevel_ot_on_h5ad"),
    "run_multilevel_ot_with_config": ("multilevel", "run_multilevel_ot_with_config"),
    "spatial_niche_validation_report": ("multilevel", "spatial_niche_validation_report"),
    "transform_h5ad_with_deep_model": ("deep", "transform_h5ad_with_deep_model"),
    "write_concern_resolution_report": ("multilevel", "write_concern_resolution_report"),
}

__version__ = "0.1.15"
__all__ = list(_LAZY_EXPORTS)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'spatial_ot' has no attribute {name!r}") from exc
    module = import_module(f".{module_name}", __name__)
    return getattr(module, attr_name)
