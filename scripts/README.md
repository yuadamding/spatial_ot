# Operational Scripts

This directory is the canonical home for runnable project helpers.
Root-level compatibility wrappers have been removed to keep the package compact.
Run helpers from this directory, for example `bash scripts/run.sh`.

Core entrypoints:

- `run.sh`: packaged cohort multilevel OT runner.
- `prepare_spatial_ot_input.sh`: pool and prepare the cohort feature cache.
- `prepare_all_spatial_ot_input.sh`: prepare and verify pooled plus sample files.
- `run_prepared_cohort_gpu.sh`: run a prepared pooled cohort on GPU, usually as the coordinate-only baseline.
- `run_deep_segmentation_cohort_gpu.sh`: run the current deep-boundary cohort profile with learned context features plus constrained joint segmentation-clustering refinement.
- `run_optimal_setting_search.sh`: launch the staged parameter search.
- `package_visium_hd_full_gene_h5ad.py`: package full-gene cell-level Visium HD data.
- `plot_sample_subregion_geometry.py`: plot sample-level subregion geometry.
