# Operational Scripts

This directory is the canonical home for runnable project helpers.
Root-level compatibility wrappers have been removed to keep the package compact.
Run helpers from this directory, for example `bash scripts/run.sh`.

Core entrypoints:

- `run.sh`: packaged cohort multilevel OT runner.
- `prepare_spatial_ot_input.sh`: pool and prepare the cohort feature cache.
- `prepare_all_spatial_ot_input.sh`: prepare and verify pooled plus sample files.
- `prepare_xenium_spatial_ot_input.sh`: pool all processed Xenium H5ADs into one sample-separated cohort input.
- `run_prepared_cohort_gpu.sh`: run a prepared pooled cohort on GPU, usually as the coordinate-only baseline.
- `run_deep_segmentation_cohort_gpu.sh`: run the current deep-boundary cohort profile with learned context features plus constrained joint segmentation-clustering refinement.
- `run_xenium_cohort_gpu.sh`: run the all-Xenium pooled cohort profile using Xenium centroid coordinates and a learned expression autoencoder latent.
- `run_optimal_setting_search.sh`: launch the staged parameter search.
