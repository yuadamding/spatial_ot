# Operational Scripts

This directory is the canonical home for runnable project helpers.
Root-level compatibility wrappers have been removed to keep the package compact.
Run helpers from this directory, for example `bash scripts/run.sh`.

Core entrypoints:

- `run.sh`: packaged cohort multilevel OT runner.
- `install_spatial_ot.sh`: install/update the local editable environment, delegating to `install_env.sh`.
- `prepare_spatial_ot_input.sh`: pool and prepare the cohort feature cache.
- `prepare_all_spatial_ot_input.sh`: prepare and verify pooled plus sample files.
- `prepare_xenium_spatial_ot_input.sh`: pool all processed Xenium H5ADs into one sample-separated cohort input.
- `run_prepared_cohort_gpu.sh`: run a prepared pooled cohort on GPU, usually as the coordinate-only baseline.
- `run_visium_hd_cohort_gpu.sh`: run whole Visium HD with the high-VRAM deep-expression profile.
- `run_deep_segmentation_cohort_gpu.sh`: run the current deep-boundary cohort profile with learned context features plus constrained joint segmentation-clustering refinement.
- `run_xenium_cohort_gpu.sh`: run Xenium with the high-VRAM deep-expression profile using Xenium centroid coordinates and a learned expression autoencoder latent.
- `run_optimal_setting_search.sh`: launch the staged parameter search.

Default inputs live under `../spatial_ot_input/`. The direct pooled inputs are
`visium_hd_spatial_ot_input_pooled.h5ad` for whole Visium HD and
`xenium_spatial_ot_input_pooled.h5ad` for Xenium; processed `hd_*_processed.h5ad`
and `xenium_*_processed.h5ad` source files are staged in the same directory.
The dataset-specific GPU runners source `_high_vram_deep_profile.sh`, which
targets about 70 GiB VRAM by default and can be reduced with
`DEEP_TARGET_VRAM_GB`, `DEEP_HIDDEN_DIM`, `DEEP_LAYERS`, and `DEEP_BATCH_SIZE`.
