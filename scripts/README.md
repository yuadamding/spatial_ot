# Scripts

These helpers support pooled inputs for the current pairwise OT workflow.

- `install_env.sh`: create/update the local Python environment.
- `install_spatial_ot.sh`: compatibility wrapper for `install_env.sh`.
- `pool_spatial_ot_input.sh`: pool sample H5AD files into one sample-aware cohort H5AD.
- `prepare_spatial_ot_input.sh`: pool inputs and build the full-gene SVD feature cache when needed.
- `prepare_all_spatial_ot_input.sh`: prepare the standard local cohorts.
- `prepare_xenium_spatial_ot_input.sh`: prepare processed Xenium inputs.

Run pairwise OT niches with:

```bash
spatial-ot pairwise-niche fit ...
```

The `cell-niche` command remains available as a descriptor/DeepSHE baseline and QC path.
