# Changelog

## 3.0.6

- Make `pairwise-niche` the primary spatial niche workflow.
- Add cohort expression embeddings, cell-centered local measures, exact blockwise Sinkhorn OT distance matrices, and distance-based clustering.
- Write `X_gene_cohort`, `cell_ot_dissimilarity`, `cell_ot_affinity`, `ot_niche`, and connected `ot_niche_instance` outputs.
- Keep `cell-niche` as a descriptor/DeepSHE baseline and QC workflow.
- Remove the legacy boundary-generation workflow, search helpers, old deep feature adapter, older run scripts, TOML configs, and matching tests.
- Keep input pooling and feature-cache preparation helpers because they are still needed for preprocessed Visium HD and Xenium cohorts.
- Preserve direct cell-level baseline outputs: `spatial_niche`, `spatial_niche_assignment_score`, and connected cell-level `spatial_niche_instance`.
- Confirm package metadata version `3.0.6`.
