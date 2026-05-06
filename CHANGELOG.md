# Changelog

## 3.0.7

- Add graph-topology fused Gromov-Wasserstein (FGW) neighborhood distances.
- Store per-cell local graph structure matrices with explicit maximum radius and maximum retained-neighbor metadata.
- Expose FGW distance controls in the pairwise niche config and CLI.
- Add FGW block and exact-matrix tests, plus full-cohort FGW output support.
- Confirm package metadata version `3.0.7`.

## 3.0.6

- Make `pairwise-niche` the primary spatial niche workflow.
- Add cohort expression embeddings, cell-centered local measures, exact blockwise Sinkhorn OT distance matrices, and distance-based clustering.
- Add sampled-median ground-cost normalization so expression dimensions do not automatically dominate relative spatial terms.
- Add exact-OT work-unit guards in addition to dense matrix-size guards.
- Rename the default debiased distance to `debiased_entropic_transport` while keeping `sinkhorn_divergence` as a compatibility alias.
- Add precomputed-embedding standardization controls, isolated-cell policy controls, local-scaled OT-kNN affinities, and a separate higher-cap graph for connected niche instances.
- Write `X_gene_cohort`, `cell_ot_dissimilarity`, `cell_ot_affinity`, `ot_niche`, and connected `ot_niche_instance` outputs.
- Keep `cell-niche` as a descriptor/DeepSHE baseline and QC workflow.
- Remove the legacy boundary-generation workflow, search helpers, old deep feature adapter, older run scripts, TOML configs, and matching tests.
- Keep input pooling and feature-cache preparation helpers because they are still needed for preprocessed Visium HD and Xenium cohorts.
- Preserve direct cell-level baseline outputs: `spatial_niche`, `spatial_niche_assignment_score`, and connected cell-level `spatial_niche_instance`.
- Confirm package metadata version `3.0.6`.
