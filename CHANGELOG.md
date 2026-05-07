# Changelog

## 3.0.14

- Add `compute_cross_ot_distance_matrix()` for query-to-reference OT/FGW distances without building a query-by-query all-pairs matrix.
- Reuse the exact pairwise block machinery for cross OT/FGW distance blocks, including active-support trimming, debiased Sinkhorn self-costs, anchor penalties, and adaptive block sizing.
- Allow cross FGW runs to use a provided reference `fgw_structure_cost_scale` so transform, medoid, landmark, and refinement workflows can stay on the fitted structure-cost scale.
- Export the cross-distance helper from `spatial_ot` and `spatial_ot.pairwise_niche`.
- Add regression tests showing cross OT/FGW distances match exact reference submatrices when using the same orientation and fitted FGW scale.

## 3.0.13

- Add sampled Sinkhorn marginal-error diagnostics to pairwise OT distance metadata.
- Warn multi-sample PCA/SVD runs that pairwise-niche records batch keys but does not perform batch correction.
- Make the CLI `--radius-um` help explicitly state the 50 µm default.
- Add regression tests for Sinkhorn diagnostics, batch-correction warnings, and the 50 µm default neighbor radius.

## 3.0.12

- Use adaptive exact-run block size before debiased Sinkhorn self-cost computation so `--block-size 0` does not fall back to one-cell self-cost batches.
- Default direct `build_local_measures()` calls to lazy structure-matrix construction; H5AD FGW runs still opt in explicitly.
- Estimate OT-kNN positive distance scale from sampled rows for large matrices instead of materializing all finite positive entries.
- Add regression tests for adaptive self-cost batching and sampled OT-kNN scale estimation.

## 3.0.11

- Replace broadcasted OT/FGW feature-cost tensors with Gram-style squared-distance blocks to reduce exact full-matrix peak memory.
- Preserve exact zero mass for padded support points in Sinkhorn and FGW kernels.
- Vectorize debiased Sinkhorn self-cost computation and trim each distance block to its active support width.
- Let Sinkhorn-only H5AD runs skip local FGW structure-matrix allocation.
- Use argpartition for OT-kNN neighbor extraction instead of full row sorts.
- Add a representative-only Visium HD full-cohort graph export mode for exactly 100,000 representative cell graphs without materializing dense all-pairs FGW.
- Add regression tests for Gram costs, batched self-costs, zero-weight padding, and lazy structure construction.

## 3.0.10

- Avoid full-matrix dense symmetrization for memmap distance outputs by symmetrizing diagonal blocks during blockwise writes.
- Keep Leiden OT-kNN clustering on sparse kNN distance/connectivity graphs instead of storing the full dense distance matrix as a sparse matrix.
- Canonicalize FGW `adjacency` mode to `binary_edge_distance` and record disconnected shortest-path structure diagnostics.
- Record FGW debiasing and diagonal-forcing semantics in distance metadata.
- Add adaptive exact-run block sizing through `--target-block-memory-gib`.
- Save reference labels, medoids, and cell IDs in the model bundle for future transform/assignment workflows.

## 3.0.9

- Default `pairwise-niche` to context-first anchor handling with `anchor_weight=0`.
- Default isolated no-neighbor local measures to `anchor_fallback` instead of a zero dummy token.
- Make FGW use expression-only node features by default and record complete spatial-structure semantics explicitly.
- Add explicit FGW structure modes: local kNN shortest path, radius-graph shortest path, adjacency, and complete Euclidean structure.
- Normalize FGW structure costs by sampled median scale before applying `fgw_alpha`.
- Add an FGW-specific work estimate/guard and export `estimate_pairwise_fgw_work`.
- Rename default model-selection metrics to pseudo/medoid/percentile forms, add percentile-Dunn, and make singleton assignment scores zero.
- Add within/between distance-ratio metadata for model selection and save/load helpers for expression embedding state.
- Remove the older descriptor/DeepSHE `cell-niche` package and CLI so the repository only exposes the current pairwise OT/FGW workflow.
- Confirm package metadata version `3.0.9`.

## 3.0.8

- Add precomputed UMAP feature opt-in for exploratory pairwise-niche runs.
- Add distance-matrix model selection over candidate cluster counts/resolutions, defaulting fixed-K selection to `5:30` when no `--n-clusters` is supplied.
- Add deterministic high-contrast niche colors, including bright orange for `ON12`.
- Add full-cohort Visium HD runner scripts for direct all-pairs FGW feasibility checks, landmark FGW, and per-sample rough clustering followed by full-cohort fine FGW.
- Confirm package metadata version `3.0.8`.

## 3.0.7

- Add fused Gromov-Wasserstein (FGW) neighborhood distances over complete local spatial-distance structures.
- Store per-cell local structure matrices with explicit maximum radius and maximum retained-neighbor metadata.
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
- Remove the legacy boundary-generation workflow, search helpers, old deep feature adapter, older run scripts, TOML configs, and matching tests.
- Keep input pooling and feature-cache preparation helpers because they are still needed for preprocessed Visium HD and Xenium cohorts.
- Confirm package metadata version `3.0.6`.
