# spatial_ot

`spatial_ot` is a compact package for **cell-centered spatial niche discovery from pairwise optimal-transport neighborhood distances**.

The primary method is:

```text
cohort gene-expression embedding
  -> same-sample cell-centered local measures or graphs
  -> pairwise OT/FGW neighborhood dissimilarity matrix
  -> distance-based niche clustering
```

The central output is the neighborhood dissimilarity matrix:

```text
adata.obsp["cell_ot_dissimilarity"]
```

For larger exact runs, the matrix can be written as a NumPy memmap and referenced from `adata.uns["cell_ot_dissimilarity_store"]`.

## Install

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
```

## Generate Cell Embeddings

`pairwise-niche` first creates a cohort-wide gene-expression embedding:

```text
x_i -> z_i = f(x_i)
```

This embedding is stored in:

```text
adata.obsm["X_gene_cohort"]
```

Spatial coordinates are not used to generate `X_gene_cohort`. They enter only later, when each cell-centered local measure is built. The compact runtime supports:

- `--embedding-method pca`: standardize expression features, run PCA, standardize the latent space.
- `--embedding-method svd`: standardize expression features, run randomized TruncatedSVD, standardize the latent space.
- `--embedding-method precomputed`: use an existing feature matrix or embedding from `--feature-obsm-key`.

For count-model embeddings such as scVI, fit the model externally across the cohort and pass the latent representation with `--embedding-method precomputed`.

Precomputed embeddings are standardized by default. Use `--no-standardize-precomputed` when the supplied latent geometry should be preserved exactly.

## Run Pairwise Niche

Example for preprocessed Visium HD input:

```bash
spatial-ot pairwise-niche fit \
  --input-h5ad ../spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad \
  --output-dir ../outputs/spatial_ot/visium_hd_pairwise_niche \
  --feature-obsm-key X_spatial_ot_x_svd_512 \
  --spatial-x-key pooled_cell_x \
  --spatial-y-key pooled_cell_y \
  --sample-obs-key sample_id \
  --spatial-scale 0.2737012522439323 \
  --embedding-method pca \
  --embedding-dim 32 \
  --radius-um 50 \
  --max-neighbors 32 \
  --include-anchor \
  --anchor-weight 0.25 \
  --expression-weight 1.0 \
  --spatial-weight 0.25 \
  --distance-weight 0.10 \
  --ground-cost-normalization sampled_median \
  --sinkhorn-epsilon 0.05 \
  --sinkhorn-iters 50 \
  --distance-mode debiased_entropic_transport \
  --pairwise-mode exact_blockwise \
  --block-size 64 \
  --max-exact-cells 4000 \
  --max-ot-work-units 5e11 \
  --cluster-method agglomerative \
  --candidate-n-clusters 5:30 \
  --model-selection-metrics silhouette,calinski_harabasz,davies_bouldin,dunn
```

For exploratory runs that intentionally use a precomputed 3D UMAP as the cell feature
space, pass it explicitly:

```bash
spatial-ot pairwise-niche fit \
  --feature-obsm-key X_umap_marker_genes_3d \
  --embedding-method precomputed \
  --allow-umap-as-feature \
  ...
```

UMAP is accepted only with this opt-in because it is useful for quick visual/debug
runs but is not generally metric-preserving enough for claim-grade OT distances.

Use `--distance-mode fused_gromov_wasserstein` to compare full local graph topology. In that mode, each cell graph is bounded by both `--radius-um` and `--max-neighbors`; the graph structure matrix stores pairwise spatial distances among the retained support nodes.

Exact all-pairs OT/FGW is quadratic in the number of cells. The command intentionally refuses oversized dense exact jobs unless `--max-exact-cells` is raised, and it also guards against excessive Sinkhorn work with `--max-ot-work-units`. For full Visium HD cohorts with hundreds of thousands of cells, use sampled/landmark subsets until the approximate sparse mode is added.

For full-cohort exploratory Visium HD runs where exact all-cell-pair FGW is infeasible, the repository includes scripts that make the approximation explicit:

- `scripts/run_visium_hd_umap3d_direct_pairwise_fgw.py`: builds the full cell graphs and writes an exact all-pairs feasibility report without using landmarks or references.
- `scripts/run_visium_hd_umap3d_sample_rough_full_fine_fgw.py`: rough-clusters cells independently per sample, compares all rough-cluster representative cell graphs with full-cohort FGW, then propagates fine labels back to cells.
- `scripts/run_visium_hd_umap3d_landmark_fgw.py`: landmark FGW assignment for faster exploratory runs.

Main outputs:

- `cells_pairwise_niche.h5ad`
- `cell_ot_dissimilarity.npy`, when `--distance-store npy_memmap` or automatic memmap output is selected.
- `summary.json`
- `ot_niche_colors.json`, a deterministic high-contrast color map for niche labels.

Key AnnData fields:

- `adata.obsm["X_gene_cohort"]`
- `adata.obsp["cell_ot_dissimilarity"]` for small in-H5AD distance matrices.
- `adata.obsp["cell_ot_affinity"]`
- `adata.obs["ot_niche"]`
- `adata.obs["ot_niche_assignment_score"]`
- `adata.obs["ot_niche_instance"]`
- `adata.obs["n_neighbors_full_r50"]`
- `adata.obs["n_neighbors_retained_r50"]`
- `adata.obs["neighbor_retention_fraction_r50"]`
- `adata.obs["local_density_full_per_um2_r50"]`

## Method Notes

For each anchor cell, `pairwise-niche` builds a same-sample local measure:

```text
M_i = sum_j a_ij delta([z_j, relative_x/r, relative_y/r, distance/r])
```

The ground cost compares expression location and relative spatial organization. Clustering is performed from the precomputed OT dissimilarity matrix using agglomerative clustering, k-medoids, or Leiden on an OT-kNN affinity graph. KMeans is intentionally not used because it does not accept a precomputed dissimilarity matrix.

For agglomerative or k-medoids runs, pass `--candidate-n-clusters 5:30` to select the cluster count across every K from 5 through 30. If `--n-clusters` is omitted and no candidate list is supplied, this 5:30 range is used by default. Candidates are ranked by a model-selection ensemble over precomputed-distance silhouette, distance-based Calinski-Harabasz, medoid Davies-Bouldin, and Dunn scores; customize the set with `--model-selection-metrics`. For Leiden runs, pass `--candidate-resolutions 0.4,0.6,0.8,1.0,1.2` to select a resolution with the same metric ensemble. The selected model and all candidate scores/ranks are written to `summary.json` and `adata.uns["pairwise_niche_clustering_summary"]`.

Niche colors are deterministic and high contrast. They are stored in `ot_niche_colors.json`, `adata.uns["pairwise_niche_color_map"]`, and `adata.uns["ot_niche_colors"]`; `ON12` is kept bright orange (`#ff7f00`).

For graph-topology runs, `--distance-mode fused_gromov_wasserstein` compares:

```text
node feature cost: ||z_a - z_b||^2
graph structure cost: (C_i[a,a'] - C_k[b,b'])^2
```

where `C_i` is the pairwise spatial-distance matrix inside anchor cell `i`'s retained local graph. `--fgw-alpha` controls the topology weight.

By default, expression, relative-xy, and radial-distance cost contributions are normalized by sampled median component costs before user weights are applied. This keeps a 32-dimensional expression embedding from automatically overwhelming the lower-dimensional spatial terms.

The default debiased distance is recorded as `debiased_entropic_transport`. It subtracts self transport costs from the entropic-plan transport cost. Metadata records that this is the plan transport cost only and does not include the entropy objective term.

If `--include-anchor` and `--anchor-weight > 0` are both used, anchor expression contributes inside the local measure and as a direct anchor penalty. Set `--anchor-weight 0` for a more context-only niche run.

## Baseline And QC Path

The descriptor/DeepSHE command is retained as a baseline and QC workflow:

```bash
spatial-ot cell-niche fit ...
```

It writes `spatial_niche` labels from descriptor or neural embeddings. The primary method for new analyses is `pairwise-niche`, where OT dissimilarity is computed before clustering and is the clustering input.

## Pool And Prepare Inputs

Pool same-schema sample H5AD files into a sample-aware cohort file:

```bash
spatial-ot pool-inputs \
  --input-dir ../spatial_ot_input \
  --output-h5ad ../spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad \
  --feature-obsm-key X \
  --sample-glob '*_cells_marker_genes_umap3d.h5ad' \
  --spatial-x-key cell_x \
  --spatial-y-key cell_y
```

Create a reusable full-gene SVD feature cache:

```bash
SPATIAL_OT_X_SVD_COMPONENTS=512 \
spatial-ot prepare-inputs \
  --input-h5ad ../spatial_ot_input/visium_hd_spatial_ot_input_pooled.h5ad \
  --feature-obsm-key X \
  --output-obsm-key X_spatial_ot_x_svd_512
```

## Package Layout

```text
spatial_ot/
  pairwise_niche/  # primary pairwise OT distance-matrix method
  cell_niche/      # descriptor/DeepSHE baseline and QC workflow
  feature_source.py
  pooling.py
  cli.py
scripts/
tests/
```

## Current Validation Status

Implemented:

- cohort-wide expression embeddings that do not use spatial coordinates
- sample-isolated local spatial measures
- shell/state-aware neighbor caps
- exact blockwise balanced Sinkhorn OT
- debiased Sinkhorn divergence option
- graph-topology fused Gromov-Wasserstein distance mode
- dense or memmap OT distance output
- distance-based clustering from the precomputed OT matrix
- connected cell-level niche instances

Still planned:

- sparse approximate OT-kNN / landmark mode for very large cohorts
- transform/predict bundles for new samples
- full null and ablation reports
- density and sample leakage reports
