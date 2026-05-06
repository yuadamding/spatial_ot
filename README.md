# spatial_ot

`spatial_ot` is a compact package for **cell-centered spatial niche discovery from pairwise optimal-transport neighborhood distances**.

The primary method is:

```text
cohort gene-expression embedding
  -> same-sample cell-centered local measures
  -> pairwise OT neighborhood dissimilarity matrix
  -> distance-based niche clustering
```

The central output is the OT distance matrix:

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
  --sinkhorn-epsilon 0.05 \
  --sinkhorn-iters 50 \
  --distance-mode sinkhorn_divergence \
  --pairwise-mode exact_blockwise \
  --block-size 64 \
  --max-exact-cells 5000 \
  --cluster-method agglomerative \
  --n-clusters 15
```

Exact all-pairs OT is quadratic in the number of cells. The command intentionally refuses oversized dense exact jobs unless `--max-exact-cells` is raised. For full Visium HD cohorts with hundreds of thousands of cells, use sampled/landmark subsets until the approximate sparse mode is added.

Main outputs:

- `cells_pairwise_niche.h5ad`
- `cell_ot_dissimilarity.npy`, when `--distance-store npy_memmap` or automatic memmap output is selected.
- `summary.json`

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
- dense or memmap OT distance output
- distance-based clustering from the precomputed OT matrix
- connected cell-level niche instances

Still planned:

- sparse approximate OT-kNN / landmark mode for very large cohorts
- Fused Gromov-Wasserstein graph-aware distance mode
- transform/predict bundles for new samples
- full null and ablation reports
- density and sample leakage reports
