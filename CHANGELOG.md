# Changelog

## 0.2.2

- Rename the current heterogeneity target implementation to `subregion_clustering_method="heterogeneity_descriptor_niche"` and keep `heterogeneity_ot_niche` only as a legacy alias until true fused-OT / FGW distances are implemented.
- Change packaged run defaults from `pooled_subregion_latent` to `heterogeneity_descriptor_niche`; the pooled summary path remains available as a composition/distribution baseline.
- Block-normalize and explicitly weight the heterogeneity descriptor blocks: soft cell-state composition, diversity/multimodality, canonical spatial-state density fields, and within-subregion state-pair co-occurrence.
- Change the pairwise co-occurrence descriptor to observed-over-expected enrichment by default, so common cell-state pairs do not dominate solely through marginal composition.
- Align package metadata with the local `0.2.2` state.
- Add `subregion_construction_method="joint_refinement"` for constrained segmentation-clustering feedback: deep graph segmentation initializes full-coverage regions, pooled subregion latent clusters provide prototypes, adjacent boundary cells may move only when cluster coherence improves after spatial/cut penalties, and the final partition is reconnected/merged to satisfy `min_cells`.
- Update the high-VRAM cohort profile to default to `SUBREGION_CONSTRUCTION_METHOD=joint_refinement`, while leaving plain `deep_segmentation` and coordinate-only construction available as ablations.
- Treat `max_subregion_area_um2` as a soft QC target for generated subregions. It is now reported in summaries and warnings, but it no longer shrinks seed scale, blocks connected-region merging, hard-splits final regions, or takes precedence over `min_cells`.
- Clarify that the current deep-boundary cohort path should use learned autoencoder context features and joint refinement, while coordinate-only construction remains the baseline/ablation.
- Change the primary subregion clustering step to `pooled_subregion_latent`: each fitted subregion now gets a raw member-cell feature-distribution latent embedding, all cohort subregion embeddings are pooled, and KMeans/model selection clusters that pooled matrix without using spatial coordinates, subregion centers, overlap edges, compressed OT supports, or OT candidate costs.
- Upgrade the pooled subregion latent from a fixed mean/std-only summary to configurable distributional modes: `mean_std_shrunk` default reliability shrinkage, `mean_std_skew_count`, `mean_std_quantile`, `codebook_histogram`, and `mean_std_codebook`.
- Make `mean_std_shrunk` sample-aware by default when `sample_id` is available: low-cell-count subregions now shrink toward a configurable sample/cohort prior instead of only a cohort prior.
- Expose `subregion_latent_heterogeneity_weight` and `subregion_latent_sample_prior_weight` through config, CLI, scripts, summaries, and diagnostics.
- Change codebook histogram modes to whiten features before fitting the cell-state codebook and use soft code assignments rather than hard nearest-centroid counts.
- Add executable validation-suite planning to concern reports, including fixed-K, shrinkage, heterogeneity-weight, codebook-size, spatial-scale/leakage, and spatial-niche validation commands.
- Add a lightweight `spatial-niche-validation` command that summarizes cluster-level subregion statistics, sample mixing, spatial adjacency homophily with permutation p-values/z-scores, and connected-component fragmentation from a finished run.
- Keep the historical OT-dictionary assignment available as `subregion_clustering_method="ot_dictionary"` while using fixed-label OT atoms/projections as diagnostics for pooled-latent labels.
- Save raw-member feature-distribution subregion latent embeddings and clustering-method metadata in summaries, H5AD metadata, subregion tables, and candidate diagnostic NPZ outputs.
- Make spot-latent cluster-anchor OT fallbacks explicit: occurrence NPZ and `summary.json` now record requested/effective anchor-distance methods, fallback matrices, fallback fractions, and solver status codes.
- Promote latent MDS quality into visible QC fields, including anchor stress status, atom high-stress cluster counts, and explicit stress/eigenvalue thresholds used by the concern report.
- Store alternate atom-posterior entropy diagnostics for fixed-temperature and cost-gap-temperature baselines alongside the default entropy-calibrated posterior.
- Block within-niche latent-heterogeneity claims when anchor OT falls back, MDS geometry is unreliable, posterior entropy is uncalibrated, or temperature calibration suggests inconsistent atom-cost scaling.
- Render the per-sample spot-latent global key as a 3D plot. The key uses 3D MDS of saved cluster-anchor distances when available, while the slide map keeps global 2D RGB scaling for visual comparability.
- Calibrate the packaged 10 GB deep-segmentation cohort profile to `DEEP_BATCH_SIZE=81920`, matching the successful pooled run that peaked at roughly 8.9 GiB reserved VRAM without OOM.

## 0.1.12

- Compact the package around the active multilevel OT workflow by removing the old teacher-student scaffold, root-level compatibility facades, legacy TOML/program-prior examples, and redundant shell aliases.
- Tune the packaged deep-segmentation cohort GPU profile for the local 10 GB RTX 3080 target: larger autoencoder defaults and a 131k batch target roughly 9 GB live VRAM without using the OOM-heavy shapes.
- Make CPU thread budgeting dynamic and multi-threaded across OpenMP, MKL, OpenBLAS, NumExpr, BLIS, Numba, Torch intraop, and Torch interop; prepared-input validation now gets the same thread env before Python starts.
- Add CUDA runtime-memory QC to packaged runs so `summary.json` records the requested VRAM target, observed peak reserved memory, and whether the run hit the target band.
- Replace expected cross-cost cluster anchors in the default spot-latent chart with balanced OT distances between fitted cluster atom measures.
- Add spot-latent MDS diagnostics for global cluster anchors and per-cluster atom embeddings, including stress and eigenvalue-mass summaries.
- Calibrate default atom-posterior temperature by target normalized entropy, while retaining fixed and cost-gap modes for diagnostics.
- Store both unweighted and confidence-weighted cell-level spot-latent previews; confidence weights are for opacity/QC rather than suppressing transitional cells.
- Make whole-sample spot-latent plots use global latent color scaling by default; within-cluster RGB rescaling is now an explicit diagnostic mode.
- Split concern reporting between subregion-clustering claims and within-niche latent-heterogeneity claims.

## 0.1.10

- Make coordinate-only data-driven subregion construction the baseline workflow; feature-aware and deep graph segmentation remain explicit sensitivity modes.
- Add mandatory concern-report generation in packaged runs, with a strict mode that fails when primary-claim blockers remain.
- Add concern-report gates for coordinate-only baselines, leakage ablations, fixed-K stability after auto-K, and OT candidate-cost comparability.
- Mark spot-level Fisher/discriminative latent maps as supervised diagnostic visualizations learned from fitted OT labels, not independent validation.
- Keep whole-sample niche and spot-latent plotting outputs organized by sample, with subregion polygon views and inherited cell-label views.
- Update package metadata to `0.1.10` and keep generated artifacts out of version control.
