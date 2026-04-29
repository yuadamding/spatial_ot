# Changelog

## 0.1.13

- Change the primary subregion clustering step to `pooled_subregion_latent`: each fitted subregion now gets a raw member-cell feature-distribution latent embedding, all cohort subregion embeddings are pooled, and KMeans/model selection clusters that pooled matrix without using spatial coordinates, subregion centers, overlap edges, compressed OT supports, or OT candidate costs.
- Upgrade the pooled subregion latent from a fixed mean/std-only summary to configurable distributional modes: `mean_std_shrunk` default reliability shrinkage, `mean_std_skew_count`, `mean_std_quantile`, `codebook_histogram`, and `mean_std_codebook`.
- Keep the historical OT-dictionary assignment available as `subregion_clustering_method="ot_dictionary"` while using fixed-label OT atoms/projections as diagnostics for pooled-latent labels.
- Save raw-member feature-distribution subregion latent embeddings and clustering-method metadata in summaries, H5AD metadata, subregion tables, and candidate diagnostic NPZ outputs.
- Make spot-latent cluster-anchor OT fallbacks explicit: occurrence NPZ and `summary.json` now record requested/effective anchor-distance methods, fallback matrices, fallback fractions, and solver status codes.
- Promote latent MDS quality into visible QC fields, including anchor stress status, atom high-stress cluster counts, and explicit stress/eigenvalue thresholds used by the concern report.
- Store alternate atom-posterior entropy diagnostics for fixed-temperature and cost-gap-temperature baselines alongside the default entropy-calibrated posterior.
- Block within-niche latent-heterogeneity claims when anchor OT falls back, MDS geometry is unreliable, posterior entropy is uncalibrated, or temperature calibration suggests inconsistent atom-cost scaling.
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
