# Changelog

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
