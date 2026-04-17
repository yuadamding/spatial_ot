# spatial_ot

`spatial_ot` is a concrete implementation scaffold for the teacher-student spatial niche model described for segmented Visium HD CRC.

The package currently realizes:

- an `8 µm` teacher branch trained on binned Visium HD counts
- an intrinsic cell branch that learns `z`
- a multi-view context/program branch that learns `s`
- teacher-to-cell distillation through `2 µm -> 8 µm -> cell` overlap mappings
- OT niche fitting on neighborhood objects built from state atoms, program activity, and shell profiles
- a COMMOT-style communication head that estimates directional sender->receiver flows over learned programs

The implementation is intentionally stage-wise:

1. teacher pretraining on `8 µm` bins
2. intrinsic cell pretraining
3. context/program training with asymmetric independence regularization
4. OT niche fitting on frozen latents
5. communication-flow fitting and residual attribution

This is a faithful realization of the proposed stack, but not a claim that the exact combined method is already a published benchmarked model.

## Layout

- `spatial_ot/`: package code
- `configs/`: runnable config files and demo prior programs
- `runs/`: experiment outputs

## Smoke run

The smoke config points at the already-prepared `P2 CRC` Visium HD sample in this workspace and keeps the subset/epoch sizes intentionally small.

```bash
cd /storage/hackathon_2026/spatial_ot
conda run -n ml1 python -m spatial_ot.cli train --config configs/p2_crc_smoke.toml
```

Expected outputs land under:

`/storage/hackathon_2026/spatial_ot/runs/p2_crc_smoke/`

Key artifacts:

- `cells_output.h5ad`
- `teacher_bins_output.h5ad`
- `summary.json`
- `niche_flows.csv`
- `top_flow_edges.parquet`
- stage checkpoints under `checkpoints/`

## Pilot config

`configs/p2_crc_pilot.toml` is a larger but still subset-based configuration intended as the next step after the smoke run.

## Prior programs

`configs/crc_demo_programs.json` is a starter library so the full path can run now. It is a demo prior set, not the final biology for your CRC project.

Once marker panels arrive, replace that file with a curated program library and rerun the same CLI.
