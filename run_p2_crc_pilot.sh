#!/usr/bin/env bash
set -euo pipefail

cd /storage/hackathon_2026/spatial_ot
conda run -n ml1 python -m spatial_ot train --config configs/p2_crc_pilot.toml
