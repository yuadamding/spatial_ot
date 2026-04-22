#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export POOL_ALL_INPUTS="${POOL_ALL_INPUTS:-0}"
export SAMPLE_KEY="${SAMPLE_KEY:-p2_crc}"
export FEATURE_OBSM_KEY="${FEATURE_OBSM_KEY:-X_umap_marker_genes_3d}"
export ALLOW_UMAP_AS_FEATURE="${ALLOW_UMAP_AS_FEATURE:-1}"
export ALLOW_OBSERVED_HULL_GEOMETRY="${ALLOW_OBSERVED_HULL_GEOMETRY:-1}"
exec bash "$SCRIPT_DIR/run.sh" "$@"
