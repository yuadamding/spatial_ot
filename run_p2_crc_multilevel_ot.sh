#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export POOL_ALL_INPUTS="${POOL_ALL_INPUTS:-0}"
export SAMPLE_KEY="${SAMPLE_KEY:-p2_crc}"
exec bash "$SCRIPT_DIR/run.sh" "$@"
