#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-ml1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
EXTRAS="${EXTRAS:-dev,viz}"

if [[ -n "$(command -v mamba || true)" ]]; then
  SOLVER=(mamba)
elif [[ -n "$(command -v conda || true)" ]]; then
  SOLVER=(conda)
else
  echo "Neither 'mamba' nor 'conda' is available in PATH." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  "${SOLVER[@]}" create -y -n "$CONDA_ENV" "python=${PYTHON_VERSION}" pip
fi

conda run -n "$CONDA_ENV" python -m pip install --upgrade pip setuptools wheel

if [[ -n "$EXTRAS" ]]; then
  PACKAGE_SPEC=".[${EXTRAS}]"
else
  PACKAGE_SPEC="."
fi

conda run -n "$CONDA_ENV" python -m pip install -e "$PACKAGE_SPEC"

echo "Installed spatial_ot into conda env '$CONDA_ENV'."
