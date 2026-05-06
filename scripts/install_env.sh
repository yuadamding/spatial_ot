#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
VENV_DIR="${VENV_DIR:-../.venv}"
EXTRAS="${EXTRAS:-dev,viz}"
MIN_PYTHON_MINOR="${MIN_PYTHON_MINOR:-10}"

default_cpu_threads() {
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN 2>/dev/null && return
  fi
  if command -v nproc >/dev/null 2>&1; then
    nproc 2>/dev/null && return
  fi
  echo 1
}

CPU_THREADS="${CPU_THREADS:-$(default_cpu_threads)}"
export CPU_THREADS
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$CPU_THREADS}"
export MAKEFLAGS="${MAKEFLAGS:--j${CPU_THREADS}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$CPU_THREADS}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-$CPU_THREADS}"

python_is_compatible() {
  local candidate="$1"
  "$candidate" - "$MIN_PYTHON_MINOR" <<'PY'
import sys

min_minor = int(sys.argv[1])
ok = sys.version_info.major == 3 and sys.version_info.minor >= min_minor
raise SystemExit(0 if ok else 1)
PY
}

python_can_create_venv() {
  local candidate="$1"
  "$candidate" - <<'PY'
try:
    import ensurepip  # noqa: F401
    import venv  # noqa: F401
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

resolve_python_bin() {
  if [[ -n "$PYTHON_BIN" ]]; then
    if [[ -z "$(command -v "$PYTHON_BIN" || true)" ]]; then
      echo "Python executable '$PYTHON_BIN' is not available in PATH." >&2
      exit 1
    fi
    if ! python_is_compatible "$PYTHON_BIN"; then
      echo "Python executable '$PYTHON_BIN' does not satisfy Python >= 3.${MIN_PYTHON_MINOR}." >&2
      exit 1
    fi
    if ! python_can_create_venv "$PYTHON_BIN"; then
      echo "Python executable '$PYTHON_BIN' cannot create a pip-enabled virtual environment." >&2
      echo "Install the corresponding venv support package or choose another interpreter, for example: PYTHON_BIN=python3.10 bash scripts/install_env.sh" >&2
      exit 1
    fi
    printf '%s\n' "$PYTHON_BIN"
    return
  fi

  local candidate
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if [[ -n "$(command -v "$candidate" || true)" ]] && python_is_compatible "$candidate" && python_can_create_venv "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  echo "Could not find a Python interpreter satisfying Python >= 3.${MIN_PYTHON_MINOR}." >&2
  echo "A usable interpreter must also support creating pip-enabled virtual environments." >&2
  echo "Set PYTHON_BIN explicitly, for example: PYTHON_BIN=python3.10 bash scripts/install_env.sh" >&2
  exit 1
}

SELECTED_PYTHON="$(resolve_python_bin)"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ -d "$VENV_DIR" && ! -x "$VENV_PYTHON" ]]; then
  echo "Existing virtual environment '$VENV_DIR' is incomplete; recreating it." >&2
  rm -rf "$VENV_DIR"
fi

if [[ -x "$VENV_PYTHON" ]] && ! python_is_compatible "$VENV_PYTHON"; then
  echo "Existing virtual environment '$VENV_DIR' uses an incompatible Python version; recreating it." >&2
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$SELECTED_PYTHON" -m venv "$VENV_DIR"
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Virtual environment python is missing: $VENV_PYTHON" >&2
  exit 1
fi

"$VENV_PYTHON" -m pip install --upgrade pip "setuptools<82" wheel

if [[ -n "$EXTRAS" ]]; then
  PACKAGE_SPEC=".[${EXTRAS}]"
else
  PACKAGE_SPEC="."
fi

"$VENV_PYTHON" -m pip install -e "$PACKAGE_SPEC"

echo "Installed spatial_ot into virtual environment '$VENV_DIR'."
echo "Python executable: $VENV_PYTHON"
echo "Python interpreter: $("$VENV_PYTHON" -c 'import sys; print(sys.version.split()[0])')"
echo "CPU build/runtime threads: $CPU_THREADS"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
