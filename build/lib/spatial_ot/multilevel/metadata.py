from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess

import anndata as ad

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def _pyproject_version() -> str | None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return None
    payload = tomllib.loads(pyproject.read_text())
    return str(payload.get("project", {}).get("version", "unknown"))


def package_version() -> str:
    local_version = _pyproject_version()
    if local_version is not None:
        return local_version
    try:
        return version("spatial-ot")
    except PackageNotFoundError:
        return "unknown"


def git_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return completed.stdout.strip() or None
    return None


def latent_source_label(feature_source: dict, deep_summary: dict) -> str:
    if bool(deep_summary.get("enabled")):
        output_embedding = deep_summary.get("output_embedding")
        return f"deep_{output_embedding}" if output_embedding is not None else "deep_unspecified"

    feature_key = str(feature_source.get("feature_key", ""))
    input_mode = str(feature_source.get("input_mode", "obsm"))
    preprocessing = str(feature_source.get("preprocessing", ""))
    if feature_key.startswith("X_spatial_ot_x_svd_"):
        return "prepared_svd"
    if input_mode == "X" and "truncated_svd" in preprocessing:
        return "runtime_svd"
    if feature_key == "X":
        return "raw_x"
    return f"obsm:{feature_key}"


def extract_count_target(adata: ad.AnnData, *, count_layer: str | None):
    if count_layer is None:
        return None, None
    layer_key = str(count_layer)
    if layer_key == "X":
        if adata.X is None:
            raise ValueError("deep.count_layer requested the primary count matrix, but adata.X is missing.")
        return adata.X, "X"
    if layer_key not in adata.layers:
        raise KeyError(f"deep.count_layer '{layer_key}' was not found in adata.layers.")
    return adata.layers[layer_key], layer_key
