from __future__ import annotations

import json
import platform
import re
import sys
from dataclasses import fields
from pathlib import Path

from .config import DeepFeatureConfig, MultilevelOTConfig


_SHELL_DEFAULT_CHECKS: tuple[tuple[str, str, str], ...] = (
    # (env_var_in_run.sh, attribute_path_on_config, expected_cast)
    ("BASIC_NICHE_SIZE_UM", "ot.basic_niche_size_um", "float"),
    ("MIN_CELLS", "ot.min_cells", "int"),
    ("MAX_SUBREGIONS", "ot.max_subregions", "int"),
    ("LAMBDA_X", "ot.lambda_x", "float"),
    ("LAMBDA_Y", "ot.lambda_y", "float"),
    ("OT_EPS", "ot.ot_eps", "float"),
    ("OVERLAP_CONSISTENCY_WEIGHT", "ot.overlap_consistency_weight", "float"),
    ("RADIUS_UM", "ot.radius_um", "float"),
    ("ALLOW_OBSERVED_HULL_GEOMETRY", "ot.allow_convex_hull_fallback", "bool01"),
    ("AUTO_N_CLUSTERS", "ot.auto_n_clusters", "bool01"),
    ("CANDIDATE_N_CLUSTERS", "ot.candidate_n_clusters", "candidate_n_clusters"),
    ("AUTO_K_MAX_SCORE_SUBREGIONS", "ot.auto_k_max_score_subregions", "int"),
    ("AUTO_K_GAP_REFERENCES", "ot.auto_k_gap_references", "int"),
    ("AUTO_K_MDS_COMPONENTS", "ot.auto_k_mds_components", "int"),
    ("AUTO_K_PILOT_N_INIT", "ot.auto_k_pilot_n_init", "int"),
    ("AUTO_K_PILOT_MAX_ITER", "ot.auto_k_pilot_max_iter", "int"),
    ("MIN_SUBREGIONS_PER_CLUSTER", "ot.min_subregions_per_cluster", "int"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _env_default(text: str, name: str) -> str | None:
    pattern = rf'{re.escape(name)}="\$\{{{re.escape(name)}:-([^"\}}]*)\}}"'
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def _cast_value(raw: str, kind: str):
    if kind == "float":
        return float(raw)
    if kind == "int":
        return int(raw)
    if kind == "bool01":
        return bool(int(raw))
    if kind == "candidate_n_clusters":
        if "-" in raw and "," not in raw:
            left, right = raw.split("-", 1)
            start = int(left.strip())
            stop = int(right.strip())
            return tuple(range(start, stop + 1))
        return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    raise ValueError(f"Unknown cast kind {kind!r}")


def _getattr_path(obj, dotted: str):
    current = obj
    for part in dotted.split("."):
        current = getattr(current, part)
    return current


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("spatial-ot")
    except Exception:
        return "unknown"


def _torch_info() -> dict[str, object]:
    try:
        import torch

        return {
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_device_names": [
                str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
        }
    except Exception as exc:
        return {"torch_version": None, "cuda_available": False, "error": str(exc)}


def run_doctor(*, verbose: bool = True) -> dict[str, object]:
    repo_root = _repo_root()
    run_sh_text = _read(repo_root / "run.sh")

    ot = MultilevelOTConfig()
    deep = DeepFeatureConfig()

    mismatches: list[dict[str, object]] = []
    for env_name, attr, kind in _SHELL_DEFAULT_CHECKS:
        raw = _env_default(run_sh_text, env_name)
        if raw is None:
            continue
        try:
            shell_value = _cast_value(raw, kind)
        except Exception as exc:
            mismatches.append(
                {
                    "env": env_name,
                    "config_attr": attr,
                    "issue": f"could not parse shell default {raw!r}: {exc}",
                }
            )
            continue
        dataclass_value = _getattr_path(
            type("_NS", (), {"ot": ot, "deep": deep})(),
            attr,
        )
        if dataclass_value != shell_value and not (
            isinstance(dataclass_value, float)
            and isinstance(shell_value, float)
            and abs(dataclass_value - shell_value) < 1e-9
        ):
            mismatches.append(
                {
                    "env": env_name,
                    "config_attr": attr,
                    "shell_default": shell_value,
                    "dataclass_default": dataclass_value,
                }
            )

    ot_defaults = {field.name: getattr(ot, field.name) for field in fields(MultilevelOTConfig)}
    deep_defaults = {field.name: getattr(deep, field.name) for field in fields(DeepFeatureConfig)}

    input_dir = repo_root.parent / "spatial_ot_input"
    input_files = sorted(p.name for p in input_dir.glob("*.h5ad")) if input_dir.exists() else []
    outputs_dir = repo_root.parent / "outputs"
    venv_dir = repo_root.parent / ".venv"

    report: dict[str, object] = {
        "spatial_ot_version": _package_version(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _torch_info(),
        "repo_root": str(repo_root),
        "spatial_ot_input_dir": {
            "path": str(input_dir),
            "exists": input_dir.exists(),
            "h5ad_count": len(input_files),
            "h5ad_files": input_files[:20],
        },
        "outputs_dir": {"path": str(outputs_dir), "exists": outputs_dir.exists()},
        "venv_dir": {"path": str(venv_dir), "exists": venv_dir.exists()},
        "shell_defaults_vs_config": {
            "checked": [env for env, *_ in _SHELL_DEFAULT_CHECKS],
            "mismatches": mismatches,
            "run_sh_found": bool(run_sh_text),
        },
        "multilevel_ot_config_defaults": ot_defaults,
        "deep_feature_config_defaults": deep_defaults,
        "status": "ok" if not mismatches else "warn",
    }

    if verbose:
        print(json.dumps(report, indent=2, default=str))
    return report
