from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from spatial_ot.doctor import run_doctor


def test_run_doctor_reports_no_shell_default_drift() -> None:
    report = run_doctor(verbose=False)
    assert report["spatial_ot_version"]
    assert report["torch"]["torch_version"]
    assert report["shell_defaults_vs_config"]["run_sh_found"] is True
    assert report["shell_defaults_vs_config"]["mismatches"] == [], report[
        "shell_defaults_vs_config"
    ]["mismatches"]
    assert report["status"] == "ok"
    assert "multilevel_ot_config_defaults" in report
    assert "deep_feature_config_defaults" in report
    assert report["multilevel_ot_config_defaults"]["basic_niche_size_um"] == 50.0
    assert report["multilevel_ot_config_defaults"]["min_cells"] == 25
    assert report["multilevel_ot_config_defaults"]["max_subregions"] == 5000
    assert (
        report["multilevel_ot_config_defaults"]["allow_convex_hull_fallback"] is False
    )
    assert report["multilevel_ot_config_defaults"]["subregion_feature_weight"] == 0.0
    assert report["multilevel_ot_config_defaults"]["subregion_feature_dims"] == 16
    assert report["multilevel_ot_config_defaults"]["candidate_n_clusters"] == (
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
    )
    assert report["multilevel_ot_config_defaults"]["min_subregions_per_cluster"] == 50
    checked = set(report["shell_defaults_vs_config"]["checked"])
    assert {
        "CANDIDATE_N_CLUSTERS",
        "MIN_SUBREGIONS_PER_CLUSTER",
        "SUBREGION_FEATURE_WEIGHT",
    }.issubset(checked)


def test_doctor_cli_exits_zero_on_clean_tree(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "spatial_ot", "doctor", "--strict"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["shell_defaults_vs_config"]["mismatches"] == []


def test_doctor_cli_tolerates_openmp_thread_lists(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "8,4"
    completed = subprocess.run(
        [sys.executable, "-m", "spatial_ot", "doctor"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
