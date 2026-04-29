from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _numeric_summary(values: object) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q10": None,
            "q25": None,
            "median": None,
            "q75": None,
            "q90": None,
            "max": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _entropy_from_counts(counts: np.ndarray) -> tuple[float, float]:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0, 0.0
    probs = counts[counts > 0] / total
    entropy = float(-np.sum(probs * np.log(probs)))
    if probs.size <= 1:
        return entropy, 0.0
    return entropy, float(entropy / np.log(probs.size))


def _cluster_stats(table: pd.DataFrame, label_col: str, sample_col: str | None) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    for label, group in table.groupby(label_col, sort=True):
        sample_counts = (
            group[sample_col].astype(str).value_counts().sort_index()
            if sample_col is not None and sample_col in group
            else pd.Series({"cohort": int(len(group))}, dtype=np.int64)
        )
        _, normalized_entropy = _entropy_from_counts(sample_counts.to_numpy())
        dominant_fraction = float(sample_counts.max() / max(int(sample_counts.sum()), 1))
        row: dict[str, object] = {
            "cluster": int(label),
            "n_subregions": int(len(group)),
            "n_samples": int(sample_counts.size),
            "sample_normalized_entropy": normalized_entropy,
            "dominant_sample_fraction": dominant_fraction,
            "sample_counts": {str(k): int(v) for k, v in sample_counts.items()},
        }
        flat: dict[str, object] = {
            "cluster": int(label),
            "n_subregions": int(len(group)),
            "n_samples": int(sample_counts.size),
            "sample_normalized_entropy": normalized_entropy,
            "dominant_sample_fraction": dominant_fraction,
        }
        for col in (
            "n_cells",
            "geometry_point_count",
            "shape_area_um2",
            "cell_density_per_um2",
            "shape_compactness",
            "shape_eccentricity",
            "subregion_latent_shrinkage_alpha",
            "subregion_latent_raw_to_shrunk_distance",
            "assignment_margin",
        ):
            if col in group:
                summary = _numeric_summary(group[col].to_numpy())
                row[col] = summary
                flat[f"{col}_median"] = summary["median"]
                flat[f"{col}_q10"] = summary["q10"]
                flat[f"{col}_q90"] = summary["q90"]
        rows.append(row)
        csv_rows.append(flat)
    return rows, pd.DataFrame(csv_rows)


def _subsample_table(table: pd.DataFrame, *, max_subregions: int, random_state: int) -> pd.DataFrame:
    if max_subregions <= 0 or len(table) <= max_subregions:
        return table
    return table.sample(n=int(max_subregions), random_state=int(random_state)).sort_index()


def _knn_label_homophily(
    table: pd.DataFrame,
    *,
    label_col: str,
    sample_col: str | None,
    k: int,
    n_permutations: int,
    random_state: int,
) -> dict[str, object]:
    from sklearn.neighbors import NearestNeighbors

    required = {"center_x_um", "center_y_um", label_col}
    if not required.issubset(table.columns):
        return {
            "available": False,
            "reason": "missing_center_or_label_columns",
            "required_columns": sorted(required),
        }

    rng = np.random.default_rng(int(random_state))
    group_key = sample_col if sample_col is not None and sample_col in table else None
    groups = table.groupby(group_key, sort=True) if group_key is not None else [("cohort", table)]
    per_sample: list[dict[str, object]] = []
    same_total = 0
    edge_total = 0
    permuted_values: list[float] = []

    for sample_id, group in groups:
        coords = group[["center_x_um", "center_y_um"]].to_numpy(dtype=np.float64, copy=False)
        labels = group[label_col].to_numpy(dtype=np.int64, copy=False)
        finite = np.isfinite(coords).all(axis=1)
        coords = coords[finite]
        labels = labels[finite]
        n = int(labels.size)
        if n < 2:
            per_sample.append(
                {
                    "sample_id": str(sample_id),
                    "n_subregions": n,
                    "homophily": None,
                    "permutation_mean": None,
                    "permutation_p95": None,
                }
            )
            continue
        local_k = min(max(int(k), 1), n - 1)
        nn = NearestNeighbors(n_neighbors=local_k + 1)
        nn.fit(coords)
        neigh = nn.kneighbors(coords, return_distance=False)[:, 1:]
        same = int(np.sum(labels[:, None] == labels[neigh]))
        edges = int(neigh.size)
        homophily = float(same / max(edges, 1))
        same_total += same
        edge_total += edges

        local_perm: list[float] = []
        for _ in range(max(int(n_permutations), 0)):
            perm_labels = rng.permutation(labels)
            value = float(np.mean(perm_labels[:, None] == perm_labels[neigh]))
            local_perm.append(value)
            permuted_values.append(value)
        per_sample.append(
            {
                "sample_id": str(sample_id),
                "n_subregions": n,
                "k": local_k,
                "homophily": homophily,
                "permutation_mean": float(np.mean(local_perm)) if local_perm else None,
                "permutation_p95": float(np.quantile(local_perm, 0.95)) if local_perm else None,
            }
        )

    observed = float(same_total / max(edge_total, 1)) if edge_total else None
    return {
        "available": observed is not None,
        "k": int(k),
        "n_neighbor_edges": int(edge_total),
        "observed": observed,
        "permutation_mean": float(np.mean(permuted_values)) if permuted_values else None,
        "permutation_p95": float(np.quantile(permuted_values, 0.95)) if permuted_values else None,
        "excess_over_permutation_mean": (
            float(observed - np.mean(permuted_values)) if observed is not None and permuted_values else None
        ),
        "per_sample": per_sample,
    }


def spatial_niche_validation_report(
    run_dir: str | Path,
    *,
    output_json: str | Path | None = None,
    output_cluster_csv: str | Path | None = None,
    max_subregions: int = 50000,
    knn: int = 6,
    n_permutations: int = 20,
    random_state: int = 0,
) -> dict[str, object]:
    """Summarize whether pooled subregion clusters behave like spatial niches."""

    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    subregion_path = run_path / "subregions_multilevel_ot.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json under run directory: {run_path}")
    if not subregion_path.exists():
        raise FileNotFoundError(f"Missing subregion table: {subregion_path}")

    summary = json.loads(summary_path.read_text())
    table = pd.read_parquet(subregion_path)
    label_col = "cluster_int" if "cluster_int" in table.columns else "argmin_cluster_int"
    if label_col not in table.columns:
        raise ValueError("subregion table must contain cluster_int or argmin_cluster_int")
    sample_col = "sample_id" if "sample_id" in table.columns else None
    sampled = _subsample_table(table, max_subregions=int(max_subregions), random_state=int(random_state))
    cluster_rows, cluster_csv = _cluster_stats(table, label_col=label_col, sample_col=sample_col)
    heavily_shrunk_clusters = []
    for row in cluster_rows:
        alpha = row.get("subregion_latent_shrinkage_alpha")
        median = alpha.get("median") if isinstance(alpha, dict) else None
        try:
            median_value = float(median)
        except (TypeError, ValueError):
            continue
        if np.isfinite(median_value) and median_value < 0.20:
            heavily_shrunk_clusters.append(
                {
                    "cluster": row["cluster"],
                    "median_shrinkage_alpha": median_value,
                    "n_subregions": row["n_subregions"],
                    "dominant_sample_fraction": row["dominant_sample_fraction"],
                }
            )
    homophily = _knn_label_homophily(
        sampled,
        label_col=label_col,
        sample_col=sample_col,
        k=int(knn),
        n_permutations=int(n_permutations),
        random_state=int(random_state),
    )
    sample_counts = table[sample_col].astype(str).value_counts().sort_index() if sample_col else pd.Series(dtype=np.int64)
    report: dict[str, object] = {
        "run_dir": str(run_path),
        "summary_json": str(summary_path),
        "subregions_parquet": str(subregion_path),
        "n_cells": summary.get("n_cells"),
        "n_subregions": int(len(table)),
        "n_clusters": int(table[label_col].nunique()),
        "label_column": label_col,
        "sample_column": sample_col,
        "sample_counts": {str(k): int(v) for k, v in sample_counts.items()},
        "subsampled_for_spatial_graph": int(len(sampled)),
        "max_subregions_for_spatial_graph": int(max_subregions),
        "cluster_statistics": cluster_rows,
        "heavily_shrunk_cluster_alpha_threshold": 0.20,
        "heavily_shrunk_clusters": heavily_shrunk_clusters,
        "spatial_adjacency_homophily": homophily,
        "interpretation": {
            "role": "spatial_organization_qc_for_pooled_feature_distribution_clusters",
            "caveat": (
                "High homophily supports spatial organization but does not by itself validate biological niche identity; "
                "combine this report with leakage, fixed-K, boundary, shrinkage, and held-out-sample checks."
            ),
        },
    }

    json_path = Path(output_json) if output_json is not None else run_path / "spatial_niche_validation.json"
    csv_path = Path(output_cluster_csv) if output_cluster_csv is not None else run_path / "spatial_niche_cluster_statistics.csv"
    json_path.write_text(json.dumps(report, indent=2))
    cluster_csv.to_csv(csv_path, index=False)
    report["outputs"] = {"json": str(json_path), "cluster_csv": str(csv_path)}
    json_path.write_text(json.dumps(report, indent=2))
    return report
