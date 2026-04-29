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


def _cluster_stats(
    table: pd.DataFrame, label_col: str, sample_col: str | None
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    for label, group in table.groupby(label_col, sort=True):
        sample_counts = (
            group[sample_col].astype(str).value_counts().sort_index()
            if sample_col is not None and sample_col in group
            else pd.Series({"cohort": int(len(group))}, dtype=np.int64)
        )
        _, normalized_entropy = _entropy_from_counts(sample_counts.to_numpy())
        dominant_fraction = float(
            sample_counts.max() / max(int(sample_counts.sum()), 1)
        )
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


def _subsample_table(
    table: pd.DataFrame, *, max_subregions: int, random_state: int
) -> pd.DataFrame:
    if max_subregions <= 0 or len(table) <= max_subregions:
        return table
    return table.sample(
        n=int(max_subregions), random_state=int(random_state)
    ).sort_index()


def _resolve_sample_column(
    table: pd.DataFrame,
    summary: dict[str, object],
    *,
    sample_obs_key: str | None,
    allow_missing_sample_key: bool,
) -> str | None:
    requested = (
        str(sample_obs_key).strip()
        if sample_obs_key is not None and str(sample_obs_key).strip()
        else ""
    )
    if not requested:
        summary_key = summary.get("sample_obs_key")
        requested = (
            str(summary_key).strip()
            if summary_key is not None and str(summary_key).strip()
            else "sample_id"
        )
    if requested in table.columns:
        return requested
    if allow_missing_sample_key:
        return None
    raise ValueError(
        f"Requested sample obs key {requested!r} is not present in subregion table columns. "
        "Pass --sample-obs-key with the correct column or --allow-missing-sample-key for cohort-wide validation."
    )


def _permutation_summary(
    observed: float | None, values: list[float]
) -> dict[str, float | int | None]:
    if observed is None or not values:
        return {
            "permutation_mean": None,
            "permutation_std": None,
            "permutation_p95": None,
            "permutation_p_value_greater": None,
            "permutation_z_score": None,
            "excess_over_permutation_mean": None,
            "n_permutations": int(len(values)),
        }
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "permutation_mean": mean,
        "permutation_std": std,
        "permutation_p95": float(np.quantile(arr, 0.95)),
        "permutation_p_value_greater": float(
            (1 + np.sum(arr >= float(observed))) / (arr.size + 1)
        ),
        "permutation_z_score": float((float(observed) - mean) / std)
        if std > 0
        else None,
        "excess_over_permutation_mean": float(float(observed) - mean),
        "n_permutations": int(arr.size),
    }


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
    groups = (
        table.groupby(group_key, sort=True)
        if group_key is not None
        else [("cohort", table)]
    )
    records: list[dict[str, object]] = []
    per_sample: list[dict[str, object]] = []
    same_total = 0
    edge_total = 0
    per_cluster_edges: dict[int, int] = {}
    per_cluster_same: dict[int, int] = {}

    for sample_id, group in groups:
        coords = group[["center_x_um", "center_y_um"]].to_numpy(
            dtype=np.float64, copy=False
        )
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
                    **_permutation_summary(None, []),
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
        for label in np.unique(labels):
            label_int = int(label)
            mask = labels == label
            label_edges = int(np.sum(mask) * local_k)
            label_same = int(np.sum(labels[mask, None] == labels[neigh[mask]]))
            per_cluster_edges[label_int] = (
                per_cluster_edges.get(label_int, 0) + label_edges
            )
            per_cluster_same[label_int] = (
                per_cluster_same.get(label_int, 0) + label_same
            )
        sample_row_index = len(per_sample)
        records.append(
            {
                "sample_id": str(sample_id),
                "labels": labels,
                "neigh": neigh,
                "local_k": local_k,
                "per_sample_index": sample_row_index,
            }
        )
        per_sample.append(
            {
                "sample_id": str(sample_id),
                "n_subregions": n,
                "k": local_k,
                "homophily": homophily,
            }
        )

    observed = float(same_total / max(edge_total, 1)) if edge_total else None
    global_permuted: list[float] = []
    per_sample_permuted: list[list[float]] = [[] for _ in per_sample]
    per_cluster_permuted: dict[int, list[float]] = {
        label: [] for label in per_cluster_edges
    }
    for _ in range(max(int(n_permutations), 0)):
        perm_same_total = 0
        perm_edge_total = 0
        perm_cluster_same: dict[int, int] = {label: 0 for label in per_cluster_edges}
        perm_cluster_edges: dict[int, int] = {label: 0 for label in per_cluster_edges}
        for record in records:
            labels = np.asarray(record["labels"], dtype=np.int64)
            neigh = np.asarray(record["neigh"], dtype=np.int64)
            local_k = int(record["local_k"])
            per_sample_index = int(record["per_sample_index"])
            perm_labels = rng.permutation(labels)
            perm_same = int(np.sum(perm_labels[:, None] == perm_labels[neigh]))
            perm_edges = int(neigh.size)
            perm_same_total += perm_same
            perm_edge_total += perm_edges
            per_sample_permuted[per_sample_index].append(
                float(perm_same / max(perm_edges, 1))
            )
            for label in np.unique(perm_labels):
                label_int = int(label)
                mask = perm_labels == label
                label_edges = int(np.sum(mask) * local_k)
                label_same = int(
                    np.sum(perm_labels[mask, None] == perm_labels[neigh[mask]])
                )
                perm_cluster_edges[label_int] = (
                    perm_cluster_edges.get(label_int, 0) + label_edges
                )
                perm_cluster_same[label_int] = (
                    perm_cluster_same.get(label_int, 0) + label_same
                )
        if perm_edge_total:
            global_permuted.append(float(perm_same_total / perm_edge_total))
        for label, label_edges in perm_cluster_edges.items():
            if label_edges > 0:
                per_cluster_permuted.setdefault(int(label), []).append(
                    float(perm_cluster_same[int(label)] / label_edges)
                )
    for index, values in enumerate(per_sample_permuted):
        per_sample[index].update(
            _permutation_summary(per_sample[index].get("homophily"), values)
        )
    per_cluster: list[dict[str, object]] = []
    for label in sorted(per_cluster_edges):
        label_observed = float(
            per_cluster_same[label] / max(per_cluster_edges[label], 1)
        )
        row: dict[str, object] = {
            "cluster": int(label),
            "n_neighbor_edges": int(per_cluster_edges[label]),
            "observed": label_observed,
        }
        row.update(
            _permutation_summary(label_observed, per_cluster_permuted.get(label, []))
        )
        per_cluster.append(row)
    perm_summary = _permutation_summary(observed, global_permuted)
    return {
        "available": observed is not None,
        "k": int(k),
        "n_neighbor_edges": int(edge_total),
        "observed": observed,
        **perm_summary,
        "strict_mode_recommended_min_permutations": 999,
        "strict_mode_permutation_count_ok": bool(int(n_permutations) >= 999),
        "per_cluster": per_cluster,
        "per_sample": per_sample,
    }


def _spatial_fragmentation(
    table: pd.DataFrame,
    *,
    label_col: str,
    sample_col: str | None,
    k: int,
) -> dict[str, object]:
    from sklearn.neighbors import NearestNeighbors

    required = {"center_x_um", "center_y_um", label_col}
    if not required.issubset(table.columns):
        return {
            "available": False,
            "reason": "missing_center_or_label_columns",
            "required_columns": sorted(required),
        }

    group_key = sample_col if sample_col is not None and sample_col in table else None
    groups = (
        table.groupby(group_key, sort=True)
        if group_key is not None
        else [("cohort", table)]
    )
    per_sample_cluster: list[dict[str, object]] = []
    aggregate: dict[int, dict[str, int]] = {}
    for sample_id, group in groups:
        coords = group[["center_x_um", "center_y_um"]].to_numpy(
            dtype=np.float64, copy=False
        )
        labels = group[label_col].to_numpy(dtype=np.int64, copy=False)
        finite = np.isfinite(coords).all(axis=1)
        coords = coords[finite]
        labels = labels[finite]
        n = int(labels.size)
        if n < 2:
            continue
        local_k = min(max(int(k), 1), n - 1)
        nn = NearestNeighbors(n_neighbors=local_k + 1)
        nn.fit(coords)
        neigh = nn.kneighbors(coords, return_distance=False)[:, 1:]
        adjacency = [set() for _ in range(n)]
        for i in range(n):
            for j in neigh[i]:
                adjacency[i].add(int(j))
                adjacency[int(j)].add(i)
        for label in np.unique(labels):
            label_int = int(label)
            nodes = set(np.flatnonzero(labels == label).tolist())
            unseen = set(nodes)
            components = 0
            while unseen:
                components += 1
                stack = [unseen.pop()]
                while stack:
                    current = stack.pop()
                    for neighbor in adjacency[current]:
                        if neighbor in unseen and int(labels[neighbor]) == label_int:
                            unseen.remove(neighbor)
                            stack.append(neighbor)
            n_nodes = int(len(nodes))
            fragmentation_index = float(components / max(n_nodes, 1))
            per_sample_cluster.append(
                {
                    "sample_id": str(sample_id),
                    "cluster": label_int,
                    "n_subregions": n_nodes,
                    "connected_components": int(components),
                    "fragmentation_index": fragmentation_index,
                }
            )
            agg = aggregate.setdefault(
                label_int,
                {"n_subregions": 0, "connected_components": 0, "occupied_samples": 0},
            )
            agg["n_subregions"] += n_nodes
            agg["connected_components"] += int(components)
            agg["occupied_samples"] += 1

    per_cluster = [
        {
            "cluster": int(label),
            "n_subregions": int(values["n_subregions"]),
            "connected_components": int(values["connected_components"]),
            "occupied_samples": int(values["occupied_samples"]),
            "components_per_100_subregions": float(
                100.0 * values["connected_components"] / max(values["n_subregions"], 1)
            ),
            "fragmentation_index": float(
                values["connected_components"] / max(values["n_subregions"], 1)
            ),
        }
        for label, values in sorted(aggregate.items())
    ]
    return {
        "available": bool(per_cluster),
        "k": int(k),
        "per_cluster": per_cluster,
        "per_sample_cluster": per_sample_cluster,
        "interpretation": "Higher component counts or fragmentation indices indicate spatially scattered cluster assignments.",
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
    sample_obs_key: str | None = None,
    allow_missing_sample_key: bool = False,
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
    label_col = (
        "cluster_int" if "cluster_int" in table.columns else "argmin_cluster_int"
    )
    if label_col not in table.columns:
        raise ValueError(
            "subregion table must contain cluster_int or argmin_cluster_int"
        )
    sample_col = _resolve_sample_column(
        table,
        summary,
        sample_obs_key=sample_obs_key,
        allow_missing_sample_key=allow_missing_sample_key,
    )
    sampled = _subsample_table(
        table, max_subregions=int(max_subregions), random_state=int(random_state)
    )
    cluster_rows, cluster_csv = _cluster_stats(
        table, label_col=label_col, sample_col=sample_col
    )
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
    fragmentation = _spatial_fragmentation(
        sampled,
        label_col=label_col,
        sample_col=sample_col,
        k=int(knn),
    )
    sample_counts = (
        table[sample_col].astype(str).value_counts().sort_index()
        if sample_col
        else pd.Series(dtype=np.int64)
    )
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
        "spatial_fragmentation": fragmentation,
        "interpretation": {
            "role": "spatial_organization_qc_for_pooled_feature_distribution_clusters",
            "caveat": (
                "High homophily supports spatial organization but does not by itself validate biological niche identity; "
                "combine this report with leakage, fixed-K, boundary, shrinkage, and held-out-sample checks."
            ),
        },
    }

    json_path = (
        Path(output_json)
        if output_json is not None
        else run_path / "spatial_niche_validation.json"
    )
    csv_path = (
        Path(output_cluster_csv)
        if output_cluster_csv is not None
        else run_path / "spatial_niche_cluster_statistics.csv"
    )
    json_path.write_text(json.dumps(report, indent=2))
    cluster_csv.to_csv(csv_path, index=False)
    report["outputs"] = {"json": str(json_path), "cluster_csv": str(csv_path)}
    json_path.write_text(json.dumps(report, indent=2))
    return report
