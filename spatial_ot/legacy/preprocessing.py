from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import torch

from ..config import ExperimentConfig
from .programs import ProgramLibrary, build_program_library, collect_program_genes, load_programs, score_programs, summarize_program_coverage


@dataclass
class PreparedSpatialOTData:
    cell_adata: ad.AnnData
    bin_adata: ad.AnnData
    gene_names: list[str]
    cell_ids: np.ndarray
    bin_ids: np.ndarray
    cell_counts: np.ndarray
    bin_counts: np.ndarray
    cell_log_counts: np.ndarray
    bin_log_counts: np.ndarray
    cell_library_panel: np.ndarray
    bin_library_panel: np.ndarray
    cell_library_full: np.ndarray
    bin_library_full: np.ndarray
    cell_coords: np.ndarray
    bin_coords: np.ndarray
    microns_per_pixel: float
    cell_aux: np.ndarray
    batch_onehot: np.ndarray
    cell_type_labels: np.ndarray
    cell_type_names: list[str]
    cell_type_onehot: np.ndarray
    program_library: ProgramLibrary
    program_scores: np.ndarray
    teacher_overlap: np.ndarray
    teacher_edge_index: np.ndarray
    context_edge_index: np.ndarray
    shell_edge_indices: tuple[np.ndarray, ...]
    ot_neighbor_index: np.ndarray
    bin_neighbor_target: np.ndarray
    cell_neighbor_target: np.ndarray
    cell_count_source: str
    bin_count_source: str
    gene_panel_report: dict
    teacher_overlap_report: dict
    graph_report: dict

    @property
    def n_cells(self) -> int:
        return self.cell_counts.shape[0]

    @property
    def n_bins(self) -> int:
        return self.bin_counts.shape[0]

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _counts_matrix(adata: ad.AnnData):
    if "counts" in adata.layers:
        return adata.layers["counts"], "layers['counts']"
    return adata.X, "X"


def _dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _row_sums_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        sums = np.asarray(matrix.sum(axis=1)).ravel()
    else:
        sums = np.asarray(matrix.sum(axis=1)).ravel()
    return sums.astype(np.float32).reshape(-1, 1)


def _validate_raw_counts(matrix, *, source_name: str, max_check_values: int = 250000) -> None:
    values = np.asarray(matrix.data if sparse.issparse(matrix) else matrix).ravel()
    if values.size == 0:
        return
    if values.size > max_check_values:
        idx = np.linspace(0, values.size - 1, num=max_check_values, dtype=np.int64)
        values = values[idx]
    values = values.astype(np.float64, copy=False)
    if np.min(values) < -1e-8:
        raise ValueError(f"{source_name} contains negative values and does not look like a raw count matrix.")
    integer_deviation = float(np.max(np.abs(values - np.rint(values))))
    if integer_deviation > 1e-4:
        raise ValueError(
            f"{source_name} is not integer-like (max fractional deviation {integer_deviation:.3g}); "
            "please provide raw counts rather than normalized/transformed values."
        )


def _resolve_raw_counts(adata: ad.AnnData, *, source_name: str):
    matrix, source = _counts_matrix(adata)
    _validate_raw_counts(matrix, source_name=f"{source_name}:{source}")
    return matrix, source


def _make_hvg_ranking(adata: ad.AnnData, n_top_genes: int) -> list[str]:
    counts_matrix, _ = _counts_matrix(adata)
    tmp = ad.AnnData(X=counts_matrix.copy(), obs=adata.obs.copy(), var=adata.var.copy())
    try:
        sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_top_genes, tmp.n_vars), flavor="cell_ranger")
        ranked = tmp.var[tmp.var["highly_variable"]].index.tolist()
        if ranked:
            return ranked
    except ValueError:
        pass
    means = np.asarray(counts_matrix.mean(axis=0)).ravel()
    order = np.argsort(-means)
    return adata.var_names[order[: min(n_top_genes, adata.n_vars)]].tolist()


def _select_gene_panel(
    cell_adata: ad.AnnData,
    bin_adata: ad.AnnData,
    program_genes: set[str],
    hvg_n: int,
    cap: int,
) -> tuple[list[str], dict]:
    shared = set(cell_adata.var_names).intersection(bin_adata.var_names)
    ranked_hvgs = [gene for gene in _make_hvg_ranking(cell_adata, hvg_n) if gene in shared]
    ranked_means = []
    cell_counts_matrix, _ = _counts_matrix(cell_adata)
    means = np.asarray(cell_counts_matrix.mean(axis=0)).ravel()
    for gene in cell_adata.var_names[np.argsort(-means)].tolist():
        if gene in shared:
            ranked_means.append(gene)

    chosen: list[str] = []
    seen: set[str] = set()
    hvg_kept: list[str] = []
    for gene in ranked_hvgs:
        if gene not in seen:
            chosen.append(gene)
            seen.add(gene)
            hvg_kept.append(gene)
        if cap > 0 and len(chosen) >= cap:
            break
    program_shared = sorted(gene for gene in program_genes if gene in shared)
    program_added: list[str] = []
    for gene in sorted(program_genes):
        if gene in shared and gene not in seen:
            chosen.append(gene)
            seen.add(gene)
            program_added.append(gene)
    for gene in ranked_means:
        if cap > 0 and len(chosen) >= cap:
            break
        if gene not in seen:
            chosen.append(gene)
            seen.add(gene)
    report = {
        "shared_gene_count": len(shared),
        "requested_hvg_count": int(hvg_n),
        "encoder_gene_cap": int(cap),
        "selected_gene_count": len(chosen),
        "cap_reached_during_hvg_stage": bool(cap > 0 and len(hvg_kept) >= cap),
        "cap_overridden_by_program_genes": bool(cap > 0 and len(chosen) > cap),
        "hvg_kept_count": len(hvg_kept),
        "program_gene_count_requested": len(program_genes),
        "program_gene_count_shared": len(program_shared),
        "program_gene_count_added": len(program_added),
        "program_genes_added": program_added,
        "program_genes_missing_from_data": sorted(program_genes - shared),
    }
    return chosen, report


def _stratified_subset(labels: pd.Series, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(labels) <= limit:
        return np.arange(len(labels), dtype=int)
    rng = np.random.default_rng(seed)
    labels = labels.astype(str).replace({"nan": "unknown", "None": "unknown"})
    picked = []
    for label in sorted(labels.unique()):
        idx = np.flatnonzero(labels.to_numpy() == label)
        keep = max(1, int(round(limit * len(idx) / len(labels))))
        keep = min(keep, len(idx))
        picked.extend(rng.choice(idx, size=keep, replace=False).tolist())
    picked = np.array(sorted(set(picked)), dtype=int)
    if picked.size > limit:
        picked = np.sort(rng.choice(picked, size=limit, replace=False))
    return picked


def _spatial_grid_subset(coords: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(coords) <= limit:
        return np.arange(len(coords), dtype=int)
    rng = np.random.default_rng(seed)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    target_patch_size = int(np.clip(round(np.sqrt(limit) * 4), 64, 256))
    n_patches = max(1, int(math.ceil(limit / target_patch_size)))
    grid_n = max(3, int(np.ceil(np.sqrt(n_patches * 2))))
    scaled = (coords - mins) / span
    x_bin = np.clip((scaled[:, 0] * grid_n).astype(int), 0, grid_n - 1)
    y_bin = np.clip((scaled[:, 1] * grid_n).astype(int), 0, grid_n - 1)
    tile_id = x_bin * grid_n + y_bin
    tile_to_idx = {tile: np.flatnonzero(tile_id == tile) for tile in np.unique(tile_id)}
    nonempty_tiles = sorted(tile_to_idx)
    n_patches = min(n_patches, len(nonempty_tiles))
    tile_weights = np.asarray([tile_to_idx[tile].size for tile in nonempty_tiles], dtype=np.float64)
    tile_weights = tile_weights / tile_weights.sum()
    chosen_tiles = rng.choice(np.asarray(nonempty_tiles, dtype=int), size=n_patches, replace=False, p=tile_weights)

    patch_base = limit // n_patches
    patch_remainder = limit % n_patches
    max_patch_budget = patch_base + (1 if patch_remainder > 0 else 0)
    nbrs = NearestNeighbors(n_neighbors=min(max_patch_budget, len(coords)))
    nbrs.fit(coords)
    selected: set[int] = set()
    for patch_idx, tile in enumerate(chosen_tiles.tolist()):
        tile_members = tile_to_idx[int(tile)]
        tile_center = coords[tile_members].mean(axis=0, keepdims=True)
        seed_local = int(tile_members[np.argmin(((coords[tile_members] - tile_center) ** 2).sum(axis=1))])
        patch_budget = patch_base + (1 if patch_idx < patch_remainder else 0)
        nearest = nbrs.kneighbors(coords[[seed_local]], n_neighbors=min(patch_budget, len(coords)), return_distance=False).ravel()
        selected.update(int(idx) for idx in nearest.tolist())

    if len(selected) < limit:
        remaining = np.setdiff1d(np.arange(len(coords), dtype=int), np.asarray(sorted(selected), dtype=int), assume_unique=False)
        if remaining.size > 0:
            extra = rng.choice(remaining, size=min(limit - len(selected), remaining.size), replace=False)
            selected.update(int(idx) for idx in np.atleast_1d(extra).tolist())

    if len(selected) > limit:
        chosen = np.asarray(sorted(selected), dtype=int)
        chosen = rng.choice(chosen, size=limit, replace=False)
        return np.asarray(sorted(chosen.tolist()), dtype=int)

    return np.asarray(sorted(selected), dtype=int)


def _read_positions(path: str | Path) -> pd.DataFrame:
    positions = pd.read_parquet(path)
    positions = positions.set_index("barcode")
    return positions


def _subset_bin_ids(
    all_bin_ids: pd.Index,
    positions: pd.DataFrame,
    mapping: pd.DataFrame | None,
    limit: int,
    seed: int,
) -> list[str]:
    candidate = list(all_bin_ids.intersection(positions.index))
    if mapping is not None and not mapping.empty:
        candidate = [barcode for barcode in mapping["square_008um"].astype(str).drop_duplicates().tolist() if barcode in candidate]
    if limit <= 0 or len(candidate) <= limit:
        return candidate
    if mapping is not None and not mapping.empty:
        sub = mapping[mapping["square_008um"].isin(candidate)].copy()
        bin_to_cells = sub.groupby("square_008um")["cell_id"].agg(list).to_dict()
        bin_counts = {barcode: len(set(cells)) for barcode, cells in bin_to_cells.items()}
        selected: list[str] = []
        remaining = set(bin_to_cells)
        uncovered = set(sub["cell_id"].astype(str).unique().tolist())
        while remaining and len(selected) < limit and uncovered:
            best_bin = None
            best_score = None
            for barcode in remaining:
                cells = set(str(cell_id) for cell_id in bin_to_cells[barcode])
                newly_covered = len(cells & uncovered)
                score = (newly_covered, bin_counts[barcode], barcode)
                if best_score is None or score > best_score:
                    best_bin = barcode
                    best_score = score
            if best_bin is None or best_score is None or best_score[0] <= 0:
                break
            selected.append(best_bin)
            uncovered -= set(str(cell_id) for cell_id in bin_to_cells[best_bin])
            remaining.remove(best_bin)
        if len(selected) >= limit or not remaining:
            return selected[:limit]
        leftover = sorted(remaining)
        pos = positions.loc[leftover]
        coords = np.stack(
            [
                np.asarray(pos["pxl_col_in_fullres"], dtype=np.float32),
                np.asarray(pos["pxl_row_in_fullres"], dtype=np.float32),
            ],
            axis=1,
        )
        keep_idx = _spatial_grid_subset(coords, limit=limit - len(selected), seed=seed)
        selected.extend(leftover[i] for i in keep_idx.tolist())
        return selected[:limit]
    pos = positions.loc[candidate]
    coords = np.stack(
        [
            np.asarray(pos["pxl_col_in_fullres"], dtype=np.float32),
            np.asarray(pos["pxl_row_in_fullres"], dtype=np.float32),
        ],
        axis=1,
    )
    keep_idx = _spatial_grid_subset(coords, limit=limit, seed=seed)
    return [candidate[i] for i in keep_idx.tolist()]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    sums = matrix.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return matrix / sums


def _build_teacher_overlap(
    mapping_path: str | None,
    cell_ids: np.ndarray,
    bin_ids: np.ndarray,
    cell_coords: np.ndarray,
    bin_coords: np.ndarray,
    allow_nearest_fallback: bool,
) -> tuple[np.ndarray, dict]:
    overlap = np.zeros((len(cell_ids), len(bin_ids)), dtype=np.float32)
    retained_mapping_rows = 0
    if mapping_path:
        mapping = pd.read_parquet(mapping_path, columns=["cell_id", "square_008um", "in_cell"])
        mapping["cell_id"] = mapping["cell_id"].astype(str)
        mapping["square_008um"] = mapping["square_008um"].astype(str)
        mapping = mapping[mapping["in_cell"]]
        mapping = mapping[mapping["cell_id"].isin(cell_ids) & mapping["square_008um"].isin(bin_ids)]
        if not mapping.empty:
            grouped = mapping.groupby(["cell_id", "square_008um"]).size().reset_index(name="count")
            retained_mapping_rows = int(len(grouped))
            cell_index = {cell_id: i for i, cell_id in enumerate(cell_ids)}
            bin_index = {bin_id: i for i, bin_id in enumerate(bin_ids)}
            for row in grouped.itertuples(index=False):
                overlap[cell_index[row.cell_id], bin_index[row.square_008um]] = float(row.count)
    zero_rows = np.flatnonzero(overlap.sum(axis=1) == 0)
    mapped_before_fallback = int(len(cell_ids) - zero_rows.size)
    if zero_rows.size:
        if not allow_nearest_fallback:
            raise ValueError(
                f"{zero_rows.size} cells had no 8 µm overlap mapping and nearest-bin fallback is disabled."
            )
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(bin_coords)
        nearest = nbrs.kneighbors(cell_coords[zero_rows], return_distance=False).ravel()
        overlap[zero_rows, nearest] = 1.0
    report = {
        "n_cells": int(len(cell_ids)),
        "n_bins": int(len(bin_ids)),
        "mapping_path_used": bool(mapping_path),
        "retained_mapping_rows": retained_mapping_rows,
        "mapped_cells_before_fallback": mapped_before_fallback,
        "unmapped_cells_before_fallback": int(zero_rows.size),
        "mapped_fraction_before_fallback": float(mapped_before_fallback / max(len(cell_ids), 1)),
        "nearest_fallback_rows": int(zero_rows.size),
        "mean_nonzero_bins_per_cell_after_fallback": float(np.mean((overlap > 0).sum(axis=1))),
    }
    return _normalize_rows(overlap), report


def _build_knn_edges(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords)))
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    src = np.repeat(np.arange(len(coords)), indices.shape[1] - 1)
    dst = indices[:, 1:].reshape(-1)
    dists = distances[:, 1:].reshape(-1)
    return src.astype(np.int64), dst.astype(np.int64), dists.astype(np.float32)


def _build_cell_graphs(coords: np.ndarray, context_k: int, ot_k: int, shell_bounds: tuple[float, ...]):
    max_k = max(context_k, ot_k)
    nbrs = NearestNeighbors(n_neighbors=min(max_k + 1, len(coords)))
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    context_idx = indices[:, 1 : min(context_k + 1, indices.shape[1])]
    context_dist = distances[:, 1 : min(context_k + 1, distances.shape[1])]
    src = np.repeat(np.arange(len(coords)), context_idx.shape[1])
    dst = context_idx.reshape(-1)
    context_edges = np.vstack([src, dst]).astype(np.int64)
    shells = []
    lower = 0.0
    for upper in shell_bounds:
        mask = (context_dist > lower) & (context_dist <= upper)
        shells.append(np.vstack([np.repeat(np.arange(len(coords)), context_idx.shape[1])[mask.reshape(-1)], context_idx.reshape(-1)[mask.reshape(-1)]]).astype(np.int64))
        lower = upper
    ot_neighbors = indices[:, 1 : min(ot_k + 1, indices.shape[1])].astype(np.int64)
    return context_edges, tuple(shells), ot_neighbors


def _graph_report(
    cell_coords: np.ndarray,
    bin_coords: np.ndarray,
    teacher_edge_index: np.ndarray,
    context_edge_index: np.ndarray,
    shell_edge_indices: tuple[np.ndarray, ...],
    ot_neighbor_index: np.ndarray,
) -> dict:
    ot_neighbor_coords = cell_coords[ot_neighbor_index]
    ot_distances = np.linalg.norm(ot_neighbor_coords - cell_coords[:, None, :], axis=2)
    return {
        "teacher_edge_count": int(teacher_edge_index.shape[1]),
        "context_edge_count": int(context_edge_index.shape[1]),
        "shell_edge_counts": [int(shell.shape[1]) for shell in shell_edge_indices],
        "ot_neighbor_k_effective": int(ot_neighbor_index.shape[1]) if ot_neighbor_index.ndim == 2 else 0,
        "ot_neighbor_distance_um_mean": float(np.mean(ot_distances)) if ot_distances.size else 0.0,
        "ot_neighbor_distance_um_median": float(np.median(ot_distances)) if ot_distances.size else 0.0,
        "ot_neighbor_distance_um_max": float(np.max(ot_distances)) if ot_distances.size else 0.0,
        "n_cells": int(cell_coords.shape[0]),
        "n_bins": int(bin_coords.shape[0]),
    }


def aggregate_mean_numpy(values: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros_like(values)
    src, dst = edge_index
    out = np.zeros_like(values, dtype=np.float32)
    deg = np.zeros((values.shape[0], 1), dtype=np.float32)
    np.add.at(out, src, values[dst])
    np.add.at(deg, src, 1.0)
    deg[deg == 0] = 1.0
    return out / deg


def aggregate_sum_numpy(values: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros_like(values)
    src, dst = edge_index
    out = np.zeros_like(values, dtype=np.float32)
    np.add.at(out, src, values[dst])
    return out


def _extract_cell_aux(adata: ad.AnnData) -> np.ndarray:
    obs = adata.obs
    cols = []
    for col in ["total_counts", "n_genes_by_counts", "umi_count", "final_level_softmax_prob"]:
        if col in obs:
            cols.append(np.asarray(obs[col], dtype=np.float32))
    if {"cell_x", "cell_y", "nucleus_x", "nucleus_y"}.issubset(obs.columns):
        dx = np.asarray(obs["cell_x"] - obs["nucleus_x"], dtype=np.float32)
        dy = np.asarray(obs["cell_y"] - obs["nucleus_y"], dtype=np.float32)
        cols.extend([dx, dy, np.sqrt(dx**2 + dy**2 + 1e-6)])
    if not cols:
        cols.append(np.zeros(adata.n_obs, dtype=np.float32))
    aux = np.vstack(cols).T.astype(np.float32)
    aux = (aux - aux.mean(axis=0, keepdims=True)) / (aux.std(axis=0, keepdims=True) + 1e-6)
    return aux


def _extract_batches(obs: pd.DataFrame) -> np.ndarray:
    if "batch_name" not in obs:
        return np.ones((len(obs), 1), dtype=np.float32)
    batch = pd.Categorical(obs["batch_name"].astype(str))
    eye = np.eye(len(batch.categories), dtype=np.float32)
    return eye[batch.codes]


def _extract_cell_types(obs: pd.DataFrame, key: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    if not key:
        raise ValueError("A non-empty cell_type_key is required for the current scaffold.")
    if key not in obs:
        available = ", ".join(map(str, obs.columns[:20]))
        raise KeyError(f"Requested cell_type_key '{key}' was not found in obs. Available columns include: {available}")
    values = obs[key].astype(str).replace({"nan": "unknown", "None": "unknown"})
    categorical = pd.Categorical(values)
    labels = categorical.codes.astype(np.int64)
    names = list(categorical.categories)
    onehot = np.eye(len(names), dtype=np.float32)[labels]
    return labels, names, onehot


def prepare_data(config: ExperimentConfig) -> PreparedSpatialOTData:
    set_seed(config.data.seed)

    cell_adata = ad.read_h5ad(config.paths.cells_h5ad)
    raw_coords = np.asarray(cell_adata.obsm["spatial"], dtype=np.float32)
    if config.data.subset_strategy == "stratified":
        if config.data.cell_type_key not in cell_adata.obs:
            available = ", ".join(map(str, cell_adata.obs.columns[:20]))
            raise KeyError(
                f"Requested cell_type_key '{config.data.cell_type_key}' was not found in obs for stratified subsetting. "
                f"Available columns include: {available}"
            )
        subset_idx = _stratified_subset(
            cell_adata.obs[config.data.cell_type_key],
            config.data.cell_subset,
            config.data.seed,
        )
    else:
        subset_idx = _spatial_grid_subset(raw_coords, config.data.cell_subset, config.data.seed)
    cell_adata = cell_adata[subset_idx].copy()

    microns_per_pixel = float(
        next(iter(cell_adata.uns["spatial"].values()))["scalefactors"].get("microns_per_pixel", 1.0)
        if "spatial" in cell_adata.uns
        else 1.0
    )

    cell_ids = cell_adata.obs_names.astype(str).to_numpy()

    mapping = None
    if config.paths.barcode_mappings:
        mapping = pd.read_parquet(config.paths.barcode_mappings, columns=["cell_id", "square_008um", "in_cell"])
        mapping = mapping[mapping["in_cell"] & mapping["cell_id"].isin(cell_ids)]

    bin_adata = sc.read_10x_h5(config.paths.bins8_h5, gex_only=False)
    bin_adata.var_names_make_unique()
    positions = _read_positions(config.paths.bins8_positions)
    selected_bin_ids = _subset_bin_ids(bin_adata.obs_names, positions, mapping, config.data.bin_subset, config.data.seed)
    bin_adata = bin_adata[selected_bin_ids].copy()
    bin_ids = bin_adata.obs_names.astype(str).to_numpy()

    programs = load_programs(config.paths.programs_json)
    gene_names, gene_panel_report = _select_gene_panel(
        cell_adata=cell_adata,
        bin_adata=bin_adata,
        program_genes=collect_program_genes(programs),
        hvg_n=config.data.hvg_n,
        cap=config.data.encoder_gene_cap,
    )
    program_library = build_program_library(programs, gene_names)
    gene_panel_report["program_coverage"] = summarize_program_coverage(programs, gene_names)

    cell_matrix_full, cell_count_source = _resolve_raw_counts(cell_adata, source_name="cells_h5ad")
    bin_matrix_full, bin_count_source = _resolve_raw_counts(bin_adata, source_name="bins8_h5")
    cell_library_full = _row_sums_float32(cell_matrix_full) + 1e-4
    bin_library_full = _row_sums_float32(bin_matrix_full) + 1e-4
    cell_gene_idx = cell_adata.var_names.get_indexer(gene_names)
    bin_gene_idx = bin_adata.var_names.get_indexer(gene_names)
    if np.any(cell_gene_idx < 0) or np.any(bin_gene_idx < 0):
        raise ValueError("Selected gene panel could not be indexed in one of the input matrices.")
    cell_counts = _dense_float32(cell_matrix_full[:, cell_gene_idx])
    bin_counts = _dense_float32(bin_matrix_full[:, bin_gene_idx])
    cell_log_counts = np.log1p(cell_counts)
    bin_log_counts = np.log1p(bin_counts)
    cell_library_panel = cell_counts.sum(axis=1, keepdims=True) + 1e-4
    bin_library_panel = bin_counts.sum(axis=1, keepdims=True) + 1e-4

    cell_coords = np.asarray(cell_adata.obsm["spatial"], dtype=np.float32) * microns_per_pixel
    bin_pos = positions.loc[bin_ids]
    bin_coords = np.stack(
        [
            np.asarray(bin_pos["pxl_col_in_fullres"], dtype=np.float32) * microns_per_pixel,
            np.asarray(bin_pos["pxl_row_in_fullres"], dtype=np.float32) * microns_per_pixel,
        ],
        axis=1,
    )

    teacher_overlap, teacher_overlap_report = _build_teacher_overlap(
        mapping_path=config.paths.barcode_mappings,
        cell_ids=cell_ids,
        bin_ids=bin_ids,
        cell_coords=cell_coords,
        bin_coords=bin_coords,
        allow_nearest_fallback=config.data.allow_nearest_overlap_fallback,
    )

    teacher_src, teacher_dst, _ = _build_knn_edges(bin_coords, config.data.teacher_knn)
    teacher_edge_index = np.vstack([teacher_src, teacher_dst]).astype(np.int64)
    context_edge_index, shell_edge_indices, ot_neighbor_index = _build_cell_graphs(
        coords=cell_coords,
        context_k=config.data.context_knn,
        ot_k=config.data.ot_knn,
        shell_bounds=tuple(config.data.shell_bounds_um),
    )

    bin_neighbor_target = aggregate_sum_numpy(bin_counts, teacher_edge_index)
    cell_neighbor_target = aggregate_sum_numpy(cell_counts, context_edge_index)
    cell_aux = _extract_cell_aux(cell_adata)
    batch_onehot = _extract_batches(cell_adata.obs)
    cell_type_labels, cell_type_names, cell_type_onehot = _extract_cell_types(cell_adata.obs, config.data.cell_type_key)
    program_scores = score_programs(cell_log_counts, gene_names, program_library)
    graph_report = _graph_report(
        cell_coords=cell_coords,
        bin_coords=bin_coords,
        teacher_edge_index=teacher_edge_index,
        context_edge_index=context_edge_index,
        shell_edge_indices=shell_edge_indices,
        ot_neighbor_index=ot_neighbor_index,
    )

    return PreparedSpatialOTData(
        cell_adata=cell_adata[:, gene_names].copy(),
        bin_adata=bin_adata[:, gene_names].copy(),
        gene_names=gene_names,
        cell_ids=cell_ids,
        bin_ids=bin_ids,
        cell_counts=cell_counts,
        bin_counts=bin_counts,
        cell_log_counts=cell_log_counts,
        bin_log_counts=bin_log_counts,
        cell_library_panel=cell_library_panel.astype(np.float32),
        bin_library_panel=bin_library_panel.astype(np.float32),
        cell_library_full=cell_library_full.astype(np.float32),
        bin_library_full=bin_library_full.astype(np.float32),
        cell_coords=cell_coords.astype(np.float32),
        bin_coords=bin_coords.astype(np.float32),
        microns_per_pixel=microns_per_pixel,
        cell_aux=cell_aux.astype(np.float32),
        batch_onehot=batch_onehot.astype(np.float32),
        cell_type_labels=cell_type_labels.astype(np.int64),
        cell_type_names=cell_type_names,
        cell_type_onehot=cell_type_onehot.astype(np.float32),
        program_library=program_library,
        program_scores=program_scores.astype(np.float32),
        teacher_overlap=teacher_overlap.astype(np.float32),
        teacher_edge_index=teacher_edge_index,
        context_edge_index=context_edge_index,
        shell_edge_indices=shell_edge_indices,
        ot_neighbor_index=ot_neighbor_index,
        bin_neighbor_target=bin_neighbor_target.astype(np.float32),
        cell_neighbor_target=cell_neighbor_target.astype(np.float32),
        cell_count_source=cell_count_source,
        bin_count_source=bin_count_source,
        gene_panel_report=gene_panel_report,
        teacher_overlap_report=teacher_overlap_report,
        graph_report=graph_report,
    )
