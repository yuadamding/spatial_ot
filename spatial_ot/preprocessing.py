from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from .config import ExperimentConfig
from .programs import ProgramLibrary, build_program_library, collect_program_genes, load_programs, score_programs


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
    cell_library: np.ndarray
    bin_library: np.ndarray
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


def _counts_matrix(adata: ad.AnnData):
    if "counts" in adata.layers:
        return adata.layers["counts"]
    return adata.X


def _dense_float32(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _make_hvg_ranking(adata: ad.AnnData, n_top_genes: int) -> list[str]:
    tmp = ad.AnnData(X=_counts_matrix(adata).copy(), obs=adata.obs.copy(), var=adata.var.copy())
    try:
        sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_top_genes, tmp.n_vars), flavor="cell_ranger")
        ranked = tmp.var[tmp.var["highly_variable"]].index.tolist()
        if ranked:
            return ranked
    except ValueError:
        pass
    means = np.asarray(_counts_matrix(adata).mean(axis=0)).ravel()
    order = np.argsort(-means)
    return adata.var_names[order[: min(n_top_genes, adata.n_vars)]].tolist()


def _select_gene_panel(
    cell_adata: ad.AnnData,
    bin_adata: ad.AnnData,
    program_genes: set[str],
    hvg_n: int,
    cap: int,
) -> list[str]:
    shared = set(cell_adata.var_names).intersection(bin_adata.var_names)
    ranked_hvgs = [gene for gene in _make_hvg_ranking(cell_adata, hvg_n) if gene in shared]
    ranked_means = []
    means = np.asarray(_counts_matrix(cell_adata).mean(axis=0)).ravel()
    for gene in cell_adata.var_names[np.argsort(-means)].tolist():
        if gene in shared:
            ranked_means.append(gene)

    chosen: list[str] = []
    seen: set[str] = set()
    for gene in ranked_hvgs:
        if gene not in seen:
            chosen.append(gene)
            seen.add(gene)
        if cap > 0 and len(chosen) >= cap:
            break
    for gene in sorted(program_genes):
        if gene in shared and gene not in seen:
            chosen.append(gene)
            seen.add(gene)
    for gene in ranked_means:
        if cap > 0 and len(chosen) >= cap:
            break
        if gene not in seen:
            chosen.append(gene)
            seen.add(gene)
    return chosen


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


def _spatial_knn_subset(coords: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or len(coords) <= limit:
        return np.arange(len(coords), dtype=int)
    center = np.median(coords, axis=0, keepdims=True)
    dist = np.linalg.norm(coords - center, axis=1)
    return np.argsort(dist)[:limit].astype(int)


def _read_positions(path: str | Path) -> pd.DataFrame:
    positions = pd.read_parquet(path)
    positions = positions.set_index("barcode")
    return positions


def _subset_bin_ids(
    all_bin_ids: pd.Index,
    positions: pd.DataFrame,
    mapping_counts: pd.Series | None,
    limit: int,
    seed: int,
) -> list[str]:
    candidate = list(all_bin_ids.intersection(positions.index))
    if mapping_counts is not None:
        candidate = [barcode for barcode in mapping_counts.index.tolist() if barcode in candidate]
    if limit <= 0 or len(candidate) <= limit:
        return candidate
    if mapping_counts is not None:
        return [barcode for barcode in mapping_counts.index.tolist() if barcode in candidate][:limit]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(candidate, dtype=object), size=limit, replace=False)
    return chosen.tolist()


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
) -> np.ndarray:
    overlap = np.zeros((len(cell_ids), len(bin_ids)), dtype=np.float32)
    if mapping_path:
        mapping = pd.read_parquet(mapping_path, columns=["cell_id", "square_008um", "in_cell"])
        mapping = mapping[mapping["in_cell"]]
        mapping = mapping[mapping["cell_id"].isin(cell_ids) & mapping["square_008um"].isin(bin_ids)]
        if not mapping.empty:
            grouped = mapping.groupby(["cell_id", "square_008um"]).size().reset_index(name="count")
            cell_index = {cell_id: i for i, cell_id in enumerate(cell_ids)}
            bin_index = {bin_id: i for i, bin_id in enumerate(bin_ids)}
            for row in grouped.itertuples(index=False):
                overlap[cell_index[row.cell_id], bin_index[row.square_008um]] = float(row.count)
    zero_rows = np.flatnonzero(overlap.sum(axis=1) == 0)
    if zero_rows.size:
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(bin_coords)
        nearest = nbrs.kneighbors(cell_coords[zero_rows], return_distance=False).ravel()
        overlap[zero_rows, nearest] = 1.0
    return _normalize_rows(overlap)


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


def aggregate_mean_numpy(values: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros_like(values)
    src, dst = edge_index
    out = np.zeros_like(values, dtype=np.float32)
    deg = np.zeros((values.shape[0], 1), dtype=np.float32)
    np.add.at(out, dst, values[src])
    np.add.at(deg, dst, 1.0)
    deg[deg == 0] = 1.0
    return out / deg


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
    if key not in obs:
        key = "cell_type" if "cell_type" in obs else obs.columns[0]
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
        subset_idx = _stratified_subset(
            cell_adata.obs.get(config.data.cell_type_key, cell_adata.obs.iloc[:, 0]),
            config.data.cell_subset,
            config.data.seed,
        )
    else:
        subset_idx = _spatial_knn_subset(raw_coords, config.data.cell_subset)
    cell_adata = cell_adata[subset_idx].copy()

    microns_per_pixel = float(
        next(iter(cell_adata.uns["spatial"].values()))["scalefactors"].get("microns_per_pixel", 1.0)
        if "spatial" in cell_adata.uns
        else 1.0
    )

    cell_ids = cell_adata.obs_names.astype(str).to_numpy()

    mapping_counts = None
    if config.paths.barcode_mappings:
        mapping = pd.read_parquet(config.paths.barcode_mappings, columns=["cell_id", "square_008um", "in_cell"])
        mapping = mapping[mapping["in_cell"] & mapping["cell_id"].isin(cell_ids)]
        mapping_counts = mapping["square_008um"].value_counts()

    bin_adata = sc.read_10x_h5(config.paths.bins8_h5, gex_only=False)
    bin_adata.var_names_make_unique()
    positions = _read_positions(config.paths.bins8_positions)
    selected_bin_ids = _subset_bin_ids(bin_adata.obs_names, positions, mapping_counts, config.data.bin_subset, config.data.seed)
    bin_adata = bin_adata[selected_bin_ids].copy()
    bin_ids = bin_adata.obs_names.astype(str).to_numpy()

    programs = load_programs(config.paths.programs_json)
    gene_names = _select_gene_panel(
        cell_adata=cell_adata,
        bin_adata=bin_adata,
        program_genes=collect_program_genes(programs),
        hvg_n=config.data.hvg_n,
        cap=config.data.encoder_gene_cap,
    )
    program_library = build_program_library(programs, gene_names)

    cell_counts = _dense_float32(_counts_matrix(cell_adata[:, gene_names]))
    bin_counts = _dense_float32(bin_adata[:, gene_names].X)
    cell_log_counts = np.log1p(cell_counts)
    bin_log_counts = np.log1p(bin_counts)
    cell_library = cell_counts.sum(axis=1, keepdims=True) + 1e-4
    bin_library = bin_counts.sum(axis=1, keepdims=True) + 1e-4

    cell_coords = np.asarray(cell_adata.obsm["spatial"], dtype=np.float32) * microns_per_pixel
    bin_pos = positions.loc[bin_ids]
    bin_coords = np.stack(
        [
            np.asarray(bin_pos["pxl_col_in_fullres"], dtype=np.float32) * microns_per_pixel,
            np.asarray(bin_pos["pxl_row_in_fullres"], dtype=np.float32) * microns_per_pixel,
        ],
        axis=1,
    )

    teacher_overlap = _build_teacher_overlap(
        mapping_path=config.paths.barcode_mappings,
        cell_ids=cell_ids,
        bin_ids=bin_ids,
        cell_coords=cell_coords,
        bin_coords=bin_coords,
    )

    teacher_src, teacher_dst, _ = _build_knn_edges(bin_coords, config.data.teacher_knn)
    teacher_edge_index = np.vstack([teacher_src, teacher_dst]).astype(np.int64)
    context_edge_index, shell_edge_indices, ot_neighbor_index = _build_cell_graphs(
        coords=cell_coords,
        context_k=config.data.context_knn,
        ot_k=config.data.ot_knn,
        shell_bounds=tuple(config.data.shell_bounds_um),
    )

    bin_neighbor_target = aggregate_mean_numpy(bin_counts, teacher_edge_index)
    cell_neighbor_target = aggregate_mean_numpy(cell_counts, context_edge_index)
    cell_aux = _extract_cell_aux(cell_adata)
    batch_onehot = _extract_batches(cell_adata.obs)
    cell_type_labels, cell_type_names, cell_type_onehot = _extract_cell_types(cell_adata.obs, config.data.cell_type_key)
    program_scores = score_programs(cell_log_counts, gene_names, program_library)

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
        cell_library=cell_library.astype(np.float32),
        bin_library=bin_library.astype(np.float32),
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
    )
