from __future__ import annotations

import numpy as np
from scipy import sparse
import torch


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested for deep features, but torch.cuda.is_available() is False."
        )
    return resolved


def standardize_features(
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (
        ((x - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def apply_standardization(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return (
        (np.asarray(features, dtype=np.float32) - mean) / np.maximum(std, 1e-6)
    ).astype(np.float32)


def iter_batches(array: np.ndarray, batch_size: int):
    batch_size = max(int(batch_size), 1)
    for start in range(0, array.shape[0], batch_size):
        yield array[start : start + batch_size]


def row_sums(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    return (
        np.asarray(matrix, dtype=np.float32).sum(axis=1).astype(np.float32, copy=False)
    )


def slice_count_chunk(
    matrix, row_index: np.ndarray, gene_index: np.ndarray
) -> np.ndarray:
    rows = np.asarray(row_index, dtype=np.int64)
    genes = np.asarray(gene_index, dtype=np.int64)
    if sparse.issparse(matrix):
        return np.asarray(matrix[rows][:, genes].toarray(), dtype=np.float32)
    return np.asarray(matrix[np.ix_(rows, genes)], dtype=np.float32)


def sample_gene_chunk(
    n_genes: int, chunk_size: int, rng: np.random.Generator
) -> np.ndarray:
    if chunk_size <= 0 or chunk_size >= n_genes:
        return np.arange(n_genes, dtype=np.int64)
    return np.sort(
        rng.choice(n_genes, size=int(chunk_size), replace=False).astype(np.int64)
    )


def seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
