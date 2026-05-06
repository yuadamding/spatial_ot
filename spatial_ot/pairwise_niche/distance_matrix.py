from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .local_measure import LocalMeasureSet
from .sinkhorn import sinkhorn_ot_block


def _resolve_device(device: str) -> torch.device:
    requested = str(device or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _estimate_dense_bytes(n_cells: int) -> int:
    return int(n_cells) * int(n_cells) * np.dtype("float32").itemsize


def _self_sinkhorn_costs(
    measures: LocalMeasureSet,
    *,
    block_size: int,
    device: torch.device,
    epsilon: float,
    n_iters: int,
) -> np.ndarray:
    n = int(measures.tokens.shape[0])
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, max(int(block_size), 1)):
        stop = min(start + max(int(block_size), 1), n)
        tokens = torch.as_tensor(measures.tokens[start:stop], dtype=torch.float32, device=device)
        weights = torch.as_tensor(measures.weights[start:stop], dtype=torch.float32, device=device)
        values = []
        for idx in range(stop - start):
            value = sinkhorn_ot_block(
                tokens[idx : idx + 1],
                weights[idx : idx + 1],
                tokens[idx : idx + 1],
                weights[idx : idx + 1],
                epsilon=epsilon,
                n_iters=n_iters,
            )
            values.append(value.reshape(()))
        out[start:stop] = torch.stack(values).detach().cpu().numpy().astype(np.float32)
    return out


def compute_pairwise_ot_distance_matrix(
    *,
    measures: LocalMeasureSet,
    anchor_embedding: np.ndarray,
    output_path: str | Path | None = None,
    block_size: int = 64,
    device: str = "auto",
    epsilon: float = 0.05,
    n_iters: int = 50,
    distance_mode: str = "sinkhorn_divergence",
    anchor_weight: float = 0.25,
    max_exact_cells: int = 5000,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute a dense exact pairwise OT matrix, optionally backed by a .npy memmap."""

    tokens = np.asarray(measures.tokens, dtype=np.float32)
    weights = np.asarray(measures.weights, dtype=np.float32)
    anchors = np.asarray(anchor_embedding, dtype=np.float32)
    n = int(tokens.shape[0])
    if anchors.shape[0] != n:
        raise ValueError("anchor_embedding must have one row per local measure.")
    if n > int(max_exact_cells):
        estimate_gb = _estimate_dense_bytes(n) / 1024**3
        raise ValueError(
            f"Exact all-pairs OT requested for {n} cells; the float32 distance matrix alone "
            f"would require about {estimate_gb:.1f} GiB. Increase --max-exact-cells only "
            "if you really want this exact dense computation, or run a smaller/landmark cohort."
        )

    out: np.ndarray
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(n, n))
    else:
        out = np.zeros((n, n), dtype=np.float32)

    resolved_device = _resolve_device(device)
    requested_mode = str(distance_mode or "sinkhorn_divergence").strip().lower()
    valid_modes = {"sinkhorn", "sinkhorn_divergence", "debiased", "debiased_sinkhorn"}
    if requested_mode not in valid_modes:
        raise ValueError("distance_mode must be sinkhorn or sinkhorn_divergence.")
    debiased = requested_mode in {"sinkhorn_divergence", "debiased", "debiased_sinkhorn"}
    self_costs = (
        _self_sinkhorn_costs(
            measures,
            block_size=max(int(block_size), 1),
            device=resolved_device,
            epsilon=float(epsilon),
            n_iters=int(n_iters),
        )
        if debiased
        else np.zeros(n, dtype=np.float32)
    )

    bs = max(int(block_size), 1)
    for a_start in range(0, n, bs):
        a_stop = min(a_start + bs, n)
        tok_a = torch.as_tensor(tokens[a_start:a_stop], dtype=torch.float32, device=resolved_device)
        w_a = torch.as_tensor(weights[a_start:a_stop], dtype=torch.float32, device=resolved_device)
        anchor_a = anchors[a_start:a_stop]
        for b_start in range(a_start, n, bs):
            b_stop = min(b_start + bs, n)
            tok_b = torch.as_tensor(tokens[b_start:b_stop], dtype=torch.float32, device=resolved_device)
            w_b = torch.as_tensor(weights[b_start:b_stop], dtype=torch.float32, device=resolved_device)
            block = sinkhorn_ot_block(
                tok_a,
                w_a,
                tok_b,
                w_b,
                epsilon=float(epsilon),
                n_iters=int(n_iters),
            ).detach().cpu().numpy().astype(np.float32)
            if debiased:
                block = block - 0.5 * self_costs[a_start:a_stop, None]
                block = block - 0.5 * self_costs[None, b_start:b_stop]
                block = np.maximum(block, 0.0).astype(np.float32, copy=False)
            if float(anchor_weight) > 0.0:
                anchor_b = anchors[b_start:b_stop]
                anchor_cost = (
                    np.sum(anchor_a * anchor_a, axis=1, keepdims=True)
                    + np.sum(anchor_b * anchor_b, axis=1, keepdims=True).T
                    - 2.0 * (anchor_a @ anchor_b.T)
                )
                block = block + float(anchor_weight) * np.maximum(anchor_cost, 0.0).astype(
                    np.float32
                )
            out[a_start:a_stop, b_start:b_stop] = block
            if b_start != a_start:
                out[b_start:b_stop, a_start:a_stop] = block.T

    out[:] = 0.5 * (out[:] + out[:].T)
    np.fill_diagonal(out, 0.0)
    if hasattr(out, "flush"):
        out.flush()
    metadata = {
        "distance_mode": "sinkhorn_divergence" if debiased else "sinkhorn",
        "epsilon": float(epsilon),
        "sinkhorn_iters": int(n_iters),
        "anchor_weight": float(anchor_weight),
        "block_size": int(bs),
        "device": str(resolved_device),
        "n_cells": int(n),
        "matrix_bytes": int(_estimate_dense_bytes(n)),
        "output_path": None if output_path is None else str(output_path),
    }
    return out, metadata
