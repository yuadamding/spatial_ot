from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .local_measure import LocalMeasureSet
from .fgw import fused_gromov_wasserstein_block
from .sinkhorn import sinkhorn_ot_block


def _resolve_device(device: str) -> torch.device:
    requested = str(device or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _estimate_dense_bytes(n_cells: int) -> int:
    return int(n_cells) * int(n_cells) * np.dtype("float32").itemsize


def estimate_pairwise_ot_work(
    *,
    n_cells: int,
    support_size: int,
    sinkhorn_iters: int,
) -> dict[str, float]:
    n_pairs = int(n_cells) * (int(n_cells) + 1) // 2
    work_units = float(n_pairs) * float(support_size) ** 2 * float(sinkhorn_iters)
    return {
        "n_pairs": float(n_pairs),
        "support_size": float(support_size),
        "sinkhorn_iters": float(sinkhorn_iters),
        "work_units": float(work_units),
        "matrix_gib": float(_estimate_dense_bytes(int(n_cells)) / 1024**3),
    }


def estimate_pairwise_fgw_work(
    *,
    n_cells: int,
    support_size: int,
    sinkhorn_iters: int,
    fgw_iters: int,
) -> dict[str, float]:
    n_pairs = int(n_cells) * (int(n_cells) + 1) // 2
    structure_units = (
        float(n_pairs) * float(fgw_iters) * float(support_size) ** 3
    )
    sinkhorn_units = (
        float(n_pairs)
        * float(fgw_iters)
        * float(support_size) ** 2
        * float(sinkhorn_iters)
    )
    return {
        "n_pairs": float(n_pairs),
        "support_size": float(support_size),
        "sinkhorn_iters": float(sinkhorn_iters),
        "fgw_iters": float(fgw_iters),
        "fgw_structure_units": float(structure_units),
        "fgw_sinkhorn_units": float(sinkhorn_units),
        "fgw_total_units": float(structure_units + sinkhorn_units),
        "matrix_gib": float(_estimate_dense_bytes(int(n_cells)) / 1024**3),
    }


def _fgw_node_feature_array(
    tokens: np.ndarray,
    metadata: dict[str, object],
    *,
    mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    requested = str(mode or "expression_only").strip().lower()
    slices = metadata.get("ground_cost_component_slices", {})
    if not isinstance(slices, dict) or "expression" not in slices:
        raise ValueError("FGW node feature selection requires local-measure component slices.")

    def component(name: str) -> np.ndarray:
        raw_slice = slices.get(name)
        if not isinstance(raw_slice, (list, tuple)) or len(raw_slice) != 2:
            raise ValueError(f"Missing local-measure component slice for {name!r}.")
        start, stop = int(raw_slice[0]), int(raw_slice[1])
        return tokens[:, :, start:stop]

    if requested == "expression_only":
        features = component("expression")
    elif requested == "expression_plus_radial":
        features = np.concatenate(
            [component("expression"), component("relative_distance")],
            axis=2,
        )
    elif requested == "full_token":
        features = tokens
    else:
        raise ValueError(
            "fgw_node_feature_mode must be expression_only, expression_plus_radial, or full_token."
        )
    return features.astype(np.float32, copy=False), {
        "fgw_node_feature_mode": requested,
        "fgw_node_feature_dim": int(features.shape[2]),
    }


def _fit_fgw_structure_scale(
    structures: np.ndarray,
    mask: np.ndarray,
    *,
    normalization: str,
    n_pairs: int,
    seed: int,
) -> float:
    requested = str(normalization or "sampled_median").strip().lower()
    if requested == "none":
        return 1.0
    if requested != "sampled_median":
        raise ValueError("fgw_structure_normalization must be none or sampled_median.")
    values: list[np.ndarray] = []
    active_mask = np.asarray(mask, dtype=bool)
    for row in range(structures.shape[0]):
        count = int(np.sum(active_mask[row]))
        if count <= 1:
            continue
        block = np.asarray(structures[row, :count, :count], dtype=np.float32)
        upper = block[np.triu_indices(count, k=1)]
        upper = upper[np.isfinite(upper)]
        if upper.size:
            values.append(upper)
    if not values:
        return 1.0
    observed = np.concatenate(values).astype(np.float32, copy=False)
    if observed.size <= 1:
        return 1.0
    rng = np.random.default_rng(int(seed))
    count = min(max(int(n_pairs), 1), max(int(observed.size) ** 2, 1))
    left = rng.integers(0, observed.size, size=count)
    right = rng.integers(0, observed.size, size=count)
    costs = (observed[left] - observed[right]) ** 2
    costs = costs[np.isfinite(costs) & (costs > 0)]
    if costs.size == 0:
        return 1.0
    return float(max(np.median(costs), 1e-8))


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
    distance_mode: str = "debiased_entropic_transport",
    anchor_weight: float = 0.0,
    fgw_alpha: float = 0.25,
    fgw_iters: int = 5,
    fgw_node_feature_mode: str = "expression_only",
    fgw_structure_normalization: str = "sampled_median",
    fgw_structure_sample_pairs: int = 10000,
    max_exact_cells: int = 5000,
    max_ot_work_units: float = 5e11,
    max_fgw_work_units: float = 1e12,
    force_large_exact_ot: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute a dense exact pairwise OT matrix, optionally backed by a .npy memmap."""

    tokens = np.asarray(measures.tokens, dtype=np.float32)
    weights = np.asarray(measures.weights, dtype=np.float32)
    anchors = np.asarray(anchor_embedding, dtype=np.float32)
    n = int(tokens.shape[0])
    if anchors.shape[0] != n:
        raise ValueError("anchor_embedding must have one row per local measure.")
    work_estimate = estimate_pairwise_ot_work(
        n_cells=n,
        support_size=int(tokens.shape[1]),
        sinkhorn_iters=int(n_iters),
    )
    if n > int(max_exact_cells) and not bool(force_large_exact_ot):
        estimate_gb = _estimate_dense_bytes(n) / 1024**3
        raise ValueError(
            f"Exact all-pairs OT requested for {n} cells; the float32 distance matrix alone "
            f"would require about {estimate_gb:.1f} GiB. Increase --max-exact-cells only "
            "if you really want this exact dense computation, or run a smaller/landmark cohort."
        )
    if work_estimate["work_units"] > float(max_ot_work_units) and not bool(force_large_exact_ot):
        raise ValueError(
            "Exact all-pairs OT work estimate is too large: "
            f"{work_estimate['work_units']:.3g} work units for {n} cells, "
            f"support size {tokens.shape[1]}, and {int(n_iters)} Sinkhorn iterations. "
            "Increase --max-ot-work-units or use --force-large-exact-ot only if this "
            "exact dense computation is intentional."
        )

    out: np.ndarray
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(n, n))
    else:
        out = np.zeros((n, n), dtype=np.float32)

    resolved_device = _resolve_device(device)
    requested_mode = str(distance_mode or "debiased_entropic_transport").strip().lower()
    valid_modes = {
        "sinkhorn",
        "sinkhorn_divergence",
        "debiased",
        "debiased_sinkhorn",
        "debiased_entropic_transport",
        "fgw",
        "fused_gromov_wasserstein",
    }
    if requested_mode not in valid_modes:
        raise ValueError(
            "distance_mode must be sinkhorn, debiased_entropic_transport, "
            "or fused_gromov_wasserstein."
        )
    debiased = requested_mode in {
        "sinkhorn_divergence",
        "debiased",
        "debiased_sinkhorn",
        "debiased_entropic_transport",
    }
    use_fgw = requested_mode in {"fgw", "fused_gromov_wasserstein"}
    if use_fgw and measures.structure_matrices is None:
        raise ValueError("fused_gromov_wasserstein requires local graph structure matrices.")
    fgw_feature_metadata: dict[str, object] = {}
    fgw_work_estimate: dict[str, float] | None = None
    if use_fgw:
        fgw_work_estimate = estimate_pairwise_fgw_work(
            n_cells=n,
            support_size=int(tokens.shape[1]),
            sinkhorn_iters=int(n_iters),
            fgw_iters=int(fgw_iters),
        )
        if (
            fgw_work_estimate["fgw_total_units"] > float(max_fgw_work_units)
            and not bool(force_large_exact_ot)
        ):
            raise ValueError(
                "Exact all-pairs FGW work estimate is too large: "
                f"{fgw_work_estimate['fgw_total_units']:.3g} work units for {n} cells, "
                f"support size {tokens.shape[1]}, {int(fgw_iters)} FGW iterations, "
                f"and {int(n_iters)} Sinkhorn iterations. Increase --max-fgw-work-units "
                "or use --force-large-exact-ot only if this exact dense computation is intentional."
            )
        tokens, fgw_feature_metadata = _fgw_node_feature_array(
            tokens,
            measures.metadata,
            mode=str(fgw_node_feature_mode),
        )
    self_costs = (
        _self_sinkhorn_costs(
            measures,
            block_size=max(int(block_size), 1),
            device=resolved_device,
            epsilon=float(epsilon),
            n_iters=int(n_iters),
        )
        if debiased and not use_fgw
        else np.zeros(n, dtype=np.float32)
    )
    structures = None
    fgw_structure_scale = 1.0
    if use_fgw:
        structures = np.asarray(measures.structure_matrices, dtype=np.float32)
        fgw_structure_scale = _fit_fgw_structure_scale(
            structures,
            measures.mask,
            normalization=str(fgw_structure_normalization),
            n_pairs=int(fgw_structure_sample_pairs),
            seed=int(measures.metadata.get("seed", 1337)),
        )
        structures = structures / np.float32(np.sqrt(max(fgw_structure_scale, 1e-8)))

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
            if use_fgw:
                assert structures is not None
                struct_a = torch.as_tensor(
                    structures[a_start:a_stop],
                    dtype=torch.float32,
                    device=resolved_device,
                )
                struct_b = torch.as_tensor(
                    structures[b_start:b_stop],
                    dtype=torch.float32,
                    device=resolved_device,
                )
                block = (
                    fused_gromov_wasserstein_block(
                        tok_a,
                        struct_a,
                        w_a,
                        tok_b,
                        struct_b,
                        w_b,
                        alpha=float(fgw_alpha),
                        epsilon=float(epsilon),
                        sinkhorn_iters=int(n_iters),
                        fgw_iters=int(fgw_iters),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            else:
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
        "distance_mode": (
            "fused_gromov_wasserstein"
            if use_fgw
            else "debiased_entropic_transport"
            if debiased
            else "sinkhorn"
        ),
        "requested_distance_mode": str(requested_mode),
        "sinkhorn_divergence_alias_used": bool(
            requested_mode in {"sinkhorn_divergence", "debiased", "debiased_sinkhorn"}
        ),
        "returns_plan_transport_cost_only": True,
        "includes_entropy_objective_term": False,
        "epsilon": float(epsilon),
        "sinkhorn_iters": int(n_iters),
        "anchor_weight": float(anchor_weight),
        "fgw_alpha": float(fgw_alpha) if use_fgw else None,
        "fgw_iters": int(fgw_iters) if use_fgw else None,
        "uses_graph_topology": bool(
            use_fgw
            and str(measures.metadata.get("fgw_structure_mode", "complete_euclidean"))
            != "complete_euclidean"
        ),
        "uses_complete_spatial_structure": bool(
            use_fgw
            and str(measures.metadata.get("fgw_structure_mode", "complete_euclidean"))
            == "complete_euclidean"
        ),
        "fgw_structure_mode": measures.metadata.get("fgw_structure_mode")
        if use_fgw
        else None,
        "fgw_structure_semantics": (
            None
            if not use_fgw
            else "complete pairwise spatial-distance structure, not adjacency/shortest-path topology"
            if str(measures.metadata.get("fgw_structure_mode")) == "complete_euclidean"
            else "explicit graph-derived structure"
        ),
        "fgw_structure_normalization": str(fgw_structure_normalization)
        if use_fgw
        else None,
        "fgw_structure_sample_pairs": int(fgw_structure_sample_pairs)
        if use_fgw
        else None,
        "fgw_structure_cost_scale": float(fgw_structure_scale) if use_fgw else None,
        "block_size": int(bs),
        "device": str(resolved_device),
        "n_cells": int(n),
        "matrix_bytes": int(_estimate_dense_bytes(n)),
        "work_estimate": work_estimate,
        "fgw_work_estimate": fgw_work_estimate,
        "max_ot_work_units": float(max_ot_work_units),
        "max_fgw_work_units": float(max_fgw_work_units),
        "force_large_exact_ot": bool(force_large_exact_ot),
        "output_path": None if output_path is None else str(output_path),
        **fgw_feature_metadata,
    }
    return out, metadata


__all__ = [
    "compute_pairwise_ot_distance_matrix",
    "estimate_pairwise_fgw_work",
    "estimate_pairwise_ot_work",
]
