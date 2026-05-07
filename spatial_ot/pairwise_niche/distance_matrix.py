from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .local_measure import LocalMeasureSet
from .fgw import fused_gromov_wasserstein_block
from .sinkhorn import (
    sinkhorn_ot_block,
    sinkhorn_ot_marginal_error_block,
    sinkhorn_self_cost_batch,
)


def _resolve_device(device: str) -> torch.device:
    requested = str(device or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


_DEBIASED_MODES = {
    "sinkhorn_divergence",
    "debiased",
    "debiased_sinkhorn",
    "debiased_entropic_transport",
}
_FGW_MODES = {"fgw", "fused_gromov_wasserstein"}
_VALID_DISTANCE_MODES = {"sinkhorn", *_DEBIASED_MODES, *_FGW_MODES}


def _parse_distance_mode(distance_mode: str) -> tuple[str, bool, bool]:
    requested = str(distance_mode or "debiased_entropic_transport").strip().lower()
    if requested not in _VALID_DISTANCE_MODES:
        raise ValueError(
            "distance_mode must be sinkhorn, debiased_entropic_transport, "
            "or fused_gromov_wasserstein."
        )
    return requested, requested in _DEBIASED_MODES, requested in _FGW_MODES


def _canonical_distance_mode(*, debiased: bool, use_fgw: bool) -> str:
    if use_fgw:
        return "fused_gromov_wasserstein"
    return "debiased_entropic_transport" if debiased else "sinkhorn"


def _active_counts(mask: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(mask, dtype=bool).sum(axis=1), 1)


def _tensor_block(
    values: np.ndarray,
    start: int,
    stop: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        values[start:stop, :width],
        dtype=torch.float32,
        device=device,
    )


def _structure_block(
    values: np.ndarray,
    start: int,
    stop: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        values[start:stop, :width, :width],
        dtype=torch.float32,
        device=device,
    )


def _distance_block(
    *,
    tok_a: torch.Tensor,
    w_a: torch.Tensor,
    tok_b: torch.Tensor,
    w_b: torch.Tensor,
    use_fgw: bool,
    struct_a: torch.Tensor | None = None,
    struct_b: torch.Tensor | None = None,
    fgw_alpha: float,
    epsilon: float,
    n_iters: int,
    fgw_iters: int,
) -> np.ndarray:
    if use_fgw:
        if struct_a is None or struct_b is None:
            raise ValueError("FGW distance blocks require structure tensors.")
        block = fused_gromov_wasserstein_block(
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
    else:
        block = sinkhorn_ot_block(
            tok_a,
            w_a,
            tok_b,
            w_b,
            epsilon=float(epsilon),
            n_iters=int(n_iters),
        )
    return block.detach().cpu().numpy().astype(np.float32)


def _add_anchor_cost(
    block: np.ndarray,
    anchor_a: np.ndarray,
    anchor_b: np.ndarray,
    anchor_weight: float,
) -> np.ndarray:
    if float(anchor_weight) <= 0.0:
        return block
    anchor_cost = (
        np.sum(anchor_a * anchor_a, axis=1, keepdims=True)
        + np.sum(anchor_b * anchor_b, axis=1, keepdims=True).T
        - 2.0 * (anchor_a @ anchor_b.T)
    )
    return block + float(anchor_weight) * np.maximum(anchor_cost, 0.0).astype(np.float32)


def _estimate_dense_bytes(n_cells: int) -> int:
    return int(n_cells) * int(n_cells) * np.dtype("float32").itemsize


def _estimate_rectangular_bytes(n_query: int, n_reference: int) -> int:
    return int(n_query) * int(n_reference) * np.dtype("float32").itemsize


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


def estimate_cross_ot_work(
    *,
    n_query: int,
    n_reference: int,
    support_size_query: int,
    support_size_reference: int,
    sinkhorn_iters: int,
) -> dict[str, float]:
    n_pairs = int(n_query) * int(n_reference)
    support_work = int(support_size_query) * int(support_size_reference)
    work_units = float(n_pairs) * float(support_work) * float(sinkhorn_iters)
    return {
        "n_query": float(n_query),
        "n_reference": float(n_reference),
        "n_cross_pairs": float(n_pairs),
        "support_size_query": float(support_size_query),
        "support_size_reference": float(support_size_reference),
        "sinkhorn_iters": float(sinkhorn_iters),
        "work_units": float(work_units),
        "matrix_gib": float(_estimate_rectangular_bytes(n_query, n_reference) / 1024**3),
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


def estimate_cross_fgw_work(
    *,
    n_query: int,
    n_reference: int,
    support_size_query: int,
    support_size_reference: int,
    sinkhorn_iters: int,
    fgw_iters: int,
) -> dict[str, float]:
    n_pairs = int(n_query) * int(n_reference)
    lq = int(support_size_query)
    lr = int(support_size_reference)
    structure_units = float(n_pairs) * float(fgw_iters) * float(lq) * float(lr) * float(max(lq, lr))
    sinkhorn_units = (
        float(n_pairs)
        * float(fgw_iters)
        * float(lq)
        * float(lr)
        * float(sinkhorn_iters)
    )
    return {
        "n_query": float(n_query),
        "n_reference": float(n_reference),
        "n_cross_pairs": float(n_pairs),
        "support_size_query": float(lq),
        "support_size_reference": float(lr),
        "sinkhorn_iters": float(sinkhorn_iters),
        "fgw_iters": float(fgw_iters),
        "fgw_structure_units": float(structure_units),
        "fgw_sinkhorn_units": float(sinkhorn_units),
        "fgw_total_units": float(structure_units + sinkhorn_units),
        "matrix_gib": float(_estimate_rectangular_bytes(n_query, n_reference) / 1024**3),
    }


def choose_pairwise_block_size(
    *,
    support_size: int,
    requested_block_size: int,
    target_memory_gib: float | None,
    use_fgw: bool,
) -> tuple[int, dict[str, object]]:
    requested = int(requested_block_size)
    if target_memory_gib is None or float(target_memory_gib) <= 0.0:
        block_size = max(requested, 1)
        return block_size, {
            "adaptive_block_size_enabled": False,
            "requested_block_size": int(requested),
            "target_block_memory_gib": None,
            "block_size_memory_cap": None,
            "block_size_live_tensor_estimate": None,
        }

    live_tensors = 12 if bool(use_fgw) else 4
    target_bytes = float(target_memory_gib) * 1024**3
    denominator = live_tensors * max(int(support_size), 1) ** 2 * np.dtype("float32").itemsize
    memory_cap = max(int(np.floor(np.sqrt(max(target_bytes, 1.0) / max(denominator, 1)))), 1)
    block_size = memory_cap if requested <= 0 else min(max(requested, 1), memory_cap)
    return block_size, {
        "adaptive_block_size_enabled": True,
        "requested_block_size": int(requested),
        "target_block_memory_gib": float(target_memory_gib),
        "block_size_memory_cap": int(memory_cap),
        "block_size_live_tensor_estimate": int(live_tensors),
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
    active_counts = _active_counts(measures.mask)
    for start in range(0, n, max(int(block_size), 1)):
        stop = min(start + max(int(block_size), 1), n)
        width = int(np.max(active_counts[start:stop]))
        tokens = _tensor_block(measures.tokens, start, stop, width, device=device)
        weights = _tensor_block(measures.weights, start, stop, width, device=device)
        out[start:stop] = (
            sinkhorn_self_cost_batch(
                tokens,
                weights,
                epsilon=epsilon,
                n_iters=n_iters,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return out


def _cached_or_computed_self_costs(
    measures: LocalMeasureSet,
    provided: np.ndarray | None,
    *,
    block_size: int,
    device: torch.device,
    epsilon: float,
    n_iters: int,
    label: str,
) -> tuple[np.ndarray, str]:
    n = int(measures.tokens.shape[0])
    if provided is None:
        return (
            _self_sinkhorn_costs(
                measures,
                block_size=int(block_size),
                device=device,
                epsilon=float(epsilon),
                n_iters=int(n_iters),
            ),
            "computed",
        )
    out = np.asarray(provided, dtype=np.float32)
    if out.shape != (n,):
        raise ValueError(f"{label}_self_costs must have shape ({n},).")
    if not np.all(np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError(f"{label}_self_costs must contain finite nonnegative values.")
    return out, "provided"


def assign_by_reference_medoids(
    cross_distance: np.ndarray,
    medoid_labels: np.ndarray,
    *,
    margin_threshold: float | None = None,
    unknown_label: object = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign each query row to its nearest reference medoid.

    Returns labels, distance-margin scores, nearest reference columns, and nearest distances.
    """

    distance = np.asarray(cross_distance, dtype=np.float32)
    labels = np.asarray(medoid_labels)
    if distance.ndim != 2:
        raise ValueError("cross_distance must be a 2D query x reference matrix.")
    if labels.shape[0] != distance.shape[1]:
        raise ValueError("medoid_labels must have one label per reference column.")
    if distance.shape[1] == 0:
        raise ValueError("cross_distance must contain at least one reference column.")
    if not np.all(np.isfinite(distance)):
        raise ValueError("cross_distance must contain only finite values.")
    if margin_threshold is not None and (
        not np.isfinite(float(margin_threshold)) or float(margin_threshold) < 0.0
    ):
        raise ValueError("margin_threshold must be a finite nonnegative value.")

    nearest = np.argmin(distance, axis=1).astype(np.int64)
    row = np.arange(distance.shape[0])
    nearest_distance = distance[row, nearest].astype(np.float32, copy=False)
    if distance.shape[1] == 1:
        score = np.ones(distance.shape[0], dtype=np.float32)
    else:
        two = np.partition(distance, kth=1, axis=1)[:, :2]
        two.sort(axis=1)
        score = ((two[:, 1] - two[:, 0]) / (two[:, 1] + 1e-8)).astype(np.float32)

    assigned = labels[nearest].copy()
    if margin_threshold is not None:
        assigned = assigned.astype(object, copy=False)
        assigned[score < float(margin_threshold)] = unknown_label
    return assigned, score, nearest, nearest_distance


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
    target_block_memory_gib: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute a dense exact pairwise OT matrix, optionally backed by a .npy memmap."""

    tokens = np.asarray(measures.tokens, dtype=np.float32)
    weights = np.asarray(measures.weights, dtype=np.float32)
    anchors = np.asarray(anchor_embedding, dtype=np.float32)
    active_counts = _active_counts(measures.mask)
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
    requested_mode, debiased, use_fgw = _parse_distance_mode(distance_mode)
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
    bs, block_metadata = choose_pairwise_block_size(
        support_size=int(tokens.shape[1]),
        requested_block_size=int(block_size),
        target_memory_gib=target_block_memory_gib,
        use_fgw=bool(use_fgw),
    )
    self_costs = (
        _self_sinkhorn_costs(
            measures,
            block_size=int(bs),
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
    sinkhorn_diagnostic_errors: list[np.ndarray] = []
    sinkhorn_diagnostic_block_limit = 4
    sinkhorn_diagnostic_cells_per_axis = 8
    for a_start in range(0, n, bs):
        a_stop = min(a_start + bs, n)
        width_a = int(np.max(active_counts[a_start:a_stop]))
        tok_a = _tensor_block(tokens, a_start, a_stop, width_a, device=resolved_device)
        w_a = _tensor_block(weights, a_start, a_stop, width_a, device=resolved_device)
        anchor_a = anchors[a_start:a_stop]
        for b_start in range(a_start, n, bs):
            b_stop = min(b_start + bs, n)
            width_b = int(np.max(active_counts[b_start:b_stop]))
            tok_b = _tensor_block(tokens, b_start, b_stop, width_b, device=resolved_device)
            w_b = _tensor_block(weights, b_start, b_stop, width_b, device=resolved_device)
            if use_fgw:
                assert structures is not None
                struct_a = _structure_block(
                    structures,
                    a_start,
                    a_stop,
                    width_a,
                    device=resolved_device,
                )
                struct_b = _structure_block(
                    structures,
                    b_start,
                    b_stop,
                    width_b,
                    device=resolved_device,
                )
            else:
                struct_a = struct_b = None
            block = _distance_block(
                tok_a=tok_a,
                w_a=w_a,
                tok_b=tok_b,
                w_b=w_b,
                use_fgw=use_fgw,
                struct_a=struct_a,
                struct_b=struct_b,
                fgw_alpha=float(fgw_alpha),
                epsilon=float(epsilon),
                n_iters=int(n_iters),
                fgw_iters=int(fgw_iters),
            )
            if not use_fgw and len(sinkhorn_diagnostic_errors) < sinkhorn_diagnostic_block_limit:
                diag_a = min(int(tok_a.shape[0]), sinkhorn_diagnostic_cells_per_axis)
                diag_b = min(int(tok_b.shape[0]), sinkhorn_diagnostic_cells_per_axis)
                errors = (
                    sinkhorn_ot_marginal_error_block(
                        tok_a[:diag_a],
                        w_a[:diag_a],
                        tok_b[:diag_b],
                        w_b[:diag_b],
                        epsilon=float(epsilon),
                        n_iters=int(n_iters),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                sinkhorn_diagnostic_errors.append(errors.reshape(-1))
            if debiased:
                block = block - 0.5 * self_costs[a_start:a_stop, None]
                block = block - 0.5 * self_costs[None, b_start:b_stop]
                block = np.maximum(block, 0.0).astype(np.float32, copy=False)
            block = _add_anchor_cost(
                block,
                anchor_a,
                anchors[b_start:b_stop],
                float(anchor_weight),
            )
            if b_start == a_start:
                block = (0.5 * (block + block.T)).astype(np.float32, copy=False)
                np.fill_diagonal(block, 0.0)
            out[a_start:a_stop, b_start:b_stop] = block
            if b_start != a_start:
                out[b_start:b_stop, a_start:a_stop] = block.T

    diag = np.arange(n)
    out[diag, diag] = 0.0
    if hasattr(out, "flush"):
        out.flush()
    if sinkhorn_diagnostic_errors and not use_fgw:
        sinkhorn_errors = np.concatenate(sinkhorn_diagnostic_errors)
        sinkhorn_diagnostics: dict[str, object] = {
            "sampled_blocks": int(len(sinkhorn_diagnostic_errors)),
            "sampled_pairs": int(sinkhorn_errors.size),
            "mean_marginal_error": float(np.mean(sinkhorn_errors)),
            "max_marginal_error": float(np.max(sinkhorn_errors)),
            "epsilon": float(epsilon),
            "sinkhorn_iters": int(n_iters),
        }
    else:
        sinkhorn_diagnostics = {
            "sampled_blocks": 0,
            "sampled_pairs": 0,
            "mean_marginal_error": None,
            "max_marginal_error": None,
            "epsilon": float(epsilon),
            "sinkhorn_iters": int(n_iters),
        }
    metadata = {
        "distance_mode": _canonical_distance_mode(debiased=debiased, use_fgw=use_fgw),
        "requested_distance_mode": str(requested_mode),
        "sinkhorn_divergence_alias_used": bool(
            requested_mode in {"sinkhorn_divergence", "debiased", "debiased_sinkhorn"}
        ),
        "returns_plan_transport_cost_only": True,
        "includes_entropy_objective_term": False,
        "blockwise_symmetric_write": True,
        "global_dense_symmetrization": False,
        "epsilon": float(epsilon),
        "sinkhorn_iters": int(n_iters),
        "sinkhorn_diagnostics": sinkhorn_diagnostics,
        "anchor_weight": float(anchor_weight),
        "fgw_alpha": float(fgw_alpha) if use_fgw else None,
        "fgw_iters": int(fgw_iters) if use_fgw else None,
        "fgw_debiased": False if use_fgw else None,
        "fgw_diagonal_forced_zero": True if use_fgw else None,
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
        "fgw_structure_requested_mode": measures.metadata.get(
            "fgw_structure_requested_mode"
        )
        if use_fgw
        else None,
        "fgw_structure_semantics": (
            None
            if not use_fgw
            else "complete pairwise spatial-distance structure, not adjacency/shortest-path topology"
            if str(measures.metadata.get("fgw_structure_mode")) == "complete_euclidean"
            else "binary edge/nonedge distance structure"
            if str(measures.metadata.get("fgw_structure_mode")) == "binary_edge_distance"
            else "explicit graph-derived shortest-path structure"
        ),
        "fgw_structure_disconnected_fraction": measures.metadata.get(
            "fgw_structure_disconnected_fraction"
        )
        if use_fgw
        else None,
        "fgw_structure_inf_fill_policy": measures.metadata.get(
            "fgw_structure_inf_fill_policy"
        )
        if use_fgw
        else None,
        "fgw_structure_disconnected_warning": measures.metadata.get(
            "fgw_structure_disconnected_warning"
        )
        if use_fgw
        else None,
        "fgw_structure_normalization": str(fgw_structure_normalization)
        if use_fgw
        else None,
        "fgw_structure_sample_pairs": int(fgw_structure_sample_pairs)
        if use_fgw
        else None,
        "fgw_structure_cost_scale": float(fgw_structure_scale) if use_fgw else None,
        "block_size": int(bs),
        "self_cost_block_size": int(bs) if debiased and not use_fgw else None,
        **block_metadata,
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


def compute_cross_ot_distance_matrix(
    *,
    query_measures: LocalMeasureSet,
    reference_measures: LocalMeasureSet,
    query_anchor_embedding: np.ndarray,
    reference_anchor_embedding: np.ndarray,
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
    fgw_structure_cost_scale: float | None = None,
    query_self_costs: np.ndarray | None = None,
    reference_self_costs: np.ndarray | None = None,
    max_cross_ot_work_units: float = 5e11,
    max_cross_fgw_work_units: float = 1e12,
    force_large_cross_ot: bool = False,
    target_block_memory_gib: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute query-to-reference OT/FGW distances without an all-by-all query matrix."""

    query_tokens = np.asarray(query_measures.tokens, dtype=np.float32)
    reference_tokens = np.asarray(reference_measures.tokens, dtype=np.float32)
    query_weights = np.asarray(query_measures.weights, dtype=np.float32)
    reference_weights = np.asarray(reference_measures.weights, dtype=np.float32)
    query_anchors = np.asarray(query_anchor_embedding, dtype=np.float32)
    reference_anchors = np.asarray(reference_anchor_embedding, dtype=np.float32)
    n_query = int(query_tokens.shape[0])
    n_reference = int(reference_tokens.shape[0])
    if query_anchors.shape[0] != n_query:
        raise ValueError("query_anchor_embedding must have one row per query measure.")
    if reference_anchors.shape[0] != n_reference:
        raise ValueError("reference_anchor_embedding must have one row per reference measure.")
    requested_mode, debiased, use_fgw = _parse_distance_mode(distance_mode)
    if use_fgw and (
        query_measures.structure_matrices is None
        or reference_measures.structure_matrices is None
    ):
        raise ValueError("fused_gromov_wasserstein requires local graph structure matrices.")
    work_estimate = estimate_cross_ot_work(
        n_query=n_query,
        n_reference=n_reference,
        support_size_query=int(query_tokens.shape[1]),
        support_size_reference=int(reference_tokens.shape[1]),
        sinkhorn_iters=int(n_iters),
    )
    fgw_work_estimate = None
    if work_estimate["work_units"] > float(max_cross_ot_work_units) and not bool(
        force_large_cross_ot
    ):
        raise ValueError(
            "Cross OT work estimate is too large: "
            f"{work_estimate['work_units']:.3g} work units for {n_query} query cells, "
            f"{n_reference} reference cells, supports {query_tokens.shape[1]} x "
            f"{reference_tokens.shape[1]}, and {int(n_iters)} Sinkhorn iterations. "
            "Increase max_cross_ot_work_units or set force_large_cross_ot=True only "
            "if this rectangular computation is intentional."
        )
    if use_fgw:
        fgw_work_estimate = estimate_cross_fgw_work(
            n_query=n_query,
            n_reference=n_reference,
            support_size_query=int(query_tokens.shape[1]),
            support_size_reference=int(reference_tokens.shape[1]),
            sinkhorn_iters=int(n_iters),
            fgw_iters=int(fgw_iters),
        )
        if fgw_work_estimate["fgw_total_units"] > float(max_cross_fgw_work_units) and not bool(
            force_large_cross_ot
        ):
            raise ValueError(
                "Cross FGW work estimate is too large: "
                f"{fgw_work_estimate['fgw_total_units']:.3g} work units for "
                f"{n_query} query cells, {n_reference} reference cells, supports "
                f"{query_tokens.shape[1]} x {reference_tokens.shape[1]}, "
                f"{int(fgw_iters)} FGW iterations, and {int(n_iters)} Sinkhorn iterations. "
                "Increase max_cross_fgw_work_units or set force_large_cross_ot=True only "
                "if this rectangular computation is intentional."
            )

    resolved_device = _resolve_device(device)
    bs, block_metadata = choose_pairwise_block_size(
        support_size=max(int(query_tokens.shape[1]), int(reference_tokens.shape[1])),
        requested_block_size=int(block_size),
        target_memory_gib=target_block_memory_gib,
        use_fgw=bool(use_fgw),
    )
    query_active = _active_counts(query_measures.mask)
    reference_active = _active_counts(reference_measures.mask)
    query_self_source = reference_self_source = None
    if debiased and not use_fgw:
        query_self, query_self_source = _cached_or_computed_self_costs(
            query_measures,
            query_self_costs,
            block_size=int(bs),
            device=resolved_device,
            epsilon=float(epsilon),
            n_iters=int(n_iters),
            label="query",
        )
        reference_self, reference_self_source = _cached_or_computed_self_costs(
            reference_measures,
            reference_self_costs,
            block_size=int(bs),
            device=resolved_device,
            epsilon=float(epsilon),
            n_iters=int(n_iters),
            label="reference",
        )
    else:
        query_self = np.zeros(n_query, dtype=np.float32)
        reference_self = np.zeros(n_reference, dtype=np.float32)

    fgw_feature_metadata: dict[str, object] = {}
    query_structures = None
    reference_structures = None
    fgw_structure_scale = 1.0
    if use_fgw:
        requested_structure_normalization = str(fgw_structure_normalization).strip().lower()
        if requested_structure_normalization not in {"none", "sampled_median"}:
            raise ValueError("fgw_structure_normalization must be none or sampled_median.")
        query_tokens, fgw_feature_metadata = _fgw_node_feature_array(
            query_tokens,
            query_measures.metadata,
            mode=str(fgw_node_feature_mode),
        )
        reference_tokens, _ = _fgw_node_feature_array(
            reference_tokens,
            reference_measures.metadata,
            mode=str(fgw_node_feature_mode),
        )
        query_structures = np.asarray(query_measures.structure_matrices, dtype=np.float32)
        reference_structures = np.asarray(
            reference_measures.structure_matrices,
            dtype=np.float32,
        )
        if fgw_structure_cost_scale is not None:
            fgw_structure_scale = float(fgw_structure_cost_scale)
            if not np.isfinite(fgw_structure_scale) or fgw_structure_scale <= 0.0:
                raise ValueError("fgw_structure_cost_scale must be a positive finite value.")
            fgw_structure_scale_source = "provided"
        elif requested_structure_normalization == "sampled_median":
            query_scale = _fit_fgw_structure_scale(
                query_structures,
                query_measures.mask,
                normalization="sampled_median",
                n_pairs=int(fgw_structure_sample_pairs),
                seed=int(query_measures.metadata.get("seed", 1337)),
            )
            reference_scale = _fit_fgw_structure_scale(
                reference_structures,
                reference_measures.mask,
                normalization="sampled_median",
                n_pairs=int(fgw_structure_sample_pairs),
                seed=int(reference_measures.metadata.get("seed", 1337)),
            )
            fgw_structure_scale = float(np.mean([query_scale, reference_scale]))
            fgw_structure_scale_source = "query_reference_sampled_median"
        else:
            fgw_structure_scale_source = "none"
        query_structures = query_structures / np.float32(
            np.sqrt(max(fgw_structure_scale, 1e-8))
        )
        reference_structures = reference_structures / np.float32(
            np.sqrt(max(fgw_structure_scale, 1e-8))
        )

    if output_path is None:
        out = np.zeros((n_query, n_reference), dtype=np.float32)
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype="float32",
            shape=(n_query, n_reference),
        )
    for q_start in range(0, n_query, bs):
        q_stop = min(q_start + bs, n_query)
        q_width = int(np.max(query_active[q_start:q_stop]))
        q_tok = _tensor_block(
            query_tokens,
            q_start,
            q_stop,
            q_width,
            device=resolved_device,
        )
        q_w = _tensor_block(
            query_weights,
            q_start,
            q_stop,
            q_width,
            device=resolved_device,
        )
        for r_start in range(0, n_reference, bs):
            r_stop = min(r_start + bs, n_reference)
            r_width = int(np.max(reference_active[r_start:r_stop]))
            r_tok = _tensor_block(
                reference_tokens,
                r_start,
                r_stop,
                r_width,
                device=resolved_device,
            )
            r_w = _tensor_block(
                reference_weights,
                r_start,
                r_stop,
                r_width,
                device=resolved_device,
            )
            if use_fgw:
                assert query_structures is not None and reference_structures is not None
                q_struct = _structure_block(
                    query_structures,
                    q_start,
                    q_stop,
                    q_width,
                    device=resolved_device,
                )
                r_struct = _structure_block(
                    reference_structures,
                    r_start,
                    r_stop,
                    r_width,
                    device=resolved_device,
                )
            else:
                q_struct = r_struct = None
            block = _distance_block(
                tok_a=q_tok,
                w_a=q_w,
                tok_b=r_tok,
                w_b=r_w,
                use_fgw=use_fgw,
                struct_a=q_struct,
                struct_b=r_struct,
                fgw_alpha=float(fgw_alpha),
                epsilon=float(epsilon),
                n_iters=int(n_iters),
                fgw_iters=int(fgw_iters),
            )
            if debiased:
                block = block - 0.5 * query_self[q_start:q_stop, None]
                block = block - 0.5 * reference_self[None, r_start:r_stop]
                block = np.maximum(block, 0.0).astype(np.float32, copy=False)
            block = _add_anchor_cost(
                block,
                query_anchors[q_start:q_stop],
                reference_anchors[r_start:r_stop],
                float(anchor_weight),
            )
            out[q_start:q_stop, r_start:r_stop] = block

    if hasattr(out, "flush"):
        out.flush()
    metadata = {
        "distance_mode": _canonical_distance_mode(debiased=debiased, use_fgw=use_fgw),
        "requested_distance_mode": str(requested_mode),
        "cross_distance": True,
        "n_query": int(n_query),
        "n_reference": int(n_reference),
        "matrix_shape": [int(n_query), int(n_reference)],
        "epsilon": float(epsilon),
        "sinkhorn_iters": int(n_iters),
        "anchor_weight": float(anchor_weight),
        "fgw_alpha": float(fgw_alpha) if use_fgw else None,
        "fgw_iters": int(fgw_iters) if use_fgw else None,
        "fgw_structure_normalization": str(fgw_structure_normalization)
        if use_fgw
        else None,
        "fgw_structure_cost_scale": float(fgw_structure_scale) if use_fgw else None,
        "fgw_structure_cost_scale_source": fgw_structure_scale_source
        if use_fgw
        else None,
        "query_self_cost_source": query_self_source,
        "reference_self_cost_source": reference_self_source,
        "block_size": int(bs),
        **block_metadata,
        "device": str(resolved_device),
        "matrix_bytes": int(_estimate_rectangular_bytes(n_query, n_reference)),
        "work_estimate": work_estimate,
        "fgw_work_estimate": fgw_work_estimate,
        "max_cross_ot_work_units": float(max_cross_ot_work_units),
        "max_cross_fgw_work_units": float(max_cross_fgw_work_units),
        "force_large_cross_ot": bool(force_large_cross_ot),
        "output_path": None if output_path is None else str(output_path),
        **fgw_feature_metadata,
    }
    return out, metadata


__all__ = [
    "assign_by_reference_medoids",
    "choose_pairwise_block_size",
    "compute_cross_ot_distance_matrix",
    "compute_pairwise_ot_distance_matrix",
    "estimate_cross_fgw_work",
    "estimate_cross_ot_work",
    "estimate_pairwise_fgw_work",
    "estimate_pairwise_ot_work",
]
