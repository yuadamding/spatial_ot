from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from ..config import ExperimentConfig
from .preprocessing import PreparedSpatialOTData


@dataclass
class CommunicationResult:
    program_names: list[str]
    incoming: np.ndarray
    outgoing: np.ndarray
    niche_flow_table: pd.DataFrame
    top_edges: pd.DataFrame
    residual_r2: dict[str, float]
    skipped: bool = False
    skip_reason: str | None = None


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, float]:
    denom = float(x.T @ x + alpha)
    if denom <= 0:
        return 0.0, 0.0
    beta = float((x.T @ y) / denom)
    pred = x * beta
    denom_r2 = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if denom_r2 <= 0 else 1.0 - float(np.sum((y - pred) ** 2)) / denom_r2
    return beta, r2


def _normalize_mass(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, None)
    x = x + 1e-6
    return x / x.sum()


def _masked_sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    mask: np.ndarray,
    reg: float,
    num_iter: int = 500,
) -> np.ndarray:
    active_rows = np.flatnonzero(mask.any(axis=1) & (a > 0))
    active_cols = np.flatnonzero(mask.any(axis=0) & (b > 0))
    if active_rows.size == 0 or active_cols.size == 0:
        return np.zeros_like(cost, dtype=np.float32)

    while True:
        mask_sub = mask[np.ix_(active_rows, active_cols)]
        keep_rows = mask_sub.any(axis=1)
        keep_cols = mask_sub.any(axis=0)
        if keep_rows.all() and keep_cols.all():
            break
        active_rows = active_rows[keep_rows]
        active_cols = active_cols[keep_cols]
        if active_rows.size == 0 or active_cols.size == 0:
            return np.zeros_like(cost, dtype=np.float32)

    a_sub = _normalize_mass(a[active_rows])
    b_sub = _normalize_mass(b[active_cols])
    cost_sub = cost[np.ix_(active_rows, active_cols)].astype(np.float64)
    mask_sub = mask[np.ix_(active_rows, active_cols)]
    kernel = np.exp(-cost_sub / max(reg, 1e-6)).astype(np.float64)
    kernel[~mask_sub] = 0.0
    if not np.any(kernel > 0):
        return np.zeros_like(cost, dtype=np.float32)
    kernel[kernel > 0] = np.maximum(kernel[kernel > 0], 1e-16)

    u = np.ones_like(a_sub, dtype=np.float64)
    v = np.ones_like(b_sub, dtype=np.float64)
    for _ in range(num_iter):
        Kv = kernel @ v
        Kv[Kv <= 1e-16] = 1e-16
        u = a_sub / Kv
        KTu = kernel.T @ u
        KTu[KTu <= 1e-16] = 1e-16
        v = b_sub / KTu

    transport_sub = (u[:, None] * kernel) * v[None, :]
    transport_sub[~mask_sub] = 0.0
    transport = np.zeros_like(cost, dtype=np.float32)
    transport[np.ix_(active_rows, active_cols)] = transport_sub.astype(np.float32)
    return transport


def fit_communication_flows(
    data: PreparedSpatialOTData,
    intrinsic_mu: np.ndarray,
    s: np.ndarray,
    niche_probs: np.ndarray,
    config: ExperimentConfig,
) -> CommunicationResult:
    program_library = data.program_library
    comm_indices = program_library.communication_program_indices
    if not comm_indices or program_library.n_programs == 0:
        return CommunicationResult(
            program_names=[],
            incoming=np.zeros((data.n_cells, 0), dtype=np.float32),
            outgoing=np.zeros((data.n_cells, 0), dtype=np.float32),
            niche_flow_table=pd.DataFrame(),
            top_edges=pd.DataFrame(),
            residual_r2={},
            skipped=True,
            skip_reason="No communication-capable prior programs were available.",
        )

    distances = pairwise_distances(data.cell_coords, metric="euclidean").astype(np.float32)
    within_radius = distances <= max(config.data.shell_bounds_um)
    np.fill_diagonal(within_radius, False)
    if not np.any(within_radius):
        return CommunicationResult(
            program_names=[],
            incoming=np.zeros((data.n_cells, 0), dtype=np.float32),
            outgoing=np.zeros((data.n_cells, 0), dtype=np.float32),
            niche_flow_table=pd.DataFrame(),
            top_edges=pd.DataFrame(),
            residual_r2={},
            skipped=True,
            skip_reason="No cell-cell pairs fell within the configured communication radius.",
        )
    costs = distances / (distances[within_radius].max() + 1e-6)
    costs[~within_radius] = 0.0

    incoming_columns: list[np.ndarray] = []
    outgoing_columns: list[np.ndarray] = []
    active_program_names: list[str] = []
    niche_rows = []
    edge_rows = []
    residual_scores: dict[str, float] = {}
    log_expr = np.log1p(data.cell_counts)

    for program_idx in comm_indices:
        program = program_library.programs[program_idx]
        activity = np.maximum(s[:, program_idx], 0.0)
        source_mask = program_library.source_mask[program_idx] > 0
        receiver_mask = program_library.receiver_target_mask[program_idx] > 0
        if not source_mask.any() or not receiver_mask.any():
            continue
        source_signal = log_expr[:, source_mask].mean(axis=1)
        receiver_signal = log_expr[:, receiver_mask].mean(axis=1)
        source_mass = activity * source_signal
        receiver_mass = activity * receiver_signal
        total_mass = min(float(source_mass.sum()), float(receiver_mass.sum()))
        if total_mass <= 0:
            continue
        a = _normalize_mass(source_mass)
        b = _normalize_mass(receiver_mass)
        transport = _masked_sinkhorn(
            a=a,
            b=b,
            cost=costs,
            mask=within_radius,
            reg=config.loss.comm_epsilon,
            num_iter=500,
        ).astype(np.float32)
        transport *= total_mass
        outgoing_columns.append(transport.sum(axis=1).astype(np.float32))
        incoming_columns.append(transport.sum(axis=0).astype(np.float32))
        active_program_names.append(program.name)

        residual = (data.cell_counts[:, receiver_mask] - intrinsic_mu[:, receiver_mask]).mean(axis=1)
        _, r2 = _ridge_fit(incoming_columns[-1], residual, alpha=config.loss.residual_ridge)
        residual_scores[program.name] = r2

        niche_flow = niche_probs.T @ transport @ niche_probs
        for sender_niche in range(niche_flow.shape[0]):
            for receiver_niche in range(niche_flow.shape[1]):
                niche_rows.append(
                    {
                        "program": program.name,
                        "sender_niche": int(sender_niche),
                        "receiver_niche": int(receiver_niche),
                        "flow": float(niche_flow[sender_niche, receiver_niche]),
                    }
                )

        if np.any(transport > 0):
            threshold = np.quantile(transport[transport > 0], 0.995)
            sender_idx, receiver_idx = np.nonzero(transport > threshold)
            flow_values = transport[sender_idx, receiver_idx]
            order = np.argsort(-flow_values)
            for edge_idx in order[:200]:
                s_idx = int(sender_idx[edge_idx])
                r_idx = int(receiver_idx[edge_idx])
                edge_rows.append(
                    {
                        "program": program.name,
                        "sender_id": data.cell_ids[s_idx],
                        "receiver_id": data.cell_ids[r_idx],
                        "sender_index": s_idx,
                        "receiver_index": r_idx,
                        "distance_um": float(distances[s_idx, r_idx]),
                        "flow": float(transport[s_idx, r_idx]),
                    }
                )

    incoming = np.column_stack(incoming_columns).astype(np.float32) if incoming_columns else np.zeros((data.n_cells, 0), dtype=np.float32)
    outgoing = np.column_stack(outgoing_columns).astype(np.float32) if outgoing_columns else np.zeros((data.n_cells, 0), dtype=np.float32)
    skipped = len(active_program_names) == 0
    top_edges = pd.DataFrame(edge_rows)
    if not top_edges.empty:
        top_edges = top_edges.sort_values(
            by=["flow", "program", "sender_index", "receiver_index"],
            ascending=[False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return CommunicationResult(
        program_names=active_program_names,
        incoming=incoming,
        outgoing=outgoing,
        niche_flow_table=pd.DataFrame(niche_rows),
        top_edges=top_edges,
        residual_r2=residual_scores,
        skipped=skipped,
        skip_reason="No active communication programs after gene-panel filtering." if skipped else None,
    )
