from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot
import pandas as pd
from sklearn.metrics import pairwise_distances

from .config import ExperimentConfig
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
    costs = distances / (distances[within_radius].max() + 1e-6)
    costs[~within_radius] = 25.0

    hard_niche = niche_probs.argmax(axis=1)
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
        transport = ot.sinkhorn(
            a,
            b,
            costs,
            reg=config.loss.comm_epsilon,
            numItermax=500,
            warn=False,
        ).astype(np.float32)
        transport *= total_mass
        transport[~within_radius] = 0.0
        outgoing_columns.append(transport.sum(axis=1).astype(np.float32))
        incoming_columns.append(transport.sum(axis=0).astype(np.float32))
        active_program_names.append(program.name)

        residual = (data.cell_counts[:, receiver_mask] - intrinsic_mu[:, receiver_mask]).mean(axis=1)
        _, r2 = _ridge_fit(incoming_columns[-1], residual, alpha=config.loss.residual_ridge)
        residual_scores[program.name] = r2

        for sender_niche in np.unique(hard_niche):
            sender_mask = hard_niche == sender_niche
            for receiver_niche in np.unique(hard_niche):
                receiver_mask_niche = hard_niche == receiver_niche
                niche_rows.append(
                    {
                        "program": program.name,
                        "sender_niche": int(sender_niche),
                        "receiver_niche": int(receiver_niche),
                        "flow": float(transport[np.ix_(sender_mask, receiver_mask_niche)].sum()),
                    }
                )

        top_indices = np.argwhere(transport > np.quantile(transport[transport > 0], 0.995) if np.any(transport > 0) else np.inf)
        for sender_idx, receiver_idx in top_indices[:200]:
            edge_rows.append(
                {
                    "program": program.name,
                    "sender_id": data.cell_ids[sender_idx],
                    "receiver_id": data.cell_ids[receiver_idx],
                    "sender_index": int(sender_idx),
                    "receiver_index": int(receiver_idx),
                    "distance_um": float(distances[sender_idx, receiver_idx]),
                    "flow": float(transport[sender_idx, receiver_idx]),
                }
            )

    incoming = np.column_stack(incoming_columns).astype(np.float32) if incoming_columns else np.zeros((data.n_cells, 0), dtype=np.float32)
    outgoing = np.column_stack(outgoing_columns).astype(np.float32) if outgoing_columns else np.zeros((data.n_cells, 0), dtype=np.float32)
    skipped = len(active_program_names) == 0
    return CommunicationResult(
        program_names=active_program_names,
        incoming=incoming,
        outgoing=outgoing,
        niche_flow_table=pd.DataFrame(niche_rows),
        top_edges=pd.DataFrame(edge_rows),
        residual_r2=residual_scores,
        skipped=skipped,
        skip_reason="No active communication programs after gene-panel filtering." if skipped else None,
    )
