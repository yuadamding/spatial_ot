from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances

from .communication import fit_communication_flows
from .config import ExperimentConfig
from .nn import (
    StudentSpatialModel,
    TeacherContextModel,
    aggregate_mean,
    cross_covariance_penalty,
    edge_bce_loss,
    kl_normal,
    nb_nll,
    sample_negative_edges,
)
from .ot import fit_niche_prototypes
from .preprocessing import PreparedSpatialOTData, prepare_data


def _device_from_config(config: ExperimentConfig) -> torch.device:
    if config.training.device != "auto":
        resolved = torch.device(config.training.device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for legacy training, but torch.cuda.is_available() is False.")
        return resolved
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def _permute_features(x: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction <= 0:
        return x
    n_genes = x.size(1)
    n_swap = max(1, int(round(n_genes * fraction)))
    cols = torch.randperm(n_genes, device=x.device)[:n_swap]
    rows = torch.randperm(x.size(0), device=x.device)
    x_perm = x.clone()
    x_perm[:, cols] = x[rows][:, cols]
    return x_perm


def _stack_shells(shells: tuple[torch.Tensor, ...], start: int, stop: int, n_nodes: int, fallback: torch.Tensor) -> torch.Tensor:
    selected = [shell for shell in shells[start:stop] if shell.numel() > 0]
    if not selected:
        return aggregate_mean(fallback, torch.empty((2, 0), dtype=torch.long, device=fallback.device))
    merged = torch.cat(selected, dim=1)
    return aggregate_mean(fallback, merged)


def _teacher_targets_to_cells(overlap: np.ndarray, teacher_u: np.ndarray, teacher_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return overlap @ teacher_u, overlap @ teacher_q


def _teacher_cluster_probs(teacher_u: np.ndarray, n_clusters: int, temperature: float, seed: int) -> np.ndarray:
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=min(2048, len(teacher_u)))
    centers = kmeans.fit(teacher_u).cluster_centers_.astype(np.float32)
    distances = pairwise_distances(teacher_u, centers, metric="euclidean").astype(np.float32)
    scaled = -distances / max(temperature, 1e-4)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
    return probs.astype(np.float32)


def _novo_l1_penalty(model: StudentSpatialModel) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for decoder in [model.self_program_decoder, model.nb_program_decoder]:
        if decoder.novo_weight is not None:
            penalty = penalty + decoder.novo_weight.abs().mean()
    return penalty


def _build_student(
    data: PreparedSpatialOTData,
    config: ExperimentConfig,
    device: torch.device,
) -> StudentSpatialModel:
    model = StudentSpatialModel(
        n_genes=data.n_genes,
        aux_dim=data.cell_aux.shape[1],
        batch_dim=data.batch_onehot.shape[1],
        teacher_dim=config.model.teacher_dim,
        z_dim=config.model.z_dim,
        prior_dim=data.program_library.n_programs,
        novo_dim=config.model.de_novo_programs,
        hidden_dim=config.model.hidden_dim,
        teacher_logit_dim=config.model.niche_prototypes,
        n_cell_types=len(data.cell_type_names),
        self_mask=data.program_library.self_mask if data.program_library.n_programs > 0 else None,
        neighborhood_mask=data.program_library.neighborhood_mask if data.program_library.n_programs > 0 else None,
        dropout=config.model.dropout,
    )
    return model.to(device)


def _validate_prepared_data(data: PreparedSpatialOTData, config: ExperimentConfig) -> None:
    if data.n_cells < 2:
        raise ValueError("Prepared dataset must contain at least 2 cells.")
    if data.n_bins < 1:
        raise ValueError("Prepared dataset must contain at least 1 teacher bin.")
    if data.n_cells < config.model.state_atoms:
        raise ValueError(
            f"Prepared dataset has {data.n_cells} cells, but state_atoms={config.model.state_atoms}. "
            "Reduce state_atoms or increase the cell subset."
        )
    if data.n_cells < config.model.niche_prototypes:
        raise ValueError(
            f"Prepared dataset has {data.n_cells} cells, but niche_prototypes={config.model.niche_prototypes}. "
            "Reduce niche_prototypes or increase the cell subset."
        )
    if data.n_bins < config.model.niche_prototypes:
        raise ValueError(
            f"Prepared dataset has {data.n_bins} teacher bins, but niche_prototypes={config.model.niche_prototypes}. "
            "Reduce niche_prototypes or increase the bin subset."
        )


def train_teacher(data: PreparedSpatialOTData, config: ExperimentConfig, device: torch.device, checkpoint_dir: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    model = TeacherContextModel(
        input_dim=data.n_genes,
        hidden_dim=config.model.hidden_dim,
        teacher_dim=config.model.teacher_dim,
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    x = _tensor(data.bin_log_counts, device)
    x_counts = _tensor(data.bin_counts, device)
    x_nb_counts = _tensor(data.bin_neighbor_target, device)
    x_nb_log = torch.log1p(x_nb_counts)
    library = _tensor(data.bin_library_panel, device)
    nb_library = x_nb_counts.sum(dim=1, keepdim=True) + 1e-4
    edge_index = torch.as_tensor(data.teacher_edge_index, dtype=torch.long, device=device)
    metrics = []

    for epoch in range(config.training.teacher_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, x_nb_log, library, nb_library)
        neg_edge = sample_negative_edges(data.n_bins, edge_index, config.data.negative_edge_ratio)
        loss_self = nb_nll(x_counts, out["mu_self"], out["theta_self"])
        loss_nb = nb_nll(x_nb_counts, out["mu_nb"], out["theta_nb"])
        loss_edge = edge_bce_loss(out["u"], edge_index, neg_edge)
        loss = (
            config.loss.teacher_self * loss_self
            + config.loss.teacher_nb * loss_nb
            + config.loss.teacher_edge * loss_edge
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()
        metrics.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "loss_self": float(loss_self.item()),
                "loss_nb": float(loss_nb.item()),
                "loss_edge": float(loss_edge.item()),
            }
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "teacher_model.pt")
    model.eval()
    with torch.no_grad():
        out = model(x, x_nb_log, library, nb_library)
    teacher_u = out["u"].cpu().numpy()
    teacher_q = _teacher_cluster_probs(
        teacher_u=teacher_u,
        n_clusters=config.model.niche_prototypes,
        temperature=config.loss.teacher_temperature,
        seed=config.data.seed,
    )
    return teacher_u, teacher_q, metrics


def train_student(
    data: PreparedSpatialOTData,
    config: ExperimentConfig,
    teacher_u: np.ndarray,
    teacher_q: np.ndarray,
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[StudentSpatialModel, dict]:
    model = _build_student(data, config, device)
    x_log = _tensor(data.cell_log_counts, device)
    x_counts = _tensor(data.cell_counts, device)
    x_nb_counts = _tensor(data.cell_neighbor_target, device)
    library = _tensor(data.cell_library_panel, device)
    nb_library = x_nb_counts.sum(dim=1, keepdim=True) + 1e-4
    aux = _tensor(data.cell_aux, device)
    batch = _tensor(data.batch_onehot, device)
    cell_type = torch.as_tensor(data.cell_type_labels, dtype=torch.long, device=device)
    teacher_u_cell = _tensor(_teacher_targets_to_cells(data.teacher_overlap, teacher_u, teacher_q)[0], device)
    teacher_q_cell = _tensor(_teacher_targets_to_cells(data.teacher_overlap, teacher_u, teacher_q)[1], device)
    composition = _tensor(data.cell_type_onehot, device)
    marker_scores = _tensor(data.program_scores, device) if data.program_scores.shape[1] > 0 else torch.zeros((data.n_cells, 0), device=device)
    context_edge = torch.as_tensor(data.context_edge_index, dtype=torch.long, device=device)
    shells = tuple(torch.as_tensor(shell, dtype=torch.long, device=device) for shell in data.shell_edge_indices)

    intrinsic_params = list(model.intrinsic_encoder.parameters()) + list(model.intrinsic_decoder.parameters()) + list(model.func_head.parameters())
    context_params = (
        list(model.view_fusion.parameters())
        + list(model.context_encoder.parameters())
        + list(model.gate.parameters())
        + list(model.teacher_proj.parameters())
        + list(model.teacher_logits.parameters())
        + list(model.self_program_decoder.parameters())
        + list(model.nb_program_decoder.parameters())
    )
    intrinsic_optimizer = torch.optim.AdamW(intrinsic_params, lr=config.training.lr, weight_decay=config.training.weight_decay)
    context_optimizer = torch.optim.AdamW(context_params, lr=config.training.context_lr, weight_decay=config.training.weight_decay)
    metrics = {"intrinsic": [], "context": []}

    for epoch in range(config.training.intrinsic_epochs):
        model.train()
        intrinsic_optimizer.zero_grad()
        x_input = _permute_features(x_log, config.training.permutation_fraction)
        z, mu_z, logvar_z = model.encode_intrinsic(x_input, aux, batch)
        mu_x, theta_x = model.decode_intrinsic(z, library)
        logits = model.func_head(z)
        rec = nb_nll(x_counts, mu_x, theta_x)
        kl = kl_normal(mu_z, logvar_z)
        cls = F.cross_entropy(logits, cell_type)
        loss = config.loss.intrinsic_rec * rec + config.loss.kl_z * kl + config.loss.marker * cls
        loss.backward()
        torch.nn.utils.clip_grad_norm_(intrinsic_params, config.training.grad_clip)
        intrinsic_optimizer.step()
        metrics["intrinsic"].append(
            {"epoch": epoch + 1, "loss": float(loss.item()), "rec": float(rec.item()), "kl": float(kl.item()), "cls": float(cls.item())}
        )

    for epoch in range(config.training.context_epochs):
        model.train()

        intrinsic_optimizer.zero_grad()
        x_input = _permute_features(x_log, config.training.permutation_fraction)
        z, mu_z, logvar_z = model.encode_intrinsic(x_input, aux, batch)
        mu_x, theta_x = model.decode_intrinsic(z, library)
        cls = F.cross_entropy(model.func_head(z), cell_type)
        loss_intr = (
            config.loss.intrinsic_rec * nb_nll(x_counts, mu_x, theta_x)
            + config.loss.kl_z * kl_normal(mu_z, logvar_z)
            + config.loss.marker * cls
        )
        loss_intr.backward()
        torch.nn.utils.clip_grad_norm_(intrinsic_params, config.training.grad_clip)
        intrinsic_optimizer.step()

        context_optimizer.zero_grad()
        with torch.no_grad():
            z_detached, _, _ = model.encode_intrinsic(x_log, aux, batch)
        short_view = _stack_shells(shells, 0, 1, data.n_cells, z_detached)
        mid_view = _stack_shells(shells, 1, len(shells), data.n_cells, z_detached)
        comp_view = aggregate_mean(composition, context_edge)
        marker_view = aggregate_mean(marker_scores, context_edge) if marker_scores.size(1) > 0 else marker_scores
        s, mu_s, logvar_s, weights = model.encode_context(z_detached, short_view, mid_view, comp_view, marker_view, teacher_u_cell, aux)
        dec = model.decode_programs(z_detached, s, library, nb_library)
        teach_emb, teach_logits = model.teacher_targets(s)
        emb = model.embedding(z_detached, s)
        neg_edge = sample_negative_edges(data.n_cells, context_edge, config.data.negative_edge_ratio)
        teacher_q_target = torch.clamp(teacher_q_cell, min=1e-8)
        teacher_q_target = teacher_q_target / teacher_q_target.sum(dim=1, keepdim=True)
        kl_teacher = F.kl_div(F.log_softmax(teach_logits, dim=1), teacher_q_target, reduction="batchmean")
        independence = cross_covariance_penalty(z_detached, s)
        loss_ctx = (
            config.loss.context_self * nb_nll(x_counts, dec["mu_self"], dec["theta_self"])
            + config.loss.context_nb * nb_nll(x_nb_counts, dec["mu_nb"], dec["theta_nb"])
            + config.loss.kl_s * kl_normal(mu_s, logvar_s)
            + config.loss.teacher_distill * F.mse_loss(teach_emb, teacher_u_cell)
            + config.loss.teacher_logits * kl_teacher
            + config.loss.independence * independence
            + config.loss.sparsity * _novo_l1_penalty(model)
            + config.loss.edge * edge_bce_loss(emb, context_edge, neg_edge)
        )
        loss_ctx.backward()
        torch.nn.utils.clip_grad_norm_(context_params, config.training.grad_clip)
        context_optimizer.step()
        metrics["context"].append(
            {
                "epoch": epoch + 1,
                "loss_intrinsic": float(loss_intr.item()),
                "loss_context": float(loss_ctx.item()),
                "view_entropy": float((-(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()).item()),
            }
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "student_model.pt")
    return model, metrics


def _final_student_outputs(
    model: StudentSpatialModel,
    data: PreparedSpatialOTData,
    teacher_u: np.ndarray,
    teacher_q: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    x_log = _tensor(data.cell_log_counts, device)
    library = _tensor(data.cell_library_panel, device)
    nb_counts = _tensor(data.cell_neighbor_target, device)
    nb_library = nb_counts.sum(dim=1, keepdim=True) + 1e-4
    aux = _tensor(data.cell_aux, device)
    batch = _tensor(data.batch_onehot, device)
    composition = _tensor(data.cell_type_onehot, device)
    marker_scores = _tensor(data.program_scores, device) if data.program_scores.shape[1] > 0 else torch.zeros((data.n_cells, 0), device=device)
    context_edge = torch.as_tensor(data.context_edge_index, dtype=torch.long, device=device)
    shells = tuple(torch.as_tensor(shell, dtype=torch.long, device=device) for shell in data.shell_edge_indices)
    teacher_u_cell, teacher_q_cell = _teacher_targets_to_cells(data.teacher_overlap, teacher_u, teacher_q)
    teacher_u_cell = _tensor(teacher_u_cell, device)
    teacher_q_cell = _tensor(teacher_q_cell, device)

    with torch.no_grad():
        _, mu_z, logvar_z = model.encode_intrinsic(x_log, aux, batch)
        z = mu_z
        short_view = _stack_shells(shells, 0, 1, data.n_cells, z)
        mid_view = _stack_shells(shells, 1, len(shells), data.n_cells, z)
        comp_view = aggregate_mean(composition, context_edge)
        marker_view = aggregate_mean(marker_scores, context_edge) if marker_scores.size(1) > 0 else marker_scores
        _, mu_s, logvar_s, view_weights = model.encode_context(z, short_view, mid_view, comp_view, marker_view, teacher_u_cell, aux)
        s = mu_s
        intrinsic_mu, _ = model.decode_intrinsic(z, library)
        program_mu = model.decode_programs(z, s, library, nb_library)
        teacher_proj, teacher_logits = model.teacher_targets(s)
    return {
        "z": z.cpu().numpy(),
        "z_mu": mu_z.cpu().numpy(),
        "z_logvar": logvar_z.cpu().numpy(),
        "s": s.cpu().numpy(),
        "s_mu": mu_s.cpu().numpy(),
        "s_logvar": logvar_s.cpu().numpy(),
        "view_weights": view_weights.cpu().numpy(),
        "intrinsic_mu": intrinsic_mu.cpu().numpy(),
        "program_mu_self": program_mu["mu_self"].cpu().numpy(),
        "program_mu_nb": program_mu["mu_nb"].cpu().numpy(),
        "teacher_proj": teacher_proj.cpu().numpy(),
        "teacher_logits": teacher_logits.cpu().numpy(),
    }


def _save_outputs(
    data: PreparedSpatialOTData,
    teacher_u: np.ndarray,
    teacher_q: np.ndarray,
    student_outputs: dict[str, np.ndarray],
    niche_result,
    communication_result,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: Path,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    print("[spatial_ot] saving cells_output.h5ad", flush=True)
    cells_out = data.cell_adata.copy()
    cells_out.obsm["spatial"] = data.cell_coords / max(data.microns_per_pixel, 1e-8)
    cells_out.obsm["z_intrinsic"] = student_outputs["z"].astype(np.float32)
    cells_out.obsm["s_program"] = student_outputs["s"].astype(np.float32)
    cells_out.obsm["p_niche"] = niche_result.niche_probs.astype(np.float32)
    cells_out.obsm["state_probs"] = niche_result.state_probs.astype(np.float32)
    cells_out.obsm["view_weights"] = student_outputs["view_weights"].astype(np.float32)
    cells_out.obs["niche_label"] = pd.Categorical(niche_result.niche_labels.astype(str))
    if communication_result.incoming.shape[1] > 0:
        cells_out.obsm["comm_incoming"] = communication_result.incoming.astype(np.float32)
        cells_out.obsm["comm_outgoing"] = communication_result.outgoing.astype(np.float32)
    cells_out.uns["spatial_ot"] = {
        "summary_json": json.dumps(summary),
        "program_names": [str(name) for name in communication_result.program_names],
        "view_names": [
            "short_shell",
            "mid_shells",
            "annotation_composition",
            "program_score_view",
            "teacher_context",
            "aux_features",
        ],
        "gene_panel_report_json": json.dumps(data.gene_panel_report),
        "teacher_overlap_report_json": json.dumps(data.teacher_overlap_report),
        "graph_report_json": json.dumps(data.graph_report),
    }
    cells_out.write_h5ad(output_dir / "cells_output.h5ad")

    print("[spatial_ot] saving teacher_bins_output.h5ad", flush=True)
    teacher_out = data.bin_adata.copy()
    teacher_out.obsm["teacher_u"] = teacher_u.astype(np.float32)
    teacher_out.obsm["teacher_q"] = teacher_q.astype(np.float32)
    teacher_out.write_h5ad(output_dir / "teacher_bins_output.h5ad")

    print("[spatial_ot] saving niche and flow tables", flush=True)
    communication_result.niche_flow_table.to_csv(output_dir / "niche_flows.csv", index=False)
    if not communication_result.top_edges.empty:
        communication_result.top_edges.to_parquet(output_dir / "top_flow_edges.parquet", index=False)
    np.savez_compressed(
        output_dir / "niche_artifacts.npz",
        state_centroids=niche_result.state_centroids,
        niche_probs=niche_result.niche_probs,
        prototype_hist=niche_result.prototype_hist,
        prototype_cov=niche_result.prototype_cov,
        prototype_shell=niche_result.prototype_shell,
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[spatial_ot] output save complete", flush=True)


def run_experiment(config: ExperimentConfig) -> dict:
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.write_resolved(output_dir / "resolved_config.json")
    device = _device_from_config(config)
    print(f"[spatial_ot] preparing data on {device}", flush=True)
    data = prepare_data(config)
    _validate_prepared_data(data, config)
    checkpoint_dir = output_dir / "checkpoints"

    print("[spatial_ot] stage A: teacher pretraining", flush=True)
    teacher_u, teacher_q, teacher_metrics = train_teacher(data, config, device, checkpoint_dir)
    print("[spatial_ot] stage B/C: student training", flush=True)
    student_model, student_metrics = train_student(data, config, teacher_u, teacher_q, device, checkpoint_dir)
    print("[spatial_ot] stage D: OT niche fitting", flush=True)
    student_outputs = _final_student_outputs(student_model, data, teacher_u, teacher_q, device)
    niche_result = fit_niche_prototypes(data, student_outputs["z"], student_outputs["s"], config)
    print("[spatial_ot] stage E: communication fitting", flush=True)
    communication_result = fit_communication_flows(
        data=data,
        intrinsic_mu=student_outputs["intrinsic_mu"],
        s=student_outputs["s"],
        niche_probs=niche_result.niche_probs,
        config=config,
    )
    print("[spatial_ot] writing outputs", flush=True)

    summary = {
        "device": str(device),
        "n_cells": data.n_cells,
        "n_bins8": data.n_bins,
        "n_genes": data.n_genes,
        "count_sources": {
            "cells": data.cell_count_source,
            "bins8": data.bin_count_source,
            "library_semantics": {
                "cell_library_panel": "sum over selected gene panel",
                "bin_library_panel": "sum over selected gene panel",
                "cell_library_full": "sum over validated raw-count matrix before gene-panel restriction",
                "bin_library_full": "sum over validated raw-count matrix before gene-panel restriction",
            },
        },
        "gene_panel_report": data.gene_panel_report,
        "teacher_overlap_report": data.teacher_overlap_report,
        "graph_report": data.graph_report,
        "n_programs_prior": data.program_library.n_programs,
        "n_programs_novo": config.model.de_novo_programs,
        "teacher_metrics": teacher_metrics,
        "student_metrics": student_metrics,
        "niche_counts": {str(k): int(v) for k, v in pd.Series(niche_result.niche_labels).value_counts().sort_index().items()},
        "communication_skipped": communication_result.skipped,
        "communication_reason": communication_result.skip_reason,
        "communication_residual_r2": communication_result.residual_r2,
    }
    _save_outputs(
        data=data,
        teacher_u=teacher_u,
        teacher_q=teacher_q,
        student_outputs=student_outputs,
        niche_result=niche_result,
        communication_result=communication_result,
        config=config,
        device=device,
        output_dir=output_dir,
        summary=summary,
    )
    return summary
