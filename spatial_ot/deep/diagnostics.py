from __future__ import annotations

import numpy as np


def correlation_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float | bool]:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return {
            "mean_abs_correlation": 0.0,
            "max_abs_correlation": 0.0,
            "fro_correlation": 0.0,
            "mean_cosine_similarity": 0.0,
            "allclose": bool(np.allclose(a, b)),
        }
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    a_scale = np.sqrt(np.var(a_centered, axis=0, keepdims=True) + 1e-4)
    b_scale = np.sqrt(np.var(b_centered, axis=0, keepdims=True) + 1e-4)
    corr = ((a_centered / a_scale).T @ (b_centered / b_scale)) / max(a.shape[0], 1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    cosine = np.sum(a * b, axis=1) / np.maximum(a_norm * b_norm, 1e-6)
    return {
        "mean_abs_correlation": float(np.mean(np.abs(corr))),
        "max_abs_correlation": float(np.max(np.abs(corr))),
        "fro_correlation": float(np.linalg.norm(corr)),
        "mean_cosine_similarity": float(np.mean(cosine)),
        "allclose": bool(np.allclose(a, b, atol=1e-5, rtol=1e-4)),
    }


def linear_r2(source: np.ndarray, target: np.ndarray) -> float:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.ndim == 1:
        source = source[:, None]
    if target.ndim == 1:
        target = target[:, None]
    if source.shape[0] < 2 or source.shape[1] == 0 or target.shape[1] == 0:
        return 0.0
    design = np.concatenate([source, np.ones((source.shape[0], 1), dtype=np.float32)], axis=1).astype(np.float64, copy=False)
    target64 = target.astype(np.float64, copy=False)
    coef, *_ = np.linalg.lstsq(design, target64, rcond=None)
    pred = design @ coef
    residual = np.mean((target64 - pred) ** 2)
    baseline = np.mean((target64 - target64.mean(axis=0, keepdims=True)) ** 2)
    if not np.isfinite(residual) or baseline <= 1e-8:
        return 0.0
    return float(np.clip(1.0 - residual / baseline, -1.0, 1.0))


def top_canonical_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    if a.shape[0] < 2 or a.shape[1] == 0 or b.shape[1] == 0:
        return 0.0
    a0 = a - a.mean(axis=0, keepdims=True)
    b0 = b - b.mean(axis=0, keepdims=True)
    denom = max(a.shape[0] - 1, 1)
    cov_aa = (a0.T @ a0) / denom
    cov_bb = (b0.T @ b0) / denom
    cov_ab = (a0.T @ b0) / denom
    eps_a = 1e-4 * np.eye(cov_aa.shape[0], dtype=np.float64)
    eps_b = 1e-4 * np.eye(cov_bb.shape[0], dtype=np.float64)
    evals_a, evecs_a = np.linalg.eigh(cov_aa + eps_a)
    evals_b, evecs_b = np.linalg.eigh(cov_bb + eps_b)
    inv_sqrt_a = evecs_a @ np.diag(1.0 / np.sqrt(np.maximum(evals_a, 1e-8))) @ evecs_a.T
    inv_sqrt_b = evecs_b @ np.diag(1.0 / np.sqrt(np.maximum(evals_b, 1e-8))) @ evecs_b.T
    whitened = inv_sqrt_a @ cov_ab @ inv_sqrt_b
    singular_values = np.linalg.svd(whitened, compute_uv=False)
    if singular_values.size == 0:
        return 0.0
    return float(np.clip(singular_values[0], 0.0, 1.0))


def latent_diagnostics(
    outputs: dict[str, np.ndarray],
    *,
    x_std: np.ndarray,
    context_std: np.ndarray,
    coords_um: np.ndarray,
    selected_embedding: str,
) -> dict[str, object]:
    intrinsic = np.asarray(outputs["intrinsic"], dtype=np.float32)
    context = np.asarray(outputs["context"], dtype=np.float32)
    joint = np.asarray(outputs["joint"], dtype=np.float32)
    selected = np.asarray(outputs[selected_embedding], dtype=np.float32)
    ic = correlation_summary(intrinsic, context)
    ij = correlation_summary(intrinsic, joint)
    cj = correlation_summary(context, joint)
    intrinsic_input_r2 = linear_r2(intrinsic, x_std)
    context_input_r2 = linear_r2(context, x_std)
    joint_input_r2 = linear_r2(joint, x_std)
    intrinsic_context_r2 = linear_r2(intrinsic, context_std)
    context_context_r2 = linear_r2(context, context_std)
    joint_context_r2 = linear_r2(joint, context_std)
    intrinsic_coord_r2 = linear_r2(intrinsic, coords_um)
    context_coord_r2 = linear_r2(context, coords_um)
    joint_coord_r2 = linear_r2(joint, coords_um)
    return {
        "selected_embedding": selected_embedding,
        "selected_mean_norm": float(np.mean(np.linalg.norm(selected, axis=1))),
        "intrinsic_reconstruction_mse": float(np.mean((np.asarray(outputs["recon"], dtype=np.float32) - x_std) ** 2)),
        "context_prediction_mse": float(np.mean((np.asarray(outputs["context_pred"], dtype=np.float32) - context_std) ** 2)),
        "intrinsic_context_mean_abs_correlation": float(ic["mean_abs_correlation"]),
        "intrinsic_context_max_abs_correlation": float(ic["max_abs_correlation"]),
        "intrinsic_context_fro_correlation": float(ic["fro_correlation"]),
        "intrinsic_context_mean_cosine_similarity": float(ic["mean_cosine_similarity"]),
        "intrinsic_context_allclose": bool(ic["allclose"]),
        "intrinsic_context_top_canonical_correlation": top_canonical_correlation(intrinsic, context),
        "intrinsic_joint_mean_abs_correlation": float(ij["mean_abs_correlation"]),
        "intrinsic_joint_max_abs_correlation": float(ij["max_abs_correlation"]),
        "intrinsic_joint_allclose": bool(ij["allclose"]),
        "intrinsic_joint_top_canonical_correlation": top_canonical_correlation(intrinsic, joint),
        "context_joint_mean_abs_correlation": float(cj["mean_abs_correlation"]),
        "context_joint_max_abs_correlation": float(cj["max_abs_correlation"]),
        "context_joint_allclose": bool(cj["allclose"]),
        "context_joint_top_canonical_correlation": top_canonical_correlation(context, joint),
        "intrinsic_input_r2": intrinsic_input_r2,
        "context_input_r2": context_input_r2,
        "joint_input_r2": joint_input_r2,
        "intrinsic_context_target_r2": intrinsic_context_r2,
        "context_context_target_r2": context_context_r2,
        "joint_context_target_r2": joint_context_r2,
        "intrinsic_coordinate_r2": intrinsic_coord_r2,
        "context_coordinate_r2": context_coord_r2,
        "joint_coordinate_r2": joint_coord_r2,
        "intrinsic_minus_context_input_r2": float(intrinsic_input_r2 - context_input_r2),
        "context_minus_intrinsic_context_target_r2": float(context_context_r2 - intrinsic_context_r2),
    }


def graph_summary(edge_index, n_nodes: int) -> dict[str, float | int]:
    import torch  # local import keeps this module importable without GPU dependencies until use

    if n_nodes <= 0:
        return {
            "edges": 0,
            "mean_degree": 0.0,
            "max_degree": 0,
            "isolated_fraction": 0.0,
        }
    if edge_index.numel() == 0:
        return {
            "edges": 0,
            "mean_degree": 0.0,
            "max_degree": 0,
            "isolated_fraction": 1.0,
        }
    dst = edge_index[1]
    deg = torch.bincount(dst, minlength=n_nodes)
    return {
        "edges": int(dst.numel()),
        "mean_degree": float(deg.float().mean().detach().cpu()),
        "max_degree": int(deg.max().detach().cpu()),
        "isolated_fraction": float((deg == 0).float().mean().detach().cpu()),
    }
