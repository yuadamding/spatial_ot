from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import warnings

import numpy as np
import ot
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances
import torch

from .geometry import (
    _normalize_hist,
    _softmax_over_negative_costs,
    _standardize_features,
    _validate_fit_inputs,
    build_subregions,
    build_composite_subregions_from_basic_niches,
    fit_ot_shape_normalizer,
    make_reference_points_unit_disk,
    sample_geometry_points,
    _region_geometries_from_members,
)
from .gpu_ot import sinkhorn_semirelaxed_unbalanced_log_torch
from .types import MultilevelOTResult, OTSolveDiagnostics, OptimizationMeasure, RegionGeometry, SubregionMeasure


_RESTART_WORKER_MEASURES: list[OptimizationMeasure] | None = None
_RESTART_WORKER_SUMMARIES: np.ndarray | None = None
_RESTART_WORKER_PARAMS: dict[str, object] | None = None


def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
    new = np.asarray(new, dtype=np.float64)
    old = np.asarray(old, dtype=np.float64)
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(old), 1e-12))


def _resolve_compute_device(device: str) -> torch.device:
    requested = str(device).strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA compute was requested for multilevel OT, but torch.cuda.is_available() is False.")
    return resolved


def _runtime_memory_snapshot(device: torch.device) -> dict[str, float | int | bool | str]:
    snapshot: dict[str, float | int | bool | str] = {
        "device": str(device),
        "cuda": bool(device.type == "cuda" and torch.cuda.is_available()),
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        return snapshot
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        free_bytes, total_bytes = 0, 0
    snapshot.update(
        {
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "memory_free_bytes": int(free_bytes),
            "memory_total_bytes": int(total_bytes),
        }
    )
    return snapshot


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return int(default) if value <= 0 else value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return float(default) if value <= 0 else float(value)


def _cuda_target_vram_gb() -> float:
    return _env_float("SPATIAL_OT_CUDA_TARGET_VRAM_GB", 50.0)


def _cuda_target_bytes(device: torch.device | None = None) -> int:
    requested = int(_cuda_target_vram_gb() * (1024**3))
    if not torch.cuda.is_available():
        return requested
    try:
        dev = device or torch.device("cuda")
        total_bytes = int(torch.cuda.get_device_properties(dev).total_memory)
    except Exception:
        return requested
    safe_bytes = int(max(total_bytes * 0.8, 1 << 30))
    return min(requested, safe_bytes)


def _cuda_cdist_row_batch_size(
    *,
    n_rows: int,
    n_cols: int,
    device: torch.device,
    per_pair_buffers: int = 3,
    min_batch: int = 256,
) -> int:
    if n_rows <= 0:
        return 1
    bytes_per = 4
    denom = max(int(per_pair_buffers) * max(int(n_cols), 1) * bytes_per, 1)
    batch = _cuda_target_bytes(device=device) // denom
    batch = max(int(min_batch), min(int(n_rows), int(batch)))
    return max(batch, 1)


def _resolve_cuda_device_pool(requested: str, n_init: int) -> list[str]:
    normalized = str(requested).strip()
    if int(n_init) <= 1:
        return [normalized]
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if not normalized.startswith("cuda"):
        return [normalized]
    if not torch.cuda.is_available():
        return [normalized]

    explicit = os.environ.get("SPATIAL_OT_CUDA_DEVICE_LIST", "").strip()
    if explicit and explicit.lower() != "all":
        devices = []
        for token in explicit.split(","):
            token = token.strip()
            if not token:
                continue
            devices.append(token if token.startswith("cuda") else f"cuda:{token}")
        return devices or [normalized]
    if normalized.startswith("cuda:"):
        return [normalized]
    visible_count = int(torch.cuda.device_count())
    if visible_count <= 1:
        return ["cuda:0"]
    return [f"cuda:{idx}" for idx in range(visible_count)]


def _resolve_parallel_restart_workers(device_pool: list[str], n_init: int) -> int:
    if len(device_pool) <= 1 or int(n_init) <= 1:
        return 1
    requested = os.environ.get("SPATIAL_OT_PARALLEL_RESTARTS", "auto").strip().lower()
    if requested == "auto":
        return max(1, min(len(device_pool), int(n_init)))
    try:
        value = int(requested)
    except ValueError:
        return max(1, min(len(device_pool), int(n_init)))
    return max(1, min(value, len(device_pool), int(n_init)))


def _make_optimization_measures(measures: list[SubregionMeasure]) -> list[OptimizationMeasure]:
    return [
        OptimizationMeasure(
            subregion_id=int(measure.subregion_id),
            canonical_coords=np.asarray(measure.canonical_coords, dtype=np.float32),
            features=np.asarray(measure.features, dtype=np.float32),
            weights=np.asarray(measure.weights, dtype=np.float32),
        )
        for measure in measures
    ]


def _configure_local_thread_budget(torch_threads: int, torch_interop_threads: int) -> None:
    torch_threads = max(int(torch_threads), 1)
    torch_interop_threads = max(int(torch_interop_threads), 1)
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[name] = str(torch_threads)
    os.environ["SPATIAL_OT_TORCH_NUM_THREADS"] = str(torch_threads)
    os.environ["SPATIAL_OT_TORCH_NUM_INTEROP_THREADS"] = str(torch_interop_threads)
    try:
        torch.set_num_threads(torch_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(torch_interop_threads)
    except Exception:
        pass


def _init_restart_worker(
    measures: list[OptimizationMeasure],
    summaries: np.ndarray,
    params: dict[str, object],
    worker_threads: int,
    worker_interop_threads: int,
) -> None:
    global _RESTART_WORKER_MEASURES, _RESTART_WORKER_SUMMARIES, _RESTART_WORKER_PARAMS
    _configure_local_thread_budget(worker_threads, worker_interop_threads)
    _RESTART_WORKER_MEASURES = measures
    _RESTART_WORKER_SUMMARIES = summaries
    _RESTART_WORKER_PARAMS = params


def _run_restart_worker(run: int, compute_device: str) -> dict[str, object]:
    if _RESTART_WORKER_MEASURES is None or _RESTART_WORKER_SUMMARIES is None or _RESTART_WORKER_PARAMS is None:
        raise RuntimeError("Restart worker was not initialized.")
    return _execute_restart(
        measures=_RESTART_WORKER_MEASURES,
        summaries=_RESTART_WORKER_SUMMARIES,
        run=run,
        compute_device=compute_device,
        **_RESTART_WORKER_PARAMS,
    )


def _cluster_cost_matrix(
    u_aligned: np.ndarray,
    y: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    *,
    cost_scale_x: float,
    cost_scale_y: float,
    lambda_x: float,
    lambda_y: float,
    compute_device: torch.device,
    atom_coords_t: torch.Tensor | None = None,
    atom_features_t: torch.Tensor | None = None,
    y_t: torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    if compute_device.type == "cpu":
        cx = ot.dist(u_aligned, atom_coords, metric="sqeuclidean") / max(cost_scale_x, 1e-12)
        cy = ot.dist(y, atom_features, metric="sqeuclidean") / max(cost_scale_y, 1e-12)
        return lambda_x * cx + lambda_y * cy

    u_aligned_t = torch.as_tensor(u_aligned, dtype=torch.float64, device=compute_device)
    if atom_coords_t is None:
        atom_coords_t = torch.as_tensor(atom_coords, dtype=torch.float64, device=compute_device)
    if atom_features_t is None:
        atom_features_t = torch.as_tensor(atom_features, dtype=torch.float64, device=compute_device)
    if y_t is None:
        y_t = torch.as_tensor(y, dtype=torch.float64, device=compute_device)
    cx_t = torch.cdist(u_aligned_t, atom_coords_t, p=2).pow(2) / max(cost_scale_x, 1e-12)
    cy_t = torch.cdist(y_t, atom_features_t, p=2).pow(2) / max(cost_scale_y, 1e-12)
    return float(lambda_x) * cx_t + float(lambda_y) * cy_t


def _pairwise_sqdist_array(
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    row_batch_size: int | None = None,
) -> np.ndarray:
    x_np = np.asarray(x, dtype=np.float32)
    y_np = np.asarray(y, dtype=np.float32)
    if x_np.size == 0 or y_np.size == 0:
        return np.zeros((x_np.shape[0], y_np.shape[0]), dtype=np.float32)
    if device.type == "cpu":
        return pairwise_distances(x_np, y_np, metric="sqeuclidean").astype(np.float32)

    x_t = torch.as_tensor(x_np, dtype=dtype, device=device)
    y_t = torch.as_tensor(y_np, dtype=dtype, device=device)
    batch = int(
        row_batch_size
        or _cuda_cdist_row_batch_size(
            n_rows=int(x_t.shape[0]),
            n_cols=int(y_t.shape[0]),
            device=device,
            per_pair_buffers=3,
            min_batch=256,
        )
    )
    chunks = []
    with torch.inference_mode():
        for start in range(0, x_t.shape[0], batch):
            dist = torch.cdist(x_t[start : start + batch], y_t, p=2).pow(2)
            chunks.append(dist.detach().cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float32, copy=False)


def _compress_measure(
    canonical_coords: np.ndarray,
    features: np.ndarray,
    weights: np.ndarray,
    m: int,
    lambda_x: float,
    lambda_y: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = canonical_coords.shape[0]
    if n <= m:
        return canonical_coords.astype(np.float32), features.astype(np.float32), _normalize_hist(weights).astype(np.float32)

    z = np.hstack([np.sqrt(lambda_x) * canonical_coords, np.sqrt(lambda_y) * features]).astype(np.float32)
    km = MiniBatchKMeans(
        n_clusters=m,
        random_state=random_state,
        batch_size=min(4096, n),
        n_init="auto",
    )
    labels = km.fit_predict(z, sample_weight=weights)
    u_centers = np.zeros((m, canonical_coords.shape[1]), dtype=np.float64)
    y_centers = np.zeros((m, features.shape[1]), dtype=np.float64)
    a_centers = np.zeros(m, dtype=np.float64)
    for j in range(m):
        mask = labels == j
        if not np.any(mask):
            continue
        a_j = float(weights[mask].sum())
        a_centers[j] = a_j
        norm = max(a_j, 1e-12)
        u_centers[j] = (weights[mask, None] * canonical_coords[mask]).sum(axis=0) / norm
        y_centers[j] = (weights[mask, None] * features[mask]).sum(axis=0) / norm
    keep = a_centers > 1e-12
    if not np.any(keep):
        return canonical_coords[:m].astype(np.float32), features[:m].astype(np.float32), _normalize_hist(weights[:m]).astype(np.float32)
    return (
        u_centers[keep].astype(np.float32),
        y_centers[keep].astype(np.float32),
        _normalize_hist(a_centers[keep]).astype(np.float32),
    )


def _build_subregion_measures(
    features: np.ndarray,
    coords_um: np.ndarray,
    centers_um: np.ndarray,
    region_geometries: list[RegionGeometry],
    geometry_reference_points: np.ndarray,
    geometry_reference_weights: np.ndarray,
    geometry_eps: float,
    geometry_samples: int,
    compressed_support_size: int,
    lambda_x: float,
    lambda_y: float,
    seed: int,
    allow_convex_hull_fallback: bool,
    compute_device: torch.device | None = None,
) -> list[SubregionMeasure]:
    measures: list[SubregionMeasure] = []
    for rid, region in enumerate(region_geometries):
        members = np.asarray(region.members, dtype=np.int32)
        local_coords = np.asarray(coords_um[members], dtype=np.float32)
        local_features = np.asarray(features[members], dtype=np.float32)
        weights = np.full(local_coords.shape[0], 1.0 / max(local_coords.shape[0], 1), dtype=np.float32)
        geom_points, geometry_source, used_fallback = sample_geometry_points(
            region,
            observed_coords=local_coords,
            n_points=max(int(geometry_samples), 32),
            seed=seed + rid,
            allow_convex_hull_fallback=allow_convex_hull_fallback,
            warn_on_fallback=False,
        )
        normalizer, diagnostics = fit_ot_shape_normalizer(
            geometry_points=geom_points,
            reference_points=geometry_reference_points,
            reference_weights=geometry_reference_weights,
            eps_geom=geometry_eps,
            compute_device=compute_device,
        )
        diagnostics.geometry_source = geometry_source
        diagnostics.used_fallback = used_fallback
        canonical_coords = normalizer.transform(local_coords)
        uc, yc, ac = _compress_measure(
            canonical_coords=canonical_coords,
            features=local_features,
            weights=weights,
            m=max(int(compressed_support_size), 1),
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            random_state=seed + 10000 + rid,
        )
        measures.append(
            SubregionMeasure(
                subregion_id=rid,
                center_um=centers_um[rid].astype(np.float32),
                members=np.asarray(members, dtype=np.int32),
                canonical_coords=uc,
                features=yc,
                weights=ac,
                geometry_point_count=int(geom_points.shape[0]),
                compressed_point_count=int(uc.shape[0]),
                normalizer=normalizer,
                normalizer_diagnostics=diagnostics,
            )
        )
    return measures


def _measure_summary(measure: SubregionMeasure) -> np.ndarray:
    z = np.hstack([measure.canonical_coords, measure.features]).astype(np.float64)
    mean = np.average(z, axis=0, weights=measure.weights)
    var = np.average((z - mean) ** 2, axis=0, weights=measure.weights)
    return np.hstack([mean, np.sqrt(np.maximum(var, 0.0))]).astype(np.float32)


def _initialize_cluster_atoms(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    n_clusters: int,
    atoms_per_cluster: int,
    lambda_x: float,
    lambda_y: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    coord_dim = measures[0].canonical_coords.shape[1]
    feat_dim = measures[0].features.shape[1]
    atom_coords = np.zeros((n_clusters, atoms_per_cluster, coord_dim), dtype=np.float32)
    atom_features = np.zeros((n_clusters, atoms_per_cluster, feat_dim), dtype=np.float32)
    betas = np.zeros((n_clusters, atoms_per_cluster), dtype=np.float32)
    sx = np.sqrt(max(float(lambda_x), 1e-12))
    sy = np.sqrt(max(float(lambda_y), 1e-12))

    for k in range(n_clusters):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            idx = np.asarray([int(rng.integers(len(measures)))], dtype=np.int32)
        u_pool = np.vstack([measures[r].canonical_coords for r in idx]).astype(np.float32)
        y_pool = np.vstack([measures[r].features for r in idx]).astype(np.float32)
        a_pool = np.concatenate([measures[r].weights for r in idx]).astype(np.float64)
        a_pool = _normalize_hist(a_pool)
        z_pool = np.hstack([np.sqrt(lambda_x) * u_pool, np.sqrt(lambda_y) * y_pool]).astype(np.float32)

        if z_pool.shape[0] >= atoms_per_cluster:
            km = KMeans(
                n_clusters=atoms_per_cluster,
                n_init=10,
                random_state=random_state + k,
            )
            local_labels = km.fit_predict(z_pool, sample_weight=a_pool)
            beta_k = np.zeros(atoms_per_cluster, dtype=np.float64)
            for ell in range(atoms_per_cluster):
                mask = local_labels == ell
                beta_k[ell] = a_pool[mask].sum()
                if beta_k[ell] <= 1e-12:
                    continue
                atom_coords[k, ell] = np.average(u_pool[mask], axis=0, weights=a_pool[mask]).astype(np.float32)
                atom_features[k, ell] = np.average(y_pool[mask], axis=0, weights=a_pool[mask]).astype(np.float32)
            dead = beta_k <= 1e-12
            if np.any(dead):
                centers = km.cluster_centers_
                atom_coords[k, dead] = (centers[dead, :coord_dim] / sx).astype(np.float32)
                atom_features[k, dead] = (centers[dead, coord_dim:] / sy).astype(np.float32)
            betas[k] = _normalize_hist(beta_k + 1e-6).astype(np.float32)
        else:
            reps = int(np.ceil(atoms_per_cluster / max(z_pool.shape[0], 1)))
            atom_coords[k] = np.tile(u_pool, (reps, 1))[:atoms_per_cluster]
            atom_features[k] = np.tile(y_pool, (reps, 1))[:atoms_per_cluster]
            betas[k] = np.full(atoms_per_cluster, 1.0 / atoms_per_cluster, dtype=np.float32)

    return atom_coords, atom_features, betas


def _estimate_cost_scales(
    measures: list[SubregionMeasure],
    max_points: int,
    random_state: int,
    compute_device: torch.device,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    u_all = np.vstack([m.canonical_coords for m in measures]).astype(np.float32)
    y_all = np.vstack([m.features for m in measures]).astype(np.float32)
    if u_all.shape[0] > max_points:
        idx = rng.choice(u_all.shape[0], size=max_points, replace=False)
        u_all = u_all[idx]
        y_all = y_all[idx]

    def _estimate_pairwise(x: np.ndarray, n_pairs: int = 200_000) -> float:
        n = x.shape[0]
        if n <= 1:
            return 1.0
        i = rng.integers(0, n, size=min(n_pairs, max(n * 8, 1024)))
        j = rng.integers(0, n, size=i.shape[0])
        if compute_device.type == "cpu":
            d2 = np.sum((x[i] - x[j]) ** 2, axis=1)
            d2 = d2[d2 > 0]
            return float(np.median(d2)) if d2.size else 1.0
        x_t = torch.as_tensor(x, dtype=torch.float32, device=compute_device)
        i_t = torch.as_tensor(i, dtype=torch.long, device=compute_device)
        j_t = torch.as_tensor(j, dtype=torch.long, device=compute_device)
        with torch.inference_mode():
            d2_t = torch.sum((x_t[i_t] - x_t[j_t]) ** 2, dim=1)
            d2_t = d2_t[d2_t > 0]
        if d2_t.numel() == 0:
            return 1.0
        return float(torch.median(d2_t).detach().cpu())

    sx = _estimate_pairwise(u_all)
    sy = _estimate_pairwise(y_all)
    return max(sx, 1e-6), max(sy, 1e-6)


def weighted_similarity_fit(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    allow_reflection: bool = True,
    allow_scale: bool = True,
    min_scale: float = 0.75,
    max_scale: float = 1.33,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = _normalize_hist(w)
    xbar = np.sum(w[:, None] * x, axis=0)
    ybar = np.sum(w[:, None] * y, axis=0)
    x0 = x - xbar
    y0 = y - ybar
    h = x0.T @ (w[:, None] * y0)
    u, _, vt = np.linalg.svd(h)
    d = np.eye(x.shape[1], dtype=np.float64)
    if not allow_reflection and np.linalg.det(u @ vt) < 0:
        d[-1, -1] = -1.0
    r = u @ d @ vt
    if allow_scale:
        denom = float(np.sum(w * np.sum(x0**2, axis=1)))
        scale = float(np.trace(r.T @ h) / max(denom, 1e-12))
        scale = max(scale, 1e-12)
    else:
        scale = 1.0
    scale = float(np.clip(scale, min_scale, max_scale))
    t = ybar - scale * xbar @ r
    return {"R": r.astype(np.float64), "scale": scale, "t": t.astype(np.float64)}


def apply_similarity(x: np.ndarray, transform: dict[str, np.ndarray | float]) -> np.ndarray:
    r = np.asarray(transform["R"], dtype=np.float64)
    scale = float(transform["scale"])
    t = np.asarray(transform["t"], dtype=np.float64)
    return (scale * np.asarray(x, dtype=np.float64) @ r + t).astype(np.float64)


def _transform_penalty(
    transform: dict[str, np.ndarray | float],
    scale_penalty: float,
    shift_penalty: float,
) -> float:
    scale = max(float(transform["scale"]), 1e-12)
    t = np.asarray(transform["t"], dtype=np.float64)
    return float(scale_penalty * (np.log(scale) ** 2) + shift_penalty * float(t @ t))


def _solve_semirelaxed_unbalanced_gpu(
    a: np.ndarray | torch.Tensor,
    beta: np.ndarray | torch.Tensor,
    c: np.ndarray | torch.Tensor,
    eps: float,
    rho: float,
    compute_device: torch.device,
) -> tuple[torch.Tensor, float, OTSolveDiagnostics]:
    regs = [max(float(eps), 1e-5)]
    regs.extend([regs[0] * 2.0, regs[0] * 4.0, regs[0] * 8.0])
    a_t = torch.as_tensor(a, dtype=torch.float32, device=compute_device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=compute_device)
    c_t = torch.as_tensor(c, dtype=torch.float32, device=compute_device)
    last_gamma: torch.Tensor | None = None
    last_objective = float("inf")
    last_reg = regs[0]
    for reg in regs:
        gamma, objective, converged, _ = sinkhorn_semirelaxed_unbalanced_log_torch(
            a_t,
            beta_t,
            c_t,
            eps=reg,
            rho=max(float(rho), 1e-6),
            num_iter=600,
            tol=1e-5,
        )
        gamma_finite = bool(torch.isfinite(gamma).all().item())
        obj_val = float(objective.detach().item())
        obj_finite = bool(np.isfinite(obj_val))
        last_gamma = gamma
        last_objective = obj_val
        last_reg = reg
        if gamma_finite and obj_finite and converged:
            return gamma, obj_val, OTSolveDiagnostics(effective_eps=float(reg), used_fallback=not np.isclose(reg, regs[0]))
    if last_gamma is None or not bool(torch.isfinite(last_gamma).all().item()):
        raise FloatingPointError("Unable to obtain a finite semi-relaxed unbalanced OT solution on GPU.")
    return (
        last_gamma,
        last_objective if np.isfinite(last_objective) else 1e12,
        OTSolveDiagnostics(effective_eps=float(last_reg), used_fallback=not np.isclose(last_reg, regs[0])),
    )


def _solve_semirelaxed_unbalanced(
    a: np.ndarray,
    beta: np.ndarray,
    c: np.ndarray,
    eps: float,
    rho: float,
    compute_device: torch.device,
) -> tuple[np.ndarray, float, OTSolveDiagnostics]:
    if compute_device.type == "cuda":
        gamma_t, objective, diag = _solve_semirelaxed_unbalanced_gpu(a, beta, c, eps=eps, rho=rho, compute_device=compute_device)
        return gamma_t.detach().cpu().numpy().astype(np.float64), float(objective), diag
    regs = [max(float(eps), 1e-5)]
    regs.extend([regs[0] * 2.0, regs[0] * 4.0, regs[0] * 8.0])
    last_gamma = None
    last_objective = 1e12
    last_reg = regs[0]
    a_backend = np.asarray(a, dtype=np.float64)
    beta_backend = np.asarray(beta, dtype=np.float64)
    for reg in regs:
        c_backend = np.asarray(c, dtype=np.float64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gamma = ot.unbalanced.sinkhorn_unbalanced(
                a_backend,
                beta_backend,
                c_backend,
                reg=reg,
                reg_m=(float("inf"), max(float(rho), 1e-6)),
                method="sinkhorn_stabilized",
                reg_type="kl",
                numItermax=3000,
                stopThr=1e-8,
            )
            objective = ot.unbalanced.sinkhorn_unbalanced2(
                a_backend,
                beta_backend,
                c_backend,
                reg=reg,
                reg_m=(float("inf"), max(float(rho), 1e-6)),
                method="sinkhorn_stabilized",
                reg_type="kl",
                returnCost="total",
                numItermax=3000,
                stopThr=1e-8,
            )
        if torch.is_tensor(gamma):
            gamma = gamma.detach().cpu().numpy()
        gamma = np.asarray(gamma, dtype=np.float64)
        if torch.is_tensor(objective):
            objective = float(objective.detach().cpu())
        else:
            objective = float(objective)
        last_gamma = gamma
        last_objective = objective
        last_reg = reg
        numeric_warn = any(
            ("Numerical errors" in str(w.message)) or ("did not converge" in str(w.message))
            for w in caught
        )
        if np.all(np.isfinite(gamma)) and np.isfinite(objective) and not numeric_warn:
            return gamma, objective, OTSolveDiagnostics(effective_eps=float(reg), used_fallback=not np.isclose(reg, regs[0]))
    if last_gamma is None or not np.all(np.isfinite(last_gamma)):
        raise FloatingPointError("Unable to obtain a finite semi-relaxed unbalanced OT solution.")
    return (
        last_gamma,
        last_objective if np.isfinite(last_objective) else 1e12,
        OTSolveDiagnostics(effective_eps=float(last_reg), used_fallback=not np.isclose(last_reg, regs[0])),
    )


def aligned_semirelaxed_ot_to_cluster(
    u: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    beta: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    n_align_iter: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
) -> tuple[float, np.ndarray, dict[str, np.ndarray | float], np.ndarray, OTSolveDiagnostics]:
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = _normalize_hist(a)
    beta = _normalize_hist(beta)
    transform: dict[str, np.ndarray | float] = {"R": np.eye(2, dtype=np.float64), "scale": 1.0, "t": np.zeros(2, dtype=np.float64)}
    atom_coords_t = torch.as_tensor(atom_coords, dtype=torch.float64, device=compute_device) if compute_device.type != "cpu" else None
    atom_features_t = torch.as_tensor(atom_features, dtype=torch.float64, device=compute_device) if compute_device.type != "cpu" else None
    y_t = torch.as_tensor(y, dtype=torch.float64, device=compute_device) if compute_device.type != "cpu" else None

    for _ in range(max(int(n_align_iter), 1)):
        u_aligned = apply_similarity(u, transform)
        c = _cluster_cost_matrix(
            u_aligned,
            y,
            atom_coords,
            atom_features,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            compute_device=compute_device,
            atom_coords_t=atom_coords_t,
            atom_features_t=atom_features_t,
            y_t=y_t,
        )
        gamma, _, _ = _solve_semirelaxed_unbalanced(a=a, beta=beta, c=c, eps=eps, rho=rho, compute_device=compute_device)
        row_mass = np.maximum(gamma.sum(axis=1), 1e-12)
        target_bary = (gamma @ atom_coords) / row_mass[:, None]
        transform = weighted_similarity_fit(
            u,
            target_bary,
            row_mass,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    u_aligned = apply_similarity(u, transform)
    c = _cluster_cost_matrix(
        u_aligned,
        y,
        atom_coords,
        atom_features,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        compute_device=compute_device,
        atom_coords_t=atom_coords_t,
        atom_features_t=atom_features_t,
        y_t=y_t,
    )
    gamma, objective, solve_diag = _solve_semirelaxed_unbalanced(a=a, beta=beta, c=c, eps=eps, rho=rho, compute_device=compute_device)
    target_mass = np.asarray(gamma.sum(axis=0), dtype=np.float64)
    theta = _normalize_hist(target_mass)
    objective += _transform_penalty(
        transform,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
    )
    if not np.isfinite(objective):
        objective = 1e12
    return objective, gamma.astype(np.float32), transform, theta.astype(np.float32), solve_diag


def _batched_weighted_similarity_fit_torch(
    x: torch.Tensor,      # (K, m, 2)
    y: torch.Tensor,      # (K, m, 2)
    w: torch.Tensor,      # (K, m)
    allow_reflection: bool,
    allow_scale: bool,
    min_scale: float,
    max_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    w_norm = w / w_sum
    xbar = (w_norm.unsqueeze(-1) * x).sum(dim=-2)   # (K, 2)
    ybar = (w_norm.unsqueeze(-1) * y).sum(dim=-2)   # (K, 2)
    x0 = x - xbar.unsqueeze(-2)
    y0 = y - ybar.unsqueeze(-2)
    h = x0.transpose(-1, -2) @ (w_norm.unsqueeze(-1) * y0)   # (K, 2, 2)
    u, _s, vt = torch.linalg.svd(h)
    d = torch.eye(2, device=x.device, dtype=x.dtype).expand(h.shape[0], 2, 2).contiguous()
    if not allow_reflection:
        det = torch.linalg.det(u @ vt)
        neg = det < 0
        if torch.any(neg):
            d = d.clone()
            d[:, -1, -1] = torch.where(neg, torch.full_like(det, -1.0), torch.full_like(det, 1.0))
    r = u @ d @ vt                                   # (K, 2, 2)
    if allow_scale:
        denom = (w_norm * (x0 ** 2).sum(dim=-1)).sum(dim=-1).clamp_min(1e-12)   # (K,)
        trace = torch.einsum("kij,kij->k", r, h)
        scale = (trace / denom).clamp_min(1e-12)
    else:
        scale = torch.ones(h.shape[0], device=x.device, dtype=x.dtype)
    scale = scale.clamp(min=float(min_scale), max=float(max_scale))
    xbar_r = torch.einsum("ki,kij->kj", xbar, r)     # (K, 2)
    t = ybar - scale.unsqueeze(-1) * xbar_r
    return r, scale, t


def _apply_similarity_batched_torch(x: torch.Tensor, r: torch.Tensor, scale: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # x: (K, m, 2), r: (K, 2, 2), scale: (K,), t: (K, 2)
    return scale.unsqueeze(-1).unsqueeze(-1) * torch.einsum("kmi,kij->kmj", x, r) + t.unsqueeze(-2)


def _aligned_semirelaxed_ot_costs_all_clusters_gpu(
    u: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    n_align_iter: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
    *,
    atom_coords_t: torch.Tensor | None = None,
    atom_features_t: torch.Tensor | None = None,
    betas_t: torch.Tensor | None = None,
) -> np.ndarray:
    """All-K batched aligned semi-relaxed OT (costs only). Returns shape (K,) numpy."""
    dtype = torch.float32
    dev = compute_device
    # tensors
    u_t = torch.as_tensor(u, dtype=dtype, device=dev)              # (m, 2)
    y_t = torch.as_tensor(y, dtype=dtype, device=dev)              # (m, f)
    a_vec = torch.as_tensor(_normalize_hist(a), dtype=dtype, device=dev)  # (m,)
    if atom_coords_t is None:
        atom_coords_t = torch.as_tensor(atom_coords, dtype=dtype, device=dev)
    if atom_features_t is None:
        atom_features_t = torch.as_tensor(atom_features, dtype=dtype, device=dev)
    if betas_t is None:
        betas_np = np.stack([_normalize_hist(b) for b in betas], axis=0)
        betas_t = torch.as_tensor(betas_np, dtype=dtype, device=dev)
    K = atom_coords_t.shape[0]
    # Broadcast u, y to (K, m, *)
    u_kb = u_t.unsqueeze(0).expand(K, -1, -1).contiguous()
    y_kb = y_t.unsqueeze(0).expand(K, -1, -1).contiguous()
    a_kb = a_vec.unsqueeze(0).expand(K, -1).contiguous()

    r = torch.eye(2, device=dev, dtype=dtype).unsqueeze(0).expand(K, 2, 2).contiguous()
    scale = torch.ones(K, device=dev, dtype=dtype)
    t = torch.zeros(K, 2, device=dev, dtype=dtype)

    sx = max(float(cost_scale_x), 1e-12)
    sy = max(float(cost_scale_y), 1e-12)
    cy_full = torch.cdist(y_kb, atom_features_t, p=2).pow(2) / sy
    cy_scaled = float(lambda_y) * cy_full                # (K, m, p) — fixed over align iters

    eps_base = max(float(eps), 1e-5)
    reg_schedule = [eps_base, 2.0 * eps_base, 4.0 * eps_base, 8.0 * eps_base]

    def _solve_once(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        last_gamma = None
        last_obj = None
        last_reg = reg_schedule[0]
        for reg in reg_schedule:
            gamma, obj, converged, _ = sinkhorn_semirelaxed_unbalanced_log_torch(
                a_kb,
                betas_t,
                cost,
                eps=reg,
                rho=max(float(rho), 1e-6),
                num_iter=600,
                tol=1e-5,
            )
            finite = torch.isfinite(gamma).all() and torch.isfinite(obj).all()
            last_gamma = gamma
            last_obj = obj
            last_reg = reg
            if bool(finite.item()) and converged:
                return gamma, obj, float(reg)
        if last_gamma is None:
            raise FloatingPointError("Unable to obtain finite batched semirelaxed OT on GPU.")
        return last_gamma, last_obj, float(last_reg)

    for _ in range(max(int(n_align_iter), 1)):
        u_aligned = _apply_similarity_batched_torch(u_kb, r, scale, t)       # (K, m, 2)
        cx = torch.cdist(u_aligned, atom_coords_t, p=2).pow(2) / sx          # (K, m, p)
        cost = float(lambda_x) * cx + cy_scaled
        gamma, _, _ = _solve_once(cost)
        row_mass = gamma.sum(dim=-1).clamp_min(1e-12)                        # (K, m)
        target_bary = torch.einsum("kmp,kpd->kmd", gamma, atom_coords_t) / row_mass.unsqueeze(-1)
        r, scale, t = _batched_weighted_similarity_fit_torch(
            u_kb,
            target_bary,
            row_mass,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    u_aligned = _apply_similarity_batched_torch(u_kb, r, scale, t)
    cx = torch.cdist(u_aligned, atom_coords_t, p=2).pow(2) / sx
    cost = float(lambda_x) * cx + cy_scaled
    _, obj, _ = _solve_once(cost)

    # transform penalty: scale_penalty * log(scale)^2 + shift_penalty * (t @ t)
    penalty = float(scale_penalty) * torch.log(scale.clamp_min(1e-12)).pow(2) + float(shift_penalty) * (t * t).sum(dim=-1)
    total = obj + penalty
    total = torch.where(torch.isfinite(total), total, torch.full_like(total, 1e12))
    return total.detach().cpu().numpy().astype(np.float64)


def _pack_measures_padded(
    measures: list[SubregionMeasure],
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad all subregion measures to shape (R, m_max, *). Zero-mass rows mark padding."""
    R = len(measures)
    if R == 0:
        raise ValueError("no subregion measures to pack")
    m_max = max(int(m.canonical_coords.shape[0]) for m in measures)
    coord_dim = int(measures[0].canonical_coords.shape[1])
    feat_dim = int(measures[0].features.shape[1])
    u = np.zeros((R, m_max, coord_dim), dtype=np.float32)
    y = np.zeros((R, m_max, feat_dim), dtype=np.float32)
    a = np.zeros((R, m_max), dtype=np.float32)
    m_r = np.zeros(R, dtype=np.int32)
    for r, meas in enumerate(measures):
        mr = int(meas.canonical_coords.shape[0])
        m_r[r] = mr
        u[r, :mr] = np.asarray(meas.canonical_coords, dtype=np.float32)
        y[r, :mr] = np.asarray(meas.features, dtype=np.float32)
        a[r, :mr] = _normalize_hist(np.asarray(meas.weights)).astype(np.float32)
    return (
        torch.as_tensor(u, dtype=dtype, device=device),
        torch.as_tensor(y, dtype=dtype, device=device),
        torch.as_tensor(a, dtype=dtype, device=device),
        torch.as_tensor(m_r, dtype=torch.int32, device=device),
    )


def _batched_weighted_similarity_fit_rk_torch(
    x: torch.Tensor,      # (R, K, m, 2)  --- x is u replicated across K
    y: torch.Tensor,      # (R, K, m, 2)  --- barycentric target
    w: torch.Tensor,      # (R, K, m)
    allow_reflection: bool,
    allow_scale: bool,
    min_scale: float,
    max_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w_sum = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    w_norm = w / w_sum
    xbar = (w_norm.unsqueeze(-1) * x).sum(dim=-2)     # (R, K, 2)
    ybar = (w_norm.unsqueeze(-1) * y).sum(dim=-2)     # (R, K, 2)
    x0 = x - xbar.unsqueeze(-2)
    y0 = y - ybar.unsqueeze(-2)
    # h[r,k] = x0^T @ (diag(w_norm) @ y0)
    h = torch.einsum("rkmi,rkmj->rkij", x0, w_norm.unsqueeze(-1) * y0)
    # batched 2x2 SVD over (R, K, 2, 2)
    u_svd, _s, vt = torch.linalg.svd(h)
    if not allow_reflection:
        det = torch.linalg.det(u_svd @ vt)
        d = torch.eye(2, device=x.device, dtype=x.dtype).expand(*h.shape[:-2], 2, 2).contiguous()
        neg = det < 0
        if torch.any(neg):
            d = d.clone()
            # set last diagonal entry to -1 where det<0
            fix = torch.where(neg, torch.full_like(det, -1.0), torch.full_like(det, 1.0))
            d[..., -1, -1] = fix
        r_mat = u_svd @ d @ vt
    else:
        r_mat = u_svd @ vt
    if allow_scale:
        denom = (w_norm * (x0 ** 2).sum(dim=-1)).sum(dim=-1).clamp_min(1e-12)
        trace = torch.einsum("rkij,rkij->rk", r_mat, h)
        scale = (trace / denom).clamp_min(1e-12)
    else:
        scale = torch.ones(h.shape[:-2], device=x.device, dtype=x.dtype)
    scale = scale.clamp(min=float(min_scale), max=float(max_scale))
    xbar_r = torch.einsum("rki,rkij->rkj", xbar, r_mat)
    t = ybar - scale.unsqueeze(-1) * xbar_r
    return r_mat, scale, t


def _apply_similarity_rk_torch(x: torch.Tensor, r: torch.Tensor, scale: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return scale.unsqueeze(-1).unsqueeze(-1) * torch.einsum("rkmi,rkij->rkmj", x, r) + t.unsqueeze(-2)


def _compute_assignment_costs_rk_gpu(
    measures: list[SubregionMeasure],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
) -> np.ndarray:
    """Fully batched (R*K, m, *) aligned semi-relaxed OT on GPU.

    This amortizes kernel-launch overhead across all (subregion, cluster) pairs.
    Returns costs of shape (R, K) as float32 numpy.
    """
    dtype = torch.float32
    dev = compute_device
    u_rm, y_rm, a_rm, _m_r = _pack_measures_padded(measures, dtype=dtype, device=dev)
    R = u_rm.shape[0]
    m = u_rm.shape[1]
    K = int(atom_coords.shape[0])
    p = int(atom_coords.shape[1])

    atom_coords_t = torch.as_tensor(atom_coords, dtype=dtype, device=dev)        # (K, p, 2)
    atom_features_t = torch.as_tensor(atom_features, dtype=dtype, device=dev)    # (K, p, f)
    betas_np = np.stack([_normalize_hist(b) for b in betas], axis=0)
    beta_k = torch.as_tensor(betas_np, dtype=dtype, device=dev)                   # (K, p)

    # Broadcast to (R, K, *)
    u_rkm = u_rm.unsqueeze(1).expand(R, K, m, u_rm.shape[-1]).contiguous()
    y_rkm = y_rm.unsqueeze(1).expand(R, K, m, y_rm.shape[-1]).contiguous()
    a_rkm = a_rm.unsqueeze(1).expand(R, K, m).contiguous()
    atom_coords_rk = atom_coords_t.unsqueeze(0).expand(R, K, p, atom_coords_t.shape[-1]).contiguous()
    atom_features_rk = atom_features_t.unsqueeze(0).expand(R, K, p, atom_features_t.shape[-1]).contiguous()
    beta_rk = beta_k.unsqueeze(0).expand(R, K, p).contiguous()

    sx = max(float(cost_scale_x), 1e-12)
    sy = max(float(cost_scale_y), 1e-12)
    cy_full = torch.cdist(y_rkm.reshape(R * K, m, y_rm.shape[-1]), atom_features_rk.reshape(R * K, p, y_rm.shape[-1]), p=2).pow(2).reshape(R, K, m, p) / sy
    cy_scaled = float(lambda_y) * cy_full

    r_mat = torch.eye(2, device=dev, dtype=dtype).expand(R, K, 2, 2).contiguous()
    scale_t = torch.ones(R, K, device=dev, dtype=dtype)
    t_t = torch.zeros(R, K, 2, device=dev, dtype=dtype)

    eps_base = max(float(eps), 1e-5)
    reg_schedule = (eps_base, 2.0 * eps_base, 4.0 * eps_base, 8.0 * eps_base)
    rho_val = max(float(rho), 1e-6)

    def _solve(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # cost shape: (R, K, m, p). Flatten batch dims for Sinkhorn.
        flat_cost = cost.reshape(R * K, m, p)
        flat_a = a_rkm.reshape(R * K, m)
        flat_b = beta_rk.reshape(R * K, p)
        last_gamma = None
        last_obj = None
        eff_reg = torch.full((R, K), reg_schedule[0], dtype=dtype, device=dev)
        converged_all = torch.zeros(R, K, dtype=torch.bool, device=dev)
        for reg in reg_schedule:
            gamma, obj, _conv, _err = sinkhorn_semirelaxed_unbalanced_log_torch(
                flat_a,
                flat_b,
                flat_cost,
                eps=reg,
                rho=rho_val,
                num_iter=600,
                tol=1e-5,
            )
            gamma = gamma.reshape(R, K, m, p)
            obj = obj.reshape(R, K)
            finite = torch.isfinite(gamma).all(dim=(-1, -2)) & torch.isfinite(obj)
            if last_gamma is None:
                last_gamma = gamma.clone()
                last_obj = obj.clone()
                converged_all = finite.clone()
                eff_reg = torch.full_like(obj, reg)
            update = finite & ~converged_all
            if update.any():
                last_gamma = torch.where(update.unsqueeze(-1).unsqueeze(-1), gamma, last_gamma)
                last_obj = torch.where(update, obj, last_obj)
                eff_reg = torch.where(update, torch.full_like(eff_reg, reg), eff_reg)
                converged_all = converged_all | update
            if bool(converged_all.all().item()):
                break
        if last_gamma is None or last_obj is None:
            raise FloatingPointError("GPU batched Sinkhorn failed")
        return last_gamma, last_obj, eff_reg

    identity_r = torch.eye(2, device=dev, dtype=dtype).expand(R, K, 2, 2).contiguous()
    for _ in range(max(int(align_iters), 1)):
        u_aligned = _apply_similarity_rk_torch(u_rkm, r_mat, scale_t, t_t)           # (R, K, m, 2)
        cx = torch.cdist(u_aligned.reshape(R * K, m, u_aligned.shape[-1]), atom_coords_rk.reshape(R * K, p, u_aligned.shape[-1]), p=2).pow(2).reshape(R, K, m, p) / sx
        cost = float(lambda_x) * cx + cy_scaled
        gamma, _obj, _reg = _solve(cost)
        gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
        row_mass = gamma.sum(dim=-1).clamp_min(1e-12)                                # (R, K, m)
        target_bary = torch.einsum("rkmp,rkpd->rkmd", gamma, atom_coords_rk) / row_mass.unsqueeze(-1)
        r_new, scale_new, t_new = _batched_weighted_similarity_fit_rk_torch(
            u_rkm,
            target_bary,
            row_mass,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        # Reset any NaN/Inf transforms to identity so we don't poison subsequent iters.
        bad = ~torch.isfinite(r_new).all(dim=(-1, -2)) | ~torch.isfinite(scale_new) | ~torch.isfinite(t_new).all(dim=-1)
        if bool(bad.any().item()):
            r_new = torch.where(bad.unsqueeze(-1).unsqueeze(-1), identity_r, r_new)
            scale_new = torch.where(bad, torch.ones_like(scale_new), scale_new)
            t_new = torch.where(bad.unsqueeze(-1), torch.zeros_like(t_new), t_new)
        r_mat, scale_t, t_t = r_new, scale_new, t_new

    u_aligned = _apply_similarity_rk_torch(u_rkm, r_mat, scale_t, t_t)
    cx = torch.cdist(u_aligned.reshape(R * K, m, u_aligned.shape[-1]), atom_coords_rk.reshape(R * K, p, u_aligned.shape[-1]), p=2).pow(2).reshape(R, K, m, p) / sx
    cost = float(lambda_x) * cx + cy_scaled
    _gamma, obj, _reg = _solve(cost)

    penalty = float(scale_penalty) * torch.log(scale_t.clamp_min(1e-12)).pow(2) + float(shift_penalty) * (t_t * t_t).sum(dim=-1)
    total = obj + penalty
    total = torch.where(torch.isfinite(total), total, torch.full_like(total, 1e12))
    return total.detach().cpu().numpy().astype(np.float32)


def _compute_assigned_artifacts_r_gpu(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray | float]], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Fully batched (over R) aligned semirelaxed OT on GPU, returning per-subregion artifacts."""
    dtype = torch.float32
    dev = compute_device
    R = len(measures)
    label_idx = torch.as_tensor(np.asarray(labels, dtype=np.int64), dtype=torch.long, device=dev)
    u_rm, y_rm, a_rm, m_r = _pack_measures_padded(measures, dtype=dtype, device=dev)
    atom_coords_t = torch.as_tensor(atom_coords, dtype=dtype, device=dev)          # (K, p, 2)
    atom_features_t = torch.as_tensor(atom_features, dtype=dtype, device=dev)      # (K, p, f)
    betas_np = np.stack([_normalize_hist(b) for b in betas], axis=0)
    beta_k = torch.as_tensor(betas_np, dtype=dtype, device=dev)                    # (K, p)

    # Gather per-r atoms using labels
    atom_coords_r = atom_coords_t[label_idx]       # (R, p, 2)
    atom_features_r = atom_features_t[label_idx]   # (R, p, f)
    beta_r = beta_k[label_idx]                     # (R, p)

    sx = max(float(cost_scale_x), 1e-12)
    sy = max(float(cost_scale_y), 1e-12)
    cy_full = torch.cdist(y_rm, atom_features_r, p=2).pow(2) / sy
    cy_scaled = float(lambda_y) * cy_full

    r_mat = torch.eye(2, device=dev, dtype=dtype).expand(R, 2, 2).contiguous()
    scale_t = torch.ones(R, device=dev, dtype=dtype)
    t_t = torch.zeros(R, 2, device=dev, dtype=dtype)

    eps_base = max(float(eps), 1e-5)
    reg_schedule = (eps_base, 2.0 * eps_base, 4.0 * eps_base, 8.0 * eps_base)
    rho_val = max(float(rho), 1e-6)

    def _solve(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        last_gamma = None
        last_obj = None
        eff_reg = torch.full((R,), reg_schedule[0], dtype=dtype, device=dev)
        converged_all = torch.zeros(R, dtype=torch.bool, device=dev)
        for reg in reg_schedule:
            gamma, obj, _conv, _err = sinkhorn_semirelaxed_unbalanced_log_torch(
                a_rm,
                beta_r,
                cost,
                eps=reg,
                rho=rho_val,
                num_iter=600,
                tol=1e-5,
            )
            finite = torch.isfinite(gamma).all(dim=(-1, -2)) & torch.isfinite(obj)
            if last_gamma is None:
                last_gamma = gamma.clone()
                last_obj = obj.clone()
                converged_all = finite.clone()
                eff_reg = torch.full_like(obj, reg)
            update = finite & ~converged_all
            if update.any():
                last_gamma = torch.where(update.unsqueeze(-1).unsqueeze(-1), gamma, last_gamma)
                last_obj = torch.where(update, obj, last_obj)
                eff_reg = torch.where(update, torch.full_like(eff_reg, reg), eff_reg)
                converged_all = converged_all | update
            if bool(converged_all.all().item()):
                break
        if last_gamma is None:
            raise FloatingPointError("GPU batched-R Sinkhorn failed")
        return last_gamma, last_obj, eff_reg

    # R-wise barycentric alignment
    identity_r = torch.eye(2, device=dev, dtype=dtype).expand(R, 2, 2).contiguous()
    for _ in range(max(int(align_iters), 1)):
        u_aligned = scale_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum("rmi,rij->rmj", u_rm, r_mat) + t_t.unsqueeze(-2)
        cx = torch.cdist(u_aligned, atom_coords_r, p=2).pow(2) / sx
        cost = float(lambda_x) * cx + cy_scaled
        gamma, _obj, _reg = _solve(cost)
        gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
        row_mass = gamma.sum(dim=-1).clamp_min(1e-12)
        target_bary = torch.einsum("rmp,rpd->rmd", gamma, atom_coords_r) / row_mass.unsqueeze(-1)
        # Batched weighted similarity fit over R (2D).
        w_sum = row_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        w_norm = row_mass / w_sum
        xbar = (w_norm.unsqueeze(-1) * u_rm).sum(dim=-2)
        ybar = (w_norm.unsqueeze(-1) * target_bary).sum(dim=-2)
        x0 = u_rm - xbar.unsqueeze(-2)
        y0 = target_bary - ybar.unsqueeze(-2)
        h = torch.einsum("rmi,rmj->rij", x0, w_norm.unsqueeze(-1) * y0)
        u_svd, _s, vt = torch.linalg.svd(h)
        if not allow_reflection:
            det = torch.linalg.det(u_svd @ vt)
            d = torch.eye(2, device=dev, dtype=dtype).expand(R, 2, 2).contiguous()
            neg = det < 0
            if torch.any(neg):
                d = d.clone()
                fix = torch.where(neg, torch.full_like(det, -1.0), torch.full_like(det, 1.0))
                d[..., -1, -1] = fix
            r_new = u_svd @ d @ vt
        else:
            r_new = u_svd @ vt
        if allow_scale:
            denom = (w_norm * (x0 ** 2).sum(dim=-1)).sum(dim=-1).clamp_min(1e-12)
            trace = torch.einsum("rij,rij->r", r_new, h)
            scale_new = (trace / denom).clamp_min(1e-12)
        else:
            scale_new = torch.ones(R, device=dev, dtype=dtype)
        scale_new = scale_new.clamp(min=float(min_scale), max=float(max_scale))
        xbar_r = torch.einsum("ri,rij->rj", xbar, r_new)
        t_new = ybar - scale_new.unsqueeze(-1) * xbar_r
        bad = ~torch.isfinite(r_new).all(dim=(-1, -2)) | ~torch.isfinite(scale_new) | ~torch.isfinite(t_new).all(dim=-1)
        if bool(bad.any().item()):
            r_new = torch.where(bad.unsqueeze(-1).unsqueeze(-1), identity_r, r_new)
            scale_new = torch.where(bad, torch.ones_like(scale_new), scale_new)
            t_new = torch.where(bad.unsqueeze(-1), torch.zeros_like(t_new), t_new)
        r_mat, scale_t, t_t = r_new, scale_new, t_new

    # Final solve
    u_aligned = scale_t.unsqueeze(-1).unsqueeze(-1) * torch.einsum("rmi,rij->rmj", u_rm, r_mat) + t_t.unsqueeze(-2)
    cx = torch.cdist(u_aligned, atom_coords_r, p=2).pow(2) / sx
    cost = float(lambda_x) * cx + cy_scaled
    gamma, obj, eff_reg = _solve(cost)
    gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)
    col_mass = gamma.sum(dim=-2)                        # (R, p)
    theta = col_mass / col_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    penalty = float(scale_penalty) * torch.log(scale_t.clamp_min(1e-12)).pow(2) + float(shift_penalty) * (t_t * t_t).sum(dim=-1)
    total = obj + penalty
    total = torch.where(torch.isfinite(total), total, torch.full_like(total, 1e12))

    # Move to CPU once
    gamma_np = gamma.detach().cpu().numpy()
    r_np = r_mat.detach().cpu().numpy()
    scale_np = scale_t.detach().cpu().numpy()
    t_np = t_t.detach().cpu().numpy()
    theta_np = theta.detach().cpu().numpy()
    total_np = total.detach().cpu().numpy().astype(np.float32)
    eff_reg_np = eff_reg.detach().cpu().numpy().astype(np.float32)
    m_r_np = m_r.detach().cpu().numpy()

    plans: list[np.ndarray] = []
    transforms: list[dict[str, np.ndarray | float]] = []
    thetas: list[np.ndarray] = []
    used_fallback_np = ~np.isclose(eff_reg_np, float(eps_base))
    for r in range(R):
        mr = int(m_r_np[r])
        plans.append(gamma_np[r, :mr, :].astype(np.float32))
        transforms.append({
            "R": r_np[r].astype(np.float64),
            "scale": float(scale_np[r]),
            "t": t_np[r].astype(np.float64),
        })
        thetas.append(theta_np[r].astype(np.float32))
    return plans, transforms, thetas, total_np, eff_reg_np, used_fallback_np.astype(bool)


def _ensure_nonempty_clusters(labels: np.ndarray, costs: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.copy()
    forced = np.zeros(labels.shape[0], dtype=bool)
    counts = np.bincount(labels, minlength=n_clusters)
    current_cost = costs[np.arange(labels.shape[0]), labels].astype(np.float64)
    for empty_k in np.where(counts == 0)[0]:
        feasible = counts[labels] > 1
        penalty = costs[:, empty_k].astype(np.float64) - current_cost
        penalty[~feasible] = np.inf
        r = int(np.argmin(penalty))
        if not np.isfinite(penalty[r]):
            raise RuntimeError("Cannot repair empty cluster without emptying another cluster.")
        donor = int(labels[r])
        counts[donor] -= 1
        labels[r] = int(empty_k)
        counts[empty_k] += 1
        current_cost[r] = float(costs[r, empty_k])
        forced[r] = True
    return labels, forced


def _compute_assignment_costs(
    measures: list[SubregionMeasure],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
) -> np.ndarray:
    n_subregions = len(measures)
    n_clusters = atom_coords.shape[0]
    costs = np.zeros((n_subregions, n_clusters), dtype=np.float64)

    if compute_device.type == "cuda":
        # Fully batched (R*K) GPU solve — amortizes kernel launch latency.
        costs_rk = _compute_assignment_costs_rk_gpu(
            measures=measures,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=compute_device,
        )
        return np.clip(costs_rk, -1e12, 1e12).astype(np.float32)

    for r, measure in enumerate(measures):
        for k in range(n_clusters):
            cost, _, _, _, _ = aligned_semirelaxed_ot_to_cluster(
                u=measure.canonical_coords,
                y=measure.features,
                a=measure.weights,
                atom_coords=atom_coords[k],
                atom_features=atom_features[k],
                beta=betas[k],
                lambda_x=lambda_x,
                lambda_y=lambda_y,
                eps=eps,
                rho=rho,
                n_align_iter=align_iters,
                allow_reflection=allow_reflection,
                allow_scale=allow_scale,
                cost_scale_x=cost_scale_x,
                cost_scale_y=cost_scale_y,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_penalty=scale_penalty,
                shift_penalty=shift_penalty,
                compute_device=compute_device,
            )
            costs[r, k] = float(np.clip(cost, -1e12, 1e12))
    return costs.astype(np.float32)


def _compute_assigned_artifacts(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    betas: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    compute_device: torch.device,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray | float]], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    if compute_device.type == "cuda":
        return _compute_assigned_artifacts_r_gpu(
            measures=measures,
            labels=labels,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=compute_device,
        )
    plans: list[np.ndarray] = []
    transforms: list[dict[str, np.ndarray | float]] = []
    thetas: list[np.ndarray] = []
    assigned_costs = np.zeros(len(measures), dtype=np.float32)
    effective_eps = np.zeros(len(measures), dtype=np.float32)
    used_fallback = np.zeros(len(measures), dtype=bool)
    for r, measure in enumerate(measures):
        k = int(labels[r])
        cost, gamma, transform, theta, solve_diag = aligned_semirelaxed_ot_to_cluster(
            u=measure.canonical_coords,
            y=measure.features,
            a=measure.weights,
            atom_coords=atom_coords[k],
            atom_features=atom_features[k],
            beta=betas[k],
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=eps,
            rho=rho,
            n_align_iter=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=compute_device,
        )
        assigned_costs[r] = cost
        effective_eps[r] = float(solve_diag.effective_eps)
        used_fallback[r] = bool(solve_diag.used_fallback)
        plans.append(gamma)
        transforms.append(transform)
        thetas.append(theta)
    return plans, transforms, thetas, assigned_costs, effective_eps, used_fallback


def _update_atoms(
    measures: list[SubregionMeasure],
    labels: np.ndarray,
    plans: list[np.ndarray],
    transforms: list[dict[str, np.ndarray | float]],
    thetas: list[np.ndarray],
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    beta_smoothing: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_clusters, atoms_per_cluster, coord_dim = atom_coords.shape
    feat_dim = atom_features.shape[2]
    new_coords = atom_coords.copy().astype(np.float32)
    new_features = atom_features.copy().astype(np.float32)
    new_betas = np.zeros((n_clusters, atoms_per_cluster), dtype=np.float32)

    for k in range(n_clusters):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            r = int(rng.integers(len(measures)))
            measure = measures[r]
            pick = rng.choice(
                measure.canonical_coords.shape[0],
                size=atoms_per_cluster,
                replace=measure.canonical_coords.shape[0] < atoms_per_cluster,
                p=measure.weights,
            )
            new_coords[k] = measure.canonical_coords[pick]
            new_features[k] = measure.features[pick]
            new_betas[k] = np.full(atoms_per_cluster, 1.0 / atoms_per_cluster, dtype=np.float32)
            continue

        coord_num = np.zeros((atoms_per_cluster, coord_dim), dtype=np.float64)
        feat_num = np.zeros((atoms_per_cluster, feat_dim), dtype=np.float64)
        denom = np.zeros(atoms_per_cluster, dtype=np.float64)
        beta_num = np.zeros(atoms_per_cluster, dtype=np.float64)

        for r in idx:
            measure = measures[r]
            gamma = np.asarray(plans[r], dtype=np.float64)
            theta = np.asarray(thetas[r], dtype=np.float64)
            transform = transforms[r]
            aligned = apply_similarity(measure.canonical_coords, transform)
            coord_num += gamma.T @ aligned
            feat_num += gamma.T @ measure.features
            denom += gamma.sum(axis=0)
            beta_num += theta

        active = denom > 1e-10
        if np.any(active):
            new_coords[k, active] = (coord_num[active] / denom[active, None]).astype(np.float32)
            new_features[k, active] = (feat_num[active] / denom[active, None]).astype(np.float32)

        if np.any(~active):
            u_pool = np.vstack([measures[r].canonical_coords for r in idx]).astype(np.float32)
            y_pool = np.vstack([measures[r].features for r in idx]).astype(np.float32)
            a_pool = _normalize_hist(np.concatenate([measures[r].weights for r in idx]))
            dead = np.where(~active)[0]
            chosen = rng.choice(
                u_pool.shape[0],
                size=dead.size,
                replace=u_pool.shape[0] < dead.size,
                p=a_pool,
            )
            new_coords[k, dead] = u_pool[chosen]
            new_features[k, dead] = y_pool[chosen]

        new_betas[k] = _normalize_hist(beta_num + beta_smoothing).astype(np.float32)

    return new_coords, new_features, new_betas


def _cell_cluster_feature_costs(
    features: np.ndarray,
    support_features: np.ndarray,
    prototype_weights: np.ndarray,
    temperature: float,
    compute_device: torch.device,
) -> np.ndarray:
    n_cells = features.shape[0]
    n_clusters = support_features.shape[0]
    temp = max(float(temperature), 1e-5)
    if compute_device.type == "cuda" and n_cells > 0 and n_clusters > 0:
        dtype = torch.float32
        feats_t = torch.as_tensor(features, dtype=dtype, device=compute_device)
        support_t = torch.as_tensor(support_features, dtype=dtype, device=compute_device)
        weights_t = torch.as_tensor(np.clip(prototype_weights, 1e-8, None), dtype=dtype, device=compute_device)
        costs_t = torch.empty((n_cells, n_clusters), dtype=dtype, device=compute_device)
        feat_dim = int(features.shape[1]) if features.ndim == 2 else 1
        support_size = int(support_features.shape[1]) if support_features.ndim >= 2 else 1
        denom = max(n_clusters * max(2 * support_size + feat_dim, 1) * 4, 1)
        batch = max(1, min(n_cells, _cuda_target_bytes(device=compute_device) // denom))
        with torch.inference_mode():
            for start in range(0, n_cells, batch):
                f_chunk = feats_t[start : start + batch].unsqueeze(0).expand(n_clusters, -1, -1)
                dist = torch.cdist(f_chunk, support_t, p=2).pow(2)                     # (K, B, p)
                scaled = torch.exp(-dist / temp) * weights_t.unsqueeze(1)              # (K, B, p)
                costs_t[start : start + batch] = (-temp * torch.log(scaled.sum(dim=-1).clamp_min(1e-8))).transpose(0, 1)
        return costs_t.detach().cpu().numpy().astype(np.float32)
    costs = np.zeros((n_cells, n_clusters), dtype=np.float32)
    for k in range(n_clusters):
        dist = _pairwise_sqdist_array(features, support_features[k], device=compute_device)
        weights = np.clip(prototype_weights[k], 1e-8, None).astype(np.float32)
        scores = np.exp(-dist / temp) * weights[None, :]
        costs[:, k] = -temp * np.log(np.maximum(scores.sum(axis=1), 1e-8))
    return costs


def _project_cells_from_subregions(
    features: np.ndarray,
    coords_um: np.ndarray,
    measures: list[SubregionMeasure],
    subregion_labels: np.ndarray,
    atom_coords: np.ndarray,
    atom_features: np.ndarray,
    prototype_weights: np.ndarray,
    assigned_transforms: list[dict[str, np.ndarray | float]],
    subregion_cluster_costs: np.ndarray,
    lambda_x: float,
    lambda_y: float,
    cost_scale_x: float,
    cost_scale_y: float,
    assignment_temperature: float,
    context_weight: float = 0.5,
    compute_device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    compute_device = compute_device or torch.device("cpu")
    support_features = atom_features
    feature_costs = _cell_cluster_feature_costs(
        features=features,
        support_features=support_features,
        prototype_weights=prototype_weights,
        temperature=assignment_temperature,
        compute_device=compute_device,
    )
    feature_probs = _softmax_over_negative_costs(feature_costs, temperature=assignment_temperature)
    subregion_probs = _softmax_over_negative_costs(subregion_cluster_costs, temperature=max(assignment_temperature, 1e-4))

    n_cells = features.shape[0]
    n_clusters = atom_coords.shape[0]
    context_probs = np.zeros((n_cells, n_clusters), dtype=np.float32)
    membership_counts = np.zeros((n_cells, 1), dtype=np.float32)
    local_model_probs = np.zeros((n_cells, n_clusters), dtype=np.float32)
    local_model_counts = np.zeros((n_cells, 1), dtype=np.float32)
    for r, measure in enumerate(measures):
        members = measure.members
        context_probs[members] += subregion_probs[r]
        membership_counts[members] += 1.0
        k = int(subregion_labels[r])
        canonical = measure.normalizer.transform(coords_um[members])
        aligned = apply_similarity(canonical, assigned_transforms[r]).astype(np.float32)
        cx = _pairwise_sqdist_array(aligned, atom_coords[k], device=compute_device) / max(float(cost_scale_x), 1e-5)
        cy = _pairwise_sqdist_array(features[members], atom_features[k], device=compute_device) / max(float(cost_scale_y), 1e-5)
        total_cost = float(lambda_x) * cx + float(lambda_y) * cy
        atom_scores = np.exp(-total_cost / max(float(assignment_temperature), 1e-5)) * np.clip(prototype_weights[k], 1e-8, None)[None, :]
        local_model_probs[members, k] += atom_scores.sum(axis=1).astype(np.float32)
        local_model_counts[members] += 1.0
    covered = membership_counts[:, 0] > 0
    if np.any(covered):
        context_probs[covered] /= membership_counts[covered]
    if np.any(~covered):
        context_probs[~covered] = feature_probs[~covered]
    local_covered = local_model_counts[:, 0] > 0
    if np.any(local_covered):
        local_model_probs[local_covered] /= local_model_counts[local_covered]
        local_model_probs[local_covered] /= np.maximum(local_model_probs[local_covered].sum(axis=1, keepdims=True), 1e-8)
    if np.any(~local_covered):
        local_model_probs[~local_covered] = feature_probs[~local_covered]

    combined = (
        feature_probs
        * np.power(np.clip(context_probs, 1e-8, None), float(context_weight))
        * np.power(np.clip(local_model_probs, 1e-8, None), 1.0 - float(context_weight))
    )
    combined = combined / np.maximum(combined.sum(axis=1, keepdims=True), 1e-8)
    labels = combined.argmax(axis=1).astype(np.int32)
    return labels, combined.astype(np.float32), local_model_probs.astype(np.float32), context_probs.astype(np.float32)


def _execute_restart(
    measures: list[OptimizationMeasure],
    summaries: np.ndarray,
    *,
    run: int,
    n_clusters: int,
    atoms_per_cluster: int,
    lambda_x: float,
    lambda_y: float,
    ot_eps: float,
    rho: float,
    align_iters: int,
    allow_reflection: bool,
    allow_scale: bool,
    cost_scale_x: float,
    cost_scale_y: float,
    min_scale: float,
    max_scale: float,
    scale_penalty: float,
    shift_penalty: float,
    max_iter: int,
    tol: float,
    seed: int,
    compute_device: str,
) -> dict[str, object]:
    resolved_compute_device = _resolve_compute_device(compute_device)
    if resolved_compute_device.type == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(resolved_compute_device)
        except Exception:
            pass
    run_seed = seed + 1000 * int(run)
    init = KMeans(n_clusters=n_clusters, n_init=20, random_state=run_seed)
    labels = init.fit_predict(summaries).astype(np.int32)
    atom_coords, atom_features, betas = _initialize_cluster_atoms(
        measures=measures,
        labels=labels,
        n_clusters=n_clusters,
        atoms_per_cluster=atoms_per_cluster,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        random_state=run_seed,
    )
    objective_history: list[dict[str, float]] = []

    for iteration in range(int(max_iter)):
        prev_coords = atom_coords.copy()
        prev_features = atom_features.copy()
        prev_labels = labels.copy()
        costs = _compute_assignment_costs(
            measures=measures,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=ot_eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=resolved_compute_device,
        )
        argmin_labels = costs.argmin(axis=1).astype(np.int32)
        labels, forced_label_mask = _ensure_nonempty_clusters(argmin_labels, costs, n_clusters)
        plans, transforms, thetas, assigned_costs, _, assigned_used_fallback = _compute_assigned_artifacts(
            measures=measures,
            labels=labels,
            atom_coords=atom_coords,
            atom_features=atom_features,
            betas=betas,
            lambda_x=lambda_x,
            lambda_y=lambda_y,
            eps=ot_eps,
            rho=rho,
            align_iters=align_iters,
            allow_reflection=allow_reflection,
            allow_scale=allow_scale,
            cost_scale_x=cost_scale_x,
            cost_scale_y=cost_scale_y,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_penalty=scale_penalty,
            shift_penalty=shift_penalty,
            compute_device=resolved_compute_device,
        )
        atom_coords, atom_features, betas = _update_atoms(
            measures=measures,
            labels=labels,
            plans=plans,
            transforms=transforms,
            thetas=thetas,
            atom_coords=atom_coords,
            atom_features=atom_features,
            beta_smoothing=1e-3,
            random_state=run_seed + iteration,
        )

        label_change_rate = float(np.mean(labels != prev_labels))
        coord_shift = _relative_change(atom_coords, prev_coords)
        feat_shift = _relative_change(atom_features, prev_features)
        mean_obj = float(np.mean(assigned_costs))
        sorted_costs = np.sort(costs, axis=1)
        mean_margin = float(np.mean(sorted_costs[:, 1] - sorted_costs[:, 0])) if sorted_costs.shape[1] >= 2 else float("nan")
        objective_history.append(
            {
                "iteration": int(iteration + 1),
                "mean_objective": mean_obj,
                "label_change_rate": label_change_rate,
                "coord_shift": coord_shift,
                "feature_shift": feat_shift,
                "mean_assignment_margin": mean_margin,
                "forced_label_count": int(forced_label_mask.sum()),
                "assigned_ot_fallback_fraction": float(np.mean(assigned_used_fallback.astype(np.float32))),
            }
        )
        if label_change_rate < 0.005 and max(coord_shift, feat_shift) < tol:
            break

    final_costs = _compute_assignment_costs(
        measures=measures,
        atom_coords=atom_coords,
        atom_features=atom_features,
        betas=betas,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        eps=ot_eps,
        rho=rho,
        align_iters=align_iters,
        allow_reflection=allow_reflection,
        allow_scale=allow_scale,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
        compute_device=resolved_compute_device,
    )
    final_argmin_labels = final_costs.argmin(axis=1).astype(np.int32)
    final_labels, _final_forced_label_mask = _ensure_nonempty_clusters(final_argmin_labels, final_costs, n_clusters)
    _final_plans, _final_transforms, _final_thetas, final_assigned_costs, _final_assigned_effective_eps, final_assigned_used_fallback = _compute_assigned_artifacts(
        measures=measures,
        labels=final_labels,
        atom_coords=atom_coords,
        atom_features=atom_features,
        betas=betas,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        eps=ot_eps,
        rho=rho,
        align_iters=align_iters,
        allow_reflection=allow_reflection,
        allow_scale=allow_scale,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
        compute_device=resolved_compute_device,
    )
    final_objective = float(np.sum(final_assigned_costs))
    return {
        "run": int(run),
        "seed": int(run_seed),
        "objective": final_objective,
        "n_iter": int(len(objective_history)),
        "mean_assigned_cost": float(np.mean(final_assigned_costs)),
        "labels": final_labels.astype(np.int32),
        "costs": final_costs.astype(np.float32),
        "atom_coords": atom_coords.astype(np.float32),
        "atom_features": atom_features.astype(np.float32),
        "betas": betas.astype(np.float32),
        "objective_history": objective_history,
        "device": str(resolved_compute_device),
        "assigned_ot_fallback_fraction": float(np.mean(final_assigned_used_fallback.astype(np.float32))),
        "runtime_memory": _runtime_memory_snapshot(resolved_compute_device),
    }


def fit_multilevel_ot(
    features: np.ndarray,
    coords_um: np.ndarray,
    *,
    subregion_members: list[np.ndarray] | None = None,
    subregion_centers_um: np.ndarray | None = None,
    n_clusters: int,
    atoms_per_cluster: int,
    radius_um: float,
    stride_um: float,
    min_cells: int,
    max_subregions: int,
    lambda_x: float,
    lambda_y: float,
    geometry_eps: float,
    ot_eps: float,
    rho: float,
    geometry_samples: int,
    compressed_support_size: int,
    align_iters: int,
    allow_reflection: bool = False,
    allow_scale: bool = False,
    min_scale: float = 0.75,
    max_scale: float = 1.33,
    scale_penalty: float = 0.05,
    shift_penalty: float = 0.05,
    n_init: int = 5,
    region_geometries: list[RegionGeometry] | None = None,
    build_grid_subregions: bool = True,
    allow_convex_hull_fallback: bool = False,
    max_iter: int = 10,
    tol: float = 1e-4,
    basic_niche_size_um: float | None = 200.0,
    seed: int = 1337,
    compute_device: str = "auto",
) -> MultilevelOTResult:
    features = np.asarray(features, dtype=np.float32)
    coords_um = np.asarray(coords_um, dtype=np.float32)
    resolved_compute_device = _resolve_compute_device(compute_device)
    _validate_fit_inputs(
        features=features,
        coords_um=coords_um,
        n_clusters=n_clusters,
        atoms_per_cluster=atoms_per_cluster,
        radius_um=radius_um,
        stride_um=stride_um,
        basic_niche_size_um=basic_niche_size_um,
        min_cells=min_cells,
        max_subregions=max_subregions,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        geometry_eps=geometry_eps,
        ot_eps=ot_eps,
        rho=rho,
        geometry_samples=geometry_samples,
        compressed_support_size=compressed_support_size,
        align_iters=align_iters,
        max_iter=max_iter,
        tol=tol,
        n_init=n_init,
        min_scale=min_scale,
        max_scale=max_scale,
    )

    features = _standardize_features(features)
    used_basic_niches = False
    basic_niche_centers_um = np.zeros((0, 2), dtype=np.float32)
    basic_niche_members: list[np.ndarray] = []
    subregion_basic_niche_ids: list[np.ndarray] = []
    if subregion_members is None:
        if region_geometries is not None:
            subregion_members = [np.asarray(region.members, dtype=np.int32) for region in region_geometries]
            subregion_basic_niche_ids = [np.asarray([], dtype=np.int32) for _ in subregion_members]
            if subregion_centers_um is None:
                subregion_centers_um = np.vstack(
                    [
                        np.asarray(coords_um[members], dtype=np.float32).mean(axis=0)
                        for members in subregion_members
                    ]
                ).astype(np.float32)
        elif build_grid_subregions:
            if basic_niche_size_um is not None:
                used_basic_niches = True
                (
                    subregion_centers_um,
                    subregion_members,
                    basic_niche_centers_um,
                    basic_niche_members,
                    subregion_basic_niche_ids,
                ) = build_composite_subregions_from_basic_niches(
                    coords_um=coords_um,
                    radius_um=radius_um,
                    stride_um=stride_um,
                    min_cells=min_cells,
                    max_subregions=max_subregions,
                    basic_niche_size_um=float(basic_niche_size_um),
                )
            else:
                subregion_centers_um, subregion_members = build_subregions(
                    coords_um=coords_um,
                    radius_um=radius_um,
                    stride_um=stride_um,
                    min_cells=min_cells,
                    max_subregions=max_subregions,
                )
                subregion_basic_niche_ids = [np.asarray([], dtype=np.int32) for _ in subregion_members]
        else:
            raise ValueError("Explicit region_geometries or subregion_members are required when build_grid_subregions=False.")
    else:
        subregion_members = [np.asarray(members, dtype=np.int32) for members in subregion_members]
        subregion_basic_niche_ids = [np.asarray([], dtype=np.int32) for _ in subregion_members]
        if subregion_centers_um is None:
            subregion_centers_um = np.vstack(
                [
                    np.asarray(coords_um[members], dtype=np.float32).mean(axis=0)
                    for members in subregion_members
                ]
            ).astype(np.float32)
    centers_um = np.asarray(subregion_centers_um, dtype=np.float32)
    if centers_um.shape[0] < n_clusters:
        raise ValueError(
            f"n_clusters={n_clusters} exceeds the number of constructed subregions={centers_um.shape[0]}."
        )

    if region_geometries is None:
        region_geometries = _region_geometries_from_members(subregion_members)
    if len(region_geometries) != len(subregion_members):
        raise ValueError("region_geometries must have the same length as the constructed subregions.")
    for rid, (region, members) in enumerate(zip(region_geometries, subregion_members, strict=False)):
        region.members = np.asarray(members, dtype=np.int32)
        if not region.region_id:
            region.region_id = f"region_{rid:04d}"

    reference_points, reference_weights = make_reference_points_unit_disk(geometry_samples)
    measures = _build_subregion_measures(
        features=features,
        coords_um=coords_um,
        centers_um=centers_um,
        region_geometries=region_geometries,
        geometry_reference_points=reference_points,
        geometry_reference_weights=reference_weights,
        geometry_eps=geometry_eps,
        geometry_samples=geometry_samples,
        compressed_support_size=compressed_support_size,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        seed=seed,
        allow_convex_hull_fallback=allow_convex_hull_fallback,
        compute_device=resolved_compute_device,
    )

    optimization_measures = _make_optimization_measures(measures)
    summaries = np.vstack([_measure_summary(m) for m in optimization_measures])
    cost_scale_x, cost_scale_y = _estimate_cost_scales(measures, max_points=5000, random_state=seed, compute_device=resolved_compute_device)
    restart_params = {
        "n_clusters": int(n_clusters),
        "atoms_per_cluster": int(atoms_per_cluster),
        "lambda_x": float(lambda_x),
        "lambda_y": float(lambda_y),
        "ot_eps": float(ot_eps),
        "rho": float(rho),
        "align_iters": int(align_iters),
        "allow_reflection": bool(allow_reflection),
        "allow_scale": bool(allow_scale),
        "cost_scale_x": float(cost_scale_x),
        "cost_scale_y": float(cost_scale_y),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "scale_penalty": float(scale_penalty),
        "shift_penalty": float(shift_penalty),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "seed": int(seed),
    }
    device_pool = _resolve_cuda_device_pool(str(compute_device), int(n_init)) if resolved_compute_device.type == "cuda" else [str(resolved_compute_device)]
    parallel_restart_workers = _resolve_parallel_restart_workers(device_pool, int(n_init))
    restart_results: list[dict[str, object]] = []
    if parallel_restart_workers > 1:
        total_torch_threads = _env_int("SPATIAL_OT_TORCH_NUM_THREADS", max(torch.get_num_threads(), 1))
        worker_threads = max(1, total_torch_threads // parallel_restart_workers)
        worker_interop_threads = 1
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=parallel_restart_workers,
            mp_context=ctx,
            initializer=_init_restart_worker,
            initargs=(optimization_measures, summaries, restart_params, worker_threads, worker_interop_threads),
        ) as executor:
            future_map = {
                executor.submit(_run_restart_worker, run, device_pool[run % len(device_pool)]): run
                for run in range(int(n_init))
            }
            for future in as_completed(future_map):
                restart_results.append(future.result())
    else:
        for run in range(int(n_init)):
            restart_results.append(
                _execute_restart(
                    measures=optimization_measures,
                    summaries=summaries,
                    run=run,
                    compute_device=device_pool[run % len(device_pool)],
                    **restart_params,
                )
            )
    restart_results.sort(key=lambda result: int(result["run"]))
    restart_summaries = [
        {
            "run": int(result["run"]),
            "seed": int(result["seed"]),
            "objective": float(result["objective"]),
            "n_iter": int(result["n_iter"]),
            "mean_assigned_cost": float(result["mean_assigned_cost"]),
            "device": str(result["device"]),
            "assigned_ot_fallback_fraction": float(result["assigned_ot_fallback_fraction"]),
            "runtime_memory": dict(result.get("runtime_memory", {})),
        }
        for result in restart_results
    ]
    best_bundle = min(restart_results, key=lambda result: float(result["objective"]))
    labels = np.asarray(best_bundle["labels"], dtype=np.int32)
    costs = np.asarray(best_bundle["costs"], dtype=np.float32)
    atom_coords = np.asarray(best_bundle["atom_coords"], dtype=np.float32)
    atom_features = np.asarray(best_bundle["atom_features"], dtype=np.float32)
    betas = np.asarray(best_bundle["betas"], dtype=np.float32)
    objective_history = list(best_bundle["objective_history"])
    best_compute_device = _resolve_compute_device(str(best_bundle["device"]))
    argmin_labels_best = costs.argmin(axis=1).astype(np.int32)
    _labels_checked, forced_label_mask_best = _ensure_nonempty_clusters(argmin_labels_best, costs, n_clusters)
    plans, transforms, thetas, _assigned_costs_best, assigned_effective_eps_best, assigned_used_fallback_best = _compute_assigned_artifacts(
        measures=measures,
        labels=labels,
        atom_coords=atom_coords,
        atom_features=atom_features,
        betas=betas,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        eps=ot_eps,
        rho=rho,
        align_iters=align_iters,
        allow_reflection=allow_reflection,
        allow_scale=allow_scale,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_penalty=scale_penalty,
        shift_penalty=shift_penalty,
        compute_device=best_compute_device,
    )

    thetas_assigned = np.vstack([np.asarray(theta, dtype=np.float32) for theta in thetas]).astype(np.float32)
    subregion_cluster_probs = _softmax_over_negative_costs(costs, temperature=max(float(np.std(costs)), 1e-3))
    cluster_supports = np.concatenate([atom_coords, atom_features], axis=2).astype(np.float32)

    cell_cluster_labels, cell_cluster_probs, cell_feature_probs, cell_context_probs = _project_cells_from_subregions(
        features=features,
        coords_um=coords_um,
        measures=measures,
        subregion_labels=labels,
        atom_coords=atom_coords,
        atom_features=atom_features,
        prototype_weights=betas,
        assigned_transforms=transforms,
        subregion_cluster_costs=costs,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        cost_scale_x=cost_scale_x,
        cost_scale_y=cost_scale_y,
        assignment_temperature=max(cost_scale_y, 1e-3),
        context_weight=0.5,
        compute_device=resolved_compute_device,
    )

    return MultilevelOTResult(
        basic_niche_size_um=float(basic_niche_size_um) if used_basic_niches and basic_niche_size_um is not None else None,
        basic_niche_centers_um=basic_niche_centers_um.astype(np.float32),
        basic_niche_members=[np.asarray(members, dtype=np.int32) for members in basic_niche_members],
        subregion_basic_niche_ids=[np.asarray(niche_ids, dtype=np.int32) for niche_ids in subregion_basic_niche_ids],
        subregion_centers_um=centers_um.astype(np.float32),
        subregion_members=[m.members for m in measures],
        subregion_argmin_labels=argmin_labels_best.astype(np.int32),
        subregion_forced_label_mask=forced_label_mask_best.astype(bool),
        subregion_geometry_point_counts=np.asarray([m.geometry_point_count for m in measures], dtype=np.int32),
        subregion_geometry_sources=[m.normalizer_diagnostics.geometry_source for m in measures],
        subregion_geometry_used_fallback=np.asarray([m.normalizer_diagnostics.used_fallback for m in measures], dtype=bool),
        subregion_normalizer_radius_p95=np.asarray([m.normalizer_diagnostics.mapped_radius_p95 if m.normalizer_diagnostics.mapped_radius_p95 is not None else np.nan for m in measures], dtype=np.float32),
        subregion_normalizer_radius_max=np.asarray([m.normalizer_diagnostics.mapped_radius_max if m.normalizer_diagnostics.mapped_radius_max is not None else np.nan for m in measures], dtype=np.float32),
        subregion_normalizer_interpolation_residual=np.asarray([m.normalizer_diagnostics.interpolation_residual if m.normalizer_diagnostics.interpolation_residual is not None else np.nan for m in measures], dtype=np.float32),
        subregion_cluster_labels=labels,
        subregion_cluster_probs=subregion_cluster_probs.astype(np.float32),
        subregion_cluster_costs=costs.astype(np.float32),
        subregion_atom_weights=thetas_assigned.astype(np.float32),
        subregion_assigned_effective_eps=assigned_effective_eps_best.astype(np.float32),
        subregion_assigned_used_ot_fallback=assigned_used_fallback_best.astype(bool),
        cluster_supports=cluster_supports.astype(np.float32),
        cluster_atom_coords=atom_coords.astype(np.float32),
        cluster_atom_features=atom_features.astype(np.float32),
        cluster_prototype_weights=betas.astype(np.float32),
        cell_feature_cluster_probs=cell_feature_probs.astype(np.float32),
        cell_context_cluster_probs=cell_context_probs.astype(np.float32),
        cell_cluster_probs=cell_cluster_probs.astype(np.float32),
        cell_cluster_labels=cell_cluster_labels.astype(np.int32),
        cost_scale_x=float(cost_scale_x),
        cost_scale_y=float(cost_scale_y),
        objective_history=objective_history,
        selected_restart=int(best_bundle["run"]),
        restart_summaries=restart_summaries,
    )
