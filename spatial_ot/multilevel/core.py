from __future__ import annotations

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
    fit_ot_shape_normalizer,
    make_reference_points_unit_disk,
    sample_geometry_points,
    _region_geometries_from_members,
)
from .types import MultilevelOTResult, OTSolveDiagnostics, RegionGeometry, SubregionMeasure


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
    batch = int(row_batch_size or max(256, min(4096, x_t.shape[0])))
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


def _solve_semirelaxed_unbalanced(
    a: np.ndarray,
    beta: np.ndarray,
    c: np.ndarray,
    eps: float,
    rho: float,
    compute_device: torch.device,
) -> tuple[np.ndarray, float, OTSolveDiagnostics]:
    regs = [max(float(eps), 1e-5)]
    regs.extend([regs[0] * 2.0, regs[0] * 4.0, regs[0] * 8.0])
    last_gamma = None
    last_objective = 1e12
    last_reg = regs[0]
    if compute_device.type == "cpu":
        a_backend = np.asarray(a, dtype=np.float64)
        beta_backend = np.asarray(beta, dtype=np.float64)
    else:
        a_backend = torch.as_tensor(a, dtype=torch.float64, device=compute_device)
        beta_backend = torch.as_tensor(beta, dtype=torch.float64, device=compute_device)
    for reg in regs:
        c_backend = np.asarray(c, dtype=np.float64) if compute_device.type == "cpu" else torch.as_tensor(c, dtype=torch.float64, device=compute_device)
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
        if compute_device.type == "cpu":
            cx = ot.dist(u_aligned, atom_coords, metric="sqeuclidean") / max(cost_scale_x, 1e-12)
            cy = ot.dist(y, atom_features, metric="sqeuclidean") / max(cost_scale_y, 1e-12)
        else:
            u_aligned_t = torch.as_tensor(u_aligned, dtype=torch.float64, device=compute_device)
            cx = (torch.cdist(u_aligned_t, atom_coords_t, p=2) ** 2 / max(cost_scale_x, 1e-12)).detach().cpu().numpy()
            cy = (torch.cdist(y_t, atom_features_t, p=2) ** 2 / max(cost_scale_y, 1e-12)).detach().cpu().numpy()
        c = lambda_x * cx + lambda_y * cy
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
    if compute_device.type == "cpu":
        cx = ot.dist(u_aligned, atom_coords, metric="sqeuclidean") / max(cost_scale_x, 1e-12)
        cy = ot.dist(y, atom_features, metric="sqeuclidean") / max(cost_scale_y, 1e-12)
    else:
        u_aligned_t = torch.as_tensor(u_aligned, dtype=torch.float64, device=compute_device)
        cx = (torch.cdist(u_aligned_t, atom_coords_t, p=2) ** 2 / max(cost_scale_x, 1e-12)).detach().cpu().numpy()
        cy = (torch.cdist(y_t, atom_features_t, p=2) ** 2 / max(cost_scale_y, 1e-12)).detach().cpu().numpy()
    c = lambda_x * cx + lambda_y * cy
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
    costs = np.zeros((n_cells, n_clusters), dtype=np.float32)
    for k in range(n_clusters):
        dist = _pairwise_sqdist_array(features, support_features[k], device=compute_device)
        weights = np.clip(prototype_weights[k], 1e-8, None).astype(np.float32)
        scores = np.exp(-dist / max(float(temperature), 1e-5)) * weights[None, :]
        costs[:, k] = -max(float(temperature), 1e-5) * np.log(np.maximum(scores.sum(axis=1), 1e-8))
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
    if subregion_members is None:
        if region_geometries is not None:
            subregion_members = [np.asarray(region.members, dtype=np.int32) for region in region_geometries]
            if subregion_centers_um is None:
                subregion_centers_um = np.vstack(
                    [
                        np.asarray(coords_um[members], dtype=np.float32).mean(axis=0)
                        for members in subregion_members
                    ]
                ).astype(np.float32)
        elif build_grid_subregions:
            subregion_centers_um, subregion_members = build_subregions(
                coords_um=coords_um,
                radius_um=radius_um,
                stride_um=stride_um,
                min_cells=min_cells,
                max_subregions=max_subregions,
            )
        else:
            raise ValueError("Explicit region_geometries or subregion_members are required when build_grid_subregions=False.")
    else:
        subregion_members = [np.asarray(members, dtype=np.int32) for members in subregion_members]
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
    )

    summaries = np.vstack([_measure_summary(m) for m in measures])
    cost_scale_x, cost_scale_y = _estimate_cost_scales(measures, max_points=5000, random_state=seed, compute_device=resolved_compute_device)
    best_bundle: dict[str, object] | None = None
    restart_summaries: list[dict[str, float | int]] = []

    for run in range(int(n_init)):
        run_seed = seed + 1000 * run
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
        final_labels, final_forced_label_mask = _ensure_nonempty_clusters(final_argmin_labels, final_costs, n_clusters)
        final_plans, final_transforms, final_thetas, final_assigned_costs, final_assigned_effective_eps, final_assigned_used_fallback = _compute_assigned_artifacts(
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
        restart_summaries.append(
            {
                "run": int(run),
                "seed": int(run_seed),
                "objective": final_objective,
                "n_iter": int(len(objective_history)),
                "mean_assigned_cost": float(np.mean(final_assigned_costs)),
            }
        )
        if best_bundle is None or final_objective < float(best_bundle["objective"]):
            best_bundle = {
                "run": int(run),
                "objective": final_objective,
                "labels": final_labels.astype(np.int32),
                "costs": final_costs.astype(np.float32),
                "plans": final_plans,
                "transforms": final_transforms,
                "thetas": final_thetas,
                "atom_coords": atom_coords.astype(np.float32),
                "atom_features": atom_features.astype(np.float32),
                "betas": betas.astype(np.float32),
                "objective_history": objective_history,
                "forced_label_mask": final_forced_label_mask.astype(bool),
                "argmin_labels": final_argmin_labels.astype(np.int32),
                "assigned_effective_eps": final_assigned_effective_eps.astype(np.float32),
                "assigned_used_fallback": final_assigned_used_fallback.astype(bool),
            }

    assert best_bundle is not None
    labels = np.asarray(best_bundle["labels"], dtype=np.int32)
    costs = np.asarray(best_bundle["costs"], dtype=np.float32)
    atom_coords = np.asarray(best_bundle["atom_coords"], dtype=np.float32)
    atom_features = np.asarray(best_bundle["atom_features"], dtype=np.float32)
    betas = np.asarray(best_bundle["betas"], dtype=np.float32)
    transforms = list(best_bundle["transforms"])
    thetas = list(best_bundle["thetas"])
    objective_history = list(best_bundle["objective_history"])
    forced_label_mask_best = np.asarray(best_bundle["forced_label_mask"], dtype=bool)
    argmin_labels_best = np.asarray(best_bundle["argmin_labels"], dtype=np.int32)
    assigned_effective_eps_best = np.asarray(best_bundle["assigned_effective_eps"], dtype=np.float32)
    assigned_used_fallback_best = np.asarray(best_bundle["assigned_used_fallback"], dtype=bool)

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
