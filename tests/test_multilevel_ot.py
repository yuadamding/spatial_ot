from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from spatial_ot.multilevel_ot import (
    RegionGeometry,
    ShapeNormalizer,
    ShapeNormalizerDiagnostics,
    SubregionMeasure,
    _build_subregion_measures,
    _compute_assignment_costs,
    _initialize_cluster_atoms,
    _project_cells_from_subregions,
    _shape_descriptor_frame,
    _shape_leakage_balanced_accuracy,
    _standardize_features,
    _ensure_nonempty_clusters,
    weighted_similarity_fit,
    sample_geometry_points,
    build_subregions,
    fit_multilevel_ot,
    fit_ot_shape_normalizer,
    make_reference_points_unit_disk,
)


def test_build_subregions_respects_min_cells() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [10.0, 11.0],
            [11.0, 11.0],
        ],
        dtype=np.float32,
    )
    centers, members = build_subregions(
        coords_um=coords,
        radius_um=2.0,
        stride_um=10.0,
        min_cells=3,
        max_subregions=10,
    )
    assert centers.shape[0] >= 2
    assert all(len(m) >= 3 for m in members)


def test_multilevel_ot_recovers_two_subregion_families() -> None:
    rng = np.random.default_rng(1337)
    group_centers = np.array(
        [
            [0.0, 0.0],
            [0.0, 30.0],
            [100.0, 0.0],
            [100.0, 30.0],
        ],
        dtype=np.float32,
    )
    true_group = np.array([0, 0, 1, 1], dtype=np.int32)

    feature_atoms = {
        0: np.array([[0.0, 0.0, 0.0], [1.0, 0.2, -0.1]], dtype=np.float32),
        1: np.array([[5.0, 5.0, 5.0], [6.0, 4.8, 5.2]], dtype=np.float32),
    }

    coords = []
    features = []
    cell_truth = []
    for group_idx, center in enumerate(group_centers):
        atoms = feature_atoms[int(true_group[group_idx])]
        mix = np.array([0.75, 0.25], dtype=np.float32) if true_group[group_idx] == 0 else np.array([0.2, 0.8], dtype=np.float32)
        n_cells = 36
        atom_ids = rng.choice(2, size=n_cells, p=mix)
        feat = atoms[atom_ids] + rng.normal(scale=0.08, size=(n_cells, 3)).astype(np.float32)
        xy = center + rng.normal(scale=1.0, size=(n_cells, 2)).astype(np.float32)
        coords.append(xy)
        features.append(feat)
        cell_truth.extend([int(true_group[group_idx])] * n_cells)

    coords = np.vstack(coords).astype(np.float32)
    features = np.vstack(features).astype(np.float32)
    cell_truth = np.asarray(cell_truth, dtype=np.int32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        n_clusters=2,
        atoms_per_cluster=2,
        radius_um=15.0,
        stride_um=30.0,
        min_cells=20,
        max_subregions=16,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=96,
        compressed_support_size=24,
        align_iters=3,
        allow_reflection=True,
        allow_scale=True,
        allow_convex_hull_fallback=True,
        max_iter=8,
        tol=1e-4,
        seed=1337,
    )

    subregion_truth = []
    for members in result.subregion_members:
        labels, counts = np.unique(cell_truth[members], return_counts=True)
        subregion_truth.append(int(labels[np.argmax(counts)]))
    subregion_truth = np.asarray(subregion_truth, dtype=np.int32)

    ari = adjusted_rand_score(subregion_truth, result.subregion_cluster_labels)
    assert ari > 0.8
    assert result.cell_cluster_probs.shape[1] == 2
    assert np.unique(result.cell_cluster_labels).size == 2
    assert np.array_equal(
        result.subregion_cluster_labels,
        result.subregion_cluster_costs.argmin(axis=1),
    )
    assert result.cluster_atom_coords.shape == (2, 2, 2)
    assert result.cluster_atom_features.shape == (2, 2, 3)
    assert result.subregion_assigned_effective_eps.shape[0] == len(result.subregion_members)
    assert np.all(result.subregion_assigned_effective_eps > 0)


def test_shape_normalizer_reduces_boundary_shape_difference() -> None:
    q, w = make_reference_points_unit_disk(128)
    circle = np.stack(
        [
            np.cos(np.linspace(0, 2 * np.pi, 180, endpoint=False)),
            np.sin(np.linspace(0, 2 * np.pi, 180, endpoint=False)),
        ],
        axis=1,
    ).astype(np.float32)
    ellipse = circle.copy()
    ellipse[:, 0] *= 2.5

    phi_circle, _ = fit_ot_shape_normalizer(circle, q, w, eps_geom=0.03)
    phi_ellipse, _ = fit_ot_shape_normalizer(ellipse, q, w, eps_geom=0.03)

    mapped_circle = phi_circle.transform(circle)
    mapped_ellipse = phi_ellipse.transform(ellipse)

    def radial_stats(x: np.ndarray) -> tuple[float, float]:
        r = np.sqrt(np.sum(x**2, axis=1))
        return float(r.mean()), float(r.std())

    mean_circle, std_circle = radial_stats(mapped_circle)
    mean_ellipse, std_ellipse = radial_stats(mapped_ellipse)

    assert abs(mean_circle - mean_ellipse) < 0.15
    assert abs(std_circle - std_ellipse) < 0.15


def test_shape_leakage_diagnostic_runs() -> None:
    coords = np.vstack(
        [
            np.random.default_rng(0).normal(loc=[0.0, 0.0], scale=0.2, size=(30, 2)),
            np.random.default_rng(1).normal(loc=[3.0, 0.0], scale=[1.2, 0.1], size=(30, 2)),
            np.random.default_rng(2).normal(loc=[0.0, 3.0], scale=0.2, size=(30, 2)),
            np.random.default_rng(3).normal(loc=[3.0, 3.0], scale=[1.2, 0.1], size=(30, 2)),
        ]
    ).astype(np.float32)
    members = [
        np.arange(0, 30, dtype=np.int32),
        np.arange(30, 60, dtype=np.int32),
        np.arange(60, 90, dtype=np.int32),
        np.arange(90, 120, dtype=np.int32),
    ]
    labels = np.array([0, 1, 0, 1], dtype=np.int32)
    shape_df = _shape_descriptor_frame(members, coords)
    score = _shape_leakage_balanced_accuracy(shape_df, labels, seed=0)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_cell_projection_boundary_uses_latent_geometry_not_subregion_centers() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0],
        ],
        dtype=np.float32,
    )
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    atom_features = np.array(
        [
            [[0.0, 0.0], [0.1, -0.1]],
            [[10.0, 10.0], [9.9, 10.1]],
        ],
        dtype=np.float32,
    )
    atom_coords = np.array(
        [
            [[0.0, 0.0], [0.1, 0.0]],
            [[1.0, 1.0], [1.1, 1.0]],
        ],
        dtype=np.float32,
    )
    prototype_weights = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    measures = [
        SubregionMeasure(
            subregion_id=0,
            center_um=np.array([0.5, 0.5], dtype=np.float32),
            members=np.array([0, 1], dtype=np.int32),
            canonical_coords=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            features=features.copy(),
            weights=np.array([0.5, 0.5], dtype=np.float32),
            geometry_point_count=2,
            compressed_point_count=2,
            normalizer=ShapeNormalizer(center=np.zeros((1, 2)), scale=1.0, interpolator=None),
            normalizer_diagnostics=ShapeNormalizerDiagnostics(
                geometry_source="test",
                used_fallback=False,
                ot_cost=None,
                sinkhorn_converged=None,
                mapped_radius_p95=None,
                mapped_radius_max=None,
                interpolation_residual=None,
            ),
        ),
        SubregionMeasure(
            subregion_id=1,
            center_um=np.array([0.5, 0.5], dtype=np.float32),
            members=np.array([0, 1], dtype=np.int32),
            canonical_coords=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            features=features.copy(),
            weights=np.array([0.5, 0.5], dtype=np.float32),
            geometry_point_count=2,
            compressed_point_count=2,
            normalizer=ShapeNormalizer(center=np.zeros((1, 2)), scale=1.0, interpolator=None),
            normalizer_diagnostics=ShapeNormalizerDiagnostics(
                geometry_source="test",
                used_fallback=False,
                ot_cost=None,
                sinkhorn_converged=None,
                mapped_radius_p95=None,
                mapped_radius_max=None,
                interpolation_residual=None,
            ),
        ),
    ]
    subregion_labels = np.array([0, 1], dtype=np.int32)
    transforms = [
        {"R": np.eye(2), "scale": 1.0, "t": np.zeros(2)},
        {"R": np.eye(2), "scale": 1.0, "t": np.zeros(2)},
    ]
    # Both cells are covered by the same two subregions, so the context alone is symmetric.
    subregion_cluster_costs = np.array(
        [
            [0.1, 1.0],
            [1.0, 0.1],
        ],
        dtype=np.float32,
    )

    labels, combined, feature_probs, context_probs = _project_cells_from_subregions(
        features=features,
        coords_um=coords,
        measures=measures,
        subregion_labels=subregion_labels,
        atom_coords=atom_coords,
        atom_features=atom_features,
        prototype_weights=prototype_weights,
        assigned_transforms=transforms,
        subregion_cluster_costs=subregion_cluster_costs,
        lambda_x=0.5,
        lambda_y=1.0,
        cost_scale_x=1.0,
        cost_scale_y=1.0,
        assignment_temperature=0.35,
        context_weight=0.5,
    )

    # Context is symmetric across the two cells because they share the same subregions.
    assert np.allclose(context_probs[0], context_probs[1])
    # The split must therefore come from latent geometry.
    assert labels.tolist() == [0, 1]
    assert feature_probs[0, 0] > feature_probs[0, 1]
    assert feature_probs[1, 1] > feature_probs[1, 0]
    assert np.allclose(combined.sum(axis=1), 1.0)


def test_initialize_cluster_atoms_respects_lambda_scaling() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ],
        dtype=np.float32,
    )
    features = np.array(
        [
            [1.0, 1.0],
            [1.2, 1.0],
            [50.0, 50.0],
            [49.8, 50.1],
        ],
        dtype=np.float32,
    )
    centers = np.array([[0.0, 0.0]], dtype=np.float32)
    regions = [RegionGeometry(region_id="r0", members=np.arange(coords.shape[0], dtype=np.int32))]
    q, w = make_reference_points_unit_disk(64)
    measures = _build_subregion_measures(
        features=features,
        coords_um=coords,
        centers_um=centers,
        region_geometries=regions,
        geometry_reference_points=q,
        geometry_reference_weights=w,
        geometry_eps=0.03,
        geometry_samples=64,
        compressed_support_size=4,
        lambda_x=0.25,
        lambda_y=4.0,
        seed=0,
        allow_convex_hull_fallback=True,
    )
    atom_coords, atom_features, betas = _initialize_cluster_atoms(
        measures=measures,
        labels=np.array([0], dtype=np.int32),
        n_clusters=1,
        atoms_per_cluster=2,
        lambda_x=0.25,
        lambda_y=4.0,
        random_state=0,
    )
    assert atom_coords.shape == (1, 2, 2)
    assert atom_features.shape == (1, 2, 2)
    assert np.max(np.abs(atom_coords[0])) < 5.0
    assert np.max(atom_features[0]) > 10.0
    assert np.allclose(betas.sum(axis=1), 1.0)


def test_sample_geometry_points_prefers_explicit_polygon_geometry() -> None:
    region = RegionGeometry(
        region_id="poly",
        members=np.arange(10, dtype=np.int32),
        polygon_vertices=np.array(
            [
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [0.0, 4.0],
            ],
            dtype=np.float32,
        ),
    )
    observed = np.column_stack([np.linspace(0.0, 0.5, 10), np.linspace(0.0, 0.5, 10)]).astype(np.float32)
    pts, source, used_fallback = sample_geometry_points(
        region,
        observed_coords=observed,
        n_points=256,
        seed=11,
        allow_convex_hull_fallback=True,
    )
    assert source == "polygon"
    assert not used_fallback
    assert pts[:, 0].max() > 3.0
    assert pts[:, 1].max() > 3.0


def test_returned_costs_match_returned_atoms() -> None:
    rng = np.random.default_rng(7)
    coords = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.6, size=(40, 2)),
            rng.normal(loc=[15.0, 0.0], scale=0.6, size=(40, 2)),
            rng.normal(loc=[0.0, 15.0], scale=0.6, size=(40, 2)),
        ]
    ).astype(np.float32)
    features = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0, 0.0], scale=0.15, size=(40, 3)),
            rng.normal(loc=[4.0, 4.0, 4.0], scale=0.15, size=(40, 3)),
            rng.normal(loc=[0.0, 4.0, 2.0], scale=0.15, size=(40, 3)),
        ]
    ).astype(np.float32)

    result = fit_multilevel_ot(
        features=features,
        coords_um=coords,
        n_clusters=3,
        atoms_per_cluster=3,
        radius_um=10.0,
        stride_um=12.0,
        min_cells=15,
        max_subregions=20,
        lambda_x=0.5,
        lambda_y=1.0,
        geometry_eps=0.03,
        ot_eps=0.03,
        rho=0.5,
        geometry_samples=64,
        compressed_support_size=16,
        align_iters=2,
        allow_reflection=False,
        allow_scale=False,
        allow_convex_hull_fallback=True,
        max_iter=3,
        tol=1e-4,
        seed=7,
    )
    q, w = make_reference_points_unit_disk(64)
    regions = [RegionGeometry(region_id=f"r{i}", members=m) for i, m in enumerate(result.subregion_members)]
    measures = _build_subregion_measures(
        features=_standardize_features(features),
        coords_um=coords,
        centers_um=result.subregion_centers_um,
        region_geometries=regions,
        geometry_reference_points=q,
        geometry_reference_weights=w,
        geometry_eps=0.03,
        geometry_samples=64,
        compressed_support_size=16,
        lambda_x=0.5,
        lambda_y=1.0,
        seed=7,
        allow_convex_hull_fallback=True,
    )
    recomputed = _compute_assignment_costs(
        measures=measures,
        atom_coords=result.cluster_atom_coords,
        atom_features=result.cluster_atom_features,
        betas=result.cluster_prototype_weights,
        lambda_x=0.5,
        lambda_y=1.0,
        eps=0.03,
        rho=0.5,
        align_iters=2,
        allow_reflection=False,
        allow_scale=False,
        cost_scale_x=result.cost_scale_x,
        cost_scale_y=result.cost_scale_y,
        min_scale=0.75,
        max_scale=1.33,
        scale_penalty=0.05,
        shift_penalty=0.05,
    )
    assert recomputed.shape == result.subregion_cluster_costs.shape
    assert np.array_equal(result.subregion_cluster_labels, result.subregion_cluster_costs.argmin(axis=1))
    assert np.allclose(recomputed, result.subregion_cluster_costs, atol=1e-4)
    assert result.subregion_assigned_used_ot_fallback.shape[0] == len(result.subregion_members)


def test_empty_mask_geometry_raises() -> None:
    region = RegionGeometry(
        region_id="empty",
        members=np.arange(5, dtype=np.int32),
        mask=np.zeros((8, 8), dtype=bool),
    )
    observed = np.random.default_rng(0).normal(size=(5, 2)).astype(np.float32)
    try:
        sample_geometry_points(
            region,
            observed_coords=observed,
            n_points=32,
            seed=0,
            allow_convex_hull_fallback=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected empty mask geometry to raise")


def test_fit_requires_explicit_fallback_flag_for_observed_hull_geometry() -> None:
    coords = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [5.0, 5.0], [5.5, 5.0], [5.0, 5.5]], dtype=np.float32)
    features = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]], dtype=np.float32)
    try:
        fit_multilevel_ot(
            features=features,
            coords_um=coords,
            n_clusters=2,
            atoms_per_cluster=1,
            radius_um=1.0,
            stride_um=5.0,
            min_cells=3,
            max_subregions=4,
            lambda_x=0.5,
            lambda_y=1.0,
            geometry_eps=0.03,
            ot_eps=0.03,
            rho=0.5,
            geometry_samples=32,
            compressed_support_size=3,
            align_iters=1,
            allow_reflection=False,
            allow_scale=False,
            max_iter=1,
            tol=1e-4,
            seed=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected fit_multilevel_ot to require an explicit hull-fallback opt-in")


def test_weighted_similarity_fit_no_reflection_keeps_positive_determinant() -> None:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    y = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    w = np.array([0.2, 0.4, 0.4], dtype=np.float64)
    transform = weighted_similarity_fit(x, y, w, allow_reflection=False, allow_scale=True)
    assert np.linalg.det(transform["R"]) > 0
    assert transform["scale"] > 0


def test_ensure_nonempty_clusters_returns_forced_mask() -> None:
    labels = np.array([0, 0, 0, 1], dtype=np.int32)
    costs = np.array(
        [
            [0.1, 1.0, 0.9],
            [0.2, 1.1, 0.8],
            [0.3, 0.9, 0.4],
            [1.0, 0.1, 0.6],
        ],
        dtype=np.float32,
    )
    repaired, forced = _ensure_nonempty_clusters(labels, costs, n_clusters=3)
    assert set(repaired.tolist()) == {0, 1, 2}
    assert forced.sum() == 1
