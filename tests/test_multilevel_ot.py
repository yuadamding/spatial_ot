from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score

from spatial_ot.multilevel_ot import (
    _project_cells_from_subregions,
    _shape_descriptor_frame,
    _shape_leakage_balanced_accuracy,
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

    phi_circle = fit_ot_shape_normalizer(circle, q, w, eps_geom=0.03)
    phi_ellipse = fit_ot_shape_normalizer(ellipse, q, w, eps_geom=0.03)

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
    subregion_members = [np.array([0, 1], dtype=np.int32), np.array([0, 1], dtype=np.int32)]
    supports = np.array(
        [
            [[0.0, 0.0], [0.1, -0.1]],
            [[10.0, 10.0], [9.9, 10.1]],
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
        subregion_members=subregion_members,
        supports=supports,
        prototype_weights=prototype_weights,
        subregion_cluster_costs=subregion_cluster_costs,
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
