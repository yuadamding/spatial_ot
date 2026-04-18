from __future__ import annotations

from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from spatial_ot.communication import _masked_sinkhorn
from spatial_ot.config import load_config
from spatial_ot.nn import aggregate_mean, sample_negative_edges
from spatial_ot.ot import _build_shell_ground
from spatial_ot.preprocessing import _extract_cell_types, _resolve_raw_counts, _spatial_grid_subset, aggregate_mean_numpy, aggregate_sum_numpy
from spatial_ot.programs import load_programs


def test_aggregate_mean_numpy_uses_neighbor_values_on_focal_rows() -> None:
    values = np.array([[1.0], [10.0], [100.0]], dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)

    mean_out = aggregate_mean_numpy(values, edge_index)
    sum_out = aggregate_sum_numpy(values, edge_index)

    assert np.allclose(mean_out[:, 0], [10.0, 100.0, 0.0])
    assert np.allclose(sum_out[:, 0], [10.0, 100.0, 0.0])


def test_aggregate_mean_torch_uses_neighbor_values_on_focal_rows() -> None:
    values = torch.tensor([[1.0], [10.0], [100.0]])
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    out = aggregate_mean(values, edge_index)

    assert torch.allclose(out[:, 0], torch.tensor([10.0, 100.0, 0.0]))


def test_shell_ground_has_zero_diagonal_and_positive_cross_shell_costs() -> None:
    ground = _build_shell_ground(n_shells=3, n_atoms=2)
    assert np.allclose(np.diag(ground), 0.0)
    assert np.all(ground >= 0.0)
    assert ground[0, 2] > 0.0
    assert ground[0, 4] > ground[0, 1]


def test_masked_sinkhorn_respects_support() -> None:
    a = np.array([0.4, 0.6], dtype=np.float32)
    b = np.array([0.5, 0.5], dtype=np.float32)
    cost = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, True]])

    transport = _masked_sinkhorn(a=a, b=b, cost=cost, mask=mask, reg=0.3, num_iter=200)

    assert transport.shape == cost.shape
    assert np.allclose(transport[~mask], 0.0)
    assert np.isclose(float(transport.sum()), 1.0, atol=1e-4)


def test_masked_sinkhorn_handles_empty_support() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    cost = np.ones((2, 2), dtype=np.float32)
    mask = np.zeros((2, 2), dtype=bool)

    transport = _masked_sinkhorn(a=a, b=b, cost=cost, mask=mask, reg=0.3, num_iter=50)

    assert transport.shape == (2, 2)
    assert np.allclose(transport, 0.0)


def test_negative_edge_sampling_excludes_positives_self_loops_and_duplicates() -> None:
    pos = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    neg = sample_negative_edges(n_nodes=4, positive_edge_index=pos, ratio=2.0)
    pairs = list(zip(neg[0].tolist(), neg[1].tolist()))
    assert len(pairs) == len(set(pairs))
    pos_pairs = set(zip(pos[0].tolist(), pos[1].tolist()))
    assert pos_pairs.isdisjoint(set(pairs))
    assert all(s != d for s, d in pairs)


def test_extract_cell_types_requires_configured_key() -> None:
    obs = pd.DataFrame({"coarse_cell_type": ["a", "b", "a"]})
    labels, names, onehot = _extract_cell_types(obs, "coarse_cell_type")
    assert list(names) == ["a", "b"]
    assert labels.tolist() == [0, 1, 0]
    assert onehot.shape == (3, 2)

    try:
        _extract_cell_types(obs, "missing_key")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError for missing configured cell_type_key")


def test_stratified_subset_requires_real_label_column() -> None:
    obs = pd.DataFrame({"other": ["a", "b", "a"]})
    try:
        _extract_cell_types(obs, "coarse_cell_type")
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError instead of fallback to the first obs column")


def test_spatial_grid_subset_preserves_local_patches() -> None:
    coords = np.stack([np.arange(1000, dtype=np.float32), np.zeros(1000, dtype=np.float32)], axis=1)
    keep = _spatial_grid_subset(coords, limit=80, seed=1337)
    assert keep.shape[0] == 80
    runs = 1 + int(np.sum(np.diff(np.sort(keep)) > 1))
    assert runs <= 4


def test_resolve_raw_counts_rejects_non_integer_like_values() -> None:
    adata = ad.AnnData(X=np.array([[0.1, 1.2], [2.0, 3.3]], dtype=np.float32))
    try:
        _resolve_raw_counts(adata, source_name="unit_test")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected raw-count validation to reject non-integer-like values")

    adata_ok = ad.AnnData(X=sparse.csr_matrix(np.array([[0, 1], [2, 3]], dtype=np.int32)))
    matrix, source = _resolve_raw_counts(adata_ok, source_name="unit_test_ok")
    assert source == "X"
    assert matrix.shape == (2, 2)


def test_program_loader_validates_duplicates() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "programs.json"
        path.write_text(
            """
            {
              "programs": [
                {"name": "a", "self_genes": ["G1"]},
                {"name": "a", "self_genes": ["G2"]}
              ]
            }
            """
        )
        try:
            load_programs(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected duplicate program names to raise")


def test_config_validation_rejects_unknown_and_removed_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.toml"
        path.write_text(
            """
            [paths]
            cells_h5ad = "cells.h5ad"
            bins8_h5 = "bins.h5"
            bins8_positions = "bins.parquet"
            bins2_h5 = "unused.h5"
            output_dir = "runs/test"

            [data]
            subset_strategy = "spatial_knn"
            cell_type_key = "coarse_cell_type"
            hvg_n = 100
            teacher_knn = 4
            context_knn = 8
            ot_knn = 8
            shell_bounds_um = [12.0, 25.0, 50.0]
            negative_edge_ratio = 1.0
            """
        )
        try:
            load_config(path)
        except KeyError:
            pass
        else:
            raise AssertionError("Expected removed inactive config fields to be rejected")


def test_config_validation_rejects_legacy_subset_strategy() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.toml"
        path.write_text(
            """
            [paths]
            cells_h5ad = "cells.h5ad"
            bins8_h5 = "bins.h5"
            bins8_positions = "bins.parquet"
            output_dir = "runs/test"

            [data]
            subset_strategy = "spatial_knn"
            cell_type_key = "coarse_cell_type"
            hvg_n = 100
            teacher_knn = 4
            context_knn = 8
            ot_knn = 8
            shell_bounds_um = [12.0, 25.0, 50.0]
            negative_edge_ratio = 1.0
            """
        )
        try:
            load_config(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected legacy subset_strategy to be rejected explicitly")
