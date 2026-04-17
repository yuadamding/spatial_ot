from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from spatial_ot.communication import _masked_sinkhorn
from spatial_ot.nn import aggregate_mean
from spatial_ot.ot import _build_shell_ground
from spatial_ot.preprocessing import _extract_cell_types, aggregate_mean_numpy, aggregate_sum_numpy


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
