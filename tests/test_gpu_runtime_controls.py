from __future__ import annotations

import numpy as np
import pytest
import torch

from spatial_ot.config import DeepFeatureConfig
from spatial_ot.deep import SpatialOTFeatureEncoder
from spatial_ot.multilevel.core import (
    _cluster_cost_matrix,
    _cuda_target_bytes,
    _pairwise_sqdist_array,
    _resolve_cuda_device_pool,
    _resolve_parallel_restart_workers,
)


class _DummyCudaProps:
    def __init__(self, total_memory: int) -> None:
        self.total_memory = total_memory


def test_resolve_cuda_device_pool_uses_all_visible_gpus(monkeypatch) -> None:
    monkeypatch.delenv("SPATIAL_OT_CUDA_DEVICE_LIST", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    devices = _resolve_cuda_device_pool("cuda", n_init=5)
    assert devices == ["cuda:0", "cuda:1"]


def test_resolve_cuda_device_pool_honors_explicit_list(monkeypatch) -> None:
    monkeypatch.setenv("SPATIAL_OT_CUDA_DEVICE_LIST", "1,3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    devices = _resolve_cuda_device_pool("cuda", n_init=5)
    assert devices == ["cuda:1", "cuda:3"]


def test_resolve_parallel_restart_workers_defaults_to_device_count(monkeypatch) -> None:
    monkeypatch.setenv("SPATIAL_OT_PARALLEL_RESTARTS", "auto")
    workers = _resolve_parallel_restart_workers(["cuda:0", "cuda:1"], n_init=5)
    assert workers == 2


def test_cuda_target_bytes_respects_visible_gpu_capacity(monkeypatch) -> None:
    monkeypatch.setenv("SPATIAL_OT_CUDA_TARGET_VRAM_GB", "50")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: _DummyCudaProps(total_memory=10 * (1024**3)),
    )

    target_bytes = _cuda_target_bytes(device=torch.device("cuda:0"))
    assert target_bytes == int(9 * (1024**3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this environment")
def test_pairwise_sqdist_array_cpu_gpu_parity() -> None:
    rng = np.random.default_rng(1234)
    x = rng.normal(size=(17, 4)).astype(np.float32)
    y = rng.normal(size=(13, 4)).astype(np.float32)

    cpu = _pairwise_sqdist_array(x, y, device=torch.device("cpu"))
    gpu = _pairwise_sqdist_array(
        x,
        y,
        device=torch.device("cuda:0"),
        dtype=torch.float64,
        row_batch_size=5,
    )
    assert np.allclose(cpu, gpu, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this environment")
def test_cluster_cost_matrix_cpu_gpu_parity() -> None:
    rng = np.random.default_rng(5678)
    u_aligned = rng.normal(size=(11, 2)).astype(np.float32)
    y = rng.normal(size=(11, 3)).astype(np.float32)
    atom_coords = rng.normal(size=(7, 2)).astype(np.float32)
    atom_features = rng.normal(size=(7, 3)).astype(np.float32)

    cpu = _cluster_cost_matrix(
        u_aligned,
        y,
        atom_coords,
        atom_features,
        cost_scale_x=1.3,
        cost_scale_y=0.7,
        lambda_x=0.5,
        lambda_y=1.1,
        compute_device=torch.device("cpu"),
    )
    gpu = _cluster_cost_matrix(
        u_aligned,
        y,
        atom_coords,
        atom_features,
        cost_scale_x=1.3,
        cost_scale_y=0.7,
        lambda_x=0.5,
        lambda_y=1.1,
        compute_device=torch.device("cuda:0"),
    )
    assert np.allclose(cpu, gpu.detach().cpu().numpy(), atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this environment")
def test_saved_encoder_transform_cpu_gpu_parity(tmp_path) -> None:
    rng = np.random.default_rng(2468)
    features = rng.normal(size=(24, 5)).astype(np.float32)
    coords = rng.normal(size=(24, 2)).astype(np.float32)
    encoder = SpatialOTFeatureEncoder(
        DeepFeatureConfig(
            method="autoencoder",
            latent_dim=3,
            hidden_dim=16,
            layers=1,
            neighbor_k=3,
            epochs=2,
            batch_size=8,
            validation="none",
            device="cpu",
            output_embedding="intrinsic",
        )
    )
    encoder.fit(features=features, coords_um=coords, seed=17)
    model_path = tmp_path / "cpu_gpu_parity_model.pt"
    encoder.save(model_path)

    cpu_loaded = SpatialOTFeatureEncoder.load(model_path, map_location="cpu", device="cpu")
    gpu_loaded = SpatialOTFeatureEncoder.load(model_path, map_location="cpu", device="cuda")
    cpu_embedding = cpu_loaded.transform(features=features, coords_um=coords)
    gpu_embedding = gpu_loaded.transform(features=features, coords_um=coords)
    assert np.allclose(cpu_embedding, gpu_embedding, atol=1e-5, rtol=1e-5)
