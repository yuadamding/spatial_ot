from __future__ import annotations

import torch

from spatial_ot.multilevel.core import _cuda_target_bytes, _resolve_cuda_device_pool, _resolve_parallel_restart_workers


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
    assert target_bytes == int(8 * (1024**3))
