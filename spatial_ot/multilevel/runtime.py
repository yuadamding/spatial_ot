from __future__ import annotations

import os

import torch

_THREADPOOL_LIMITS = None


def relative_change(new, old) -> float:
    import numpy as np

    new = np.asarray(new, dtype=np.float64)
    old = np.asarray(old, dtype=np.float64)
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(old), 1e-12))


def resolve_compute_device(device: str) -> torch.device:
    requested = str(device).strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA compute was requested for multilevel OT, but torch.cuda.is_available() is False."
        )
    return resolved


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return int(default) if value <= 0 else value


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return float(default) if value <= 0 else float(value)


def cuda_target_vram_gb() -> float:
    return env_float("SPATIAL_OT_CUDA_TARGET_VRAM_GB", 70.0)


def cuda_target_bytes(device: torch.device | None = None) -> int:
    requested = int(cuda_target_vram_gb() * (1024**3))
    if not torch.cuda.is_available():
        return requested
    try:
        dev = device or torch.device("cuda")
        total_bytes = int(torch.cuda.get_device_properties(dev).total_memory)
    except Exception:
        return requested
    max_fraction = min(
        max(env_float("SPATIAL_OT_CUDA_MAX_TARGET_FRACTION", 0.9), 0.1), 0.98
    )
    safe_bytes = int(max(total_bytes * max_fraction, 1 << 30))
    return min(requested, safe_bytes)


def cuda_cdist_row_batch_size(
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
    batch = cuda_target_bytes(device=device) // denom
    batch = max(int(min_batch), min(int(n_rows), int(batch)))
    return max(batch, 1)


def resolve_cuda_device_pool(requested: str, n_init: int) -> list[str]:
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


def resolve_parallel_restart_workers(device_pool: list[str], n_init: int) -> int:
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


def configure_local_thread_budget(
    torch_threads: int, torch_interop_threads: int
) -> None:
    global _THREADPOOL_LIMITS

    torch_threads = max(int(torch_threads), 1)
    torch_interop_threads = max(int(torch_interop_threads), 1)
    for name in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ]:
        os.environ[name] = str(torch_threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["SPATIAL_OT_TORCH_NUM_THREADS"] = str(torch_threads)
    os.environ["SPATIAL_OT_TORCH_NUM_INTEROP_THREADS"] = str(torch_interop_threads)
    os.environ["SPATIAL_OT_CPU_THREADS"] = str(torch_threads)
    try:
        torch.set_num_threads(torch_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(torch_interop_threads)
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITS = threadpool_limits(limits=torch_threads)
    except Exception:
        _THREADPOOL_LIMITS = None
