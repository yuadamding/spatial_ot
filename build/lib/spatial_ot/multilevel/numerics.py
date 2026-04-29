from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances
import torch

from .runtime import cuda_cdist_row_batch_size as _cuda_cdist_row_batch_size


def pairwise_sqdist_array(
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
