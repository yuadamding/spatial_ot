from __future__ import annotations

import numpy as np


def apply_similarity(
    x: np.ndarray, transform: dict[str, np.ndarray | float]
) -> np.ndarray:
    r = np.asarray(transform["R"], dtype=np.float64)
    scale = float(transform["scale"])
    t = np.asarray(transform["t"], dtype=np.float64)
    return (scale * np.asarray(x, dtype=np.float64) @ r + t).astype(np.float64)
