from __future__ import annotations

import numpy as np


def normalize_mass(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, None)
    x = x + 1e-6
    return x / x.sum()
