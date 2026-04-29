from __future__ import annotations

import torch


def runtime_memory_snapshot(device: torch.device) -> dict[str, float | int | bool | str]:
    snapshot: dict[str, float | int | bool | str] = {
        "device": str(device),
        "cuda": bool(device.type == "cuda" and torch.cuda.is_available()),
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        return snapshot
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        free_bytes, total_bytes = 0, 0
    snapshot.update(
        {
            "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "memory_free_bytes": int(free_bytes),
            "memory_total_bytes": int(total_bytes),
        }
    )
    return snapshot
