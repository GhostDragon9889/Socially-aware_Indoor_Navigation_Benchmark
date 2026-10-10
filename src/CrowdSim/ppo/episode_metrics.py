"""Count finished episodes separately from overlapping terminal events."""

import torch


def terminal_masks(done: torch.Tensor, info: dict) -> dict[str, torch.Tensor]:
    done = done.to(dtype=torch.bool)
    events = {}
    for key in ("reached", "collision", "timeout", "stuck"):
        value = info.get(key)
        value = torch.zeros_like(done) if value is None else torch.as_tensor(value, device=done.device, dtype=torch.bool)
        if value.shape != done.shape:
            raise ValueError(f"Terminal event {key} must have shape {tuple(done.shape)}")
        events[key] = value & done
    # Reaching on the final allowed step is a success; a simultaneous collision
    # prevents success. Raw event masks remain available for diagnostics.
    events["success"] = events["reached"] & ~events["collision"]
    events["episodes"] = done
    return events


def terminal_counts(done: torch.Tensor, info: dict) -> dict[str, torch.Tensor]:
    return {key: mask.to(dtype=torch.float32).sum() for key, mask in terminal_masks(done, info).items()}
