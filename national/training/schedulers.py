"""Learning-rate schedulers for national-scale training."""

from __future__ import annotations

import math

import torch


def cosine_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a linear-warmup, cosine-decay LambdaLR scheduler."""
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps must be <= total_steps.")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}.")

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


__all__ = ["cosine_warmup"]
