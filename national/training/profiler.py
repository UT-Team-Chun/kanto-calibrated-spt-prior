"""Scaling profiler for the foundation model.

Runs the SVGP forward+backward at varying ``batch_size`` and
``n_inducing`` settings and reports per-step time and peak GPU memory.
Used in two places:

- Locally (CPU/MPS), as a sanity check before submitting an HPC job.
- On Miyabi-G via ``infra/miyabi/scaling/profile_throughput.py``, to
  produce the scaling-curve evidence the U-Tokyo allocation review asks
  for. The PBS/SLURM templates in ``infra/miyabi/`` reference this CLI
  by path, so any change here propagates automatically.

The profiler intentionally uses synthetic data so the experiment is
hermetic; the resource numbers it reports depend only on model shape and
hardware, not on the boring database state.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from national.models.foundation import (
    EncoderSpec,
    FoundationModel,
    FoundationSpec,
    SVGPSpec,
    init_inducing_points,
)
from national.training.losses import elbo_with_regime_weights

LOG = logging.getLogger("national.training.profiler")


@dataclass
class ProfilePoint:
    """One row of the scaling table."""

    n_inducing: int
    batch_size: int
    forward_ms: float
    backward_ms: float
    step_ms: float
    peak_memory_mb: float
    out_of_memory: bool = False


@dataclass
class ProfileResult:
    """Aggregate profiling output across a sweep."""

    device: str
    n_features: int
    points: list[ProfilePoint]

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "n_features": self.n_features,
            "points": [asdict(p) for p in self.points],
        }


def profile_grid(
    *,
    inducing_counts: Iterable[int],
    batch_sizes: Iterable[int],
    n_features: int = 16,
    n_warmup: int = 2,
    n_steps: int = 6,
    device: str = "cpu",
    seed: int = 0,
) -> ProfileResult:
    """Run a (n_inducing, batch_size) sweep and report timings + memory.

    Each cell does ``n_warmup`` untimed iterations followed by ``n_steps``
    timed iterations; the cell is reported with the median timing.

    OOM is reported per-cell so the sweep can continue across the OOM frontier
    instead of bailing on the first failure.
    """
    dev = torch.device(device)
    points: list[ProfilePoint] = []
    for n_ind in inducing_counts:
        for bs in batch_sizes:
            try:
                pt = _profile_one(
                    n_inducing=int(n_ind),
                    batch_size=int(bs),
                    n_features=int(n_features),
                    n_warmup=int(n_warmup),
                    n_steps=int(n_steps),
                    device=dev,
                    seed=int(seed),
                )
            except torch.cuda.OutOfMemoryError:
                LOG.warning("OOM at n_inducing=%d batch_size=%d", n_ind, bs)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                pt = ProfilePoint(
                    n_inducing=int(n_ind),
                    batch_size=int(bs),
                    forward_ms=float("nan"),
                    backward_ms=float("nan"),
                    step_ms=float("nan"),
                    peak_memory_mb=float("nan"),
                    out_of_memory=True,
                )
            LOG.info(
                "n_inducing=%d batch_size=%d step=%.2fms mem=%.1fMB oom=%s",
                pt.n_inducing,
                pt.batch_size,
                pt.step_ms,
                pt.peak_memory_mb,
                pt.out_of_memory,
            )
            points.append(pt)
    return ProfileResult(device=str(dev), n_features=int(n_features), points=points)


def _profile_one(
    *,
    n_inducing: int,
    batch_size: int,
    n_features: int,
    n_warmup: int,
    n_steps: int,
    device: torch.device,
    seed: int,
) -> ProfilePoint:
    torch.manual_seed(seed)

    spec = FoundationSpec(
        encoder=EncoderSpec(
            n_input=n_features,
            n_output=min(32, max(4, n_features // 2)),
            n_layers=4,
            hidden=128,
            batchnorm=True,
            dropout=0.0,
            fourier_bands=8,
        ),
        svgp=SVGPSpec(
            n_inducing=n_inducing,
            learn_inducing=True,
            whitened=True,
            inducing_init="random",
            kernel_type="matern52",
            add_residual_geo=True,
        ),
        regime_dim=8,
        depth_scale_m=30.0,
    )

    # Build synthetic [lat, lon, depth, *covariates] in Japan-ish range.
    lat = torch.rand(batch_size) * 2.0 + 35.0
    lon = torch.rand(batch_size) * 2.0 + 139.0
    depth = torch.rand(batch_size) * 30.0
    covariates = torch.randn(batch_size, n_features - 3) if n_features > 3 else torch.empty(batch_size, 0)
    x = torch.cat([lat[:, None], lon[:, None], depth[:, None], covariates], dim=1).to(device)
    y = (5.0 + 0.1 * (lat + lon)).to(device)

    # Inducing points sampled from an independently-sized pool so the
    # n_inducing sweep is decoupled from the batch_size sweep.
    pool_size = max(n_inducing, batch_size)
    pool_lat = torch.rand(pool_size) * 2.0 + 35.0
    pool_lon = torch.rand(pool_size) * 2.0 + 139.0
    pool_dep = torch.rand(pool_size) * 30.0
    pool_cov = (
        torch.randn(pool_size, n_features - 3) if n_features > 3 else torch.empty(pool_size, 0)
    )
    pool_x = torch.cat(
        [pool_lat[:, None], pool_lon[:, None], pool_dep[:, None], pool_cov], dim=1
    )
    inducing = init_inducing_points(pool_x, n_inducing=n_inducing, method="random").to(device)
    model = FoundationModel(spec, inducing).to(device)
    model.set_target_stats(float(y.mean()), float(y.std() + 1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    forward_times: list[float] = []
    backward_times: list[float] = []
    step_times: list[float] = []

    for i in range(n_warmup + n_steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_start = time.perf_counter()
        out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_fwd = time.perf_counter()
        loss = elbo_with_regime_weights(
            out,
            model.likelihood,
            (y - model.target_mean) / model.target_std,
            num_data=batch_size * 100,  # synthetic "dataset" so loss magnitude is realistic
            model=model.gp,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_bwd = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_end = time.perf_counter()
        if i >= n_warmup:
            forward_times.append((t_fwd - t_start) * 1000.0)
            backward_times.append((t_bwd - t_fwd) * 1000.0)
            step_times.append((t_end - t_start) * 1000.0)

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return ProfilePoint(
        n_inducing=n_inducing,
        batch_size=batch_size,
        forward_ms=float(_median(forward_times)),
        backward_ms=float(_median(backward_times)),
        step_ms=float(_median(step_times)),
        peak_memory_mb=float(peak_mem_mb),
    )


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def _parse_int_list(s: str) -> list[int]:
    return [int(v) for v in s.split(",") if v]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inducing", type=_parse_int_list, default=[256, 1024, 4096])
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[512, 2048, 8192])
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=6)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    result = profile_grid(
        inducing_counts=args.inducing,
        batch_sizes=args.batch_sizes,
        n_features=args.n_features,
        n_warmup=args.n_warmup,
        n_steps=args.n_steps,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))
    LOG.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())


__all__ = ["ProfilePoint", "ProfileResult", "profile_grid", "main"]
