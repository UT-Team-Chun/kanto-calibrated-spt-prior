"""Hydra entrypoint for national-scale training.

The driver composes a config from ``conf/``, instantiates the
:class:`FoundationModel` + :class:`FoundationTrainer`, and runs the
training loop. The same entrypoint serves local smoke runs (``training=single_gpu``)
and Miyabi-G distributed runs (``training=miyabi_8gpu`` / ``miyabi_64gpu``).

CLI examples::

    # validate config only
    uv run python -m national.training.driver +dry_run=true region=abukuma model=svgp_separable5d

    # local smoke training with a synthetic dataset
    uv run python -m national.training.driver region=debug_5km \
        model=dkl_svgp training=single_gpu run.name=smoke_local +synthetic=true

    # Miyabi-G distributed (under torchrun)
    torchrun --nnodes=8 --nproc_per_node=1 -m national.training.driver \
        region=japan model=dkl_svgp training=miyabi_8gpu io=miyabi_scratch
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

LOG = logging.getLogger("national.training.driver")

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONF_DIR = _PROJECT_ROOT / "conf"


@hydra.main(version_base=None, config_path=str(_CONF_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    LOG.info("Composed Hydra config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    _validate(cfg)

    if cfg.get("dry_run", False):
        LOG.info("dry_run=true -> exiting after config validation.")
        return

    if cfg.get("synthetic", False):
        _train_on_synthetic(cfg)
        return

    LOG.warning(
        "Real-data training requires the covariate registry and BoringDataset "
        "(Phase B-6 / Phase B-7). Until those land, only +synthetic=true runs "
        "end-to-end. See docs/architecture.md for the phase roadmap."
    )


def _train_on_synthetic(cfg: DictConfig) -> None:
    """Run a tiny end-to-end smoke fit on synthetic data.

    Useful for verifying the driver/trainer/model wiring without depending on
    the covariate pipeline. The dataset draws ``(lat, lon, depth, covariates)``
    from a uniform spatial box and a noisy ground truth.
    """
    import torch

    from national.models.foundation import (
        EncoderSpec,
        FoundationModel,
        FoundationSpec,
        SVGPSpec,
        init_inducing_points,
    )
    from national.training.trainer import FoundationTrainer

    torch.manual_seed(int(cfg.run.seed))

    n_data = 1024
    n_covariates = 4
    lat = torch.rand(n_data) * (cfg.region.bounds.lat_max - cfg.region.bounds.lat_min) + cfg.region.bounds.lat_min
    lon = torch.rand(n_data) * (cfg.region.bounds.lon_max - cfg.region.bounds.lon_min) + cfg.region.bounds.lon_min
    depth = torch.rand(n_data) * 30.0
    covs = torch.randn(n_data, n_covariates)
    x = torch.stack([lat, lon, depth], dim=1)
    x = torch.cat([x, covs], dim=1).float()
    y = (
        5.0
        + 0.3 * (lat - lat.mean())
        - 0.2 * (lon - lon.mean())
        + 0.05 * depth
        + 0.4 * covs[:, 0]
        + 0.2 * torch.randn(n_data)
    ).float()
    regime = torch.randint(0, 4, (n_data,)).long()

    spec = FoundationSpec(
        encoder=EncoderSpec(n_input=3 + n_covariates, n_output=16, n_layers=2, hidden=64),
        svgp=SVGPSpec(
            n_inducing=64,
            learn_inducing=True,
            whitened=True,
            inducing_init="random",
            kernel_type="matern52",
            add_residual_geo=True,
        ),
        regime_dim=8,
        depth_scale_m=30.0,
    )
    inducing = init_inducing_points(x, n_inducing=spec.svgp.n_inducing, method="random")
    model = FoundationModel(spec, inducing_points=inducing)
    model.set_target_stats(float(y.mean()), float(y.std() + 1e-6))

    class _MemDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return n_data

        def __getitem__(self, idx: int) -> dict:
            return {
                "x": x[idx],
                "y": (y[idx] - y.mean()) / (y.std() + 1e-6),
                "regime": regime[idx],
            }

    trainer = FoundationTrainer(
        model=model,
        dataset=_MemDataset(),
        cfg=cfg,
        device="cpu",
    )
    output = trainer.fit()
    LOG.info("synthetic training done: final_loss=%.4f", output.final_loss)


def _validate(cfg: DictConfig) -> None:
    required_groups = ("region", "covariates", "model", "training", "prediction", "io")
    missing = [k for k in required_groups if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config groups: {missing!r}")

    if cfg.training.world_size < 1:
        raise ValueError(f"training.world_size must be >= 1, got {cfg.training.world_size}")
    if cfg.training.batch_size < 1:
        raise ValueError("training.batch_size must be positive")
    if cfg.run.seed is None or cfg.run.seed < 0:
        raise ValueError("run.seed must be a non-negative int")


if __name__ == "__main__":
    sys.exit(main())  # type: ignore[func-returns-value]
