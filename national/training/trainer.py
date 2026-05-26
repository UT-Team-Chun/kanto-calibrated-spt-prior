"""Distributed training loop for the foundation model.

The trainer is structured so the *same* code path works for:

- Single-CPU / single-MPS local smoke runs (no ``torch.distributed`` at all).
- Single-node multi-CPU (DDP over Gloo).
- Multi-node Miyabi-G runs launched under ``torchrun`` (DDP over NCCL).

Distributed setup is detected from the ``WORLD_SIZE``/``RANK``/``LOCAL_RANK``
environment variables (set by ``torchrun``). When those are absent we run in
single-process mode and skip the ``init_process_group`` call so that local
tests don't need a distributed launcher.

Checkpoints are atomic (write to ``.tmp`` then ``os.replace``) and contain
model weights, optimizer state, scheduler state, RNG state, and the trainer
``TrainerState`` (epoch, step, best metric). Resume support is opt-in via
``--resume-if-exists`` in the Hydra driver -- the trainer reads the newest
``latest.pt`` if present.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from national.models.foundation import FoundationModel
from national.training.losses import elbo_with_regime_weights, mmd_regularizer
from national.training.schedulers import cosine_warmup

LOG = logging.getLogger("national.training.trainer")


@dataclass
class TrainerState:
    """Mutable training progress tracked across checkpoints."""

    epoch: int = 0
    step: int = 0
    best_metric: float | None = None
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class TrainerOutput:
    """Returned by :meth:`FoundationTrainer.fit`."""

    final_loss: float
    state: TrainerState
    last_checkpoint: Path | None


def _is_distributed_environment() -> bool:
    """``True`` if torchrun-style env vars suggest we're in DDP."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


class FoundationTrainer:
    """SVGP training loop with DDP, mini-batches, checkpointing, logging."""

    def __init__(
        self,
        model: FoundationModel,
        dataset: Dataset,
        cfg: Any,
        *,
        device: torch.device | str = "cpu",
        sample_weight_fn=None,  # noqa: ANN001
        log_every: int = 50,
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.log_every = int(log_every)
        self.sample_weight_fn = sample_weight_fn
        self.state = TrainerState()
        self._wandb_run = None  # populated by _maybe_init_wandb on the master rank

        self.rank, self.local_rank, self.world_size = self._setup_distributed()
        self.is_master = self.rank == 0
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)

        self.model_module = model.to(self.device)
        if self.world_size > 1:
            self.model = DistributedDataParallel(
                self.model_module,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        else:
            self.model = self.model_module

        # FoundationModel.parameters() already traverses encoder, GP, likelihood,
        # and the regime FiLM submodule, so a single sweep covers everything.
        params = list(self.model_module.parameters())
        self.optimizer = torch.optim.Adam(
            params,
            lr=float(getattr(cfg.training, "lr", 5e-3)),
            betas=(
                float(getattr(cfg.training, "beta1", 0.9)),
                float(getattr(cfg.training, "beta2", 0.999)),
            ),
            weight_decay=float(getattr(cfg.training, "weight_decay", 1e-5)),
        )

        total_steps = max(
            1,
            int(getattr(cfg.training, "n_epochs", 100))
            * max(1, len(self.dataset) // max(1, int(getattr(cfg.training, "batch_size", 1024)))),
        )
        self.scheduler = cosine_warmup(
            self.optimizer,
            warmup_steps=int(getattr(cfg.training, "warmup_steps", 100)),
            total_steps=total_steps,
            min_lr_ratio=0.01,
        )

        self._mll_num_data = len(self.dataset)

    # ------------------------------------------------------------------ fit
    def fit(self) -> TrainerOutput:
        """Run training. Returns final loss and the (saved) checkpoint path."""
        seed = int(getattr(self.cfg.run, "seed", 42))
        self._set_deterministic(seed)

        batch_size = int(getattr(self.cfg.training, "batch_size", 1024))
        n_epochs = int(getattr(self.cfg.training, "n_epochs", 100))
        mmd_weight = float(getattr(self.cfg.training, "mmd_weight", 0.0))
        checkpoint_dir = Path(getattr(self.cfg.io, "checkpoint_root", "./checkpoints"))
        checkpoint_every_min = float(getattr(self.cfg.training, "checkpoint_every_min", 30))
        if self.is_master:
            self._maybe_init_wandb()

        if self.world_size > 1:
            sampler: DistributedSampler | None = DistributedSampler(
                self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=int(getattr(self.cfg.training, "num_workers", 0)),
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            collate_fn=_collate,
        )

        last_ckpt: Path | None = None
        last_ckpt_at = time.monotonic()
        final_loss = float("nan")

        self.model.train()
        self.model_module.likelihood.train()
        for epoch in range(self.state.epoch, n_epochs):
            self.state.epoch = epoch
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                x = batch["x"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                regime = batch.get("regime")
                if regime is not None:
                    regime = regime.to(self.device, non_blocking=True)
                sample_weights = (
                    self.sample_weight_fn(regime) if self.sample_weight_fn is not None else None
                )

                self.optimizer.zero_grad(set_to_none=True)
                with gpytorch.settings.num_likelihood_samples(8):
                    pred_dist = self.model(x)
                    # Heteroscedastic path: the noise head produces a
                    # per-point variance that FixedNoiseGaussianLikelihood
                    # consumes. Homoscedastic path keeps noise=None and
                    # uses the likelihood's own learned noise.
                    noise = (
                        self.model_module.predict_noise_variance(x)
                        if hasattr(self.model_module, "predict_noise_variance")
                        else None
                    )
                    loss = elbo_with_regime_weights(
                        pred_dist,
                        self.model_module.likelihood,
                        y,
                        num_data=self._mll_num_data,
                        model=self.model_module.gp,
                        sample_weights=sample_weights,
                        noise=noise,
                    )
                    if mmd_weight > 0.0:
                        phi = self.model_module.encoder(x)
                        phi_ref = torch.randn_like(phi)
                        loss = loss + mmd_weight * mmd_regularizer(phi, phi_ref)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                self.state.step += 1
                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1

                if self.is_master and self.state.step % self.log_every == 0:
                    mean_loss = epoch_loss / max(1, n_batches)
                    lr_now = self.scheduler.get_last_lr()[0]
                    LOG.info(
                        "epoch=%d step=%d loss=%.4f lr=%.3e",
                        epoch,
                        self.state.step,
                        mean_loss,
                        lr_now,
                    )
                    self._wandb_log(
                        {"epoch": epoch, "loss": mean_loss, "lr": lr_now},
                        step=self.state.step,
                    )

                if self.is_master and (time.monotonic() - last_ckpt_at) >= checkpoint_every_min * 60:
                    last_ckpt = self._save_checkpoint(checkpoint_dir, name="latest.pt")
                    last_ckpt_at = time.monotonic()

            final_loss = epoch_loss / max(1, n_batches)
            self.state.history.append({"epoch": epoch, "loss": final_loss})
            if self.is_master:
                LOG.info("epoch %d done; mean loss=%.4f", epoch, final_loss)
                self._wandb_log({"epoch_loss": final_loss}, step=self.state.step)

        # Always end with a final checkpoint.
        if self.is_master:
            last_ckpt = self._save_checkpoint(checkpoint_dir, name="final.pt")
            self._wandb_finish()

        return TrainerOutput(final_loss=final_loss, state=self.state, last_checkpoint=last_ckpt)

    # ------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path: Path, *, step: int) -> Path:
        """Public wrapper kept for backward compat with the Phase A interface."""
        self.state.step = int(step)
        return self._save_checkpoint(Path(path).parent, name=Path(path).name)

    def _save_checkpoint(self, directory: Path, *, name: str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "version": 1,
            "state": asdict(self.state),
            "model_state_dict": self.model_module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            payload["rng_cuda"] = torch.cuda.get_rng_state_all()
        torch.save(payload, tmp)
        os.replace(tmp, path)
        LOG.info("Saved checkpoint to %s", path)
        return path

    def maybe_load_checkpoint(self, path: Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model_module.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        state = payload.get("state", {})
        self.state = TrainerState(
            epoch=int(state.get("epoch", 0)),
            step=int(state.get("step", 0)),
            best_metric=state.get("best_metric"),
            history=list(state.get("history", [])),
        )
        if "rng_torch" in payload:
            torch.set_rng_state(payload["rng_torch"])
        if "rng_numpy" in payload:
            np.random.set_state(payload["rng_numpy"])
        if "rng_cuda" in payload and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
        LOG.info("Resumed from checkpoint %s (step=%d)", path, self.state.step)
        return self.state.step

    # ------------------------------------------------------------- internals
    def _setup_distributed(self) -> tuple[int, int, int]:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if _is_distributed_environment() and torch.distributed.is_available():
            if not torch.distributed.is_initialized():
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                torch.distributed.init_process_group(
                    backend=backend, rank=rank, world_size=world_size
                )
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size

    # ----------------------------------------------------------------- wandb
    def _maybe_init_wandb(self) -> None:
        """Initialize a Weights & Biases run if cfg.io.wandb is configured.

        Honors offline mode so Miyabi-G compute nodes (no outbound HTTPS) can
        write run logs to ``cfg.io.run_root`` for post-job ``wandb sync``.
        """
        wb_cfg = getattr(getattr(self.cfg, "io", None), "wandb", None)
        if wb_cfg is None:
            return
        mode = str(getattr(wb_cfg, "mode", "online")).lower()
        if mode == "disabled":
            return
        try:
            import wandb  # heavy; only imported when actually used
        except ImportError:
            LOG.warning("wandb not installed; skipping W&B logging.")
            return
        run_name = str(getattr(self.cfg.run, "name", "default"))
        project = str(getattr(wb_cfg, "project", "geo-estimation-national"))
        self._wandb_run = wandb.init(
            project=project,
            name=run_name,
            mode=mode,
            dir=str(getattr(self.cfg.io, "run_root", ".")),
            config=_omegaconf_to_dict(self.cfg),
            reinit=True,
        )

    def _wandb_log(self, metrics: dict, step: int) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log(metrics, step=int(step))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("wandb log failed: %s", exc)

    def _wandb_finish(self) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.finish()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("wandb finish failed: %s", exc)
        finally:
            self._wandb_run = None

    def _set_deterministic(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cudnn.deterministic is intentionally NOT forced -- it costs ~2x on
        # Hopper and the SVGP loss is already mostly non-deterministic from
        # mini-batch shuffling. Use ``torch.use_deterministic_algorithms(True)``
        # in tests if you need bit-reproducibility.


def _omegaconf_to_dict(cfg: Any) -> dict:
    """Best-effort conversion of an OmegaConf or dict into a plain dict."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    except ImportError:
        pass
    if isinstance(cfg, dict):
        return cfg
    return {"cfg_repr": repr(cfg)}


def _collate(samples: Iterable[dict]) -> dict:
    """Stack a list of ``BoringSample``-style dicts into a batched dict.

    Each sample must expose at least an ``x`` (shape ``(D,)``) and a ``y``
    (scalar). Optionally a ``regime`` (int scalar) is stacked. Extra string
    metadata is dropped.
    """
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    regimes: list[torch.Tensor] = []
    for s in samples:
        xs.append(s["x"])
        ys.append(s["y"].reshape(()))
        if "regime" in s:
            regimes.append(s["regime"].reshape(()))
    out: dict[str, torch.Tensor] = {
        "x": torch.stack(xs, dim=0),
        "y": torch.stack(ys, dim=0),
    }
    if regimes:
        out["regime"] = torch.stack(regimes, dim=0).long()
    return out


__all__ = ["TrainerState", "TrainerOutput", "FoundationTrainer"]
