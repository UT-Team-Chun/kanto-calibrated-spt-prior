#!/usr/bin/env python
"""End-to-end Phase B smoke train on real Kanto borings.

This is the first script in the project that trains the foundation
model on actual KuniJiban data instead of synthetic samples. The
runtime is short (~5 minutes on a laptop CPU) but it exercises every
production code path: ``BoringDataset`` -> ``FoundationModel`` ->
``FoundationTrainer`` -> ``FoundationModel.predict`` -> spatial K-fold
RMSE.

Output: a JSON summary at ``data/runs/<run_name>/summary.json`` plus a
saved foundation artifact. Re-runs the same hyperparameters
deterministically given the same ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from national.data.boring_dataset import BoringDataset
from national.evaluation.spatial_kfold import spatial_kfold_split
from national.models.foundation import (
    EncoderSpec,
    FoundationModel,
    FoundationSpec,
    SVGPSpec,
    init_inducing_points,
)
from national.training.trainer import FoundationTrainer

LOG = logging.getLogger("scripts.train_kanto_smoke")


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=repo / "data/features/borings_kanto.parquet")
    parser.add_argument("--output-dir", type=Path, default=repo / "data/runs/kanto_smoke")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-inducing", type=int, default=2000)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.05,
        help="Random subsample fraction for the smoke run (default: 5%).",
    )
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | mps | cuda. 'auto' picks cuda > mps > cpu.",
    )
    parser.add_argument(
        "--regime-one-hot",
        action="store_true",
        help=(
            "Append 8-way regime one-hot to encoder input (in addition to the "
            "FiLM block). Significantly increases the regime signal at the "
            "cost of 8 extra encoder input columns."
        ),
    )
    parser.add_argument(
        "--target-transform",
        choices=("none", "log1p"),
        default="none",
        help=(
            "Forward transform applied to the regression target before "
            "standardization. ``log1p`` matches the SPT N-value's "
            "heavy-right-tail distribution to a Gaussian likelihood much "
            "more cleanly than the raw N. Inverse transform is applied "
            "automatically for the K-fold RMSE / MAE reporting."
        ),
    )
    parser.add_argument(
        "--heteroscedastic-noise",
        action="store_true",
        help=(
            "Enable the heteroscedastic NoiseHead -- a small MLP that "
            "predicts log_variance from (depth_norm, regime_one_hot). "
            "Switches the model's likelihood from GaussianLikelihood "
            "(homoscedastic) to FixedNoiseGaussianLikelihood. Addresses "
            "the alpha=0.50 calibration over-cautiousness and the per-depth "
            "RMSE inflation."
        ),
    )
    parser.add_argument(
        "--kernel-type",
        choices=["matern52", "matern32", "matern12", "rbf"],
        default="matern52",
        help=(
            "GP kernel family on the encoder output. matern52 (default) is "
            "twice-differentiable; matern32 is once-differentiable; matern12 "
            "is non-differentiable (rough); rbf is infinitely smooth. The "
            "right choice depends on how rough the regression target is in "
            "the encoded feature space."
        ),
    )
    parser.add_argument(
        "--mean-type",
        choices=["constant", "linear"],
        default="constant",
        help=(
            "Prior mean function. 'constant' learns a single bias; 'linear' "
            "learns a linear combination of the encoder output dimensions "
            "(gpytorch.means.LinearMean). Useful when the target has a "
            "monotonic trend in the encoded representation."
        ),
    )
    parser.add_argument(
        "--inducing-init",
        choices=["random", "kmeans_pp", "kmeans_pp_stratified"],
        default="random",
        help=(
            "Inducing point initialization strategy. random is the historical "
            "default; kmeans_pp gives more uniformly-spread inducing; "
            "kmeans_pp_stratified balances across AIST regimes."
        ),
    )
    parser.add_argument(
        "--likelihood-type",
        choices=["gaussian", "studentt", "censored"],
        default="gaussian",
        help=(
            "Observation likelihood. gaussian (default) is homoscedastic "
            "Gaussian. studentt switches to a Student-t likelihood that "
            "natively handles the heavy-tail residual structure we observed "
            "(kurtosis ≈ 9.3). censored uses a right-censored Gaussian for "
            "the N≤100 cap (see --censored-cap). Ignored if "
            "--heteroscedastic-noise is also set "
            "(that path forces FixedNoiseGaussianLikelihood)."
        ),
    )
    parser.add_argument(
        "--censored-cap",
        type=float,
        default=100.0,
        help=(
            "Right-censoring threshold in raw N units. Used only when "
            "--likelihood-type=censored. Defaults to 100 to match the "
            "SPT N cap used throughout the paper."
        ),
    )
    parser.add_argument(
        "--buffer-meshes",
        type=int,
        default=0,
        help=(
            "If > 0, use spatial_kfold_split_buffered with the given ring "
            "size (in secondary-mesh cells) to exclude train rows within "
            "that distance from any test mesh. 0 (default) means use the "
            "plain spatial_kfold_split."
        ),
    )
    parser.add_argument(
        "--leave-prefecture",
        type=str,
        default="",
        help=(
            "If non-empty, switch from K-fold to leave-prefecture-out "
            "evaluation; the named Kanto prefecture is the held-out test "
            "set, the rest is train. One of {tokyo, kanagawa, saitama, "
            "chiba, ibaraki, tochigi, gunma}."
        ),
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=None,
        help=(
            "Explicit list of derived feature column names to use (in "
            "addition to the mandatory lat/lon/depth). Default is "
            "['absolute_elevation', 'river_distance_km', 'coast_distance_km']. "
            "Pass an empty list to ablate all derived features."
        ),
    )
    parser.add_argument(
        "--zero-fourier",
        action="store_true",
        help=(
            "Zero out the random-Fourier features over (lat, lon) in the "
            "encoder. Counter-test for 'is the encoder just memorising "
            "spatial coordinates?'"
        ),
    )
    parser.add_argument(
        "--kfold-test-fold",
        type=int,
        default=-1,
        help=(
            "If >= 0, switch from train-on-all + report-K-fold-metrics "
            "to a single proper hold-out run: train on all folds EXCEPT "
            "the given one, and evaluate on that one. Enables proper "
            "buffered / leave-prefecture / nested spatial cross-validation "
            "by launching one job per held-out fold. Default -1 preserves "
            "the historical train-on-all + report-K-fold behaviour."
        ),
    )
    parser.add_argument(
        "--fold-assignment",
        choices=["random", "contiguous"],
        default="random",
        help=(
            "Spatial K-fold mesh assignment. 'random' (default) is the "
            "load-balanced random shuffle of mesh codes across folds, "
            "which interleaves folds spatially; 'contiguous' uses "
            "k-means on mesh centroids to produce geographic-block folds. "
            "Random shuffle is biased toward optimistic K-fold metrics "
            "(test rows surrounded by training neighbours); contiguous "
            "is the stricter reviewer-defensible variant."
        ),
    )
    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=24,
        help=(
            "Encoder output dimensionality (the latent feeding the GP). "
            "Default 24 is what every published ablation used. Wider (32, 48, 64) "
            "tests whether LinearMean's win was an encoder-capacity gap."
        ),
    )
    parser.add_argument(
        "--studentt-df",
        type=float,
        default=4.0,
        help=(
            "Initial degrees of freedom ν for Student-t likelihood. The value "
            "is learnable during training, constrained to ν > 2 so the marginal "
            "variance stays finite. Default 4 is a pragmatic init for moderate "
            "heavy-tail (kurtosis 6 at ν=4). Ignored unless "
            "--likelihood-type=studentt."
        ),
    )
    parser.add_argument(
        "--pred-batch",
        type=int,
        default=20_000,
        help=(
            "Batch size for the K-fold posterior prediction at the end of "
            "training. 50k fits on a 48 GB GPU; drop to 10–20k on a 12 GB "
            "GPU (RTX 4070 Ti) to avoid OOM in the matern kernel materialisation."
        ),
    )
    parser.add_argument(
        "--baseline-pred-train",
        type=Path,
        default=None,
        help=(
            "Path to .npy of CatBoost/LightGBM mean predictions on the "
            "spatial out-of-bag training rows (one prediction per index in "
            "--baseline-idx-train). When set, switches the SVGP target to "
            "residuals y - baseline_pred for hybrid tree+GP training. The "
            "baseline mean is added back at inference time."
        ),
    )
    parser.add_argument(
        "--baseline-pred-test",
        type=Path,
        default=None,
        help=(
            "Path to .npy of CatBoost/LightGBM mean predictions on the "
            "held-out test rows (one prediction per index in "
            "--baseline-idx-test). Re-added to the SVGP residual prediction "
            "at fold-evaluation time."
        ),
    )
    parser.add_argument(
        "--baseline-idx-train",
        type=Path,
        default=None,
        help="Row indices into the parquet for --baseline-pred-train.",
    )
    parser.add_argument(
        "--baseline-idx-test",
        type=Path,
        default=None,
        help="Row indices into the parquet for --baseline-pred-test.",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="catboost",
        help=(
            "Identifier of the teacher baseline (catboost / lightgbm). Used "
            "for provenance only; the actual predictions are loaded from "
            "--baseline-pred-{train,test}."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases run logging (requires WANDB_API_KEY env).",
    )
    parser.add_argument(
        "--wandb-project",
        default="geo-estimation-national",
        help="W&B project name. Used only when --wandb is set.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name. Defaults to output-dir basename.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    LOG.info("Using device=%s", args.device)

    # MPS does not support float64 (Apple Silicon GPUs are float32 only). GPyTorch's
    # variational strategy defaults its Cholesky factorization to float64 for
    # numerical stability; we force float32 globally before any model is built so
    # the SVGP forward pass works on MPS. Note: this can in theory hurt stability
    # on very ill-conditioned kernel matrices -- on a healthy SVGP (well-spaced
    # inducing points, reasonable lengthscales) the difference is negligible.
    if args.device == "mps":
        import gpytorch.settings as gp_settings
        gp_settings._linalg_dtype_cholesky._global_value = torch.float32
        gp_settings._linalg_dtype_symeig._global_value = torch.float32

    # kmeans_pp_stratified inducing init can produce near-degenerate K_zz
    # in some regimes (high-density clusters → close inducing points). The
    # gpytorch default jitter 1e-8 / max_tries=3 then fails with NotPSDError.
    # Bump both unconditionally — extra jitter is cheap and only matters
    # near the singular regime.
    import gpytorch.settings as gp_settings  # noqa: E402
    gp_settings.cholesky_jitter._global_float_value = 1e-4
    gp_settings.cholesky_max_tries._global_value = 10

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load dataset --------------------------------------------------------
    if args.feature_cols is None:
        feature_cols = ["absolute_elevation", "river_distance_km", "coast_distance_km"]
    elif len(args.feature_cols) == 1 and args.feature_cols[0].upper() == "NONE":
        # Sentinel for "no derived features" — POSIX argparse cannot
        # accept zero-argument nargs="+", so we use NONE.
        feature_cols = []
    else:
        feature_cols = list(args.feature_cols)
    LOG.info("Using feature_cols=%s", feature_cols)
    # Hybrid mode: the BoringDataset target is residual y - baseline_pred,
    # not raw y. The baseline predictions are required for both the training
    # rows (residual target) and the eventual held-out test rows
    # (re-addition at inference). Both --baseline-pred-{train,test} +
    # --baseline-idx-{train,test} must be supplied together.
    hybrid_mode = args.baseline_pred_train is not None
    if hybrid_mode:
        if args.baseline_idx_train is None:
            raise ValueError(
                "--baseline-pred-train requires --baseline-idx-train"
            )
        if args.baseline_pred_test is None or args.baseline_idx_test is None:
            raise ValueError(
                "Hybrid mode also requires --baseline-pred-test and "
                "--baseline-idx-test for inference-time mean re-addition"
            )
        # log1p is non-monotonic on signed residuals (they can be negative).
        # Force the residual target to the no-transform path.
        if args.target_transform != "none":
            LOG.warning(
                "Hybrid mode forces target_transform='none' "
                "(requested %r overridden)", args.target_transform,
            )
            args.target_transform = "none"
        LOG.info(
            "Hybrid mode ON: baseline=%s, residual target switched on",
            args.baseline_name,
        )
    dataset = BoringDataset(
        args.parquet,
        feature_columns=feature_cols,
        depth_scale_m=30.0,
        standardize_target=True,
        regime_one_hot=args.regime_one_hot,
        target_transform=args.target_transform,
        baseline_pred_npy=args.baseline_pred_train if hybrid_mode else None,
        baseline_idx_npy=args.baseline_idx_train if hybrid_mode else None,
        baseline_pred_test_npy=args.baseline_pred_test if hybrid_mode else None,
        baseline_idx_test_npy=args.baseline_idx_test if hybrid_mode else None,
    )
    LOG.info(
        "Loaded BoringDataset: %d rows, %d features (mean=%.2f std=%.2f)",
        len(dataset),
        dataset.n_features,
        dataset.target_mean,
        dataset.target_std,
    )

    # 2. Subsample for the smoke run ----------------------------------------
    n_total = len(dataset)
    n_smoke = int(n_total * args.train_fraction)
    rng = np.random.default_rng(args.seed)
    smoke_idx = rng.choice(n_total, size=n_smoke, replace=False)
    smoke_idx.sort()
    sub_x = torch.from_numpy(dataset._x[smoke_idx]).float()
    # ``dataset._y_raw`` is the *original-unit* target, preserved for
    # K-fold metric reporting (RMSE / MAE are computed in N-value units).
    # ``dataset._y`` is post-transform AND post-standardization, the
    # quantity the GP actually fits against.
    sub_y_raw = torch.from_numpy(dataset._y_raw[smoke_idx]).float()
    sub_y_std = torch.from_numpy(dataset._y[smoke_idx]).float()
    sub_regime = torch.from_numpy(dataset._regime[smoke_idx].astype(np.int64))
    # Hybrid mode: baseline_pred_per_row is in raw-N units (same as _y_raw).
    # We slice to the smoke subsample so the inference path can add it back.
    if hybrid_mode:
        sub_baseline_pred = dataset.baseline_pred_per_row[smoke_idx].astype(np.float32)
    else:
        sub_baseline_pred = None
    LOG.info("Subsampled %d / %d rows for smoke training", n_smoke, n_total)

    # Optional proper hold-out: train on rows OUTSIDE the held-out fold
    # so we can measure spatial-generalisation under buffered CV,
    # leave-prefecture-out, or nested-spatial conformal.
    holdout_active = (
        args.kfold_test_fold >= 0
        or bool(args.leave_prefecture)
    )
    holdout_test_idx: np.ndarray | None = None
    if holdout_active:
        sub_df_for_split = pd.DataFrame(
            {
                "latitude_deg": sub_x[:, 0].numpy(),
                "longitude_deg": sub_x[:, 1].numpy(),
                "n_value": sub_y_raw.numpy(),
            }
        )
        if args.leave_prefecture:
            from national.evaluation.prefecture_regions import leave_prefecture_out_split

            split_iter = list(
                leave_prefecture_out_split(
                    sub_df_for_split, prefectures=[args.leave_prefecture]
                )
            )
            if not split_iter:
                raise RuntimeError(
                    f"leave_prefecture_out_split produced no fold for "
                    f"prefecture {args.leave_prefecture!r}; check bbox."
                )
            _, train_keep, holdout_test_idx = split_iter[0]
            LOG.info(
                "Hold-out (leave-prefecture=%s): train_keep=%d, test=%d",
                args.leave_prefecture, len(train_keep), len(holdout_test_idx),
            )
        else:
            if args.buffer_meshes > 0:
                from national.evaluation.spatial_kfold import (
                    spatial_kfold_split_buffered,
                )

                fold_splits = spatial_kfold_split_buffered(
                    sub_df_for_split, n_folds=3, mesh_level=2,
                    buffer_meshes=args.buffer_meshes, seed=args.seed,
                    base_split=args.fold_assignment,
                )
            elif args.fold_assignment == "contiguous":
                from national.evaluation.spatial_kfold import (
                    spatial_kfold_split_contiguous,
                )

                fold_splits = spatial_kfold_split_contiguous(
                    sub_df_for_split, n_folds=3, mesh_level=2, seed=args.seed,
                )
            else:
                fold_splits = spatial_kfold_split(
                    sub_df_for_split, n_folds=3, mesh_level=2, seed=args.seed,
                )
            if args.kfold_test_fold >= len(fold_splits):
                raise ValueError(
                    f"--kfold-test-fold={args.kfold_test_fold} out of range "
                    f"for n_folds={len(fold_splits)}"
                )
            train_keep, holdout_test_idx = fold_splits[args.kfold_test_fold]
            LOG.info(
                "Hold-out (fold=%d, buffer=%d): train_keep=%d, test=%d",
                args.kfold_test_fold, args.buffer_meshes,
                len(train_keep), len(holdout_test_idx),
            )

        # Restrict training tensors to the hold-out's training side
        sub_x_full = sub_x
        sub_y_raw_full = sub_y_raw
        sub_y_std_full = sub_y_std
        sub_regime_full = sub_regime

        sub_x = sub_x_full[train_keep]
        sub_y_raw = sub_y_raw_full[train_keep]
        sub_y_std = sub_y_std_full[train_keep]
        sub_regime = sub_regime_full[train_keep]
        n_smoke = len(sub_x)
        LOG.info("After hold-out, training set is %d rows", n_smoke)

    class _SubDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return n_smoke

        def __getitem__(self, idx: int) -> dict:
            return {
                "x": sub_x[idx],
                "y": sub_y_std[idx],
                "regime": sub_regime[idx],
            }

    sub_dataset = _SubDataset()

    # 3. Build the foundation model -----------------------------------------
    n_input = dataset.n_features
    encoder = EncoderSpec(
        n_input=n_input,
        n_output=args.encoder_dim,
        n_layers=4,
        hidden=128,
        batchnorm=True,
        dropout=0.0,
        fourier_bands=12,
        zero_fourier=args.zero_fourier,
    )
    svgp = SVGPSpec(
        n_inducing=min(args.n_inducing, n_smoke),
        learn_inducing=True,
        whitened=True,
        inducing_init=args.inducing_init,
        kernel_type=args.kernel_type,
        mean_type=args.mean_type,
        likelihood_type=args.likelihood_type,
        studentt_deg_free=args.studentt_df,
        censored_cap=args.censored_cap,
        add_residual_geo=True,
    )
    from national.models.foundation import NoiseHeadSpec

    spec = FoundationSpec(
        encoder=encoder,
        svgp=svgp,
        regime_dim=8,
        depth_scale_m=30.0,
        noise_head=NoiseHeadSpec(enabled=args.heteroscedastic_noise),
    )
    # kmeans_pp_stratified requires regime codes so it can budget inducing
    # points per AIST regime. The other methods ignore the kwarg.
    inducing = init_inducing_points(
        sub_x,
        n_inducing=svgp.n_inducing,
        method=args.inducing_init,
        regime_codes=sub_regime if args.inducing_init == "kmeans_pp_stratified" else None,
    )
    model = FoundationModel(spec, inducing_points=inducing)
    model.set_target_stats(dataset.target_mean, dataset.target_std)

    # 4. Trainer config -----------------------------------------------------
    run_name = args.wandb_run_name or args.output_dir.name
    io_cfg: dict = {
        "checkpoint_root": str(args.output_dir / "checkpoints"),
        "run_root": str(args.output_dir),
    }
    if args.wandb:
        io_cfg["wandb"] = {
            "project": args.wandb_project,
            "mode": "online",
        }
    cfg = OmegaConf.create(
        {
            "training": {
                "lr": args.lr,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "warmup_steps": 20,
                "weight_decay": 1e-5,
                "num_workers": 0,
                "checkpoint_every_min": 9999,
                "mmd_weight": 0.0,
                "beta1": 0.9,
                "beta2": 0.999,
            },
            "run": {"seed": args.seed, "name": run_name},
            "io": io_cfg,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = FoundationTrainer(model=model, dataset=sub_dataset, cfg=cfg, device=args.device)
    LOG.info("Training start...")
    t_start = time.perf_counter()
    output = trainer.fit()
    t_train = time.perf_counter() - t_start
    LOG.info(
        "Trained %d epochs in %.1fs; final_loss=%.4f",
        args.n_epochs,
        t_train,
        output.final_loss,
    )

    # 5. Spatial K-fold RMSE on the subsample -------------------------------
    if holdout_active:
        # Proper hold-out: model was trained on (sub_x, sub_y_*) only;
        # evaluate predictions on the held-out rows from the original
        # full subsample (sub_x_full etc.) and report a single fold.
        eval_x = sub_x_full[holdout_test_idx]
        eval_y_raw = sub_y_raw_full[holdout_test_idx]
        # Align the per-row baseline prediction with the held-out test rows
        # so the hybrid inference path can re-add the CatBoost mean at
        # exactly the right rows. Without this slice the assertion in step
        # 6 below trips when sub_baseline_pred has length len(sub_x_full)
        # while pred_mean has length len(holdout_test_idx).
        if hybrid_mode and sub_baseline_pred is not None:
            sub_baseline_pred = sub_baseline_pred[holdout_test_idx]
        eval_label = (
            f"prefecture={args.leave_prefecture}"
            if args.leave_prefecture
            else f"fold={args.kfold_test_fold}"
            + (f" buffered={args.buffer_meshes}" if args.buffer_meshes > 0 else "")
        )
        LOG.info("Hold-out evaluation set: %d rows (%s)", len(eval_x), eval_label)
        folds = [(np.arange(len(sub_x)), np.arange(len(eval_x)))]
    else:
        sub_df = pd.DataFrame(
            {
                "latitude_deg": sub_x[:, 0].numpy(),
                "longitude_deg": sub_x[:, 1].numpy(),
                "n_value": sub_y_raw.numpy(),
            }
        )
        if args.buffer_meshes > 0:
            from national.evaluation.spatial_kfold import spatial_kfold_split_buffered

            folds = spatial_kfold_split_buffered(
                sub_df, n_folds=3, mesh_level=2,
                buffer_meshes=args.buffer_meshes, seed=args.seed,
                base_split=args.fold_assignment,
            )
            LOG.info(
                "Buffered spatial K-fold (assignment=%s, buffer_meshes=%d): "
                "fold sizes %s",
                args.fold_assignment, args.buffer_meshes,
                [(len(tr), len(te)) for tr, te in folds],
            )
        elif args.fold_assignment == "contiguous":
            from national.evaluation.spatial_kfold import spatial_kfold_split_contiguous

            folds = spatial_kfold_split_contiguous(
                sub_df, n_folds=3, mesh_level=2, seed=args.seed,
            )
        else:
            folds = spatial_kfold_split(sub_df, n_folds=3, mesh_level=2, seed=args.seed)
        eval_x = sub_x
        eval_y_raw = sub_y_raw
    LOG.info("Computing posterior predictions for K-fold RMSE (batched)...")
    # Predict in batches so a 495k × 6k SVGP kernel matrix does not OOM
    # on CUDA. A 50k batch keeps peak memory < ~3 GB on the matern path.
    pred_means: list[np.ndarray] = []
    pred_stds: list[np.ndarray] = []
    pred_batch = int(args.pred_batch)
    with torch.no_grad():
        for start in range(0, eval_x.shape[0], pred_batch):
            end = min(start + pred_batch, eval_x.shape[0])
            pred_chunk = model.predict(eval_x[start:end])
            pred_means.append(pred_chunk.mean.cpu().numpy())
            pred_stds.append(pred_chunk.std.cpu().numpy())
    pred_mean_trans = np.concatenate(pred_means, axis=0)
    pred_std_trans = np.concatenate(pred_stds, axis=0)
    # Map the prediction back into the original N-value units so the
    # K-fold metrics are comparable across runs regardless of which
    # target transform was applied during training.
    from national.data.boring_dataset import invert_target_transform_moments

    pred_mean, pred_std = invert_target_transform_moments(
        pred_mean_trans.astype(np.float64),
        pred_std_trans.astype(np.float64),
        args.target_transform,
    )
    pred_mean = pred_mean.astype(np.float32)
    pred_std = pred_std.astype(np.float32)
    # Hybrid mode: pred_mean is the residual prediction in raw-N units
    # (because we forced target_transform='none'). Re-add the baseline
    # mean to recover the total prediction. pred_std stays as the GP
    # residual sigma — CatBoost contributes no variance.
    if hybrid_mode:
        if sub_baseline_pred is None or sub_baseline_pred.shape[0] != pred_mean.shape[0]:
            raise RuntimeError(
                "Hybrid inference: baseline-prediction slice does not align "
                f"with pred_mean (got {None if sub_baseline_pred is None else sub_baseline_pred.shape}"
                f" vs {pred_mean.shape})"
            )
        pred_mean_residual = pred_mean.copy()
        pred_mean = (pred_mean_residual + sub_baseline_pred).astype(np.float32)
        LOG.info(
            "Hybrid inference: residual prediction range [%.2f, %.2f]; "
            "baseline mean range [%.2f, %.2f]; combined mean range [%.2f, %.2f]",
            float(pred_mean_residual.min()), float(pred_mean_residual.max()),
            float(sub_baseline_pred.min()), float(sub_baseline_pred.max()),
            float(pred_mean.min()), float(pred_mean.max()),
        )
    y_true = eval_y_raw.numpy()
    fold_metrics = []
    for fi, (train_idx, test_idx) in enumerate(folds):
        rmse = float(np.sqrt(((pred_mean[test_idx] - y_true[test_idx]) ** 2).mean()))
        mae = float(np.abs(pred_mean[test_idx] - y_true[test_idx]).mean())
        std_mean = float(pred_std[test_idx].mean())
        fold_metrics.append(
            {"fold": fi, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "rmse": rmse, "mae": mae, "std_mean": std_mean}
        )
        LOG.info("fold %d: RMSE=%.2f MAE=%.2f mean_std=%.2f", fi, rmse, mae, std_mean)

    # Per-regime breakdown of the smoke subset -- highlights whether the
    # AIST cache is actually resolving the regime column to anything other
    # than UNKNOWN. The summary stores both the absolute counts and the
    # fraction so it survives configuration changes.
    regime_codes = sub_regime.numpy()
    regime_counts: dict[str, int] = {}
    for code in regime_codes:
        regime_counts[str(int(code))] = regime_counts.get(str(int(code)), 0) + 1

    summary = {
        "run_name": "kanto_smoke",
        "n_smoke": n_smoke,
        "n_features": n_input,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "n_inducing": svgp.n_inducing,
        "final_loss": output.final_loss,
        "training_time_seconds": t_train,
        "target_mean": dataset.target_mean,
        "target_std": dataset.target_std,
        "target_transform": args.target_transform,
        "regime_one_hot": args.regime_one_hot,
        "heteroscedastic_noise": args.heteroscedastic_noise,
        "kernel_type": args.kernel_type,
        "mean_type": args.mean_type,
        "inducing_init": args.inducing_init,
        "likelihood_type": args.likelihood_type,
        "studentt_df_init": args.studentt_df,
        "encoder_dim": args.encoder_dim,
        "spatial_kfold": fold_metrics,
        "regime_distribution": regime_counts,
        "device": args.device,
        "hybrid_mode": hybrid_mode,
        "baseline_name": args.baseline_name if hybrid_mode else None,
        "baseline_pred_train": str(args.baseline_pred_train) if hybrid_mode else None,
        "baseline_pred_test": str(args.baseline_pred_test) if hybrid_mode else None,
        "feature_columns": list(feature_cols),
        "zero_fourier": bool(args.zero_fourier),
        "fold_assignment": args.fold_assignment,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s", summary_path)

    # Persist per-row predictions for downstream Phase-2 analyses
    # (hybrid conformal recalibration, Mondrian conformal subgroup tables,
    # locally-weighted conformal, threshold-classifier comparison). The
    # smoke trainer's K-fold loop already populated `pred_mean` and
    # `pred_std` over `eval_x` rows.
    np.savez(
        args.output_dir / "predictions.npz",
        pred_mean=pred_mean,
        pred_std=pred_std,
        y_true=y_true,
        regime=sub_regime.numpy(),
        baseline_pred=(sub_baseline_pred if hybrid_mode else np.zeros_like(pred_mean)),
        hybrid_mode=np.array([1 if hybrid_mode else 0], dtype=np.int32),
    )
    LOG.info("Wrote %s", args.output_dir / "predictions.npz")

    # 6. Save the foundation artifact ---------------------------------------
    artifact_path = args.output_dir / "foundation_model.pt"
    model.save(artifact_path)
    LOG.info("Saved foundation artifact to %s", artifact_path)

    # 7. Auto-generate the diagnostic plot so every training run leaves a
    # visualisable trail. Done in a subprocess to keep the heavy
    # matplotlib import out of the training memory footprint.
    try:
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "scripts.visualize_results",
            "--run-dir",
            str(args.output_dir),
            "--train-fraction",
            str(args.train_fraction),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        # The visualizer needs to know if the dataset was loaded with
        # regime one-hot so its own subsample matches the training one.
        if args.regime_one_hot:
            cmd.append("--regime-one-hot")
        subprocess.run(cmd, check=False)
    except Exception as exc:  # noqa: BLE001 -- viz failure shouldn't break training
        LOG.warning("Auto-visualisation failed (non-fatal): %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
