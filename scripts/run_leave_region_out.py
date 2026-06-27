#!/usr/bin/env python
"""Leave-region-out (LRO) national evaluation — the Phase C exit gate.

Fits a model on all-but-one region (or geological block), conformal-calibrates
on a mesh-disjoint nested subset, and evaluates cross-region transfer on the
held-out region. Reports per-region + per-regime RMSE/MAE and interval coverage
plus a gate verdict.

This harness builds and tests on the Kanto Parquet today; point ``--parquet`` at
the national Parquet once it is built to run the real Phase C gate.

Run:
  cd backend
  .venv/bin/python -m scripts.run_leave_region_out \
      --parquet ../data/features/borings_kanto_aist.parquet \
      --partition region --model gpboost
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from national.evaluation.leave_region_out_runner import (
    KANTO_CONTIG_GPBOOST_RMSE,
    run_leave_region_out,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = PROJECT_ROOT / "data/features/borings_kanto_aist.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/runs/leave_region_out"

LOG = logging.getLogger("leave_region_out")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--partition",
        choices=["region", "block", "prefecture"],
        default="region",
        help="region/block need national data; 'prefecture' works on Kanto today.",
    )
    p.add_argument(
        "--prefectures",
        nargs="+",
        default=None,
        help="Subset of Kanto prefectures to hold out for --partition prefecture "
             "(default: all seven). Per-prefecture folds are independent, so a subset "
             "gives identical per-fold results --- used to parallelise the LPO sweep "
             "across nodes.",
    )
    p.add_argument(
        "--model",
        choices=["gpboost", "rf", "hgb", "lightgbm", "xgboost", "catboost"],
        default="gpboost",
    )
    p.add_argument("--cal-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-neighbors", type=int, default=20,
                   help="GPBoost Vecchia neighbourhood size.")
    p.add_argument("--n-boost-iter", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--reference-rmse", type=float, default=KANTO_CONTIG_GPBOOST_RMSE)
    p.add_argument("--rmse-rel-tol", type=float, default=0.30)
    p.add_argument("--quick", type=int, default=0,
                   help="Subsample to N rows for a smoke run (0 = full).")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    LOG.info("Loading %s", args.parquet)
    df = pd.read_parquet(args.parquet)
    if args.quick:
        df = df.sample(int(args.quick), random_state=args.seed).reset_index(drop=True)
    LOG.info("Loaded %d rows; partition=%s model=%s", len(df), args.partition, args.model)

    gpboost_kwargs = {
        "num_neighbors": args.num_neighbors,
        "n_boost_iter": args.n_boost_iter,
        "learning_rate": args.learning_rate,
    } if args.model == "gpboost" else None

    result = run_leave_region_out(
        df,
        partition=args.partition,
        model=args.model,
        cal_fraction=args.cal_fraction,
        seed=args.seed,
        gpboost_kwargs=gpboost_kwargs,
        reference_rmse=args.reference_rmse,
        rmse_rel_tol=args.rmse_rel_tol,
        prefectures=args.prefectures,
    )

    out_dir = args.out_dir / f"{args.partition}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(result, indent=2))
    LOG.info("Wrote %s", out_path)

    for f in result["per_fold"]:
        LOG.info("  %-16s rmse=%.3f mae=%.3f n_test=%d",
                 f["fold"], f["rmse"], f["mae"], f["n_test"])
    agg = result["aggregate"]
    gate = result["gate"]
    LOG.info("Aggregate: rmse=%.3f ± %.3f (worst %.3f)  cov95=%.3f",
             agg["rmse_mean"], agg["rmse_std"], agg["rmse_worst"],
             gate["mean_coverage_95"])
    LOG.info("GATE: pass=%s (rmse %.3f <= %.3f: %s; cov95 %.3f within ±%.2f: %s)",
             gate["pass"], gate["mean_test_rmse"], gate["rmse_threshold"],
             gate["pass_rmse"], gate["mean_coverage_95"],
             gate["coverage_abs_tol"], gate["pass_coverage"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
