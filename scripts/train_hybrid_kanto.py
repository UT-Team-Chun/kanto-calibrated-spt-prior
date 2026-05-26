#!/usr/bin/env python
"""Convenience wrapper that launches :mod:`scripts.train_kanto_smoke` in
hybrid mode (CatBoost / LightGBM mean + SVGP residual) by resolving the
required baseline-prediction `.npy` file paths from a single
``--baseline-dir`` and ``--baseline-name`` argument pair.

Expected directory layout produced by ``run_advanced_baselines.py
--save-fold-predictions``::

    <baseline_dir>/
        <baseline>_pred_test_fold{0,1,2}.npy
        <baseline>_pred_train_oob_fold{0,1,2}.npy
        <baseline>_idx_test_fold{0,1,2}.npy
        <baseline>_idx_train_fold{0,1,2}.npy

All other arguments are passed through to ``train_kanto_smoke.py`` verbatim,
so the full CLI of the smoke trainer (kernel, mean, inducing, encoder dim,
fold assignment, ...) is available via this wrapper.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_baseline_paths(
    baseline_dir: Path, baseline_name: str, fold: int
) -> dict[str, Path]:
    suffix = f"_fold{fold}.npy"
    pairs = {
        "baseline_pred_train": baseline_dir / f"{baseline_name}_pred_train_oob{suffix}",
        "baseline_pred_test":  baseline_dir / f"{baseline_name}_pred_test{suffix}",
        "baseline_idx_train":  baseline_dir / f"{baseline_name}_idx_train{suffix}",
        "baseline_idx_test":   baseline_dir / f"{baseline_name}_idx_test{suffix}",
    }
    missing = [str(p) for p in pairs.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing baseline prediction artefacts (did you run "
            "`run_advanced_baselines.py --save-fold-predictions`?):\n  "
            + "\n  ".join(missing)
        )
    return pairs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-dir", type=Path, required=True,
                   help="Directory containing CatBoost/LightGBM .npy artefacts")
    p.add_argument("--baseline-name", choices=["catboost", "lightgbm"],
                   default="catboost",
                   help="Which teacher's predictions to consume")
    p.add_argument("--fold", type=int, required=True,
                   help="Which outer fold's artefacts to load (matched to "
                        "--kfold-test-fold downstream)")
    p.add_argument("--smoke-args", default="",
                   help="Quoted string of extra args to forward verbatim "
                        "to train_kanto_smoke.py (e.g. \"--parquet ... "
                        "--n-epochs 50 --kernel-type rbf --mean-type linear "
                        "--n-inducing 6000 --encoder-dim 24 --device cuda "
                        "--kfold-test-fold N --output-dir ...\")")
    a = p.parse_args()

    pairs = _resolve_baseline_paths(a.baseline_dir, a.baseline_name, a.fold)
    forwarded = shlex.split(a.smoke_args) if a.smoke_args else []

    cmd = [
        sys.executable, "-m", "scripts.train_kanto_smoke",
        "--baseline-name", a.baseline_name,
        "--baseline-pred-train", str(pairs["baseline_pred_train"]),
        "--baseline-pred-test",  str(pairs["baseline_pred_test"]),
        "--baseline-idx-train",  str(pairs["baseline_idx_train"]),
        "--baseline-idx-test",   str(pairs["baseline_idx_test"]),
        *forwarded,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT / "backend"))
    print("[train_hybrid_kanto] launching:", " ".join(shlex.quote(x) for x in cmd))
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
