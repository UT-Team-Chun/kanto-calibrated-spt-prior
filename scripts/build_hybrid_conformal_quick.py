#!/usr/bin/env python
"""Quick hybrid + conditional conformal evaluation across available runs.

Reads each hybrid_<prot>_f<k>/predictions.npz under --runs-root and:
  1. Reports raw hybrid RMSE / MAE / 95% Gaussian coverage / interval score.
  2. Fits *marginal* split conformal on a mesh-disjoint cal subset of each
     test fold (inner 20% of unique mesh codes in the test fold). Reports
     conformal-calibrated 95% coverage and width.
  3. Fits *Mondrian* conformal on the same cal subset, grouped by
     (regime, depth-bin) joint and by regime alone. Reports group-
     conditional N>=30 coverage at 95%.
  4. Locally-weighted conformal in geographic feature space (lat, lon,
     depth, regime one-hot) with a single bandwidth (0.1) for the same
     evaluation (capped to 1500 test rows for cost).

The cal/test split within a fold is a known approximation — proper
nested cal would use mesh-disjoint cal on the *training* side. This
script reports the in-fold approximation as a faster-to-compute proxy
that still surfaces the relative ordering between marginal, Mondrian,
and locally-weighted conformal.

Outputs:
  data/runs/hybrid_conformal_summary.json
  docs/paper/paper_1_kanto/tables/hybrid_conformal.tex
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("hybrid_conformal_quick")

DEPTH_EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf"))


def _interval_score(y, lo, hi, alpha):
    width = hi - lo
    miss_lo = np.maximum(lo - y, 0.0)
    miss_hi = np.maximum(y - hi, 0.0)
    return float(np.mean(width + (2.0 / (1.0 - alpha)) * (miss_lo + miss_hi)))


def _qhat(s, alpha):
    s_sorted = np.sort(s)
    n = len(s_sorted)
    k = min(max(int(np.ceil((n + 1) * alpha)), 1), n)
    return float(s_sorted[k - 1])


def _per_fold_eval(pred_npz_path: Path, df: pd.DataFrame, fold_arr: np.ndarray,
                    fold_k: int, mondrian_min_group_n: int = 30,
                    rng_seed: int = 42, lw_bandwidth: float = 0.1,
                    lw_sample_cap: int = 1500) -> dict:
    z = np.load(pred_npz_path)
    pred_mean = z["pred_mean"].astype(np.float64)
    pred_std = np.maximum(z["pred_std"].astype(np.float64), 1e-3)
    y_true = z["y_true"].astype(np.float64)

    test_mask = fold_arr == fold_k
    n_test = int(test_mask.sum())
    if n_test != pred_mean.shape[0]:
        # KMeans contiguous cluster IDs can differ between the run_advanced_baselines
        # `assign_folds` and the train_kanto_smoke `spatial_kfold_split_contiguous`
        # call. Look for a fold whose size matches the predictions, then map.
        for k_alt in range(int(fold_arr.max()) + 1):
            if int((fold_arr == k_alt).sum()) == pred_mean.shape[0]:
                LOG.warning(
                    "fold %d size (%d) mismatch; remapping to fold %d (size %d)",
                    fold_k, n_test, k_alt, pred_mean.shape[0],
                )
                fold_k = k_alt
                test_mask = fold_arr == fold_k
                n_test = int(test_mask.sum())
                break
        else:
            raise RuntimeError(
                f"fold {fold_k} test size ({n_test}) != predictions size "
                f"({pred_mean.shape[0]}); no alternative fold matches either"
            )
    test_idx = np.where(test_mask)[0]
    test_lat = df["latitude_deg"].values[test_idx]
    test_lon = df["longitude_deg"].values[test_idx]
    test_depth = df["depth_from_surface"].values[test_idx]
    test_regime = df["regime_code"].values[test_idx]
    depth_bin = np.clip(np.digitize(test_depth, np.asarray(DEPTH_EDGES)) - 1, 0, len(DEPTH_EDGES) - 2)

    # Inner cal split: 20% of unique mesh codes in the test fold
    from run_advanced_baselines import secondary_mesh_code
    codes_test = np.array([secondary_mesh_code(la, lo) for la, lo in zip(test_lat, test_lon)])
    unique_codes, code_inverse = np.unique(codes_test, return_inverse=True)
    rng = np.random.default_rng(rng_seed + fold_k)
    n_cal_meshes = max(1, int(unique_codes.size * 0.2))
    cal_mesh_idx = rng.choice(unique_codes.size, size=n_cal_meshes, replace=False)
    cal_mesh_mask = np.zeros(unique_codes.size, dtype=bool)
    cal_mesh_mask[cal_mesh_idx] = True
    cal_in_test = cal_mesh_mask[code_inverse]
    eval_in_test = ~cal_in_test

    rec: dict = {"fold": fold_k, "n_test": n_test,
                  "n_cal": int(cal_in_test.sum()),
                  "n_eval": int(eval_in_test.sum())}

    y_cal, mu_cal, sigma_cal = y_true[cal_in_test], pred_mean[cal_in_test], pred_std[cal_in_test]
    y_ev, mu_ev, sigma_ev = y_true[eval_in_test], pred_mean[eval_in_test], pred_std[eval_in_test]
    regime_cal = test_regime[cal_in_test]
    regime_ev = test_regime[eval_in_test]
    depth_bin_cal = depth_bin[cal_in_test]
    depth_bin_ev = depth_bin[eval_in_test]

    rec["raw_rmse"] = float(np.sqrt(np.mean((mu_ev - y_ev) ** 2)))
    rec["raw_mae"] = float(np.mean(np.abs(mu_ev - y_ev)))
    raw_lo = mu_ev - 1.96 * sigma_ev
    raw_hi = mu_ev + 1.96 * sigma_ev
    rec["raw95_cov"] = float(((y_ev >= raw_lo) & (y_ev <= raw_hi)).mean())
    rec["raw95_width"] = float((raw_hi - raw_lo).mean())
    rec["raw95_interval_score"] = _interval_score(y_ev, raw_lo, raw_hi, 0.95)

    # Marginal conformal
    s = np.abs(y_cal - mu_cal) / sigma_cal
    q95 = _qhat(s, 0.95)
    cf_lo = mu_ev - q95 * sigma_ev
    cf_hi = mu_ev + q95 * sigma_ev
    rec["marginal95_cov"] = float(((y_ev >= cf_lo) & (y_ev <= cf_hi)).mean())
    rec["marginal95_width"] = float((cf_hi - cf_lo).mean())
    rec["marginal95_interval_score"] = _interval_score(y_ev, cf_lo, cf_hi, 0.95)

    # Headline conditional coverage on N>=30 + igneous + metamorphic
    for tag, mask in [
        ("n_ge_30", y_ev >= 30.0),
        ("regime_igneous", regime_ev == 6),
        ("regime_metamorphic", regime_ev == 5),
        ("depth_ge_20", test_depth[eval_in_test] >= 20.0),
    ]:
        if mask.sum() < 10:
            continue
        rec[f"marginal95_cov_{tag}"] = float(
            ((y_ev[mask] >= cf_lo[mask]) & (y_ev[mask] <= cf_hi[mask])).mean()
        )
        rec[f"n_{tag}"] = int(mask.sum())

    # Mondrian by regime
    regime_qhat: dict[int, float] = {}
    for g in np.unique(regime_cal):
        gm = regime_cal == g
        if gm.sum() < mondrian_min_group_n:
            continue
        regime_qhat[int(g)] = _qhat(np.abs(y_cal[gm] - mu_cal[gm]) / sigma_cal[gm], 0.95)
    q_per_row = np.full(len(y_ev), q95)
    for g, qg in regime_qhat.items():
        q_per_row[regime_ev == g] = qg
    mo_lo = mu_ev - q_per_row * sigma_ev
    mo_hi = mu_ev + q_per_row * sigma_ev
    rec["mondrian_regime95_cov"] = float(((y_ev >= mo_lo) & (y_ev <= mo_hi)).mean())
    rec["mondrian_regime95_width"] = float((mo_hi - mo_lo).mean())
    rec["mondrian_regime95_interval_score"] = _interval_score(y_ev, mo_lo, mo_hi, 0.95)
    for tag, mask in [
        ("n_ge_30", y_ev >= 30.0),
        ("regime_igneous", regime_ev == 6),
        ("regime_metamorphic", regime_ev == 5),
        ("depth_ge_20", test_depth[eval_in_test] >= 20.0),
    ]:
        if mask.sum() < 10:
            continue
        rec[f"mondrian_regime95_cov_{tag}"] = float(
            ((y_ev[mask] >= mo_lo[mask]) & (y_ev[mask] <= mo_hi[mask])).mean()
        )

    # Mondrian by (regime, depth_bin) joint
    joint_cal = regime_cal * 100 + depth_bin_cal
    joint_ev = regime_ev * 100 + depth_bin_ev
    joint_qhat: dict[int, float] = {}
    for g in np.unique(joint_cal):
        gm = joint_cal == g
        if gm.sum() < mondrian_min_group_n:
            continue
        joint_qhat[int(g)] = _qhat(np.abs(y_cal[gm] - mu_cal[gm]) / sigma_cal[gm], 0.95)
    q_per_row = np.full(len(y_ev), q95)
    for g, qg in joint_qhat.items():
        q_per_row[joint_ev == g] = qg
    mj_lo = mu_ev - q_per_row * sigma_ev
    mj_hi = mu_ev + q_per_row * sigma_ev
    rec["mondrian_regime_depth95_cov"] = float(((y_ev >= mj_lo) & (y_ev <= mj_hi)).mean())
    rec["mondrian_regime_depth95_width"] = float((mj_hi - mj_lo).mean())
    rec["mondrian_regime_depth95_interval_score"] = _interval_score(y_ev, mj_lo, mj_hi, 0.95)
    for tag, mask in [
        ("n_ge_30", y_ev >= 30.0),
        ("regime_igneous", regime_ev == 6),
        ("regime_metamorphic", regime_ev == 5),
        ("depth_ge_20", test_depth[eval_in_test] >= 20.0),
    ]:
        if mask.sum() < 10:
            continue
        rec[f"mondrian_regime_depth95_cov_{tag}"] = float(
            ((y_ev[mask] >= mj_lo[mask]) & (y_ev[mask] <= mj_hi[mask])).mean()
        )

    # Locally-weighted (geographic, capped)
    if eval_in_test.sum() > lw_sample_cap:
        sample_idx_local = rng.choice(eval_in_test.sum(), size=lw_sample_cap, replace=False)
    else:
        sample_idx_local = np.arange(eval_in_test.sum())
    geo_cal = np.stack([test_lat[cal_in_test], test_lon[cal_in_test],
                         test_depth[cal_in_test], regime_cal], axis=1).astype(np.float64)
    geo_ev = np.stack([test_lat[eval_in_test], test_lon[eval_in_test],
                        test_depth[eval_in_test], regime_ev], axis=1).astype(np.float64)
    g_mean = geo_cal.mean(axis=0)
    g_std = geo_cal.std(axis=0) + 1e-9
    geo_cal_z = (geo_cal - g_mean) / g_std
    geo_ev_z = (geo_ev - g_mean) / g_std
    sample_geo = geo_ev_z[sample_idx_local]
    sample_y = y_ev[sample_idx_local]
    sample_mu = mu_ev[sample_idx_local]
    sample_sigma = sigma_ev[sample_idx_local]
    h2 = 2.0 * (lw_bandwidth ** 2)
    cal_scores = np.abs(y_cal - mu_cal) / sigma_cal
    q_per_row = np.empty(len(sample_y))
    for j in range(len(sample_y)):
        d2 = np.sum((geo_cal_z - sample_geo[j]) ** 2, axis=1)
        w = np.exp(-d2 / h2)
        order = np.argsort(cal_scores)
        cumw = np.cumsum(w[order]) / max(w.sum(), 1e-12)
        idx = int(np.searchsorted(cumw, 0.95, side="left"))
        idx = min(idx, len(cal_scores) - 1)
        q_per_row[j] = float(cal_scores[order][idx])
    lw_lo = sample_mu - q_per_row * sample_sigma
    lw_hi = sample_mu + q_per_row * sample_sigma
    rec["lw_geo95_cov"] = float(((sample_y >= lw_lo) & (sample_y <= lw_hi)).mean())
    rec["lw_geo95_width"] = float((lw_hi - lw_lo).mean())
    rec["lw_geo95_interval_score"] = _interval_score(sample_y, lw_lo, lw_hi, 0.95)
    return rec


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--parquet", type=Path,
                   default=PROJECT_ROOT / "data/features/borings_kanto_aist.parquet")
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--label-prefix", default="hybrid",
                   help="Run-dir label prefix. Default 'hybrid' (i.e.\\ "
                        "expects hybrid_random_f{k} / hybrid_contig_f{k}). "
                        "For Phase C1 cross-fit, use 'hybrid_xfit'.")
    a = p.parse_args()

    df = pd.read_parquet(a.parquet, columns=[
        "latitude_deg", "longitude_deg", "depth_from_surface",
        "regime_code", "n_value",
    ])
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "backend" / "scripts"))
    from run_advanced_baselines import assign_folds
    from national.evaluation.spatial_kfold import spatial_kfold_split_contiguous, spatial_kfold_split

    out: dict = {"protocols": {}}
    for protocol in ("random", "contiguous"):
        if protocol == "contiguous":
            # Match train_kanto_smoke's centroid-from-mesh-bounds path so
            # fold IDs agree with the hybrid run's holdout split.
            sub_df = df[["latitude_deg", "longitude_deg"]].copy()
            sub_df["n_value"] = df["n_value"]
            fold_splits = spatial_kfold_split_contiguous(sub_df, n_folds=3, mesh_level=2, seed=42)
            fold_arr = np.empty(len(df), dtype=np.int64)
            for k, (_, test_idx) in enumerate(fold_splits):
                fold_arr[test_idx] = k
        else:
            fold_arr = assign_folds(df, n_folds=3, seed=42, assignment=protocol)
        label_pat = f"{a.label_prefix}_{'random' if protocol == 'random' else 'contig'}_f{{}}"
        folds: list[dict] = []
        for k in range(3):
            run_dir = a.runs_root / label_pat.format(k)
            pred_npz = run_dir / "predictions.npz"
            if not pred_npz.exists():
                LOG.warning("Missing %s; skipping", pred_npz)
                continue
            LOG.info("Evaluating %s", run_dir.name)
            try:
                rec = _per_fold_eval(pred_npz, df, fold_arr, fold_k=k)
            except Exception as e:  # noqa
                LOG.error("Failed on %s: %s", run_dir.name, e)
                continue
            rec["label"] = run_dir.name
            folds.append(rec)
        out["protocols"][protocol] = folds

    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(out, indent=2))
    LOG.info("Wrote %s", a.out_json)


if __name__ == "__main__":
    main()
