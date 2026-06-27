#!/usr/bin/env python
"""Phase R (review response, R1.2) — per-lithology error breakdown.

Reviewer 1 asks which soil types carry the largest prediction errors. The
released v4 feature schema carries a granular AIST lithology macro code
(`aist_litho_macro_code`, the 15-way `AistLithoMacro` classification) derived
from the AIST seamless geological map -- a per-location SURFACE-geology proxy
assigned at each borehole's (lat, lon) and constant across its depth rows (NOT
depth-resolved), preferable to a fuzzy keyword join against free-text borehole
descriptions but not a substitute for a depth-aligned soil-description taxonomy. We compute out-of-fold CatBoost (the recommended
within-network point regressor) point errors under the random spatial 3-fold
and report RMSE / MAE / signed bias (r = y - yhat; +ve = under-prediction) per
lithology macro. This complements the 8-way AIST surface-regime breakdown of
Table~depth_regime_metrics with the finer lithology view the reviewer requested.

Output:
  docs/paper/paper_1_kanto/tables/lithology_breakdown.tex

Run:
  cd backend
  uv run python -m scripts.build_lithology_breakdown_phase_r
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from national.data.derived.aist_granular import AistLithoMacro
from national.evaluation.baselines import fit_predict_catboost
from national.evaluation.spatial_kfold import spatial_kfold_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = PROJECT_ROOT / "data/features/borings_japan_v4.parquet"
TABLES_DIR = PROJECT_ROOT / "docs/paper/paper_1_kanto/tables"
KANTO_BBOX = (34.85, 37.20, 138.40, 141.00)
FEATURES = [
    "latitude_deg", "longitude_deg", "depth_from_surface",
    "absolute_elevation", "river_distance_km", "coast_distance_km",
    "regime_code",
]
LOG = logging.getLogger("lithology_breakdown")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-n", type=int, default=50,
                   help="Minimum rows for a lithology macro to be reported.")
    args = p.parse_args(argv)

    df = pd.read_parquet(args.parquet)
    la, lb, lo, lp = KANTO_BBOX
    df = df[(df.latitude_deg.between(la, lb)) & (df.longitude_deg.between(lo, lp))]
    df = df.dropna(subset=["aist_litho_macro_code"]).reset_index(drop=True)
    LOG.info("Kanto rows with lithology macro: %d", len(df))

    x = df[FEATURES].to_numpy(np.float64)
    y = df["n_value"].to_numpy(np.float64)
    litho = df["aist_litho_macro_code"].to_numpy(int)

    # Out-of-fold CatBoost predictions on the random spatial 3-fold.
    oof = np.full(len(df), np.nan)
    for k, (tr, te) in enumerate(spatial_kfold_split(df, n_folds=3, mesh_level=2, seed=args.seed)):
        LOG.info("fold %d: train=%d test=%d", k, len(tr), len(te))
        oof[te] = fit_predict_catboost(x[tr], y[tr], x[te])
    err = y - oof  # +ve = under-prediction

    names = {int(m): m.name.replace("_", " ").title() for m in AistLithoMacro}
    rows = []
    for code in sorted(set(litho.tolist())):
        m = litho == code
        if m.sum() < args.min_n:
            continue
        rows.append({
            "name": names.get(code, f"code {code}"),
            "n": int(m.sum()),
            "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
            "mae": float(np.mean(np.abs(err[m]))),
            "bias": float(np.mean(err[m])),
            "mean_y": float(np.mean(y[m])),
        })
    rows.sort(key=lambda r: r["rmse"], reverse=True)

    lines = [
        r"\begin{table}[H]",
        r"  \caption{Per-lithology point-prediction error (R1.2), using the",
        r"           granular AIST macro-lithology code in the released v4 feature",
        r"           schema (15-way \texttt{AistLithoMacro} classification) --- a",
        r"           per-location surface-geology proxy from the AIST seamless map,",
        r"           assigned at each borehole location and constant with depth (not a",
        r"           depth-aligned soil-description taxonomy).",
        r"           Out-of-fold CatBoost predictions under the",
        r"           random spatial 3-fold; RMSE / MAE / signed mean residual",
        r"           $\overline{r}=y-\hat{y}$ (+ve $=$ under-prediction) in raw",
        rf"           \Nblow{{}} units. Lithologies with $<{args.min_n}$ Kanto rows",
        r"           are omitted. This is the finer lithology view of the",
        r"           per-regime errors in Table~\ref{tab:depth_regime_metrics};",
        r"           the largest errors fall on the sparse hard-rock and",
        r"           coarse-clastic lithologies (Metamorphic, Volcanic-pyroclastic,",
        r"           Granitic; high in-class $N$ variance), while the soft alluvial",
        r"           lithology that dominates the corpus ($n\approx275{,}000$) is",
        r"           predicted most accurately and near-unbiased",
        r"           ($\overline{r}=-0.14$). Lithologies with only a few hundred",
        r"           rows (Loess, Volcanic-lava) carry noisy, non-representative",
        r"           statistics.}",
        r"  \label{tab:lithology_breakdown}",
        r"  \centering\small",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    AIST macro-lithology & $n$ & RMSE & MAE & $\overline{r}$ \\",
        r"    \midrule",
    ]
    for r in rows:
        lines.append(
            f"    {r['name']} & {r['n']:,} & {r['rmse']:.2f} & {r['mae']:.2f} & {r['bias']:+.2f} \\\\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (TABLES_DIR / "lithology_breakdown.tex").write_text("\n".join(lines) + "\n")
    LOG.info("Wrote lithology_breakdown.tex (%d lithologies)", len(rows))
    for r in rows:
        LOG.info("  %-22s n=%-7d rmse=%.2f bias=%+.2f", r["name"], r["n"], r["rmse"], r["bias"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
