#!/usr/bin/env python
"""Aggregate threshold-classifier results into the §5 comparison table.

Inputs:
  run_threshold_classifiers.py output:
    data/runs/threshold_classifiers_{random,contig}/summary.json
    data/runs/threshold_classifiers_{random,contig}/<label>_pred[_iso]_fold*.npy

Outputs:
  - data/runs/threshold_classifiers_{...}/analysis.json
  - docs/paper/paper_1_kanto/tables/threshold_classifiers.tex
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("threshold_analysis")


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.3f}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                   help="One per protocol (random + contig).")
    p.add_argument("--out-tex", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    a = p.parse_args()

    aggregated: list[dict] = []
    for run_dir in a.run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        fold_assignment = summary.get("fold_assignment", run_dir.name)
        for tinfo in summary["thresholds"]:
            row = {
                "fold_assignment": fold_assignment,
                "threshold_label": tinfo["label"],
                "threshold": tinfo["threshold"],
                "mode": tinfo["mode"],
                "positive_rate": tinfo["positive_rate"],
                "mean_brier_raw": tinfo["mean_brier_raw"],
                "mean_brier_iso": tinfo["mean_brier_iso"],
                "mean_ece_raw": tinfo["mean_ece_raw"],
                "mean_ece_iso": tinfo["mean_ece_iso"],
                "mean_auc_raw": tinfo["mean_auc_raw"],
            }
            aggregated.append(row)

    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps({"rows": aggregated}, indent=2))

    # Build LaTeX
    lines: list[str] = [
        r"\begin{table}[H]",
        r"  \caption{Direct threshold-probability classifiers (CatBoost / "
        r"LightGBM with isotonic recalibration) replacing the Gaussian-"
        r"CDF-from-regression baseline. Brier / ECE / AUC are fold-mean "
        r"across $K = 3$ proper spatial K-fold; positive class rate "
        r"reflects the corpus prior. Lower Brier and ECE are better; "
        r"higher AUC is better. Random and contiguous fold protocols "
        r"are reported side-by-side; the contiguous protocol is the "
        r"stricter test of spatial extrapolation.}",
        r"  \label{tab:threshold_classifiers}",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{llrrrrr}",
        r"    \toprule",
        r"    Protocol & Threshold & $P_+$ & Brier (raw) & Brier (iso) & "
        r"ECE (raw $\to$ iso) & AUC \\",
        r"    \midrule",
    ]
    for r in aggregated:
        ece_str = f"{r['mean_ece_raw']:.3f} $\\to$ {r['mean_ece_iso']:.3f}"
        lines.append(
            f"    {r['fold_assignment']} & {r['mode']} {r['threshold']:.0f} & "
            f"{_fmt(r['positive_rate'])} & "
            f"{_fmt(r['mean_brier_raw'])} & {_fmt(r['mean_brier_iso'])} & "
            f"{ece_str} & {_fmt(r['mean_auc_raw'])} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])

    a.out_tex.parent.mkdir(parents=True, exist_ok=True)
    a.out_tex.write_text("\n".join(lines) + "\n")
    LOG.info("Wrote %s and %s", a.out_tex, a.out_json)


if __name__ == "__main__":
    main()
