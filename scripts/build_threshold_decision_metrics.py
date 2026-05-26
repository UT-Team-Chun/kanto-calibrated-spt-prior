#!/usr/bin/env python
"""Extend tab:threshold_classifiers with PR-AUC, top-k precision,
recall at fixed FPR, Brier skill score, and decision-curve net benefit
for the rare-class threshold classifier outputs.

Inputs (already produced by Phase 1d):
  data/runs/threshold_classifiers_{random,contig}/
    {label}_pred[_iso]_fold{k}.npy
    {label}_y_fold{k}.npy
where label in {lt5, lt10, lt15, gte30}.

Outputs:
  data/runs/threshold_decision_metrics.json
  docs/paper/paper_1_kanto/tables/threshold_decision.tex
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("threshold_decision")


def _pr_auc(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def _top_k_precision(p: np.ndarray, y: np.ndarray, k_frac: float) -> float:
    n = len(p)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-p)[:k]
    return float(y[order].sum() / k)


def _recall_at_fpr(p: np.ndarray, y: np.ndarray, target_fpr: float) -> float:
    from sklearn.metrics import roc_curve

    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def _brier_skill(p: np.ndarray, y: np.ndarray) -> float:
    """1 - Brier(model) / Brier(climatology). Climatology = corpus prior."""
    p_clim = float(y.mean())
    brier_model = float(np.mean((p - y) ** 2))
    brier_clim = float(np.mean((p_clim - y) ** 2))
    if brier_clim < 1e-9:
        return float("nan")
    return 1.0 - brier_model / brier_clim


def _decision_curve_net_benefit(p: np.ndarray, y: np.ndarray,
                                 p_threshold: float) -> float:
    """Vickers 2006: NB = TP/n - FP/n * pt/(1-pt)."""
    pos = p >= p_threshold
    tp = int(((y == 1) & pos).sum())
    fp = int(((y == 0) & pos).sum())
    n = len(p)
    if p_threshold >= 1.0:
        return float("nan")
    return tp / n - fp / n * p_threshold / (1.0 - p_threshold)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                   help="One per protocol (random + contig).")
    p.add_argument("--thresholds", nargs="+",
                   default=["lt5", "lt10", "lt15", "gte30"])
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-tex", type=Path, required=True)
    a = p.parse_args()

    rows: list[dict] = []
    for run_dir in a.run_dirs:
        protocol = "contig" if "contig" in run_dir.name else "random"
        for label in a.thresholds:
            preds_iso = []
            ys = []
            for k in range(a.n_folds):
                pi = run_dir / f"{label}_pred_iso_fold{k}.npy"
                yp = run_dir / f"{label}_y_fold{k}.npy"
                if not (pi.exists() and yp.exists()):
                    LOG.warning("Missing %s or %s; skipping fold %d", pi, yp, k)
                    continue
                preds_iso.append(np.load(pi))
                ys.append(np.load(yp))
            if not preds_iso:
                continue
            p_all = np.concatenate(preds_iso)
            y_all = np.concatenate(ys)
            row = {
                "protocol": protocol,
                "threshold": label,
                "positive_rate": float(y_all.mean()),
                "pr_auc": _pr_auc(p_all, y_all),
                "top_1pct_precision": _top_k_precision(p_all, y_all, 0.01),
                "top_5pct_precision": _top_k_precision(p_all, y_all, 0.05),
                "top_10pct_precision": _top_k_precision(p_all, y_all, 0.10),
                "recall_at_fpr_1pct": _recall_at_fpr(p_all, y_all, 0.01),
                "recall_at_fpr_5pct": _recall_at_fpr(p_all, y_all, 0.05),
                "recall_at_fpr_10pct": _recall_at_fpr(p_all, y_all, 0.10),
                "brier_skill_score": _brier_skill(p_all, y_all),
                "decision_nb_pt05": _decision_curve_net_benefit(p_all, y_all, 0.05),
                "decision_nb_pt10": _decision_curve_net_benefit(p_all, y_all, 0.10),
                "decision_nb_pt20": _decision_curve_net_benefit(p_all, y_all, 0.20),
            }
            rows.append(row)
            LOG.info("%s %s: PR-AUC %.3f, top-5%% prec %.3f, recall@10%%FPR %.3f, BSS %.3f",
                     protocol, label, row["pr_auc"], row["top_5pct_precision"],
                     row["recall_at_fpr_10pct"], row["brier_skill_score"])

    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps({"rows": rows}, indent=2))

    # Build LaTeX table
    lines = [
        r"\begin{table}[H]",
        r"  \caption{Rare-class decision metrics for the isotonic-",
        r"           recalibrated threshold classifiers of",
        r"           Section~\ref{sec:results:threshold_classifiers}.",
        r"           PR-AUC (average precision) is informative on the",
        r"           rare $P(N \geq 30)$ class (positive rate 6.4\,\%)",
        r"           where ROC-AUC is dominated by the dominant negative.",
        r"           ``Top-5\,\%'' is precision in the top decile of",
        r"           predicted probability (planning: how many of the",
        r"           highest-risk meshes are true positives).",
        r"           ``Recall@10\,\% FPR'' is the fraction of true",
        r"           positives captured at the operating point that",
        r"           accepts 10\,\% false-positive rate. Brier skill",
        r"           score (BSS) compares to a constant-rate climatology;",
        r"           BSS $\leq 0$ means the classifier is no better than",
        r"           predicting the corpus prior. Decision-curve net",
        r"           benefit (NB) at threshold $p_t = 0.10$ is the",
        r"           per-row utility relative to ``treat all'' / ``treat",
        r"           none'' under a 1\,:\,9 cost ratio.}",
        r"  \label{tab:threshold_decision}",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{llrrrrr}",
        r"    \toprule",
        r"    Protocol & Threshold & $P_+$ & PR-AUC & Top-5\% prec & Recall@10\% FPR & BSS \\",
        r"    \midrule",
    ]
    for r in rows:
        thr_disp = {"lt5": r"$N<5$", "lt10": r"$N<10$",
                    "lt15": r"$N<15$", "gte30": r"$N \geq 30$"}.get(r["threshold"], r["threshold"])
        lines.append(
            f"    {r['protocol']} & {thr_disp} & "
            f"{r['positive_rate']:.3f} & {r['pr_auc']:.3f} & "
            f"{r['top_5pct_precision']:.3f} & "
            f"{r['recall_at_fpr_10pct']:.3f} & "
            f"{r['brier_skill_score']:.3f} \\\\"
        )
    lines.extend([r"    \bottomrule", r"  \end{tabular}", r"\end{table}"])
    a.out_tex.parent.mkdir(parents=True, exist_ok=True)
    a.out_tex.write_text("\n".join(lines) + "\n")
    LOG.info("Wrote %s and %s", a.out_tex, a.out_json)


if __name__ == "__main__":
    main()
