#!/usr/bin/env python
"""Build docs/paper/paper_1_kanto/tables/hurdle_comparison.tex from
hurdle_models_{random,contig}/summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fmt(x: float | None, prec: int = 3) -> str:
    if x is None:
        return "TBD"
    try:
        if x != x:  # NaN
            return "TBD"
    except TypeError:
        return "TBD"
    return f"{x:.{prec}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root", type=Path,
                   default=PROJECT_ROOT / "data/runs")
    p.add_argument("--out-tex", type=Path,
                   default=PROJECT_ROOT / "docs/paper/paper_1_kanto/tables/hurdle_comparison.tex")
    p.add_argument("--out-tex-ja", type=Path,
                   default=PROJECT_ROOT / "docs/paper/paper_1_kanto_ja/tables/hurdle_comparison.tex")
    a = p.parse_args()

    rows: dict[str, dict] = {}
    for protocol_label, dir_suffix in [("random", "random"), ("contig", "contig")]:
        summary_path = a.run_root / f"hurdle_models_{dir_suffix}/summary.json"
        if not summary_path.exists():
            print(f"  [skip] {summary_path} missing")
            rows[protocol_label] = {}
            continue
        s = json.loads(summary_path.read_text())
        rows[protocol_label] = {
            "rmse_h":    s.get("rmse_hurdle_mean"),
            "rmse_s":    s.get("rmse_single_mean"),
            "stiff_h":   s.get("rmse_stiff_hurdle_mean"),
            "stiff_s":   s.get("rmse_stiff_single_mean"),
            "cov95_m":   s.get("cov95_marginal_mean"),
            "cov95_s":   s.get("cov95_stiff_mean"),
            "cov95_mm":  s.get("cov95_mondrian_mean"),
            "cov95_ms":  s.get("cov95_mondrian_stiff_mean"),
        }

    def render(jp: bool) -> str:
        if jp:
            caption = (
                r"二段階 hurdle と単一回帰の点推定 + conformal カバレッジ比較。"
                r"``Hurdle'' は Section~\ref{sec:results:hurdle} の構成; "
                r"``Single'' は同じ stage-2-train サブセットで学習した CatBoost "
                r"回帰で、apples-to-apples 比較を可能にする。RMSE は $K=3$ "
                r"外側 fold の平均。``Stiff RMSE'' は $\Nblow \geq 30$ の "
                r"テスト行に制限。``Cov95 marginal'' は内側 conformal-cal "
                r"分割上の実現 95\,\% カバレッジ; ``stiff'' は真の stiff "
                r"テスト行に制限; ``$p$-quintile Mondrian'' は "
                r"$p_{\text{stiff}}(x)$ の 5 分位 bucket に bucket 固有の "
                r"conformal radius を割り当ててカバレッジを評価する。"
            )
            cols = (
                r"Protocol & RMSE H & RMSE S & Stiff H & Stiff S & "
                r"Cov$_{95}$ marg / stiff & $p$-Mondrian marg / stiff \\"
            )
        else:
            caption = (
                r"Two-stage hurdle vs.\ single-regressor point estimate "
                r"and conformal coverage. ``Hurdle'' is the "
                r"Section~\ref{sec:results:hurdle} construction; "
                r"``Single'' is a CatBoost regressor trained on the same "
                r"stage-2-train subset for like-for-like comparison. RMSE "
                r"columns are mean over $K=3$ outer folds. ``Stiff RMSE'' "
                r"is restricted to test-set rows with $\Nblow \geq 30$. "
                r"``Cov95 marginal'' is the realised 95\,\% conformal "
                r"coverage on the held-out conformal-cal split; "
                r"``stiff'' restricts to true-stiff test rows; "
                r"``$p$-quintile Mondrian'' uses five buckets of "
                r"$p_{\text{stiff}}(x)$ to assign a bucket-specific "
                r"conformal radius before evaluating coverage."
            )
            cols = (
                r"Protocol & RMSE H & RMSE S & Stiff H & Stiff S & "
                r"Cov$_{95}$ marg / stiff & $p$-Mondrian marg / stiff \\"
            )

        lines = [
            r"\begin{table}[H]",
            r"  \caption{" + caption + r"}",
            r"  \label{tab:hurdle_comparison}",
            r"  \centering",
            r"  \small",
            r"  \begin{tabular}{lrrrrrr}",
            r"    \toprule",
            "    " + cols,
            r"    \midrule",
        ]
        for protocol_label in ("random", "contig"):
            r = rows.get(protocol_label, {})
            lines.append(
                f"    {protocol_label} & {_fmt(r.get('rmse_h'))} & "
                f"{_fmt(r.get('rmse_s'))} & {_fmt(r.get('stiff_h'))} & "
                f"{_fmt(r.get('stiff_s'))} & "
                f"{_fmt(r.get('cov95_m'))} / {_fmt(r.get('cov95_s'))} & "
                f"{_fmt(r.get('cov95_mm'))} / {_fmt(r.get('cov95_ms'))} \\\\"
            )
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines) + "\n"

    a.out_tex.parent.mkdir(parents=True, exist_ok=True)
    a.out_tex.write_text(render(jp=False))
    print(f"Wrote {a.out_tex}")
    a.out_tex_ja.parent.mkdir(parents=True, exist_ok=True)
    a.out_tex_ja.write_text(render(jp=True))
    print(f"Wrote {a.out_tex_ja}")


if __name__ == "__main__":
    main()
