# kanto-calibrated-spt-prior

**Companion repository for the paper**:

> Okauchi, R. and Chun, P.-J. (2026). _A calibrated regional
> screening framework for SPT N-values from public borehole
> records under strict spatial cross-validation._ Submitted to
> _Computers and Geotechnics_.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.20376580-1682d4.svg)](https://doi.org/10.5281/zenodo.20376580)

This repository is the **canonical reproduction package** for
every numerical result, table, and figure in the paper. The paper
proposes a *framework*, not a single model: a spatial validation
protocol, a model-agnostic split-conformal calibration layer with
a Mondrian conditional-coverage decomposition, and a
right-censored survival reformulation of the depth-to-first-stiff
endpoint, instantiated on the public KuniJiban borehole archive
over the Japanese Kanto plain (495,725 measurements from 21,031
unique boreholes). The framework treats the regressor as a
configurable component and is instantiated with six interchangeable
candidate regressors (DKL+SVGP, CatBoost, LightGBM, XGBoost,
LightGBM-CQR, GPBoost) plus three survival predictors (Cox PH,
Weibull AFT, Random Survival Forest).

Released artefacts include trained model checkpoints,
spatial-fold mesh-assignment indices, conformal calibration
outputs, Mondrian conditional radii, Cloud-Optimised GeoTIFF
threshold-exceedance maps, per-borehole site-scale depth
profiles, and the two Supplementary Material specifications.

## Quick reference: paper headline numbers ↔ files

Every number cited below is a fair K-fold cell from Table 6 in
the paper (or, where noted, a survival diagnostic).

| Headline number (paper) | Canonical file |
|---|---|
| **No single regressor dominates** (random vs contiguous) — see Table 6 | `data/provenance/tree_conformal/results.json` |
| CatBoost+conformal RMSE 9.751 (random, **best random-fold point**) | `models/catboost/nested_kfold_summary.json` |
| GPBoost+conformal RMSE **10.744** (contiguous, **best contiguous-fold point**) | `models/gpboost/nested_kfold_summary.json` |
| LightGBM-CQR stiff coverage 0.704 (random, **best stiff-layer screening**) | `models/lightgbm_cqr/nested_kfold_summary.json` |
| DKL+SVGP+conformal RMSE 11.719 / 13.733 (fair K-fold) | `models/dkl_svgp/nested_kfold_summary.json` |
| GPBoost − CatBoost paired contiguous BCa 95% CI [−3.55, −1.87] | `data/provenance/stat_significance/tree_conformal_gpboost_vs_catboost.json` |
| Random Survival Forest C-index 0.727 / 0.574 (**released survival predictor**) | `models/rsf/nested_kfold_summary.json` |
| Schoenfeld p < 10⁻¹² rejects Cox PH (across all spatial folds) | `models/cox_ph/schoenfeld_test.json` |
| Released foundation-model training-fit diagnostic RMSE 5.875 (Appendix A) | `models/dkl_svgp/summary.json` |

The full `\macro → script → measurement` mapping lives in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Layout

```
kanto-calibrated-spt-prior/
├── README.md                    # this file
├── LICENSE                      # MIT
├── CITATION.cff                 # GitHub citation widget + Zenodo metadata
├── AUTHORS.md                   # contributors + data sources
├── REPRODUCIBILITY.md           # macro ↔ script ↔ JSON map
├── .env.template                # environment variables (KUNIJIBAN_DATA_DIR, …)
├── pyproject.toml               # Python package (`national`) + dependencies
├── national/                    # Python package — implementation
│   ├── data/                    # boring datasets, covariate enrichment
│   ├── tiling/                  # mesh grid, halo, regime classifier
│   ├── models/                  # DKL+SVGP foundation model + heads
│   ├── training/                # training driver, fold orchestrator
│   ├── prediction/              # tiled inference, GeoTIFF / Zarr output
│   ├── evaluation/              # spatial K-fold, baselines, conformal
│   └── api/                     # (optional) FastAPI prediction endpoints
├── scripts/                     # CLI entry-points calling national.*
│   ├── prepare_training_data.py
│   ├── train_kanto_smoke.py     # DKL+SVGP
│   ├── run_baselines.py         # CatBoost / LightGBM / XGBoost / HGB
│   ├── run_advanced_baselines.py
│   ├── train_gpboost.py         # GPBoost (Vecchia-approximated GP)
│   ├── train_lightgbm_cqr.py    # CQR-LightGBM
│   ├── train_hybrid_kanto.py    # cross-fit hybrid (k=5)
│   ├── train_hurdle_model.py    # two-stage hurdle
│   ├── train_survival_*.py      # Cox PH / Weibull AFT / RSF
│   ├── run_threshold_classifiers.py
│   ├── compute_locally_weighted_conformal.py  # split + Mondrian
│   ├── compute_tree_conformal_paired_bca.py   # BCa CI for paired diff
│   ├── build_paper_figures.py   # all auto-generated PDFs
│   └── build_paper_tables.py    # auto-generated LaTeX tables
├── data/
│   ├── fold_indices/            # spatial K-fold mesh assignments (random / contig)
│   └── provenance/              # per-run summary.json + nested_kfold_summary.json
├── models/                      # trained checkpoints (released via GitHub Release)
│   ├── dkl_svgp/                # foundation_model.pt + summary.json + meta
│   ├── catboost/, lightgbm/, xgboost/, hgb/
│   ├── gpboost/                 # Vecchia-approximated GP regressor
│   ├── lightgbm_cqr/            # CQR-quantile LightGBM
│   ├── conformal/               # split + Mondrian quantiles
│   ├── threshold_classifiers/   # isotonic-recalibrated CatBoost (P[N<c], P[N≥30])
│   ├── hurdle/                  # two-stage hurdle artefacts
│   ├── hybrid_xfit/             # cross-fit hybrid (k=5)
│   ├── cox_ph/                  # parametric reference baseline
│   ├── weibull_aft/             # parametric reference baseline
│   └── rsf/                     # Random Survival Forest (RELEASED survival predictor)
├── outputs/
│   ├── exceedance_maps_raw_N/   # COG slices for P(N<c), P(N≥30)
│   └── site_profiles/           # per-borehole depth profiles
└── supplementary/
    ├── S1_provenance_log_spec.pdf   # mirror of paper Supp. Mat. S1
    └── S2_evaluation_module_spec.pdf
```

## How to reproduce

> **Note**: Raw KuniJiban borehole XML records cannot be
> redistributed under KuniJiban's terms of use. To reproduce the
> headline numbers, first obtain the same XML records from
> <https://www.kunijiban.pwri.go.jp/> following the extraction
> procedure in
> [`supplementary/S1_provenance_log_spec.pdf`](supplementary/S1_provenance_log_spec.pdf),
> then point this repository's scripts at your local extract.

```bash
# 1. Install (Python 3.12 recommended)
python -m venv .venv
source .venv/bin/activate
pip install -e .[baselines]

# 2. Configure the path to your local KuniJiban extract
cp .env.template .env
# Edit .env to set KUNIJIBAN_DATA_DIR

# 3. Reproduce the headline framework: spatial folds → six regressors → conformal → endpoints
python -m scripts.prepare_training_data --region kanto
python -m scripts.train_kanto_smoke      \
    --parquet data/features/borings_kanto_aist.parquet \
    --output-dir data/runs/kanto_full_6k_50ep_linear_rbf \
    --kernel rbf --mean linear --n-inducing 6000 \
    --n-epochs 50 --batch-size 4096 --lr 5e-3 --regime-one-hot
python -m scripts.run_baselines          # CatBoost / LightGBM / XGBoost / HGB
python -m scripts.train_gpboost          # GPBoost (Vecchia approx)
python -m scripts.train_lightgbm_cqr     # CQR-LightGBM
python -m scripts.compute_locally_weighted_conformal
python -m scripts.compute_tree_conformal_paired_bca  # BCa CI

# 4. Survival reformulation of depth-to-first-stiff endpoint
python -m scripts.train_survival_cox_ph
python -m scripts.train_survival_weibull_aft
python -m scripts.train_survival_rsf     # RELEASED predictor

# 5. Engineering-facing primitives
python -m scripts.run_threshold_classifiers
python -m scripts.train_endpoint_models
python -m scripts.train_hurdle_model
python -m scripts.train_hybrid_kanto --inner-k 5

# 6. Regional screening outputs + paper figures/tables
python -m scripts.build_paper_figures
python -m scripts.build_paper_tables
```

Each run writes a `summary.json` whose schema is specified in
`supplementary/S1_provenance_log_spec.pdf`. The complete
`paper-macro → script → JSON` map is in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Large artefacts

Trained model checkpoints (largest is the DKL+SVGP foundation
model at ~145 MB) are distributed via **GitHub Releases**, not
committed to the repository tree, to keep `git clone` fast.
Download from:

> <https://github.com/UT-Team-Chun/kanto-calibrated-spt-prior/releases/tag/v0.1.0-cg-submission>

The release asset is a single archive; extract into the
repository root and the layout above is recovered. A Zenodo
archive of the same release is available at
[10.5281/zenodo.20376580](https://doi.org/10.5281/zenodo.20376580).

## On the `national/` package naming

The Python package is named [`national/`](national/) because it
implements a *national-scale* geotechnical foundation framework;
the present paper instantiates and validates that framework on
the Kanto subset of the KuniJiban archive as Phase 1. A
national-scale extension is the topic of follow-up work; the
package is intentionally retained under the same name to keep
the codebase forward-compatible.

## Relation to the upstream development repository

The foundation-model pipeline was developed in
[`UT-Team-Chun/geo-estimation`](https://github.com/UT-Team-Chun/geo-estimation),
which carries the full application code (national-scale
prediction service, frontend, internal experiment logs, etc.).
**The canonical reference for reproducing this paper is the
present repository**; the upstream may evolve beyond the paper's
scope in follow-up work.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

```bibtex
@article{okauchi2026kanto,
  title   = {A calibrated regional screening framework for
             {SPT} {$N$}-values from public borehole records
             under strict spatial cross-validation},
  author  = {Okauchi, Ryota and Chun, Pang-jo},
  journal = {Computers and Geotechnics},
  year    = {2026},
  note    = {Submitted}
}
```

A [`CITATION.cff`](CITATION.cff) file is provided for GitHub's
citation widget and Zenodo metadata.
