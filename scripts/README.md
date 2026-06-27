# `scripts/` — paper reproduction entry points

This directory holds the entry-point scripts that produce every
numerical result, table, and figure in the companion paper
(Okauchi & Chun, 2026, *Computers and Geotechnics*, submitted).
Each script is a thin CLI wrapper over the modules in
[`../national/`](../national/) (the implementation package).

## Scripts by paper component

### Data preparation

| Script | What it does |
|---|---|
| `prepare_training_data.py` | Convert raw KuniJiban XML into the model-ready Parquet (cleans, joins AIST geology, MLIT W05 river / C23 coast distances; applies QC filters per §2). |
| `enrich_borings.py` | Same as above, packaged for batch runs. |
| `make_fold_indices.py` | Generate canonical spatial K-fold mesh assignments (random / contiguous, K=3, seed=42). |

### Six interchangeable candidate regressors

| Script | Regressor | Released suite role |
|---|---|---|
| `train_kanto_smoke.py` | DKL+SVGP foundation model | Smooth probabilistic 3D priors |
| `run_baselines.py` | CatBoost / LightGBM / XGBoost / HGB | Tuned tree-ensemble candidates |
| `run_advanced_baselines.py` | Extended hyperparameter sweeps | (architectural diagnostics) |
| `train_gpboost.py` | GPBoost (Vecchia-approximated GP) | Out-of-network regional prediction |
| `train_lightgbm_cqr.py` | LightGBM-CQR (CQR-quantile LightGBM) | Stiff-layer subgroup screening |

### Conformal calibration layer (model-agnostic)

| Script | What it does |
|---|---|
| `compute_locally_weighted_conformal.py` | Split conformal + Mondrian (regime × depth) conditional-coverage decomposition. |
| `compute_tree_conformal_paired_bca.py` | Fair Table 6 cross-regressor cells + mesh-level BCa 95% CI for the paired contiguous-fold RMSE differences (GPBoost vs CatBoost: [−3.55, −1.87], excludes zero). |

### Right-censored survival reformulation

| Script | Survival model | Released suite role |
|---|---|---|
| `train_survival_cox_ph.py` | Cox proportional-hazards | Parametric reference baseline |
| `train_survival_weibull_aft.py` | Weibull AFT | Parametric reference baseline |
| `train_survival_rsf.py` | Random Survival Forest | **Released survival predictor** (Schoenfeld test rejects PH at p < 10⁻¹²) |

### Engineering-facing primitives

| Script | What it does |
|---|---|
| `run_threshold_classifiers.py` | Direct binary CatBoost classifiers for `P(N<c)` and `P(N≥30)` with isotonic recalibration on a mesh-disjoint inner split. |
| `train_endpoint_models.py` | Per-borehole regressors: soft-layer thickness, upper-10 m mean / minimum. |
| `train_hybrid_kanto.py` | Cross-fit CatBoost teacher + SVGP residual hybrid; `k=3` (null) and `k=5` (`HybridXfitRMSEContig`) protocols. Supplementary architectural diagnostic. |
| `train_hurdle_model.py` | Two-stage hurdle (Stage 1: P(N≥30) classifier; Stage 2: separated soft/stiff regressors). Supplementary architectural diagnostic. |

### Auxiliary spatial-validation geometries (recommended deployment regressors)

| Script | Geometry |
|---|---|
| `run_buffered_baselines.py` | Buffered (1-mesh ring) contiguous CV for GPBoost / CatBoost. |
| `run_leave_region_out.py --partition prefecture` | Leave-prefecture-out over all seven Kanto prefectures (administrative-polygon containment) for GPBoost / CatBoost; `--prefectures` runs a subset. |

### Review-response analyses

| Script | Purpose |
|---|---|
| `run_correction_sensitivity.py` | Correction-metadata audit + partial-correction sensitivity (raw vs C_N vs C_N·C_R). |
| `build_feature_importance_phase_r.py` | Tree-SHAP + permutation importance (random vs contiguous folds). |
| `build_lithology_breakdown_phase_r.py` | Per-macro-lithology OOF error breakdown. |
| `build_bucket_b_analyses.py` | Per-depth / per-regime signed-bias diagnostics. |

### Aggregators and figure / table builders

| Script | Output |
|---|---|
| `build_threshold_analysis.py` | Aggregates threshold-classifier metrics (PR-AUC, Brier skill score, decision curves). |
| `build_threshold_reliability_plots.py` | Reliability diagrams for raw / isotonic-recalibrated threshold classifiers. |
| `build_threshold_decision_metrics.py` | Net-benefit / decision-curve summary table. |
| `build_hybrid_evaluation.py` | Cross-fit hybrid evaluation aggregator. |
| `build_hybrid_conformal_quick.py` | Hybrid + conformal coverage quick-check. |
| `build_hurdle_table.py` | Hurdle vs DKL-only comparison table. |
| `build_paper_figures.py` | Top-level figure generation entry point (Figures 3–11). |
| `build_paper_user_figures.py` | Study-area + spatial-fold geometry figure (Figure 1). |
| `build_paper_tables.py` | Auto-generated LaTeX tables (e.g. Appendix A full ablation). |

## Conventional run order

```bash
# 1. Preprocess raw KuniJiban → Parquet
python -m scripts.prepare_training_data --region kanto

# 2. Canonical spatial K-fold mesh assignments
python -m scripts.make_fold_indices --K 3 --seed 42

# 3. Released foundation model (DKL+SVGP)
python -m scripts.train_kanto_smoke \
    --parquet data/features/borings_kanto_aist.parquet \
    --output-dir data/runs/kanto_full_6k_50ep_linear_rbf \
    --kernel rbf --mean linear \
    --n-inducing 6000 --n-epochs 50 \
    --batch-size 4096 --lr 5e-3 \
    --regime-one-hot

# 4. Six candidate regressors fair K-fold comparison
python -m scripts.run_baselines           # CatBoost / LightGBM / XGBoost / HGB
python -m scripts.run_advanced_baselines  # extended sweeps
python -m scripts.train_gpboost
python -m scripts.train_lightgbm_cqr
python -m scripts.compute_tree_conformal_paired_bca  # fair Table 6 + BCa CI

# 5. Survival reformulation (Schoenfeld → RSF released)
python -m scripts.train_survival_cox_ph
python -m scripts.train_survival_weibull_aft
python -m scripts.train_survival_rsf

# 6. Engineering primitives
python -m scripts.run_threshold_classifiers
python -m scripts.train_endpoint_models
python -m scripts.train_hurdle_model
python -m scripts.train_hybrid_kanto --inner-k 5

# 7. Conformal + Mondrian decomposition
python -m scripts.compute_locally_weighted_conformal

# 8. Auxiliary spatial-validation geometries (recommended deployment regressors)
python -m scripts.run_buffered_baselines --model gpboost  --base-split contiguous
python -m scripts.run_buffered_baselines --model catboost --base-split contiguous
python -m scripts.run_leave_region_out --partition prefecture --model gpboost
python -m scripts.run_leave_region_out --partition prefecture --model catboost

# 9. Paper tables and figures
python -m scripts.build_paper_tables
python -m scripts.build_paper_figures
python -m scripts.build_paper_user_figures
python -m scripts.build_threshold_reliability_plots
python -m scripts.build_threshold_decision_metrics
python -m scripts.build_hybrid_evaluation
python -m scripts.build_hurdle_table
```

Every command writes a `summary.json` /
`nested_kfold_summary.json` to its output directory; the schema
is documented in
[Supplementary Material S1](../supplementary/S1_provenance_log_spec.pdf).

## Macro → script map

See [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the
machine-checkable map from each headline numerical macro
(`\KantoNRows`, `\OperationalRMSE`, `\BufferedRMSE`,
`\HybridXfitRMSEContig`, `\GpboostMinusCatboostBcaLo/Hi`, …) to
the producing script and the `summary.json` it lands in.
