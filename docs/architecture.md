# Architecture — calibrated regional screening framework

This document describes the four-layer framework that the
companion paper (Okauchi & Chun, 2026, *Computers and
Geotechnics*, submitted) instantiates on the KuniJiban Kanto
borehole archive. The package is named [`national/`](../national/)
because the same framework is designed to scale to the full
national archive in follow-up work; the present paper validates
it on the Kanto subset as Phase 1.

## Framework overview

The released regional screening framework is a four-layer
construction. The regressor is a *configurable component*
between a spatial-validation layer and a post-hoc conformal
calibration layer; engineering-facing endpoints sit downstream.

```
                              Calibrated regional screening framework
                              ──────────────────────────────────────

KuniJiban borehole archive ──► Spatial K-fold ──► Candidate regressor ──► Split conformal ──► Engineering endpoints
21,031 boreholes               random / contig    {DKL+SVGP, CatBoost,    + Mondrian            threshold maps,
495,725 measurements           buffered / LPO     LightGBM, XGBoost,      marginal q_α          profile regressors,
                                                  LightGBM-CQR,           per-bin q_α^(c)       right-censored
                                                  GPBoost}                                      survival (RSF)
                                                  ▲
                                          configurable component
```

### Layer 1 — Spatial validation

Mesh-keyed K-fold protocol over the Japan-standard secondary
mesh (`MeshLevel`). Four geometries:

| Geometry | Description | Applied to |
|---|---|---|
| **random** | Load-balanced mesh shuffle; interleaved fold geometry; in-distribution diagnostic. | all six candidate regressors |
| **contiguous** | `k`-means on secondary-mesh centroids; geographic-block folds; the stricter spatial-extrapolation test. | all six candidate regressors |
| **buffered** | 1-mesh ring removed from each fold's training set; characterises worst-case spatial leakage. | DKL+SVGP + recommended regressors (GPBoost, CatBoost) |
| **leave-prefecture-out (LPO)** | Whole prefecture withheld (all seven Kanto prefectures, administrative-polygon containment); characterises out-of-distribution boundary. | GPBoost, CatBoost |

`K = 3`, seed `42`. Canonical assignments are persisted to
[`data/fold_indices/`](../data/fold_indices/).

### Layer 2 — Candidate regressors

Six interchangeable point or quantile regressors plus the
released DKL+SVGP foundation model. All share the same input
contract (`14-D covariate vector x → predicted N`):

| Regressor | Released suite role | Conformity score |
|---|---|---|
| **DKL+SVGP** (released foundation model) | Smooth probabilistic 3D priors | normalised residual `|y−μ|/σ` |
| **CatBoost** | Within-network point estimate | absolute residual `|y−ŷ|` |
| **LightGBM** | (tree-ensemble alternative) | absolute residual |
| **XGBoost** | (tree-ensemble alternative) | absolute residual |
| **LightGBM-CQR** | Stiff-layer (N≥30) screening | CQR score `max(q̂_lo−y, y−q̂_hi)` |
| **GPBoost** | Out-of-network regional prediction | absolute residual |

The 14-D covariate stack is `(lat, lon, depth, abs_elev,
river_dist, coast_dist)` plus the 8-way AIST surface-regime
one-hot encoding. The DKL+SVGP encoder additionally applies a
random-Fourier expansion (`B = 12`) on `(lat, lon)` before
feeding the ResMLP encoder; see Methods §3 of the paper.

### Layer 3 — Model-agnostic split-conformal calibration

A held-out mesh-disjoint 20 % calibration subsample of each
outer training fold produces:

- **Marginal radius** `q_α` for split-conformal coverage.
- **Mondrian per-bin radii** `q_α^(c)` on the partition
  `C = AIST regime (8-way) × depth stratum ({<10 m, 10–20 m, ≥20 m})`,
  for conditional-coverage diagnostics that localise where
  the interval is trustworthy.

Conformity scores are model-aware (see Layer 2 table). Under
proper spatial K-fold the conformal layer substantially reduces
the marginal coverage deficit, reaching near-nominal 95 %
coverage under random folds while exposing residual
under-coverage under contiguous spatial extrapolation; the
Mondrian decomposition localises this residual under-coverage
in the point estimator rather than in the conformal radius.

### Layer 4 — Engineering endpoints

Downstream of the calibrated predictive distribution:

| Endpoint | Implementation | Released suite role |
|---|---|---|
| `P(N < c)`, `P(N ≥ 30)` threshold maps | Direct binary CatBoost classifier + isotonic recalibration on a mesh-disjoint inner split. | Stiff-layer / soft-layer screening maps. |
| Per-borehole soft-layer thickness, upper-10 m mean / minimum | CatBoost point + LightGBM quantile (P10 / P50 / P90). | Site-investigation-planning input. |
| Depth-to-first-stiff (right-censored) | Cox PH, Weibull AFT, **Random Survival Forest**. The Schoenfeld test rejects PH at `p < 10⁻¹²` across all folds; RSF is the released predictor. | Pre-investigation stiff-layer reach screening. |

A do-not-trust mask flags grid cells more than 5 km from any
training borehole as requiring site-specific investigation; it
is overlaid on the threshold-exceedance maps as a 55 %-opacity
neutral-grey layer.

## Statistical-significance protocol

The cross-regressor RMSE differences in Table 6 of the paper are
accompanied by a paired mesh-level **BCa bootstrap 95 %
confidence interval** (B = 500). For the contiguous-fold
difference GPBoost − CatBoost the CI is **[−3.55, −1.87]**,
excluding zero — confirming the GPBoost contiguous-fold win is
not a `K = 3` artefact.

## Repository layout

See [`../README.md#layout`](../README.md#layout) for the
directory tree. The high-level mapping is:

- [`../national/`](../national/) — Python package; one
  subpackage per framework layer.
- [`../scripts/`](../scripts/) — CLI entry points wrapping
  `national.*`.
- [`../data/`](../data/) — fold indices, per-run provenance
  JSON.
- [`../models/`](../models/) — trained checkpoints (released
  via GitHub Release).
- [`../supplementary/`](../supplementary/) — S1 provenance log
  spec, S2 evaluation module spec.

## Reading order for newcomers

1. The paper itself (`main.pdf`), especially Methods §3 and
   Results §5–6.
2. [`../README.md`](../README.md) for the layout + headline
   numbers.
3. This file for the four-layer framework view.
4. [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for the
   `paper-macro → script → JSON` map.
5. [Supplementary Material S1](../supplementary/S1_provenance_log_spec.pdf)
   for the per-run provenance schema.
6. [Supplementary Material S2](../supplementary/S2_evaluation_module_spec.pdf)
   for the evaluation module API surface.

## Follow-up work

The framework is designed for national-scale extension. The
upstream development repository
([`UT-Team-Chun/geo-estimation`](https://github.com/UT-Team-Chun/geo-estimation))
carries the work-in-progress toward the full KuniJiban archive
(~175 k boreholes, ~2.7 M measurements), the multi-output
extension (qc, w%, fines content), the FastAPI prediction
endpoint, and the MapLibre frontend. The companion paper for
those national-scale results will appear separately.
