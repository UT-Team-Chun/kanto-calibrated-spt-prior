# Authors

## Lead developer & paper first author

**Ryota Okauchi**
([@R-Okauchi](https://github.com/R-Okauchi),
[ORCID 0009-0001-5391-7133](https://orcid.org/0009-0001-5391-7133))
The University of Tokyo

Designed and implemented the entire calibrated SPT-prior framework
as **sole developer**:

- **Spatial validation protocol** — mesh-keyed K-fold with four
  geometries: random (load-balanced mesh shuffle), contiguous
  (k-means on secondary-mesh centroids), buffered (1-mesh ring),
  and leave-prefecture-out. Random and contiguous are applied
  across all six candidate regressors; buffered and
  leave-prefecture-out characterise the operational DKL+SVGP
  envelope.
- **Six interchangeable candidate regressors** — Deep-kernel
  sparse variational GP (DKL+SVGP) with the full architecture
  (encoder design, mean-function ablations, inducing-initialisation
  ablations); tuned tree ensembles (CatBoost, LightGBM, XGBoost,
  HGB); LightGBM-CQR (conformalized quantile regression);
  GPBoost (Vecchia-approximated scalable GP).
- **Model-agnostic split-conformal calibration layer** — with
  model-aware conformity scores (normalised residual for
  DKL+SVGP, absolute residual for tree ensembles and GPBoost,
  CQR score for LightGBM-CQR) and a Mondrian (regime by depth)
  conditional-coverage decomposition.
- **Paired statistical-significance protocol** — mesh-level BCa
  bootstrap on the contiguous-fold paired RMSE difference; the
  95 % CI for GPBoost − CatBoost = [−3.55, −1.87] excludes zero.
- **Right-censored survival reformulation** of the
  depth-to-first-stiff endpoint, instantiated with Cox PH,
  Weibull AFT, and Random Survival Forest. The Schoenfeld
  residual test rejects the Cox PH assumption at `p < 10⁻¹²`
  across all spatial folds; the released survival predictor is
  Random Survival Forest. Cox PH and Weibull AFT are retained
  as parametric reference baselines.
- **Direct binary threshold-probability classifier pipeline**,
  including isotonic recalibration on a mesh-disjoint inner
  calibration split (for stiff-layer and soft-layer screening
  maps).
- **Per-borehole endpoint regressors** — soft-layer thickness,
  upper-10 m mean / minimum N.
- **Cross-fit CatBoost teacher + SVGP residual hybrid**
  (`k = 5` inner OOB) and the two-stage hurdle alternative,
  reported as supplementary architectural diagnostics.
- **Do-not-trust mask** (> 5 km from any training borehole) to
  flag out-of-network locations as requiring site-specific
  investigation.
- All evaluation harnesses, figures, tables, and the paper text
  in both English and Japanese.

## Supervisor

**Prof. Pang-jo Chun (全 邦釘)** — The University of Tokyo,
[Team Chun](https://github.com/UT-Team-Chun).
Research direction and paper review.

## Data source

**国土地盤情報検索サイト「KuniJiban」** — operated jointly by the
Ministry of Land, Infrastructure, Transport and Tourism (MLIT),
the Public Works Research Institute (PWRI), and the Port and
Airport Research Institute (PARI). KuniJiban provides public
access to borehole records used in this study. The raw borehole
XML records are **not** redistributed in this repository;
readers can retrieve the same records by following the
extraction procedure documented in
[Supplementary Material S1](supplementary/S1_provenance_log_spec.pdf).

## Additional data sources

- **Geospatial Information Authority of Japan (GSI)** — digital
  elevation model (DEM).
- **National Institute of Advanced Industrial Science and
  Technology (AIST)** — 1:200,000 seamless geology map (used as
  the AIST regime label in the 14-D covariate stack).
- **MLIT W05** — Class-1 river polylines (used for
  river-distance feature).
- **MLIT C23** — coastline polylines (used for coast-distance
  feature).

## AI assistance disclosure

Per Elsevier's *Declaration of generative-AI and AI-assisted
technologies in the writing process* requirement: the authors
used AI-assisted tools (Anthropic Claude and OpenAI ChatGPT) for
English-language polishing and for simulating critical reviewer
commentary on draft text during manuscript preparation. The
authors reviewed, edited, and verified all AI-assisted output,
and take full responsibility for the final scientific content
of the manuscript. The tools were not used to generate or
analyse data, to design experiments, or to interpret results.
