# `data/provenance/` — per-run summary.json

Every model training and evaluation run shipped with this paper
writes a `summary.json` provenance log here. The directory naming
mirrors the canonical `run_name` referenced from the paper.

## Schema

The schema is documented in
[Supplementary Material S1](../../supplementary/S1_provenance_log_spec.pdf)
and includes:

- Run configuration (`kernel_type`, `mean_type`, `inducing_init`,
  `n_inducing`, `n_epochs`, etc.).
- Spatial K-fold per-fold metrics
  (`spatial_kfold[].{fold, n_train, n_test, rmse, mae, std_mean}`).
- Regime distribution.
- Device + wall-clock training time.

## Headline runs

The headline DKL+SVGP operational run is
[`kanto_full_6k_50ep_linear_rbf/`](kanto_full_6k_50ep_linear_rbf/).
The macro → canonical-file map is in
[`../../REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).

## Folds are deterministic from seed

This directory does **not** contain per-row spatial fold-assignment
CSVs. The fold geometry is fully determined by:

1. The set of secondary-mesh codes present in the Kanto extract.
2. The fold seed (default `42`, recorded in every `summary.json`).
3. The protocol (`random` load-balanced vs `contig` k-means).

Given the same KuniJiban extract and seed, the
[`national.evaluation.spatial_kfold`](../../national/evaluation/spatial_kfold.py)
module reproduces the exact same fold IDs bit-for-bit. The per-fold
row counts (`n_train`, `n_test`) recorded in the `summary.json`
files here are the authoritative cross-check.
