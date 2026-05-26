# `models/` — trained model artefacts

Large model checkpoints (notably the ~145 MB DKL+SVGP foundation
model) are distributed as **GitHub Release assets**, not as
git-tracked files, to keep `git clone` fast.

## Where to find each artefact

| Artefact | Size | Distribution |
|---|---:|---|
| `dkl_svgp/foundation_model.pt` | ~145 MB | GitHub Release `v0.1.0-cg-submission` |
| `dkl_svgp/foundation_model.meta.json` | <1 KB | in this repo at `data/provenance/kanto_full_6k_50ep_linear_rbf/foundation_model.meta.json` |
| `catboost/`, `lightgbm/`, `xgboost/`, `hgb/` | ~10–50 MB each | GitHub Release |
| `lightgbm_cqr/` | ~10 MB | GitHub Release |
| `gpboost/` | ~20 MB | GitHub Release |
| `threshold_classifiers/N_*/*.cbm` | ~5 MB each | GitHub Release |
| `cox_ph/model.pkl`, `weibull_aft/model.pkl` | <1 MB each | GitHub Release |
| `rsf/model.pkl` | ~30 MB | GitHub Release |
| `hurdle/*.cbm`, `hybrid_xfit/*.cbm` | varies | GitHub Release |
| `conformal/calibration_chosen.json`, `mondrian_quantiles.json` | <1 KB | in this repo at `data/provenance/<run>/calibration_chosen.json` |

The released survival predictor is **Random Survival Forest**
(`rsf/`); Cox PH (`cox_ph/`) and Weibull AFT (`weibull_aft/`)
are retained as parametric reference baselines after the
Schoenfeld test rejects the PH assumption at p < 10⁻¹² across
all spatial folds.

## Downloading

```bash
gh release download v0.1.0-cg-submission \
  --repo UT-Team-Chun/kanto-calibrated-spt-prior \
  --dir models/
# or manually:
# https://github.com/UT-Team-Chun/kanto-calibrated-spt-prior/releases/tag/v0.1.0-cg-submission
```

The Zenodo archive of the same release is available at
[10.5281/zenodo.20376580](https://doi.org/10.5281/zenodo.20376580)
(concept DOI; resolves to the latest tag).

Extract the downloaded archive into this directory; the layout
matches the cross-reference table in
[`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

> **Note for paper-companion repo readers**: This `models/`
> directory is *intentionally* near-empty in the git tree. Large
> binaries live only in releases so that the source-of-truth
> Python code in [`../national/`](../national/) and
> [`../scripts/`](../scripts/) can be cloned without bandwidth
> concern. Provenance metadata (`summary.json`,
> `nested_kfold_summary.json`) for every release artefact is in
> [`../data/provenance/`](../data/provenance/).
