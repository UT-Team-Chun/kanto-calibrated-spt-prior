# `data/` — derived datasets and provenance

This directory holds derived artefacts that are not subject to
KuniJiban's redistribution clause:

| Subdirectory | Contents |
|---|---|
| [`provenance/`](provenance/) | Per-run `summary.json` provenance logs for every model training and evaluation run reported in the paper. |

## Raw KuniJiban records — not included

The raw KuniJiban borehole XML records cannot be redistributed
under KuniJiban's terms of use
(https://www.kunijiban.pwri.go.jp/). To reproduce the published
numerical results, follow the extraction procedure in
[Supplementary Material S1](../supplementary/S1_provenance_log_spec.pdf)
to retrieve the same records from the KuniJiban portal, then point
this repository's `scripts/prepare_training_data.py` at your local
extract via the `KUNIJIBAN_DATA_DIR` environment variable in
[`../.env`](../.env.template).

The derived Parquet
(`borings_kanto_aist.parquet`,
~$495{,}725$ rows $\times$ 8 features, ~14 MB) reproduces
deterministically from the raw extract using the same
preprocessing pipeline.
