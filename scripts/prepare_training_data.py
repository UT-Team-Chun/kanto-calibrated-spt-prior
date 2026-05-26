"""Helper script to prepare full dataset for GPyTorch training.

This script automates the steps needed to prepare the CSV used by the
training script `poc.scripts.train_gpytorch_model`.

Steps performed:
  - Optionally download river data (simple or full)
  - Run verification script to produce `location_n_values_with_river.csv` for dataset
  - Validate output CSV contains expected columns
  - Print final instructions to run training script

Note: This script does not run the model training itself. It's intended to
ensure data is fully prepared so training is a single command away.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "outputs" / "location_n_values_with_river.csv"


def _run_module(module: str, args: Sequence[str]) -> int:
    env = {**dict(), **{"PYTHONPATH": str(Path(__file__).resolve().parents[3])}}
    cmd = [sys.executable, "-m", module, *args]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, env={**env, **dict(PATH=sys.executable)})


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare training data for GPyTorch model")
    parser.add_argument("--download-river", choices=["none", "simple", "full"], default="none",
                        help="Whether to download river data before extraction")
    parser.add_argument("--force", action="store_true", help="Always regenerate output CSV even if present")
    parser.add_argument("--threshold", type=float, default=None, help="River proximity threshold in km (optional)")
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH, help="Output CSV path (default data/outputs/location_n_values_with_river.csv)")

    args = parser.parse_args(argv)

    # Download river data if requested
    if args.download_river == "simple":
        print("Downloading river data (simple)...")
        rc = _run_module("poc.scripts.download_river_simple", [])
        if rc != 0:
            print("Warning: download_river_simple exited with non-zero status")
    elif args.download_river == "full":
        print("Downloading river data (full) — this might take a long time...")
        rc = _run_module("poc.scripts.download_river_data", [])
        if rc != 0:
            print("Warning: download_river_data exited with non-zero status")

    # Run verification script to extract all xml -> CSV
    output_path = args.output_path
    if output_path.exists() and not args.force:
        print(f"Output CSV already exists at {output_path}. Use --force to regenerate.")
    else:
        print("Extracting location and SPT data from XML dataset (full) ...")
        extract_args = ["--source", "dataset"]
        if args.threshold is not None:
            extract_args += ["--threshold", str(args.threshold)]
        # If the target output path is not default, instruct the extraction script
        # to write directly to the requested location using --output
        if args.output_path != OUTPUT_PATH:
            extract_args += ["--output", str(args.output_path)]
        rc = _run_module("verification.r_okauchi.v2_extract_location_with_river", extract_args)
        if rc != 0:
            print("ERROR: extraction script failed with non-zero exit code")
            sys.exit(rc)

    # Validate output
    print("Validating output CSV...")
    if not output_path.exists():
        print(f"ERROR: Expected output file not found: {output_path}")
        sys.exit(1)

    try:
        import pandas as pd

        df = pd.read_csv(output_path)
        print(f"Loaded output CSV with {len(df)} rows and {len(df.columns)} columns")
        required_cols = {"longitude_deg", "latitude_deg", "spt_start_depth", "n_value"}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"ERROR: missing expected columns: {missing}")
            sys.exit(1)
        else:
            print("Output CSV contains required columns")
    except Exception as exc:
        print(f"WARNING: Could not validate CSV using pandas: {exc}")

    print("\nData preparation complete. To start training, run:")
    print("PYTHONPATH=backend python3 -m poc.scripts.train_gpytorch_model --data-path ", output_path)


if __name__ == "__main__":
    main()
