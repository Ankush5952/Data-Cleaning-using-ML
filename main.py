"""Run the complete invoice cleaning and classification pipeline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from cleaner import clean_invoices
from classifier import classify_invoices
from generate_data import generate_rows
from rules import apply_rules


def ensure_input_file(input_path: Path) -> pd.DataFrame:
    """Load the input CSV or create a fresh randomized dataset when absent."""
    if input_path.exists():
        return pd.read_csv(input_path)

    input_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(None)
    np.random.seed(None)
    generated_data = generate_rows()
    generated_data.to_csv(input_path, index=False)
    print(f"Input file was missing; generated fresh test data at {input_path}")
    return generated_data


def run_pipeline(
    raw_data: pd.DataFrame,
    output_path: Path,
    window_days: int = 7,
    standard_deviations: float = 3.0,
) -> pd.DataFrame:
    """Clean, classify, flag, and export invoice data."""
    cleaned_data = clean_invoices(raw_data)
    classified_data = classify_invoices(cleaned_data)
    final_data = apply_rules(
        classified_data,
        window_days=window_days,
        standard_deviations=standard_deviations,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_data.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return final_data


def print_summary(raw_rows: int, final_data: pd.DataFrame) -> None:
    """Print a concise report for the completed pipeline."""
    print("\nInvoice processing summary")
    print("--------------------------")
    print(f"Input rows:       {raw_rows}")
    print(f"Output rows:      {len(final_data)}")
    print(f"Removed rows:     {raw_rows - len(final_data)}")
    print(f"Duplicates:       {int(final_data['Is_Duplicate'].sum())}")
    print(f"Anomalies:        {int(final_data['Is_Anomaly'].sum())}")
    print("\nCategories:")
    print(final_data["Category"].value_counts().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean, classify, and flag invoice transactions."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("raw_invoices.csv"),
        help="Input CSV path (default: raw_invoices.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("cleaned_invoices.csv"),
        help="Output CSV path (default: cleaned_invoices.csv)",
    )
    parser.add_argument(
        "--th", #threshold window
        type=int,
        default=7,
        help="Duplicate date window in days (default: 7)",
    )
    parser.add_argument(
        "--sd", #standard deviation threshold
        type=float,
        default=3.0,
        help="Anomaly threshold above category mean (default: 3)",
    )
    args = parser.parse_args()

    raw_data = ensure_input_file(args.input)
    final_data = run_pipeline(
        raw_data=raw_data,
        output_path=args.output,
        window_days=args.th, 
        standard_deviations=args.sd,
    )
    print_summary(len(raw_data), final_data)
    print(f"\nWrote final report to {args.output}")


if __name__ == "__main__":
    main()
