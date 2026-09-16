"""Apply duplicate and amount-anomaly rules to classified invoices."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cleaner import clean_invoices
from classifier import classify_invoices


def flag_duplicates(
    data: pd.DataFrame,
    window_days: int = 7,
) -> pd.DataFrame:
    """Flag later matching vendor/amount transactions within a date window."""
    required_columns = {"Vendor", "Amount", "Transaction_Date"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")
    if window_days < 0:
        raise ValueError("window_days must be non-negative")

    flagged = data.copy()
    flagged["_original_order"] = range(len(flagged))
    flagged = flagged.sort_values(
        ["Vendor", "Amount", "Transaction_Date", "_original_order"]
    )
    previous_date = flagged.groupby(["Vendor", "Amount"], sort=False)[
        "Transaction_Date"
    ].shift(1)
    within_window = (
        previous_date.notna()
        & (
            flagged["Transaction_Date"] - previous_date
        ).dt.days.le(window_days)
    )
    flagged["Is_Duplicate"] = within_window.astype(bool)
    return flagged.sort_values("_original_order").drop(columns="_original_order")


def flag_anomalies(
    data: pd.DataFrame,
    standard_deviations: float = 3.0,
) -> pd.DataFrame:
    """Flag amounts above their category mean plus N standard deviations."""
    required_columns = {"Category", "Amount"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")
    if standard_deviations < 0:
        raise ValueError("standard_deviations must be non-negative")

    flagged = data.copy()
    category_mean = flagged.groupby("Category")["Amount"].transform("mean")
    category_std = flagged.groupby("Category")["Amount"].transform("std").fillna(0)
    threshold = category_mean + standard_deviations * category_std
    flagged["Is_Anomaly"] = flagged["Amount"].gt(threshold).astype(bool)
    return flagged


def apply_rules(
    data: pd.DataFrame,
    window_days: int = 7,
    standard_deviations: float = 3.0,
) -> pd.DataFrame:
    """Apply both duplicate and anomaly detection rules."""
    return flag_anomalies(
        flag_duplicates(data, window_days=window_days),
        standard_deviations=standard_deviations,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag duplicate and anomalous invoice transactions."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("raw_invoices.csv"),
        help="Input raw CSV path (default: raw_invoices.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("flagged_invoices.csv"),
        help="Output CSV path (default: flagged_invoices.csv)",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Duplicate date window in days (default: 7)",
    )
    parser.add_argument(
        "--standard-deviations",
        type=float,
        default=3.0,
        help="Anomaly threshold above category mean (default: 3)",
    )
    args = parser.parse_args()

    cleaned = clean_invoices(pd.read_csv(args.input))
    classified = classify_invoices(cleaned)
    flagged = apply_rules(
        classified,
        window_days=args.window_days,
        standard_deviations=args.standard_deviations,
    )
    flagged.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Flagged {len(flagged)} rows into {args.output}")
    print(f"Duplicates: {int(flagged['Is_Duplicate'].sum())}")
    print(f"Anomalies: {int(flagged['Is_Anomaly'].sum())}")


if __name__ == "__main__":
    main()
