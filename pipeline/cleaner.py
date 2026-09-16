"""Clean and standardize the raw invoice CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "Invoice_ID",
    "Transaction_Date",
    "Vendor",
    "Amount",
    "Description",
    "Payment_Method",
}


def clean_vendor(value: object) -> str:
    """Remove payment-terminal prefixes and transaction IDs from a vendor."""
    if pd.isna(value): # type: ignore
        return "Unknown Vendor"

    vendor = str(value).strip().upper()
    vendor = re.sub(r"^\s*POS[-\s]*", "", vendor)
    vendor = re.sub(r"\s*[*#_ -]\s*[A-Z0-9]{4,}\s*$", "", vendor)
    vendor = vendor.replace("_", " ")
    vendor = re.sub(r"[^A-Z0-9& ]+", " ", vendor)
    vendor = re.sub(r"\s+", " ", vendor).strip()
    return vendor.title() or "Unknown Vendor"


def clean_invoices(data: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of an invoice DataFrame."""
    cleaned = data.copy()
    cleaned.columns = cleaned.columns.astype(str).str.strip()

    missing_columns = REQUIRED_COLUMNS.difference(cleaned.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    cleaned["Transaction_Date"] = pd.to_datetime(
        cleaned["Transaction_Date"],
        errors="coerce",
        format="mixed",
    )
    cleaned = cleaned.dropna(subset=["Transaction_Date"]).copy()
    cleaned["Amount"] = pd.to_numeric(cleaned["Amount"], errors="coerce")
    cleaned["Amount"] = cleaned["Amount"].fillna(cleaned["Amount"].median())
    cleaned["Vendor"] = cleaned["Vendor"].map(clean_vendor)
    cleaned["Description"] = (
        cleaned["Description"].fillna("Unknown description").astype(str).str.strip()
    )
    cleaned["Payment_Method"] = (
        cleaned["Payment_Method"].fillna("Unknown").astype(str).str.strip()
    )

    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean a raw invoice CSV file.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("input data/raw_invoices.csv"),
        help="Input CSV path (default: input data/raw_invoices.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output data/cleaned_invoices.csv"),
        help="Output CSV path (default: output data/cleaned_invoices.csv)",
    )
    args = parser.parse_args()

    raw_data = pd.read_csv(args.input)
    cleaned_data = clean_invoices(raw_data)
    cleaned_data.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Cleaned {len(cleaned_data)} rows into {args.output}")


if __name__ == "__main__":
    main()
