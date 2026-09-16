"""Classify cleaned invoice vendors into expense categories."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cleaner import clean_invoices


SEED_EXAMPLES = pd.DataFrame(
    {
        "text": [
            "AWS cloud hosting",
            "Microsoft Azure cloud services",
            "Slack team collaboration software",
            "Dropbox online file storage",
            "GitHub software development platform",
            "Uber ride",
            "Lyft ride share",
            "Delta Airlines flight",
            "Airbnb lodging",
            "Shell fuel station",
            "Staples office supplies",
            "Office Depot stationery",
            "Amazon Marketplace office supplies",
            "Walmart office supplies",
            "Target office supplies",
            "Comcast internet service",
            "Verizon phone service",
            "Duke Energy electricity",
            "City Water utility bill",
            "AT&T telecommunications service",
        ],
        "category": [
            "SaaS/Cloud",
            "SaaS/Cloud",
            "SaaS/Cloud",
            "SaaS/Cloud",
            "SaaS/Cloud",
            "Travel/Transport",
            "Travel/Transport",
            "Travel/Transport",
            "Travel/Transport",
            "Travel/Transport",
            "Office Supplies",
            "Office Supplies",
            "Office Supplies",
            "Office Supplies",
            "Office Supplies",
            "Utilities",
            "Utilities",
            "Utilities",
            "Utilities",
            "Utilities",
        ],
    }
)


def build_classifier() -> Pipeline:
    """Create a TF-IDF and logistic-regression classification pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                ),
            ),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )


def classify_invoices(data: pd.DataFrame) -> pd.DataFrame:
    """Train on seed examples and add category predictions to invoice data."""
    if "Vendor" not in data.columns:
        raise ValueError("Input data must contain a cleaned 'Vendor' column")

    model = build_classifier()
    model.fit(SEED_EXAMPLES["text"], SEED_EXAMPLES["category"])

    classified = data.copy()
    vendor_text = classified["Vendor"].fillna("Unknown Vendor").astype(str)
    classified["Category"] = model.predict(vendor_text) # type: ignore
    return classified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify cleaned invoice vendors."
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
        default=Path("classified_invoices.csv"),
        help="Output CSV path (default: classified_invoices.csv)",
    )
    args = parser.parse_args()

    cleaned_data = clean_invoices(pd.read_csv(args.input))
    classified_data = classify_invoices(cleaned_data)
    classified_data.to_csv(args.output, index=False, date_format="%Y-%m-%d")
    print(f"Classified {len(classified_data)} rows into {args.output}")
    print(classified_data["Category"].value_counts().to_string())


if __name__ == "__main__":
    main()
