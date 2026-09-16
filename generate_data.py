"""Generate a synthetic, intentionally messy invoice dataset."""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


MERCHANTS = {
    "SaaS/Cloud": [
        "AWS",
        "Microsoft Azure",
        "Slack",
        "Dropbox",
        "GitHub",
    ],
    "Travel/Transport": [
        "Uber",
        "Lyft",
        "Delta Airlines",
        "Airbnb",
        "Shell",
    ],
    "Office Supplies": [
        "Staples",
        "Office Depot",
        "Amazon Marketplace",
        "Walmart",
        "Target",
    ],
    "Utilities": [
        "Comcast",
        "Verizon",
        "Duke Energy",
        "City Water",
        "AT&T",
    ],
}


def build_transaction_string(merchant: str, row_number: int) -> str:
    """Add payment-provider formatting and an identifier to a merchant name."""
    transaction_id = f"{random.getrandbits(24):06X}"
    formats = [
        f"{merchant.upper()}_{transaction_id}",
        f"{merchant.upper()} *{transaction_id}",
        f"{merchant.title()} #{row_number:04d}",
        f" POS-{merchant.upper()}-{transaction_id}",
    ]
    return random.choice(formats)


def make_date(row_number: int) -> str | float:
    """Return a valid date most of the time, with occasional corrupted values."""
    transaction_date = date(2025, 1, 1) + timedelta(days=random.randint(0, 364))
    corrupted_dates = [
        transaction_date.strftime("%m/%d/%Y"),
        transaction_date.strftime("%Y-%m-%d"),
        transaction_date.strftime("%d-%b-%Y"),
        "not-a-date",
        "2025/99/99",
        np.nan,
    ]

    # Every tenth row is intentionally noisy; the rest uses a valid ISO date.
    if row_number % 10 == 0:
        return random.choice(corrupted_dates)
    return transaction_date.isoformat()


def generate_rows(row_count: int = 200) -> pd.DataFrame:
    """Build rows containing missing values, duplicates, and amount outliers."""
    rows: list[dict[str, object]] = []
    category_names = list(MERCHANTS)

    for row_number in range(1, row_count + 1):
        category = random.choice(category_names)
        merchant = random.choice(MERCHANTS[category])
        amount = round(float(np.random.lognormal(mean=3.4, sigma=0.65)), 2)

        # A few unusually large values will later be detected as anomalies.
        if row_number % 47 == 0:
            amount = round(amount * 25, 2)

        rows.append(
            {
                "Invoice_ID": f"INV-{row_number:05d}",
                "Transaction_Date": make_date(row_number),
                "Vendor": build_transaction_string(merchant, row_number),
                "Amount": amount,
                "Description": f"{category} purchase",
                "Payment_Method": random.choice(
                    ["Corporate Card", "ACH", "Wire", "Corporate Card", None]
                ),
            }
        )

    data = pd.DataFrame(rows)

    # Missing vendor and amount values exercise different cleaning strategies.
    data.loc[17, "Vendor"] = np.nan
    data.loc[83, "Amount"] = np.nan

    # Duplicate complete transactions, including their original invoice IDs.
    duplicate_rows = data.iloc[[5, 42, 101]].copy()
    duplicate_rows["Invoice_ID"] = [
        f"INV-{row_count + 1:05d}",
        f"INV-{row_count + 2:05d}",
        f"INV-{row_count + 3:05d}",
    ]
    data = pd.concat([data, duplicate_rows], ignore_index=True)

    return data.sample(frac=1, random_state=42).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic messy invoice CSV file."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("raw_invoices.csv"),
        help="Output CSV path (default: raw_invoices.csv)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=200,
        help="Number of base rows before adding duplicates (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable output (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = generate_rows(args.rows)
    data.to_csv(args.output, index=False)
    print(f"Generated {len(data)} rows in {args.output}")


if __name__ == "__main__":
    main()
