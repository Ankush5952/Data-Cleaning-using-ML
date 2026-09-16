# Automated Vendor Invoice & Expense Classifier

A local, modular data-operations pipeline that turns messy vendor transaction data into a standardized, categorized invoice report.

This project demonstrates practical data engineering and machine-learning techniques using Pandas, NumPy, regular expressions, and Scikit-Learn:

- Synthetic messy invoice generation
- Missing-value and date handling
- Vendor-name normalization with Regex
- TF-IDF text classification
- Duplicate transaction detection
- Category-based amount anomaly detection
- A configurable command-line workflow

## Project workflow

```sh
input data/raw_invoices.csv
        |
        v
    pipeline/cleaner.py       Normalize dates, amounts, vendors, and missing values
        |
        v
  pipeline/classifier.py    Predict an expense category from the cleaned vendor text
        |
        v
     pipeline/rules.py         Detect duplicates and unusually large category amounts(outliers)
        |
        v
     main.py          Run the full pipeline and export the final CSV

```

## Technology

- Python 3.14+
- Pandas
- NumPy
- Regular expressions (`re`)
- Scikit-Learn
   - `TfidfVectorizer`
   - `LogisticRegression`

## Repository structure

| File | Purpose |
| --- | --- |
| `pipeline/generate_data.py` | Generates synthetic, intentionally messy invoice data |
| `pipeline/cleaner.py` | Standardizes columns, dates, amounts, vendor names, and missing values |
| `pipeline/classifier.py` | Trains the seed text classifier and predicts expense categories |
| `pipeline/rules.py` | Applies duplicate and anomaly detection rules |
| `main.py` | Provides the end-to-end CLI |
| `setup/requirements.txt` | Lists required Python packages |
| `setup/setup_environment.py` | Checks and optionally installs dependencies |
| `pipeline/prompt.md` | Original project requirements and learning blueprint |

Generated CSV files are ignored by Git through `.gitignore`.

## Setup

Clone the repository and open its directory:

```powershell
git clone https://github.com/Ankush5952/Data-Cleaning-using-ML.git
cd "Data Cleaning using ML"

```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```

Install the dependencies:

```powershell
python -m pip install -r .\setup\requirements.txt
```

Alternatively, use the interactive dependency checker:

```powershell
python .\setup\setup_environment.py
```

The checker verifies whether NumPy, Pandas, and Scikit-Learn can be imported. If any are missing, it lists the packages and asks for confirmation before running `pip install -r .\setup\requirements.txt`. Choosing `N` or pressing Enter leaves the environment unchanged.

## Quick start

Run the complete pipeline with its defaults:

```powershell
python .\main.py

```

The default behavior is:

1. Look for `input data/raw_invoices.csv`.
2. If it does not exist, generate a fresh randomized test dataset.
3. Clean and standardize the data.
4. Predict one of four expense categories.
5. Detect duplicates and anomalies internally.
6. Write the result to `output data/cleaned_invoices.csv`.
7. Print a processing summary.

The missing-input test feature creates the parent directories automatically. Existing input files are loaded without being overwritten. Each newly generated missing input uses a fresh random state, so repeated runs can produce different test data.

## Command-line options

```text
python .\main.py [OPTIONS]

```

| Option | Default | Description |
| --- | --- | --- |
| `-i`, `--input` | `input data/raw_invoices.csv` | Input CSV path |
| `-o`, `--output` | `output data/cleaned_invoices.csv` | Output CSV path |
| `--th` | `7` | Duplicate matching window in days |
| `--sd` | `3.0` | Standard-deviation threshold for anomalies |
| `--show_metrics`, `--smc` | Disabled | Include `Is_Duplicate` and `Is_Anomaly` in the output |

Example with custom paths:

```powershell
python .\main.py `
  --input ".\input data\incoming.csv" `
  --output ".\output data\cleaned_invoices.csv"

```

Example with review metrics visible:

```powershell
python .\main.py `
  --input ".\input data\raw_invoices.csv" `
  --output ".\output data\invoice_review.csv" `
  --th 14 `
  --sd 2.5 `
  --show_metrics

```

## Output behavior

By default, the exported file contains the cleaned business data and the predicted category:

```text
Invoice_ID
Transaction_Date
Vendor
Amount
Description
Payment_Method
Category

```

The duplicate and anomaly rules still run internally, but their machine-generated review columns are hidden by default. Use `--show_metrics` or `--smc` to include:

```text
Is_Duplicate
Is_Anomaly

```

The terminal summary also reports duplicate and anomaly counts only when metrics are enabled.

## Data generation

To generate a raw dataset directly:

```powershell
python .\pipeline\generate_data.py

```

This creates approximately 200 base rows plus three duplicate records. The generated data includes:

- Mixed and invalid date formats
- Missing vendor, amount, and payment-method values
- Payment-provider prefixes and transaction IDs in vendor strings
- Duplicate transactions with ordinary invoice IDs
- Large amount outliers

The generator is deterministic when called directly with its default seed:

```powershell
python .\pipeline\generate_data.py --seed 42

```

The automatic missing-input path in `main.py` intentionally uses fresh randomness for test runs.

## Cleaning rules

`pipeline/cleaner.py`:

- Strips whitespace from column names
- Parses mixed date formats
- Drops rows with invalid or missing transaction dates
- Converts amounts to numeric values
- Replaces missing amounts with the median amount
- Normalizes vendors such as `POS-LYFT-7C52FA` to `Lyft`
- Replaces missing text fields with explicit values such as `Unknown`

## Classification approach

The classifier uses a small labeled seed dataset containing representative vendor descriptions for:

- `SaaS/Cloud`
- `Travel/Transport`
- `Office Supplies`
- `Utilities`

`TfidfVectorizer` converts text into weighted word and two-word phrase features. `LogisticRegression` then predicts the category for each cleaned vendor.

The model deliberately excludes `Invoice_ID` and the generated `Description` field. This avoids learning from identifiers or from a description that already contains the target category.

## Detection rules

### Duplicates

A later transaction is marked as a duplicate when it has:

- The same cleaned vendor
- The same amount
- A transaction date within the configured `--th` window

### Anomalies

For each predicted category, the pipeline calculates:

```text
category mean + (--sd × category standard deviation)

```

Amounts above that threshold receive `Is_Anomaly = True`.

## Example terminal output

```text
Invoice processing summary
--------------------------
Input rows:       203
Output rows:      191
Removed rows:     12
Duplicates:       3
Anomalies:        6

Categories:
Category
Office Supplies     51
SaaS/Cloud          46
Travel/Transport    48
Utilities           46

```

Counts vary when `main.py` creates a new randomized input file.

## Design notes and limitations

- The classifier is an educational seed-model workflow, not a production-trained model.
- The seed examples should be expanded with verified historical labels before business use.
- Invalid dates are dropped because date-based duplicate detection requires a trustworthy date.
- Missing amounts are imputed with the dataset median; a production pipeline may instead quarantine those records for review.
- Duplicate and anomaly flags are rule-based and should be validated against domain-specific accounting policies.
- No external service or cloud storage is required; processing occurs locally.

## Learning outcomes

This repository is designed to demonstrate:

- DataFrame loading and transformation
- Null handling and type conversion
- Regular-expression-based text normalization
- Feature extraction with TF-IDF
- Supervised text classification
- Boolean indexing and grouped statistics
- Modular Python design
- CLI argument parsing and configurable data workflows
