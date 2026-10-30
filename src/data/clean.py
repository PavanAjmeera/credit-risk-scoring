"""
Validate and clean the raw loan data.

Steps (in this order, and the order matters):
  1. Fix formats      - text typos, income stored as "45,000"
  2. Remove duplicates - same application loaded twice
  3. Fix impossible values - age 250, utilization 180%  -> treated as missing
  4. Fill missing values   - median for numbers, "Unknown" for text
  5. Cap extreme outliers  - very high incomes clipped at the 99th percentile
  6. Validate             - final checks; the script stops if anything is wrong

Output: data/processed/loans_clean.csv  (the contract other members use)
        data/processed/quality_report.json (what we found and fixed)

Run from the project root:  python -m src.data.clean
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/loans_raw.csv")
CLEAN_PATH = Path("data/processed/loans_clean.csv")
REPORT_PATH = Path("data/processed/quality_report.json")

NUMERIC_COLUMNS = [
    "age", "years_employed", "income_monthly", "loan_amount", "tenure_months",
    "emi", "existing_loans", "late_payments_6m", "bureau_score",
    "enquiries_6m", "credit_utilization",
]
TEXT_COLUMNS = ["gender", "employment_type", "loan_purpose"]

# Business rules: the valid range for each column
VALID_RANGES = {
    "age": (18, 75),
    "bureau_score": (300, 900),
    "credit_utilization": (0, 1),
    "income_monthly": (1, None),
}

EMPLOYMENT_MAP = {
    "salaried": "Salaried",
    "self employed": "Self-Employed",
    "self-employed": "Self-Employed",
    "business": "Business",
}


def profile(df: pd.DataFrame) -> dict:
    """Quick data-quality snapshot, used before and after cleaning."""
    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
    }


def fix_formats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # "45,000" -> 45000 ; anything unreadable becomes NaN instead of crashing
    df["income_monthly"] = pd.to_numeric(
        df["income_monthly"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    # " SALARIED", "self employed" -> "Salaried", "Self-Employed"
    cleaned = df["employment_type"].str.strip().str.lower()
    df["employment_type"] = cleaned.map(EMPLOYMENT_MAP)
    for col in ["gender", "loan_purpose"]:
        df[col] = df[col].str.strip().str.title()
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # One row per applicant: keep the first copy
    return df.drop_duplicates(subset="applicant_id", keep="first").reset_index(drop=True)


def flag_impossible_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Values outside business rules are data-entry errors -> set to missing."""
    df = df.copy()
    counts = {}
    for col, (low, high) in VALID_RANGES.items():
        bad = pd.Series(False, index=df.index)
        if low is not None:
            bad |= df[col] < low
        if high is not None:
            bad |= df[col] > high
        counts[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    return df, counts


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        # Median, not mean: income is skewed, so the mean is pulled up by rich applicants
        df[col] = df[col].fillna(df[col].median())
    for col in TEXT_COLUMNS:
        df[col] = df[col].fillna("Unknown")
    return df


def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise income: very high earners are real, but extreme values distort models."""
    df = df.copy()
    upper = df["income_monthly"].quantile(0.99)
    df["income_monthly"] = df["income_monthly"].clip(upper=upper)
    return df


def validate(df: pd.DataFrame) -> None:
    """Final gate: stop the pipeline loudly if the clean data breaks any rule."""
    assert df["applicant_id"].is_unique, "Duplicate applicant IDs found"
    assert df[NUMERIC_COLUMNS + TEXT_COLUMNS].notna().all().all(), "Missing values remain"
    assert df["age"].between(18, 75).all(), "Age out of range"
    assert df["bureau_score"].between(300, 900).all(), "Bureau score out of range"
    assert df["credit_utilization"].between(0, 1).all(), "Utilization out of range"
    assert set(df["default"].unique()) <= {0, 1}, "Target must be 0 or 1"
    allowed = {"Salaried", "Self-Employed", "Business", "Unknown"}
    assert set(df["employment_type"]) <= allowed, "Unexpected employment type"


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"before": profile(raw)}
    df = fix_formats(raw)
    df = remove_duplicates(df)
    df, impossible = flag_impossible_values(df)
    report["impossible_values_fixed"] = impossible
    df = fill_missing(df)
    df = cap_outliers(df)
    for col in ["age", "tenure_months", "existing_loans", "late_payments_6m", "enquiries_6m"]:
        df[col] = df[col].astype(int)
    validate(df)
    report["after"] = profile(df)
    report["default_rate"] = round(float(df["default"].mean()), 4)
    return df, report


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    df, report = clean(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Rows: {report['before']['rows']:,} raw -> {report['after']['rows']:,} clean")
    print(f"Impossible values fixed: {report['impossible_values_fixed']}")
    print(f"Default rate: {report['default_rate']:.1%}")
    print(f"Saved {CLEAN_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
