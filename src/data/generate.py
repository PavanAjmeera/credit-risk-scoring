"""
Generate a realistic synthetic loan dataset.

Why synthetic? Real bank data is private. Banks use synthetic data for demos too.
We control every column the problem statement asks for, and anyone who runs
this script gets exactly the same data (fixed random seed = reproducible).

We also add realistic MESS on purpose (missing values, duplicates, typos,
impossible ages) so the cleaning step has real work to do.

Run from the project root:  python -m src.data.generate
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/loans_raw.csv")
N_APPLICANTS = 10_000
SEED = 42  # same seed -> same data every time


def monthly_emi(principal, annual_rate, months):
    """Standard loan EMI formula: P * r * (1+r)^n / ((1+r)^n - 1)."""
    r = annual_rate / 12
    factor = (1 + r) ** months
    return principal * r * factor / (factor - 1)


def make_clean_applicants(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create applicants whose default risk follows sensible real-world patterns."""
    age = rng.integers(21, 66, n)
    gender = rng.choice(["Male", "Female"], n, p=[0.6, 0.4])
    employment_type = rng.choice(
        ["Salaried", "Self-Employed", "Business"], n, p=[0.6, 0.25, 0.15]
    )
    years_employed = np.clip(rng.normal(age - 23, 4, n), 0, None).round(1)

    # Income in rupees per month: skewed (a few people earn a lot), like real life
    income_monthly = np.round(rng.lognormal(mean=10.8, sigma=0.5, size=n), -2)

    loan_purpose = rng.choice(
        ["Home", "Car", "Personal", "Education", "Business"], n,
        p=[0.3, 0.2, 0.3, 0.1, 0.1],
    )
    tenure_months = rng.choice([12, 24, 36, 48, 60], n)
    loan_amount = np.round(income_monthly * rng.uniform(2, 20, n), -3)
    emi = np.round(monthly_emi(loan_amount, 0.12, tenure_months), 0)

    existing_loans = rng.poisson(1.0, n)
    late_payments_6m = rng.poisson(0.6, n)
    bureau_score = np.clip(rng.normal(710, 70, n), 300, 900).round(0)
    enquiries_6m = rng.poisson(1.5, n)
    credit_utilization = np.clip(rng.beta(2, 3, n), 0, 1).round(3)

    # The hidden "true" risk. Gender is deliberately NOT used here,
    # so a fair model should give men and women similar scores.
    debt_to_income = (emi / income_monthly).clip(0, 3)
    logit = (
        -2.8
        + 3.0 * (debt_to_income - 0.35)
        + 0.55 * late_payments_6m
        - 0.012 * (bureau_score - 710)
        + 0.15 * enquiries_6m
        + 1.8 * (credit_utilization - 0.4)
        + 0.25 * existing_loans
        - 0.03 * years_employed
        + np.where(employment_type == "Self-Employed", 0.3, 0.0)
    )
    probability = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, probability)

    return pd.DataFrame({
        "applicant_id": [f"APP{100000 + i}" for i in range(n)],
        "age": age,
        "gender": gender,
        "employment_type": employment_type,
        "years_employed": years_employed,
        "income_monthly": income_monthly,
        "loan_purpose": loan_purpose,
        "loan_amount": loan_amount,
        "tenure_months": tenure_months,
        "emi": emi,
        "existing_loans": existing_loans,
        "late_payments_6m": late_payments_6m,
        "bureau_score": bureau_score,
        "enquiries_6m": enquiries_6m,
        "credit_utilization": credit_utilization,
        "default": default,
    })


def add_realistic_mess(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject the kinds of problems real bank data has."""
    df = df.copy()
    n = len(df)

    def pick(fraction):
        return rng.choice(n, int(n * fraction), replace=False)

    # 1. Missing values
    df.loc[pick(0.03), "income_monthly"] = np.nan
    df.loc[pick(0.02), "bureau_score"] = np.nan
    df.loc[pick(0.01), "employment_type"] = np.nan

    # 2. Inconsistent text (typed differently by different branch staff)
    variants = {"Salaried": ["salaried", " SALARIED", "Salaried "],
                "Self-Employed": ["self employed", "Self employed", "SELF-EMPLOYED"]}
    for clean_value, typos in variants.items():
        rows = df.index[df["employment_type"] == clean_value]
        chosen = rng.choice(rows, int(len(rows) * 0.05), replace=False)
        df.loc[chosen, "employment_type"] = rng.choice(typos, len(chosen))

    # 3. Income stored as text with commas in some rows, e.g. "45,000"
    df["income_monthly"] = df["income_monthly"].astype(object)
    for i in pick(0.02):
        if pd.notna(df.at[i, "income_monthly"]):
            df.at[i, "income_monthly"] = f"{int(df.at[i, 'income_monthly']):,}"

    # 4. Impossible values (data-entry errors)
    df.loc[pick(0.003), "age"] = rng.choice([0, 150, 250, -5], int(n * 0.003))
    df.loc[pick(0.003), "credit_utilization"] = 1.8  # utilization above 100%

    # 5. Duplicate rows (same application loaded twice)
    duplicates = df.iloc[pick(0.01)]
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle so the duplicates are not all at the bottom
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    clean = make_clean_applicants(N_APPLICANTS, rng)
    raw = add_realistic_mess(clean, rng)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(raw):,} rows to {RAW_PATH}")
    print(f"Default rate: {clean['default'].mean():.1%}")


if __name__ == "__main__":
    main()
