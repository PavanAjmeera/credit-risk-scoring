import numpy as np
import pandas as pd

from src.data.clean import clean


def make_raw():
    """Three tiny applicants with the exact problems the real data has."""
    return pd.DataFrame({
        "applicant_id": ["A1", "A2", "A2", "A3"],          # A2 is duplicated
        "age": [30, 250, 250, 45],                           # 250 is impossible
        "gender": ["Male", "Female", "Female", "Male"],
        "employment_type": [" SALARIED", "self employed", "self employed", np.nan],
        "years_employed": [5.0, 3.0, 3.0, 20.0],
        "income_monthly": ["45,000", np.nan, np.nan, 90000],  # text and missing
        "loan_purpose": ["Car", "Home", "Home", "Personal"],
        "loan_amount": [300000, 900000, 900000, 200000],
        "tenure_months": [36, 60, 60, 24],
        "emi": [9964, 20020, 20020, 9415],
        "existing_loans": [0, 1, 1, 2],
        "late_payments_6m": [0, 2, 2, 1],
        "bureau_score": [750, np.nan, np.nan, 680],
        "enquiries_6m": [1, 3, 3, 0],
        "credit_utilization": [0.3, 1.8, 1.8, 0.5],          # 1.8 is impossible
        "default": [0, 1, 1, 0],
    })


def test_duplicates_removed():
    df, _ = clean(make_raw())
    assert df["applicant_id"].tolist() == ["A1", "A2", "A3"]


def test_formats_fixed():
    df, _ = clean(make_raw())
    assert df.loc[0, "income_monthly"] == 45000
    assert df["employment_type"].tolist() == ["Salaried", "Self-Employed", "Unknown"]


def test_impossible_values_replaced():
    df, report = clean(make_raw())
    assert df["age"].between(18, 75).all()
    assert df["credit_utilization"].between(0, 1).all()
    assert report["impossible_values_fixed"]["age"] == 1


def test_no_missing_values_left():
    df, _ = clean(make_raw())
    assert df.notna().all().all()
