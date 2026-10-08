"""
src/features.py
---------------
Central place for feature engineering. Both training notebooks and the
FastAPI layer import from here so preprocessing is EXACTLY the same
in training and inference.
"""

from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

FEATURE_ORDER: List[str] = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount_scaled", "Hour_scaled",
]

RAW_REQUIRED_COLUMNS: List[str] = (
    [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
)


def add_hour_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Time (seconds) into Hour-of-day (0-23)."""
    df = df.copy()
    df["Hour"] = (df["Time"] / 3600) % 24
    df = df.drop(columns=["Time"])
    return df


def scale_amount_and_hour(
    df: pd.DataFrame,
    scaler_amount: RobustScaler,
    scaler_hour: RobustScaler
) -> pd.DataFrame:
    """Apply two separate RobustScalers (matching notebook 02)."""
    df = df.copy()
    df["Amount_scaled"] = scaler_amount.transform(df[["Amount"]])
    df["Hour_scaled"]   = scaler_hour.transform(df[["Hour"]])
    df = df.drop(columns=["Amount", "Hour"])
    return df


def enforce_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to match training-time FEATURE_ORDER exactly."""
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return df[FEATURE_ORDER]


def prepare_features(
    raw_df: pd.DataFrame,
    scaler_amount: RobustScaler,
    scaler_hour: RobustScaler
) -> pd.DataFrame:
    """Full preprocessing pipeline: raw -> model-ready."""
    df = add_hour_feature(raw_df)
    df = scale_amount_and_hour(df, scaler_amount, scaler_hour)
    df = enforce_feature_order(df)
    return df


def validate_raw_input(raw_df: pd.DataFrame) -> None:
    """Raise clear error if input is missing required raw columns."""
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(
            f"Missing raw columns in input: {missing}. "
            f"Expected: {RAW_REQUIRED_COLUMNS}"
        )