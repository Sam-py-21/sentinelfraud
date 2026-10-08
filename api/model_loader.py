"""
api/model_loader.py
-------------------
Loads models and exposes the API's inference functions.

NOTE: We import from `src/` so preprocessing logic lives in ONE place
and is shared between training and serving — no drift.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path so `src` package is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features import prepare_features, enforce_feature_order, FEATURE_ORDER
from src.model_loader import load_all_models, predict_ensemble


# ---------- Module-level singleton ----------
# Load models ONCE at import time. FastAPI reuses them for every request.

_MODELS = None
_ENSEMBLE_THRESHOLD = None


def get_models() -> dict:
    """Lazy-load models on first call; return cached afterwards."""
    global _MODELS, _ENSEMBLE_THRESHOLD
    if _MODELS is None:
        _MODELS = load_all_models()
        _ENSEMBLE_THRESHOLD = _MODELS["ensemble_config"]["ensemble_threshold"]
    return _MODELS


def get_threshold() -> float:
    get_models()
    return _ENSEMBLE_THRESHOLD


# ---------- Prediction ----------

def predict_transaction(raw_dict: dict) -> dict:
    """
    Accept a raw transaction dict (with Time, V1-V28, Amount),
    preprocess, run ensemble, return scores + fraud flag.
    """
    models = get_models()
    threshold = get_threshold()

    raw_df = pd.DataFrame([raw_dict])
    X_prep = prepare_features(raw_df, models["scaler_amount"], models["scaler_hour"])

    result = predict_ensemble(models, X_prep)

    ensemble_score = float(result["ensemble_score"][0])
    return {
        "ensemble_score": ensemble_score,
        "is_fraud": ensemble_score >= threshold,
        "threshold": float(threshold),
        "xgb_score": float(result["xgb_score"][0]),
        "rf_score": float(result["rf_score"][0]),
        "ae_raw_score": float(result["ae_raw_score"][0]),
    }


# ---------- Explain ----------

def explain_transaction(raw_dict: dict, top_n: int = 5) -> dict:
    """
    Run prediction, then compute SHAP values for this transaction and
    return the top N contributing features in human-readable form.
    """
    models = get_models()
    threshold = get_threshold()

    raw_df = pd.DataFrame([raw_dict])
    X_prep = prepare_features(raw_df, models["scaler_amount"], models["scaler_hour"])

    # Predict
    pred = predict_transaction(raw_dict)

    # SHAP values for this single row
    explainer = models["shap_explainer"]
    shap_vals = explainer.shap_values(X_prep)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]  # some SHAP versions return a list

    shap_row = shap_vals[0]
    values   = X_prep.iloc[0].values
    feature_names = X_prep.columns.tolist()

    # Rank by |SHAP|
    order = np.argsort(np.abs(shap_row))[::-1][:top_n]

    reasons = []
    for idx in order:
        reasons.append({
            "feature": feature_names[idx],
            "value": float(values[idx]),
            "shap_value": float(shap_row[idx]),
            "direction": "increased" if shap_row[idx] > 0 else "decreased",
        })

    return {
        "ensemble_score": pred["ensemble_score"],
        "is_fraud": pred["is_fraud"],
        "base_value": float(explainer.expected_value)
            if not isinstance(explainer.expected_value, np.ndarray)
            else float(explainer.expected_value[0]),
        "top_reasons": reasons,
    }


# ---------- Metrics ----------

def get_metrics() -> dict:
    """Return metrics.json contents for the /metrics endpoint."""
    models = get_models()
    m = models["metrics"]
    return {
        "best_model": m.get("best_model", "unknown"),
        "selection_metric": m.get("selection_metric", "unknown"),
        "models": {k: v for k, v in m.items() if isinstance(v, dict)},
    }