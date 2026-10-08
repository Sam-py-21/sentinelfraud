"""
src/model_loader.py
-------------------
Loads all trained models and scalers. Used by the FastAPI layer.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from sklearn.preprocessing import RobustScaler

from .features import FEATURE_ORDER


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")



def _load_scalers() -> Tuple[RobustScaler, RobustScaler]:
    """
    Load pre-fit RobustScalers from disk. No CSV needed at runtime —
    critical for Docker/Render deployment where creditcard.csv isn't shipped.
    """
    scaler_amount = joblib.load(os.path.join(MODELS_DIR, "scaler_amount.pkl"))
    scaler_hour = joblib.load(os.path.join(MODELS_DIR, "scaler_hour.pkl"))
    return scaler_amount, scaler_hour

def load_all_models() -> dict:
    """Load every trained artifact needed for inference."""
    device = torch.device("cpu")

    scaler_amount, scaler_hour = _load_scalers()

    models = {
        "xgb": joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl")),
        "rf":  joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl")),
        "scaler_amount": scaler_amount,
        "scaler_hour":   scaler_hour,
        "shap_explainer": joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl")),
    }

    ae = Autoencoder(input_dim=30).to(device)
    ae.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "autoencoder.pt"),
        map_location=device
    ))
    ae.eval()
    models["autoencoder"] = ae
    models["device"] = device

    with open(os.path.join(MODELS_DIR, "ensemble_config.json")) as f:
        models["ensemble_config"] = json.load(f)
    with open(os.path.join(MODELS_DIR, "shap_meta.json")) as f:
        models["shap_meta"] = json.load(f)
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        models["metrics"] = json.load(f)

    return models


def predict_ensemble(models: dict, X_prepared: pd.DataFrame) -> dict:
    """Compute ensemble fraud probability for a batch of prepared features."""
    cfg = models["ensemble_config"]
    w = cfg["weights"]
    device = models["device"]

    xgb_score = models["xgb"].predict_proba(X_prepared)[:, 1]
    rf_score = models["rf"].predict_proba(X_prepared)[:, 1]

    X_t = torch.tensor(X_prepared.values.astype("float32")).to(device)
    with torch.no_grad():
        recon = models["autoencoder"](X_t)
        ae_raw = torch.mean((X_t - recon) ** 2, dim=1).cpu().numpy()

    ae_norm = ae_raw / (ae_raw.max() + 1e-10) if ae_raw.max() > 0 else ae_raw

    ensemble = (w["xgb"] * xgb_score) + (w["rf"] * rf_score) + (w["ae"] * ae_norm)

    return {
        "ensemble_score": ensemble,
        "xgb_score": xgb_score,
        "rf_score": rf_score,
        "ae_raw_score": ae_raw,
        "ae_norm_score": ae_norm,
    }