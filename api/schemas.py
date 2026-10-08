"""
api/schemas.py
--------------
Pydantic schemas for request/response validation. FastAPI uses these to
validate incoming JSON and auto-generate OpenAPI docs at /docs.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Transaction(BaseModel):
    """A single raw credit-card transaction (before scaling)."""
    Time: float = Field(..., description="Seconds since first transaction in dataset")
    V1:  float; V2:  float; V3:  float; V4:  float; V5:  float
    V6:  float; V7:  float; V8:  float; V9:  float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., description="Raw transaction amount")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0, "V1": -1.359807, "V2": -0.072781,
                "V3": 2.536347, "V4": 1.378155, "V5": -0.338321,
                "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
                "V9": 0.363787, "V10": 0.090794, "V11": -0.551600,
                "V12": -0.617801, "V13": -0.991390, "V14": -0.311169,
                "V15": 1.468177, "V16": -0.470401, "V17": 0.207971,
                "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
                "V21": -0.018307, "V22": 0.277838, "V23": -0.110474,
                "V24": 0.066928, "V25": 0.128539, "V26": -0.189115,
                "V27": 0.133558, "V28": -0.021053, "Amount": 149.62
            }
        }


class PredictionResponse(BaseModel):
    """Response from /predict."""
    ensemble_score: float
    is_fraud: bool
    threshold: float
    xgb_score: float
    rf_score: float
    ae_raw_score: float


class Reason(BaseModel):
    """A single reason behind a fraud prediction."""
    feature: str
    value: float
    shap_value: float
    direction: str  # "increased" or "decreased"


class ExplainResponse(BaseModel):
    """Response from /explain."""
    ensemble_score: float
    is_fraud: bool
    base_value: float
    top_reasons: List[Reason]


class MetricsResponse(BaseModel):
    """Response from /metrics."""
    best_model: str
    selection_metric: str
    models: Dict[str, dict]