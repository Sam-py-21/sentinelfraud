"""
api/main.py
-----------
FastAPI app exposing SentinelFraud's hybrid fraud detection system.

Run locally:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    Transaction,
    PredictionResponse,
    ExplainResponse,
    MetricsResponse,
)
from .model_loader import (
    predict_transaction,
    explain_transaction,
    get_metrics,
)


app = FastAPI(
    title="SentinelFraud API",
    description=(
        "Hybrid fraud detection: XGBoost + Random Forest + PyTorch Autoencoder, "
        "with SHAP explainability."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check — returns 200 if the service is up."""
    return {"status": "ok", "service": "SentinelFraud"}


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: Transaction):
    """Score a single transaction. Returns ensemble fraud probability."""
    try:
        return predict_transaction(tx.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/explain", response_model=ExplainResponse)
def explain(tx: Transaction):
    """Explain a prediction using SHAP — top reasons for the flag."""
    try:
        return explain_transaction(tx.model_dump(), top_n=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {e}")


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    """Return performance metrics of every trained model."""
    return get_metrics()