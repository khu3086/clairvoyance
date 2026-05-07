"""Pydantic schemas for API request and response bodies.

Keeping these separate from the FastAPI app so they can be reused by tests
and (eventually) generated into TypeScript types for the React dashboard.
"""
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_artifact: Optional[str] = None


class SymbolInfo(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


class Prediction(BaseModel):
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    prob: float = Field(..., ge=0.0, le=1.0,
                        description="Probability the symbol outperforms SPY by ≥0.5% over 5 trading days")
    as_of: str  # ISO date of the most recent bar used for prediction


class PredictionDetail(Prediction):
    """Single-symbol prediction with extra context."""
    model_version: Optional[str] = None
    horizon_days: int = 5
    threshold: float = 0.005


class RecommendationsResponse(BaseModel):
    n: int
    generated_at: str
    model_version: str
    recommendations: list[Prediction]


class ModelInfo(BaseModel):
    artifact: str
    saved_at: str
    feature_cols: list[str]
    seq_len: int
    metrics: dict
    hyperparams: dict
