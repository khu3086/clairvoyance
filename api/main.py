"""FastAPI application — Clairvoyance recommendation API."""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ingestion.db import get_prices, get_securities

from .inference import ModelService
from .schemas import (
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictionDetail,
    RecommendationsResponse,
    SymbolInfo,
)

logger = logging.getLogger(__name__)

service = ModelService(cache_ttl_seconds=300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Clairvoyance API starting up...")
    try:
        service.load()
    except FileNotFoundError as e:
        logger.error("No model artifact found: %s", e)
        logger.error("Train one with: python -m training.train")
    except Exception:
        logger.exception("Model load failed")
    yield
    logger.info("Clairvoyance API shutting down")


app = FastAPI(
    title="Clairvoyance API",
    description="ML-driven investment recommendation API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _securities_map() -> dict[str, dict]:
    return {
        s["symbol"]: s
        for s in get_securities().find(
            {}, {"_id": 0, "symbol": 1, "name": 1, "sector": 1, "industry": 1}
        )
    }


def _to_prediction(p: dict, sec_map: dict[str, dict]) -> Prediction:
    sec = sec_map.get(p["symbol"], {})
    return Prediction(
        symbol=p["symbol"],
        name=sec.get("name"),
        sector=sec.get("sector"),
        prob=p["prob"],
        as_of=p["as_of"],
    )


# ---------- endpoints ----------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok" if service.is_ready else "degraded",
        model_loaded=service.is_ready,
        model_artifact=service.artifact_dir.name if service.artifact_dir else None,
    )


@app.get("/symbols", response_model=list[SymbolInfo], tags=["data"])
def list_symbols():
    rows = list(get_securities().find({}, {"_id": 0}).sort("symbol", 1))
    return [SymbolInfo(**r) for r in rows]


@app.get("/predict/{symbol}", response_model=PredictionDetail, tags=["predict"])
def predict_one(symbol: str):
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    symbol = symbol.upper()
    pred = service.predict_symbol(symbol)
    if pred is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction for '{symbol}'. Either it's not in the universe, "
                   f"or there isn't enough history yet (needs ≥{service.seq_len} bars).",
        )

    sec = get_securities().find_one({"symbol": symbol}, {"_id": 0}) or {}
    return PredictionDetail(
        symbol=pred["symbol"],
        name=sec.get("name"),
        sector=sec.get("sector"),
        prob=pred["prob"],
        as_of=pred["as_of"],
        model_version=service.artifact_dir.name if service.artifact_dir else None,
        horizon_days=5,
        threshold=0.005,
    )


@app.get("/recommend", response_model=RecommendationsResponse, tags=["predict"])
def recommend(
    n: int = Query(5, ge=1, le=50, description="How many top picks to return"),
):
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    all_preds = service.predict_universe()
    if not all_preds:
        raise HTTPException(status_code=503, detail="No predictions available")

    sec_map = _securities_map()
    top = [_to_prediction(p, sec_map) for p in all_preds[:n]]

    return RecommendationsResponse(
        n=n,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model_version=service.artifact_dir.name if service.artifact_dir else "unknown",
        recommendations=top,
    )


@app.get("/history/{symbol}", tags=["data"])
def history(
    symbol: str,
    days: int = Query(252, ge=10, le=2520, description="Trading days to return"),
):
    """Recent OHLCV bars for a symbol. Powers the price chart on the dashboard.

    Default 252 ≈ 1 trading year. Returns oldest-first so the chart can render directly.
    """
    symbol = symbol.upper()

    # Pull a buffer to account for weekends / holidays, then slice
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=int(days * 1.6))
    cursor = get_prices().find(
        {"symbol": symbol, "date": {"$gte": cutoff}},
        {"_id": 0, "ingested_at": 0},
    ).sort("date", 1)

    bars = []
    for doc in cursor:
        bars.append({
            "date": doc["date"].strftime("%Y-%m-%d"),
            "open": doc["open"],
            "high": doc["high"],
            "low": doc["low"],
            "close": doc["close"],
            "adj_close": doc["adj_close"],
            "volume": doc["volume"],
        })

    if not bars:
        raise HTTPException(status_code=404, detail=f"No price history for '{symbol}'")

    return {"symbol": symbol, "bars": bars[-days:], "count": min(len(bars), days)}


@app.get("/models/current", response_model=ModelInfo, tags=["meta"])
def current_model():
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="No model loaded")
    return ModelInfo(
        artifact=service.artifact_dir.name,
        saved_at=service.metadata["saved_at"],
        feature_cols=service.metadata["feature_cols"],
        seq_len=service.metadata["seq_len"],
        metrics=service.metadata.get("metrics", {}),
        hyperparams=service.metadata.get("hyperparams", {}),
    )


@app.post("/admin/reload", tags=["admin"])
def reload_model():
    try:
        service.load()
        service.invalidate_cache()
        return {"status": "reloaded", "artifact": service.artifact_dir.name}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/admin/cache/clear", tags=["admin"])
def clear_cache():
    service.invalidate_cache()
    return {"status": "cleared"}
