"""Data source adapters.

Today: yfinance only. Tomorrow: add Alpha Vantage / Polygon by exposing
the same `fetch_history` / `fetch_metadata` interface and switching at
the call site.
"""
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def fetch_history(symbol: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV bars for a single symbol over [start, end]."""
    end = end or _utcnow()
    ticker = yf.Ticker(symbol)
    df = ticker.history(
        start=start.strftime("%Y-%m-%d"),
        # yfinance treats `end` as exclusive — bump by a day to include today
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,  # we want both Close and Adj Close
    )
    if df.empty:
        logger.warning("No history returned for %s", symbol)
        return df

    df = df.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df["symbol"] = symbol
    return df


def fetch_metadata(symbol: str) -> dict:
    """Fetch security-level metadata. yfinance's .info can be slow/flaky;
    we fall back to symbol-only metadata if it fails."""
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as e:  # noqa: BLE001 - yfinance raises various things
        logger.warning("Metadata fetch failed for %s: %s", symbol, e)
        info = {}

    return {
        "symbol": symbol,
        "name": info.get("longName") or info.get("shortName") or symbol,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "exchange": info.get("exchange"),
        "currency": info.get("currency", "USD"),
    }


def df_to_price_docs(df: pd.DataFrame) -> list[dict]:
    """Convert a yfinance dataframe to MongoDB-ready documents.

    Each doc represents a single (symbol, date) bar. Dates are normalized
    to naive midnight UTC so the unique index works deterministically.
    """
    if df.empty:
        return []

    now = _utcnow()
    docs: list[dict] = []
    for row in df.itertuples(index=False):
        # `date` may be a pandas Timestamp (possibly tz-aware) or datetime
        ts = pd.Timestamp(row.date)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        date = ts.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)

        adj_close = float(getattr(row, "adj_close", row.close))
        volume = int(row.volume) if not pd.isna(row.volume) else 0

        docs.append({
            "symbol": row.symbol,
            "date": date,
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "adj_close": adj_close,
            "volume": volume,
            "ingested_at": now,
        })
    return docs
