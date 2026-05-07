"""Smoke tests for the ingestion pipeline.

Run with:
    pytest tests/

The DB-touching test will be skipped if MONGODB_URI isn't set.
"""
import os
from datetime import datetime, timedelta, timezone

import pytest

from ingestion.fetchers import df_to_price_docs, fetch_history


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def test_fetch_history_returns_data():
    df = fetch_history("AAPL", _utcnow() - timedelta(days=10))
    assert not df.empty
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)


def test_df_to_docs_shape():
    df = fetch_history("AAPL", _utcnow() - timedelta(days=10))
    docs = df_to_price_docs(df)
    assert len(docs) > 0

    d = docs[0]
    assert d["symbol"] == "AAPL"
    assert isinstance(d["close"], float)
    assert isinstance(d["volume"], int)
    assert d["date"].hour == 0  # normalized to midnight


@pytest.mark.skipif(not os.getenv("MONGODB_URI"), reason="MONGODB_URI not set")
def test_db_connection_and_indexes():
    from ingestion.db import ensure_indexes, get_db

    db = get_db()
    ensure_indexes()

    indexes = list(db["prices"].list_indexes())
    names = {idx["name"] for idx in indexes}
    assert "symbol_date_uniq" in names
