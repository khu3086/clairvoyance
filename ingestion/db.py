"""MongoDB connection and schema setup.

We use a singleton client. Indexes are created idempotently on every run
so you never have to think about migrations during early development.
"""
import logging
from typing import Optional

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from .config import MONGODB_DB, MONGODB_URI

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    """Return a process-wide MongoClient. Pings the server on first call."""
    global _client
    if _client is None:
        if not MONGODB_URI:
            raise RuntimeError(
                "MONGODB_URI is not set. Copy .env.example to .env and fill it in."
            )
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10_000)
        _client.admin.command("ping")  # fail fast if creds/network are wrong
        logger.info("Connected to MongoDB")
    return _client


def get_db() -> Database:
    return get_client()[MONGODB_DB]


def get_securities() -> Collection:
    return get_db()["securities"]


def get_prices() -> Collection:
    return get_db()["prices"]


def ensure_indexes() -> None:
    """Create the indexes the rest of the pipeline assumes exist.

    - prices.(symbol, date) is unique: prevents duplicate bars on re-runs
    - prices.date: speeds up cross-sectional queries ("all closes on 2024-06-01")
    - securities.symbol unique: one row per ticker
    """
    prices = get_prices()
    prices.create_index(
        [("symbol", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="symbol_date_uniq",
    )
    prices.create_index([("date", ASCENDING)], name="date_idx")

    securities = get_securities()
    securities.create_index(
        [("symbol", ASCENDING)], unique=True, name="symbol_uniq"
    )

    logger.info("Indexes ensured")
