"""Daily incremental ingest.

Run this on a schedule (cron, systemd timer, GitHub Actions, etc.) after
market close. It fetches only bars newer than what's already in Mongo,
so re-running is cheap and safe.

Usage:
    python -m ingestion.ingest
"""
import logging
from datetime import datetime, timedelta, timezone

from pymongo import DESCENDING, UpdateOne

from .config import LOG_LEVEL, UNIVERSE
from .db import ensure_indexes, get_prices
from .fetchers import df_to_price_docs, fetch_history

logger = logging.getLogger(__name__)


def latest_date(symbol: str) -> datetime | None:
    doc = get_prices().find_one({"symbol": symbol}, sort=[("date", DESCENDING)])
    return doc["date"] if doc else None


def ingest_one(symbol: str) -> int:
    """Pull bars since the most recent stored date (or last 30 days if empty)."""
    last = latest_date(symbol)
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    start = (last + timedelta(days=1)) if last else today - timedelta(days=30)

    if start.date() > today.date():
        logger.info("Up to date: %s", symbol)
        return 0

    df = fetch_history(symbol, start)
    docs = df_to_price_docs(df)
    if not docs:
        return 0

    ops = [
        UpdateOne(
            {"symbol": d["symbol"], "date": d["date"]},
            {"$set": d},
            upsert=True,
        )
        for d in docs
    ]
    result = get_prices().bulk_write(ops, ordered=False)
    n = result.upserted_count + result.modified_count
    logger.info("Ingested %s: %d bars", symbol, n)
    return n


def main() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_indexes()

    total = 0
    failed: list[str] = []
    for symbol in UNIVERSE:
        try:
            total += ingest_one(symbol)
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to ingest %s: %s", symbol, e)
            failed.append(symbol)

    logger.info("Daily ingest complete: %d bars, %d failed", total, len(failed))
    if failed:
        logger.warning("Failed symbols: %s", ", ".join(failed))


if __name__ == "__main__":
    main()
