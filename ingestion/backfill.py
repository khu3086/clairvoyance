"""One-shot historical backfill.

Usage:
    python -m ingestion.backfill              # backfill the full UNIVERSE
    python -m ingestion.backfill AAPL MSFT    # backfill specific symbols
"""
import logging
import sys
from datetime import datetime, timedelta, timezone

from pymongo import UpdateOne

from .config import BACKFILL_YEARS, LOG_LEVEL, UNIVERSE
from .db import ensure_indexes, get_prices, get_securities
from .fetchers import df_to_price_docs, fetch_history, fetch_metadata

logger = logging.getLogger(__name__)


def backfill_one(symbol: str, years: int = BACKFILL_YEARS) -> int:
    """Backfill `years` of history for a single symbol. Idempotent — safe to re-run."""
    start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=365 * years)

    # 1. Upsert metadata
    meta = fetch_metadata(symbol)
    get_securities().update_one(
        {"symbol": symbol},
        {
            "$set": meta,
            "$setOnInsert": {"added_at": datetime.now(timezone.utc).replace(tzinfo=None)},
        },
        upsert=True,
    )

    # 2. Fetch and upsert prices
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
    logger.info("Backfilled %s: %d bars (%d new, %d updated)",
                symbol, n, result.upserted_count, result.modified_count)
    return n


def main(symbols: list[str] | None = None) -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_indexes()

    targets = symbols or UNIVERSE
    logger.info("Starting backfill: %d symbols, %d years", len(targets), BACKFILL_YEARS)

    total = 0
    failed: list[str] = []
    for symbol in targets:
        try:
            total += backfill_one(symbol)
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to backfill %s: %s", symbol, e)
            failed.append(symbol)

    logger.info("Backfill complete: %d total bars, %d symbols, %d failed",
                total, len(targets) - len(failed), len(failed))
    if failed:
        logger.warning("Failed symbols: %s", ", ".join(failed))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else None)
