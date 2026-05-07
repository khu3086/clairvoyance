"""Centralized configuration. Reads from .env via python-dotenv."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- MongoDB ---
MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "clairvoyance")

# --- Backfill window ---
BACKFILL_YEARS = int(os.getenv("BACKFILL_YEARS", "5"))

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(exist_ok=True, parents=True)

# --- Initial ticker universe ---
# Mix of sectors plus benchmarks. Expand later by adding to this list —
# the ingestion pipeline is universe-agnostic.
UNIVERSE: list[str] = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "LLY",
    # Energy
    "XOM", "CVX",
    # Consumer
    "WMT", "COST", "HD",
    # Benchmarks
    "SPY", "QQQ",
]
