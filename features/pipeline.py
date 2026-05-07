"""End-to-end feature pipeline: Mongo → DataFrame with features and targets.

The same pipeline runs at training time AND inference time. Any divergence
creates train/serve skew, which is the second-most common cause of ML
production failures (after look-ahead bias). Keep this module as the single
source of truth for what 'a feature' means.
"""
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from ingestion.db import get_prices
from .indicators import macd, realized_vol, rsi, sma, zscore
from .targets import relative_outperformance_label

logger = logging.getLogger(__name__)

BENCHMARK = "SPY"
TARGET_HORIZON = 5            # 5 trading days ≈ 1 week
TARGET_THRESHOLD = 0.005      # 50bps outperformance required to count as positive
EXCLUDE_FROM_TRAINING = {BENCHMARK}

FEATURE_COLS: list[str] = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "close_to_sma20",
    "close_to_sma50",
    "rsi_14",
    "macd_hist",
    "vol_20d",
    "vol_zscore_20",
    "rel_ret_spy_1d",
]


def load_prices(symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load prices from Mongo as a long-format DataFrame."""
    query = {"symbol": {"$in": list(symbols)}} if symbols else {}
    cursor = get_prices().find(query, {"_id": 0}).sort("date", 1)
    df = pd.DataFrame(list(cursor))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def _compute_per_symbol_features(group: pd.DataFrame) -> pd.DataFrame:
    """Compute per-symbol features. `group` is one symbol's bars sorted by date."""
    g = group.copy()
    close = g["adj_close"]

    # Returns (log)
    log_close = np.log(close)
    g["ret_1d"] = log_close.diff(1)
    g["ret_5d"] = log_close.diff(5)
    g["ret_20d"] = log_close.diff(20)

    # Moving-average ratios — bounded, scale-free
    g["close_to_sma20"] = close / sma(close, 20) - 1
    g["close_to_sma50"] = close / sma(close, 50) - 1

    # RSI scaled to [0, 1]
    g["rsi_14"] = rsi(close, 14) / 100.0

    # MACD histogram, normalized by price for cross-asset comparability
    macd_df = macd(close)
    g["macd_hist"] = macd_df["macd_hist"] / close

    # Volatility and its regime (z-score over 60d window)
    g["vol_20d"] = realized_vol(g["ret_1d"], 20)
    g["vol_zscore_20"] = zscore(g["vol_20d"], 60)

    return g


def _add_market_features(df: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.DataFrame:
    """Cross-sectional features that depend on the benchmark."""
    bench = (
        df[df["symbol"] == benchmark][["date", "ret_1d"]]
        .rename(columns={"ret_1d": "spy_ret_1d"})
    )
    df = df.merge(bench, on="date", how="left")
    df["rel_ret_spy_1d"] = df["ret_1d"] - df["spy_ret_1d"]
    return df


def _add_targets(
    df: pd.DataFrame,
    horizon: int = TARGET_HORIZON,
    benchmark: str = BENCHMARK,
    threshold: float = TARGET_THRESHOLD,
) -> pd.DataFrame:
    bench_close = (
        df[df["symbol"] == benchmark][["date", "adj_close"]]
        .rename(columns={"adj_close": "benchmark_close"})
    )
    df = df.merge(bench_close, on="date", how="left")

    pieces = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        g["target"] = relative_outperformance_label(
            g["adj_close"], g["benchmark_close"], horizon, threshold
        )
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)


def build_dataset(symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """Full pipeline: load → features → targets, ready for the Dataset class."""
    df = load_prices(symbols)
    if df.empty:
        raise RuntimeError("No price data found. Run `python -m ingestion.backfill` first.")

    # Per-symbol features. Manual loop instead of groupby.apply to avoid the
    # pandas 2.2+ FutureWarning about implicit grouping-column inclusion.
    pieces = [_compute_per_symbol_features(g) for _, g in df.groupby("symbol", sort=False)]
    df = pd.concat(pieces, ignore_index=True)

    # Cross-sectional features (need ret_1d to be computed first)
    df = _add_market_features(df)

    # Targets (need adj_close from both stock and benchmark)
    df = _add_targets(df)

    # Drop benchmark rows from training data
    df = df[~df["symbol"].isin(EXCLUDE_FROM_TRAINING)]

    keep_cols = ["symbol", "date"] + FEATURE_COLS + ["target"]
    df = df[keep_cols].dropna().reset_index(drop=True)

    pos_rate = df["target"].mean()
    logger.info(
        "Dataset built: %d rows, %d symbols, %d features, target horizon %dd, threshold %.3f, pos_rate=%.3f",
        len(df), df["symbol"].nunique(), len(FEATURE_COLS), TARGET_HORIZON, TARGET_THRESHOLD, pos_rate,
    )
    return df
