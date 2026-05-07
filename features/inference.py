"""Feature pipeline for inference.

Differs from `pipeline.build_dataset()` in two ways:
1. No targets — we predict on bars where the future is unknown.
2. Doesn't drop the last few rows per symbol (those are the rows we WANT).

Reuses the same per-symbol and market-feature functions, so train/serve
parity is guaranteed by construction.
"""
import logging
from typing import Iterable

import numpy as np
import pandas as pd

from training.dataset import FeatureScaler

from .pipeline import (
    EXCLUDE_FROM_TRAINING,
    FEATURE_COLS,
    _add_market_features,
    _compute_per_symbol_features,
    load_prices,
)

logger = logging.getLogger(__name__)


def build_inference_features(symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """Like build_dataset(), but no targets and we keep the latest bars."""
    df = load_prices(symbols)
    if df.empty:
        raise RuntimeError("No price data available. Did you run `python -m ingestion.ingest`?")

    pieces = [_compute_per_symbol_features(g) for _, g in df.groupby("symbol", sort=False)]
    df = pd.concat(pieces, ignore_index=True)
    df = _add_market_features(df)

    keep_cols = ["symbol", "date"] + FEATURE_COLS
    df = df[keep_cols].dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df


def latest_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: list[str],
    scaler: FeatureScaler | None = None,
) -> dict[str, dict]:
    """For each non-benchmark symbol, return the most recent seq_len-day window."""
    out: dict[str, dict] = {}
    for symbol, g in df.groupby("symbol", sort=False):
        if symbol in EXCLUDE_FROM_TRAINING:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < seq_len:
            logger.warning("Not enough history for %s (have %d, need %d)", symbol, len(g), seq_len)
            continue
        features = g[feature_cols].to_numpy(dtype=np.float32)
        if scaler is not None:
            features = scaler.transform(features)
        out[symbol] = {
            "sequence": features[-seq_len:],
            "as_of": g["date"].iloc[-1],
        }
    return out
