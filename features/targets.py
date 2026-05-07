"""Target labeling.

The label at time t answers: 'will this stock outperform SPY by at least
`threshold` over the next H days?'

Why a threshold? Without one, the target treats +0.001% and +5% the same.
Most days the spread between a stock and SPY is tiny noise. A small positive
threshold (default 0.5% over 5 days) drops the noisy middle and gives the
model a cleaner signal to learn.
"""
import numpy as np
import pandas as pd


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """log(close[t+horizon] / close[t]). NaN for the last `horizon` rows."""
    return np.log(close.shift(-horizon) / close)


def relative_outperformance_label(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    horizon: int = 5,
    threshold: float = 0.005,
) -> pd.Series:
    """1 if stock beats benchmark by at least `threshold` over `horizon` days, else 0.
    NaN where the forward window is unknown."""
    stock_ret = forward_log_return(stock_close, horizon)
    bench_ret = forward_log_return(benchmark_close, horizon)
    diff = stock_ret - bench_ret
    label = (diff > threshold).astype(float)
    label[diff.isna()] = np.nan
    return label
