"""Time-series train/val/test splits with a purge gap.

Why a gap matters: a sample at time t has a target computed from the window
[t+1, t+horizon]. If t sits at the end of train and val starts at t+1, the
training labels overlap with validation features — that's leakage. We leave
`gap_days` between splits to prevent this.

For v1 we use a single chronological split. The richer setup (expanding
walk-forward CV producing multiple folds) is a swap-in for splits.py later.
"""
import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    gap_days: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Single chronological train/val/test split with a purge gap of `gap_days`
    unique trading days between train→val and val→test."""
    df = df.sort_values("date").copy()
    dates = df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(dates)

    train_end_idx = int(n * train_frac)
    val_end_idx = int(n * (train_frac + val_frac))

    if train_end_idx + gap_days >= val_end_idx or val_end_idx + gap_days >= n:
        raise ValueError("Not enough data for the requested split fractions and gap.")

    train_end = dates.iloc[train_end_idx]
    val_start = dates.iloc[train_end_idx + gap_days]
    val_end = dates.iloc[val_end_idx]
    test_start = dates.iloc[val_end_idx + gap_days]

    train = df[df["date"] <= train_end]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
    test = df[df["date"] >= test_start]

    return train, val, test
