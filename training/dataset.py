"""PyTorch Dataset wrapping the feature pipeline.

Adds a FeatureScaler that fits on the training set ONLY and is applied to
val/test/inference. Stats are persisted with the model artifact so the
serving layer can apply identical transformations to live features.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from features.pipeline import FEATURE_COLS

SEQ_LEN = 60


class FeatureScaler:
    """Mean/std scaler. Fit on train only — never on val/test (leakage)."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "FeatureScaler":
        self.mean = df[feature_cols].mean().to_numpy(dtype=np.float32)
        self.std = df[feature_cols].std().to_numpy(dtype=np.float32)
        # Avoid divide-by-zero for any near-constant feature
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler not fit. Call .fit() or .from_dict() first.")
        return ((X - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        return cls(
            mean=np.array(d["mean"], dtype=np.float32),
            std=np.array(d["std"], dtype=np.float32),
        )


class SequenceDataset(Dataset):
    """Sliding-window sequences from a long-format feature DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = SEQ_LEN,
        feature_cols: list[str] | None = None,
        scaler: FeatureScaler | None = None,
    ):
        self.seq_len = seq_len
        self.feature_cols = feature_cols or FEATURE_COLS

        self._sequences: list[np.ndarray] = []
        self._targets: list[float] = []
        self._meta: list[tuple[str, pd.Timestamp]] = []

        for symbol, g in df.groupby("symbol", sort=False):
            g = g.sort_values("date").reset_index(drop=True)
            features = g[self.feature_cols].to_numpy(dtype=np.float32)
            if scaler is not None:
                features = scaler.transform(features)
            targets = g["target"].to_numpy(dtype=np.float32)
            dates = g["date"].to_numpy()

            for i in range(seq_len - 1, len(g)):
                self._sequences.append(features[i - seq_len + 1 : i + 1])
                self._targets.append(targets[i])
                self._meta.append((symbol, pd.Timestamp(dates[i])))

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._sequences[idx]),
            torch.tensor(self._targets[idx], dtype=torch.float32),
        )

    def metadata(self, idx: int) -> tuple[str, pd.Timestamp]:
        return self._meta[idx]
