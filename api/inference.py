"""Model service: loads the trained model artifact and runs predictions."""
import logging
import threading
from pathlib import Path

import numpy as np
import torch
from cachetools import TTLCache

from features.inference import build_inference_features, latest_sequences
from models.registry import latest_artifact, load_model
from training.dataset import FeatureScaler

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, cache_ttl_seconds: int = 300):
        self.model: torch.nn.Module | None = None
        self.metadata: dict | None = None
        self.scaler: FeatureScaler | None = None
        self.artifact_dir: Path | None = None
        self._predictions_cache: TTLCache = TTLCache(maxsize=1, ttl=cache_ttl_seconds)
        self._cache_lock = threading.Lock()

    def load(self, artifact_dir: Path | None = None) -> None:
        if artifact_dir is None:
            artifact_dir = latest_artifact("lstm_baseline")
        self.artifact_dir = artifact_dir
        self.model, self.metadata = load_model(artifact_dir)
        if self.metadata.get("scaler"):
            self.scaler = FeatureScaler.from_dict(self.metadata["scaler"])
        logger.info("Loaded model from %s", artifact_dir)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def feature_cols(self) -> list[str]:
        return self.metadata["feature_cols"]

    @property
    def seq_len(self) -> int:
        return self.metadata["seq_len"]

    def predict_universe(self) -> list[dict]:
        with self._cache_lock:
            cached = self._predictions_cache.get("all")
            if cached is not None:
                return cached

        if not self.is_ready:
            raise RuntimeError("Model not loaded.")

        df = build_inference_features()
        seqs = latest_sequences(df, self.seq_len, self.feature_cols, self.scaler)
        if not seqs:
            return []

        symbols = list(seqs.keys())
        batch = np.stack([seqs[s]["sequence"] for s in symbols])

        with torch.no_grad():
            logits = self.model(torch.from_numpy(batch))
            probs = torch.sigmoid(logits).numpy()

        results = [
            {"symbol": s, "prob": float(p), "as_of": seqs[s]["as_of"].isoformat()}
            for s, p in zip(symbols, probs)
        ]
        results.sort(key=lambda r: r["prob"], reverse=True)

        with self._cache_lock:
            self._predictions_cache["all"] = results

        logger.info("Computed predictions for %d symbols", len(results))
        return results

    def predict_symbol(self, symbol: str) -> dict | None:
        symbol = symbol.upper()
        for p in self.predict_universe():
            if p["symbol"] == symbol:
                return p
        return None

    def invalidate_cache(self) -> None:
        with self._cache_lock:
            self._predictions_cache.clear()
        logger.info("Predictions cache cleared")