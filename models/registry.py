"""Save and load model artifacts with metadata.

Each saved model is a directory containing:
- model.pt        : the state dict
- metadata.json   : feature schema, scaler stats, training metrics, hyperparams

The scaler stats are saved here (not as a separate file) so that loading is a
single read and there's no ambiguity about which scaler goes with which model.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .lstm import LSTMClassifier

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def save_model(
    model: torch.nn.Module,
    name: str,
    feature_cols: list[str],
    seq_len: int,
    metrics: dict[str, float],
    hyperparams: dict[str, Any],
    scaler_dict: dict | None = None,
) -> Path:
    """Save model + sidecar metadata. Returns the artifact directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    artifact_dir = ARTIFACTS_DIR / f"{name}_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), artifact_dir / "model.pt")

    metadata = {
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": feature_cols,
        "seq_len": seq_len,
        "metrics": metrics,
        "hyperparams": hyperparams,
        "scaler": scaler_dict,
    }
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Cross-platform "latest" pointer (no symlink perms needed on Windows)
    (ARTIFACTS_DIR / f"{name}_latest.txt").write_text(artifact_dir.name)

    logger.info("Saved model to %s", artifact_dir)
    return artifact_dir


def latest_artifact(name: str = "lstm_baseline") -> Path:
    """Return the most recent artifact directory for `name`."""
    marker = ARTIFACTS_DIR / f"{name}_latest.txt"
    if marker.exists():
        path = ARTIFACTS_DIR / marker.read_text().strip()
        if path.exists():
            return path
    candidates = sorted(p for p in ARTIFACTS_DIR.glob(f"{name}_*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No artifacts found for '{name}'. Train a model first.")
    return candidates[-1]


def load_model(artifact_dir: Path) -> tuple[LSTMClassifier, dict]:
    """Load model + metadata from an artifact directory."""
    with open(artifact_dir / "metadata.json") as f:
        metadata = json.load(f)

    hp = metadata["hyperparams"]
    model = LSTMClassifier(
        input_size=len(metadata["feature_cols"]),
        hidden_size=hp["hidden_size"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
    )
    model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu"))
    model.eval()
    return model, metadata
