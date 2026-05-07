"""Evaluate a trained model with a simple long-only backtest."""
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from features.pipeline import build_dataset
from models.registry import latest_artifact, load_model
from .dataset import FeatureScaler, SequenceDataset
from .splits import chronological_split

logger = logging.getLogger(__name__)


def predict(model: torch.nn.Module, ds: SequenceDataset, batch_size: int = 256) -> pd.DataFrame:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    rows = []
    idx = 0
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            probs = torch.sigmoid(model(X)).numpy()
            for p, t in zip(probs, y.numpy()):
                symbol, date = ds.metadata(idx)
                rows.append({
                    "symbol": symbol,
                    "date": date,
                    "prob": float(p),
                    "target": float(t),
                })
                idx += 1
    return pd.DataFrame(rows)


def topn_backtest(predictions: pd.DataFrame, n: int = 3) -> dict:
    hits = trades = days = 0
    for _, g in predictions.groupby("date"):
        if len(g) < n:
            continue
        top = g.nlargest(n, "prob")
        hits += int(top["target"].sum())
        trades += len(top)
        days += 1
    return {
        "hit_rate": hits / trades if trades else 0.0,
        "trades": trades,
        "days": days,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    artifact_dir = latest_artifact("lstm_baseline")
    logger.info("Loading model from %s", artifact_dir)
    model, metadata = load_model(artifact_dir)

    scaler = None
    if metadata.get("scaler"):
        scaler = FeatureScaler.from_dict(metadata["scaler"])
        logger.info("Loaded feature scaler from artifact metadata.")

    df = build_dataset()
    _, _, test_df = chronological_split(df)
    test_ds = SequenceDataset(
        test_df,
        seq_len=metadata["seq_len"],
        feature_cols=metadata["feature_cols"],
        scaler=scaler,
    )

    preds = predict(model, test_ds)
    logger.info(
        "Generated %d predictions across %d days, %d symbols",
        len(preds), preds["date"].nunique(), preds["symbol"].nunique(),
    )

    for n in (1, 3, 5):
        bt = topn_backtest(preds, n=n)
        logger.info(
            "Top-%d hit rate: %.4f  (%d trades over %d days)",
            n, bt["hit_rate"], bt["trades"], bt["days"],
        )

    print("\nMost recent predictions (top 15 by prob on the last test day):")
    last_day = preds["date"].max()
    print(
        preds[preds["date"] == last_day]
        .sort_values("prob", ascending=False)
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
