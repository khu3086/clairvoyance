"""Train the baseline LSTM with MLflow tracking.

Usage:
    python -m training.train

Then view results:
    mlflow ui
    # browser: http://127.0.0.1:5000
"""
import logging

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from features.pipeline import FEATURE_COLS, build_dataset
from models.lstm import LSTMClassifier
from models.registry import save_model
from .dataset import SEQ_LEN, FeatureScaler, SequenceDataset
from .splits import chronological_split

logger = logging.getLogger(__name__)

EXPERIMENT = "clairvoyance-baseline"

# Hyperparameters — tuned for CPU + small dataset, with feature scaling
HPARAMS: dict = {
    "hidden_size": 64,        # was 32 — more capacity once features are scaled
    "num_layers": 2,          # was 1
    "dropout": 0.3,           # was 0.2 — stronger regularization for the bigger net
    "learning_rate": 5e-4,    # was 1e-3 — smaller for the bigger model
    "batch_size": 128,
    "epochs": 60,             # was 30 — let it run longer
    "weight_decay": 1e-4,     # was 1e-5
    "patience": 10,           # was 5
    "grad_clip": 1.0,
}


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    auc = float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.5
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "auc": auc,
        "loss": float(
            nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(logits), torch.from_numpy(targets)
            )
        ),
        "pos_rate": float(targets.mean()),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    device = "cpu"
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Build dataset
    logger.info("Building dataset from MongoDB...")
    df = build_dataset()
    train_df, val_df, test_df = chronological_split(df)
    logger.info(
        "Splits — train: %d rows (%s → %s), val: %d, test: %d",
        len(train_df), train_df["date"].min().date(), train_df["date"].max().date(),
        len(val_df), len(test_df),
    )

    # 2. Fit scaler on train ONLY — no val/test leakage
    scaler = FeatureScaler().fit(train_df, FEATURE_COLS)
    logger.info(
        "Scaler fit on train. Means: %s",
        np.round(scaler.mean, 4).tolist(),
    )

    train_ds = SequenceDataset(train_df, seq_len=SEQ_LEN, scaler=scaler)
    val_ds = SequenceDataset(val_df, seq_len=SEQ_LEN, scaler=scaler)
    test_ds = SequenceDataset(test_df, seq_len=SEQ_LEN, scaler=scaler)
    logger.info(
        "Sequences — train: %d, val: %d, test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    train_loader = DataLoader(train_ds, batch_size=HPARAMS["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=HPARAMS["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=HPARAMS["batch_size"], shuffle=False, num_workers=0)

    # 3. Model
    model = LSTMClassifier(
        input_size=len(FEATURE_COLS),
        hidden_size=HPARAMS["hidden_size"],
        num_layers=HPARAMS["num_layers"],
        dropout=HPARAMS["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters", n_params)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HPARAMS["learning_rate"],
        weight_decay=HPARAMS["weight_decay"],
    )
    criterion = nn.BCEWithLogitsLoss()

    # 4. Train under MLflow
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params({
            **HPARAMS,
            "n_params": n_params,
            "n_features": len(FEATURE_COLS),
            "seq_len": SEQ_LEN,
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "test_rows": len(test_ds),
        })

        best_val_auc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        best_state = None

        for epoch in range(1, HPARAMS["epochs"] + 1):
            model.train()
            running_loss, n_batches = 0.0, 0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=HPARAMS["grad_clip"])
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            train_loss = running_loss / max(n_batches, 1)

            val_metrics = evaluate(model, val_loader, device)

            mlflow.log_metrics(
                {"train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}},
                step=epoch,
            )
            logger.info(
                "Epoch %02d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_auc=%.4f",
                epoch, train_loss, val_metrics["loss"], val_metrics["accuracy"], val_metrics["auc"],
            )

            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= HPARAMS["patience"]:
                    logger.info("Early stopping at epoch %d (best epoch was %d)", epoch, best_epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # 5. Test
        test_metrics = evaluate(model, test_loader, device)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        logger.info(
            "Test | loss=%.4f | acc=%.4f | auc=%.4f | pos_rate=%.4f",
            test_metrics["loss"], test_metrics["accuracy"],
            test_metrics["auc"], test_metrics["pos_rate"],
        )

        # 6. Persist artifact (now includes scaler stats)
        artifact_dir = save_model(
            model,
            name="lstm_baseline",
            feature_cols=FEATURE_COLS,
            seq_len=SEQ_LEN,
            metrics={
                "best_val_auc": best_val_auc,
                "best_epoch": best_epoch,
                "test_auc": test_metrics["auc"],
                "test_accuracy": test_metrics["accuracy"],
            },
            hyperparams=HPARAMS,
            scaler_dict=scaler.to_dict(),
        )
        mlflow.log_artifacts(str(artifact_dir), artifact_path="model")
        logger.info("MLflow run id: %s", run.info.run_id)

    logger.info("Training complete. Run `mlflow ui` and open http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
