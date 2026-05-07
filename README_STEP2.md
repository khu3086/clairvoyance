# Step 2 — Features + PyTorch Model + MLflow

This adds the ML core on top of the data pipeline from Step 1.

## What this does

1. **Feature pipeline** — reads bars from MongoDB, computes returns, moving-average ratios, RSI, MACD, volatility, and SPY-relative features. Same module is used at training and inference time (no train/serve skew).
2. **Target labeling** — for each (symbol, day), a binary label: *did this stock outperform SPY over the next 5 trading days?*
3. **Walk-forward split** — train / val / test split chronologically, with a 5-day purge gap between splits to prevent label leakage.
4. **PyTorch LSTM classifier** — 60-day input window, ~10k parameters, CPU-friendly.
5. **Training loop** — Adam + BCE loss + early stopping on val AUC, with full MLflow tracking.
6. **Evaluation** — test-set AUC plus a simple top-N hit-rate backtest.

## Install (additions over Step 1)

From your project root, with the venv activated:

```powershell
pip install -r requirements.txt
```

This adds `torch`, `mlflow`, and `scikit-learn`. PyTorch CPU wheel is ~200MB so the install takes a minute. If it's slow or fails, install the CPU wheel explicitly (smaller and more reliable):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mlflow scikit-learn
```

## Train

```powershell
python -m training.train
```

You'll see something like:

```
Building dataset from MongoDB...
Dataset built: ~24000 rows, 20 symbols, 10 features, target horizon 5d
Splits — train: ~17000 rows ... val: ~3500 ... test: ~3500
Sequences — train: ~16000 ... val: ~2500 ... test: ~2500
Model: 10145 parameters
Epoch 01 | train_loss=0.6912 | val_loss=0.6905 | val_acc=0.5240 | val_auc=0.5310
Epoch 02 | ...
...
Test | loss=0.6890 | acc=0.5380 | auc=0.5420 | pos_rate=0.4870
Saved model to artifacts/lstm_baseline_20260506T...
```

Training takes 3–8 minutes on CPU, depending on your machine. Models save to `artifacts/`.

### What numbers to expect

- **Val/test AUC of 0.52–0.58** is realistic for this universe and target. Markets are mostly efficient at this horizon — anything substantially above 0.5 is genuinely informative.
- AUC under 0.50 isn't a bug; it can happen on small datasets or specific regimes. Try a different random seed or larger universe.
- **Don't expect 0.7+ AUC.** If you see that, suspect a look-ahead bug.

## View MLflow runs

In a second terminal (keep the venv active):

```powershell
mlflow ui
```

Open http://127.0.0.1:5000. You'll see the `clairvoyance-baseline` experiment with one run per training invocation: hyperparameters, metric curves over epochs, and the saved model artifacts.

## Evaluate the latest model

```powershell
python -m training.evaluate
```

This loads the most recent artifact, runs predictions across the test set, and prints the top-N hit rate plus a sample of recent picks.

## File layout (Step 2 additions)

```
clairvoyance/
├── features/
│   ├── indicators.py       # SMA, EMA, RSI, MACD, vol, zscore
│   ├── targets.py          # forward-return labels
│   └── pipeline.py         # Mongo → features+targets DataFrame
├── models/
│   ├── lstm.py             # baseline classifier
│   └── registry.py         # save/load with metadata
├── training/
│   ├── splits.py           # chronological split with purge gap
│   ├── dataset.py          # PyTorch Dataset, sliding windows
│   ├── train.py            # main training loop + MLflow
│   └── evaluate.py         # backtest + diagnostic predictions
├── artifacts/              # saved models land here
└── mlruns/                 # MLflow tracking data (auto-created)
```

## Iteration ideas (after baseline works)

These are good directions for v2 — only chase them once the v1 loop is solid:

- **More features.** Add cross-sectional rank features (where does this stock rank in 1m momentum across the universe?), sector dummies, lagged volume.
- **Bigger universe.** Expand `UNIVERSE` in `ingestion/config.py` to 100–200 names. Re-run backfill, retrain.
- **Better target.** Multi-class (under-perform / market / out-perform) or regression on log-return spread.
- **Walk-forward CV.** Replace `chronological_split` with multiple expanding folds and average metrics.
- **Better model.** Try a small Transformer (works fine on CPU at this size).
- **Probability calibration.** Apply Platt scaling on the validation set so predicted probabilities are interpretable.

## Next step (after this works)

Step 3 is the **serving layer**: a FastAPI service that loads the artifact and exposes `/predict/{symbol}` and `/recommend/{user_id}` endpoints, plus the Node.js gateway that fronts it.
