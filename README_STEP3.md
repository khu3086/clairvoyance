# Step 3 — FastAPI Serving Layer

REST API that loads the trained PyTorch model and serves predictions / recommendations. This is the layer the React dashboard (Step 5) and Node.js gateway (later) will talk to.

## What this adds

- `features/inference.py` — feature pipeline variant for inference (no targets, keeps the latest bars)
- `api/inference.py` — `ModelService` that loads the artifact, runs predictions, caches results
- `api/schemas.py` — Pydantic models for request/response bodies
- `api/main.py` — FastAPI application with health, predict, recommend, model-info endpoints

## Install

The dependencies came along as transitive deps of MLflow, but let's lock them in:

```powershell
pip install -r requirements.txt
```

## Run

Make sure your venv is active and you've trained a model (you have — `artifacts/lstm_baseline_*` exists).

```powershell
uvicorn api.main:app --reload --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [...]
2026-... [INFO] api.main: Clairvoyance API starting up...
2026-... [INFO] api.inference: Loaded model from artifacts\lstm_baseline_...
2026-... [INFO] api.main: Application startup complete.
```

Open http://127.0.0.1:8000/docs for the auto-generated Swagger UI. Every endpoint is callable from there with one click.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness + whether a model is loaded |
| GET | `/symbols` | List of all tracked securities with sector / industry |
| GET | `/predict/{symbol}` | Outperformance probability for one symbol |
| GET | `/recommend?n=5` | Top-N predicted outperformers across the universe |
| GET | `/models/current` | Metadata of the loaded model (features, metrics, hyperparams) |
| POST | `/admin/reload` | Reload the latest artifact from disk (after retraining) |
| POST | `/admin/cache/clear` | Force re-computation on the next request |

## Try it from the terminal

In a second PowerShell with the venv active:

```powershell
# Health
curl http://127.0.0.1:8000/health

# Predict one
curl http://127.0.0.1:8000/predict/AAPL

# Top 5 recommendations
curl "http://127.0.0.1:8000/recommend?n=5"

# What model is loaded
curl http://127.0.0.1:8000/models/current
```

(On older PowerShell, `curl` is aliased to `Invoke-WebRequest`. Use `curl.exe` explicitly to get real curl, or just open the URLs in your browser.)

## Caching

`/recommend` and `/predict/{symbol}` share an in-memory cache with a 5-minute TTL. Predictions only meaningfully change after market close + new ingest, so this is safe and saves CPU cycles. Force a refresh with `POST /admin/cache/clear`.

## When you retrain

After running `python -m training.train` again, the API still has the *old* model in memory. Either restart uvicorn, or hit `POST /admin/reload`:

```powershell
curl.exe -X POST http://127.0.0.1:8000/admin/reload
```

## CORS

Currently set to `allow_origins=["*"]` for local development convenience. **Change this** before exposing the API publicly — set it to your dashboard origin, e.g. `["http://localhost:5173"]`.

## What's next

Once the API is running and `/recommend` returns sensible JSON, we move to:

- **Step 4** — Node.js gateway (auth, user portfolios, MCP server)
- **Step 5** — React dashboard consuming `/recommend`

Or, alternatively: skip Node.js for now and have the React dashboard call FastAPI directly. The gateway is mostly about adding auth and user-data persistence, which you may not need for a portfolio demo.
