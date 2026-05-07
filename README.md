# Clairvoyance — Step 1: Data Foundation

Ingestion pipeline that pulls daily OHLCV bars from yfinance into MongoDB Atlas. This is the foundation for everything downstream (features, models, recommendations).

## Setup

### 1. MongoDB Atlas

1. Create a free account at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
2. Create an **M0 (free)** cluster — pick the region closest to you
3. **Database Access** → add a database user with read/write
4. **Network Access** → allow your IP (or `0.0.0.0/0` while developing — lock down later)
5. **Connect** → **Drivers** → **Python** → copy the connection string

### 2. Local environment

```bash
cd clairvoyance
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Open .env and paste your Atlas connection string into MONGODB_URI
# Replace <password> in the URI with your actual DB user password
```

### 3. Verify connection

```bash
python -c "from ingestion.db import get_client; get_client(); print('Connected ✓')"
```

## Run

### Backfill historical data (one-time, ~5 minutes)

```bash
python -m ingestion.backfill
```

This pulls 5 years of daily bars for all 21 tickers in `UNIVERSE`. Idempotent — safe to re-run.

To backfill specific symbols only:

```bash
python -m ingestion.backfill TSLA NFLX
```

### Daily incremental ingest

```bash
python -m ingestion.ingest
```

Pulls only bars newer than what's already stored. Schedule this after market close (e.g. cron at 5pm ET on weekdays).

### Tests

```bash
pytest tests/ -v
```

## What you should see in Atlas

After backfill, in your Atlas cluster:

- `clairvoyance.securities` — 21 documents, one per ticker, with sector/industry metadata
- `clairvoyance.prices` — ~26,000 documents (21 tickers × ~1260 trading days)

A sanity check from the mongo shell or Compass:

```javascript
db.prices.countDocuments({ symbol: "AAPL" })       // ~1260
db.prices.find({ symbol: "AAPL" }).sort({ date: -1 }).limit(1)
```

## Project layout

```
clairvoyance/
├── ingestion/
│   ├── config.py       # env vars + ticker universe
│   ├── db.py           # Mongo client + indexes
│   ├── fetchers.py     # yfinance adapters
│   ├── backfill.py     # historical one-shot
│   └── ingest.py       # daily incremental
├── tests/
│   └── test_ingest.py
├── .env.example
├── requirements.txt
└── README.md
```

## Schema

**`securities`**
```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "sector": "Technology",
  "industry": "Consumer Electronics",
  "exchange": "NMS",
  "currency": "USD",
  "added_at": "2026-05-06T..."
}
```

**`prices`** (one doc per symbol-date, unique on `(symbol, date)`)
```json
{
  "symbol": "AAPL",
  "date": "2024-01-15T00:00:00",
  "open": 185.92,
  "high": 186.40,
  "low": 183.62,
  "close": 185.59,
  "adj_close": 185.59,
  "volume": 47000000,
  "ingested_at": "2026-05-06T..."
}
```

## Next step

Once your data is in Atlas, we move to **feature engineering + the PyTorch predictive model**: rolling returns, technical indicators, walk-forward train/val splits, and a baseline LSTM that predicts 5-day return direction.
