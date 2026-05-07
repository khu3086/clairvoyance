# Step 5 — React Dashboard

A polished single-page dashboard that consumes the FastAPI service. Dark, editorial, financial-research aesthetic — not generic SaaS gradient slop.

## Backend change

One small addition to FastAPI: a `/history/{symbol}` endpoint that returns OHLCV bars for the price chart. **Replace** your existing `api/main.py` with the new one I generated.

Then restart uvicorn — or hit the reload button if `--reload` is on.

## Dashboard setup

The dashboard lives in a separate sibling folder, not inside the Python project. You'll create it with Vite, then drop in the source files I generated.

### 1. Scaffold the Vite project

In a new terminal (does not need the venv):

```powershell
cd C:\Users\khush\OneDrive\Documents
npm create vite@latest dashboard -- --template react-ts
cd dashboard
```

When prompted by Vite, accept the default options.

### 2. Install dependencies

```powershell
# Core deps
npm install

# Routing + data fetching
npm install react-router-dom @tanstack/react-query

# Tailwind + shadcn deps
npm install -D tailwindcss postcss autoprefixer tailwindcss-animate
npm install class-variance-authority clsx tailwind-merge lucide-react

# Charts
npm install lightweight-charts

# Init Tailwind (this generates tailwind.config.js — overwrite it with the one I provide)
npx tailwindcss init -p
```

### 3. Drop in the generated files

The structure should end up looking like this:

```
dashboard/
├── components.json                   ← provided
├── index.html                        ← REPLACE the Vite default with mine
├── postcss.config.js                 ← provided
├── tailwind.config.js                ← REPLACE the auto-generated one
├── tsconfig.json                     ← REPLACE
├── vite.config.ts                    ← REPLACE
└── src/
    ├── App.tsx                       ← REPLACE
    ├── main.tsx                      ← REPLACE
    ├── index.css                     ← REPLACE
    ├── lib/
    │   ├── api.ts
    │   ├── types.ts
    │   └── utils.ts
    ├── components/
    │   ├── Header.tsx
    │   ├── ModelInfoCard.tsx
    │   ├── PriceChart.tsx
    │   ├── ProbabilityBar.tsx
    │   └── RecommendationCard.tsx
    └── pages/
        ├── Dashboard.tsx
        └── SymbolDetail.tsx
```

You can delete `src/App.css` (Vite default), and you don't need `src/assets/` for now.

### 4. Run it

Make sure your FastAPI service is running on port 8000 in another terminal. Then:

```powershell
npm run dev
```

Vite will open http://localhost:5173 automatically. You should see the Clairvoyance dashboard.

## What you'll see

**Home (`/`):** Editorial intro, top 12 recommendations as cards in a 3-column grid, sidebar with model info. Click any card to drill in.

**Symbol detail (`/symbol/AAPL`):** Big mono ticker, full company name in italic serif, model probability as a large display number with sentiment label, 12-month adjusted-close chart (TradingView lightweight-charts), 1m/3m/6m/1y returns strip.

**Header:** Brand mark in italic display serif, live model-status indicator that polls `/health` every 15 seconds.

## Design choices, briefly

- **Typography:** Instrument Serif (italic) for the brand and big editorial moments, Manrope for body, JetBrains Mono with tabular figures for every number. No Inter, no Roboto, no Arial.
- **Color:** A single warm amber accent (`#F5B544`) instead of the typical purple/blue gradient. Green/red are reserved strictly for performance directionality.
- **Surfaces:** No big rounded cards — corners are 4px sharp. Hairline 1px borders. Subtle film-grain overlay in `body::before` for atmosphere.
- **Motion:** Staggered fade-up on the recommendation grid (60ms delay per card). Subtle pulse on the live indicator. No bouncing or scattered micro-interactions.
- **Charts:** Lightweight-charts with the dark theme tokens injected at runtime so the chart matches the rest of the design without manual color sync.

## When the API isn't running

The dashboard surfaces this gracefully: header shows a red "Degraded" indicator, and the recommendations area shows a helpful error explaining that uvicorn needs to be running.

## Customizing

- **Change the universe size shown on the home page:** `src/pages/Dashboard.tsx`, the `api.recommend(12)` call.
- **Change the chart history length:** `src/pages/SymbolDetail.tsx`, the `api.history(sym, 252)` call.
- **Change accent color:** edit `--accent` in `src/index.css` (HSL components).
- **Switch to light theme:** remove `class="dark"` from `<html>` in `index.html`.

## Building for production (optional)

```powershell
npm run build
```

Outputs static files to `dist/`. You can serve them with any static host. For a portfolio demo, `npm run dev` is fine.

## Demo workflow

When you want to show this off:

1. Terminal 1: `uvicorn api.main:app --port 8000` (in the Python project)
2. Terminal 2: `npm run dev` (in the dashboard)
3. Browser opens to `http://localhost:5173` — already styled, recommendations loading

Take screenshots of the dashboard + the symbol detail page for your README/portfolio.

## What's left after this

You now have:
- ✅ Data ingestion → MongoDB
- ✅ PyTorch model with proper validation
- ✅ MLflow experiment tracking
- ✅ FastAPI service
- ✅ MCP server (just wire into Claude Desktop)
- ✅ React dashboard with chart and routing

Optional remaining work:
- **Node.js gateway** — only worth it if you want auth + persistent user data
- **More features in the model** — sector momentum rank, volume features, news sentiment
- **Bigger universe** — expand `UNIVERSE` in `ingestion/config.py`, re-backfill, retrain
- **Deployment** — Render or Fly.io for the API + Vercel for the dashboard
