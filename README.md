# ProphetAI — NBA Player Prop Prediction System

ProphetAI is an end-to-end machine learning system for predicting NBA player prop
bets (OVER / UNDER on lines such as points, rebounds, assists, etc.). It combines a
trained ensemble of models, a FastAPI prediction service, an interactive Streamlit
front end, and a scraping + backtesting pipeline for evaluating pick quality against
real game results.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [How It Works](#how-it-works)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Running the Scraper & Analysis Pipeline](#running-the-scraper--analysis-pipeline)
- [Model Training Notebook](#model-training-notebook)
- [Environment Variables](#environment-variables)

---

## Architecture Overview

The project is split into two independent pieces that share a common set of models
and feature definitions:

1. **The prediction application** (repo root) — the parts needed to serve and use
   the models interactively:
   - `backend/` — a FastAPI service that loads the trained models and exposes a
     `/predict` endpoint.
   - `app.py` — a Streamlit UI that collects a player + stat + line, builds the
     feature vector, calls the backend, and visualizes the result.
   - `utils.py` / `stat_utils.py` — feature-engineering helpers shared by the app
     and the backend.

2. **The scraper & analysis pipeline** (`scraper_analysis/`) — everything used to
   pull daily prop lines, generate picks in bulk, verify them against actual box
   scores, and analyze historical accuracy. This piece is decoupled from the UI but
   talks to the same backend API.

```
                 ┌─────────────────────────┐
                 │  ProphetAI_NBA_Models    │   trains + exports models
                 │        .ipynb            │
                 └────────────┬─────────────┘
                              │  (writes model artifacts)
                              ▼
                 ┌─────────────────────────┐
                 │      backend/models/     │
                 └────────────┬─────────────┘
                              │
                 ┌────────────▼─────────────┐
                 │   backend/main.py (API)  │  FastAPI @ :8000
                 └───────┬──────────┬───────┘
                         │          │
          ┌──────────────▼──┐   ┌───▼───────────────────────┐
          │    app.py       │   │  scraper_analysis/         │
          │  (Streamlit UI) │   │  scraper.py → picks →      │
          └─────────────────┘   │  verify_picks.py →         │
                                │  analyze_picks.py          │
                                └────────────────────────────┘
```

---

## How It Works

1. **Model training** happens in `ProphetAI_NBA_Models.ipynb`. It engineers features
   from historical NBA data and trains four "champion" model families, exported into
   `backend/models/`:
   - `CHAMPION_FULL_ALL`
   - `CHAMPION_FULL_TIGHT`
   - `CHAMPION_REDUCED_ALL`
   - `CHAMPION_REDUCED_TIGHT`

   (`FULL` / `REDUCED` refer to the feature set size; `ALL` / `TIGHT` refer to the
   training/threshold strategy.)

2. **The backend** (`backend/main.py`) loads these models at startup and serves
   predictions. For a given player/stat/line it returns per-family predictions plus a
   **consensus** that aggregates them into a single OVER/UNDER call with a confidence
   score, agreement ratio, rank score, and a betting recommendation. It also pulls
   live sportsbook lines from The Odds API when a key is configured.

3. **The Streamlit app** (`app.py`) builds the full feature vector for a chosen
   player via `utils.get_input(...)`, posts it to `/predict`, and renders the pick,
   confidence, model-family breakdown, signal-strength metrics, and historical charts
   (last-10 over rate, matchup history, season averages, volatility, momentum).

4. **The pipeline** (`scraper_analysis/`) scales this up: it scrapes daily lines,
   requests a prediction for every player/stat, writes the picks to CSV, verifies
   them against real box scores, and produces accuracy/backtest reports.

---

## Repository Layout

```
nbasportsbetting/
├── app.py                     # Streamlit front end
├── utils.py                   # Feature engineering / input builder (shared)
├── stat_utils.py              # Stat column definitions (shared by app + backend)
├── requirements.txt           # Python dependencies (whole project)
├── ProphetAI_NBA_Models.ipynb # Model training & research notebook
├── README.md
│
├── backend/                   # FastAPI prediction service
│   ├── main.py                # API app: /, /health, /models, /predict, /predict_batch
│   ├── schema.py              # Pydantic request schema (builds feature fields)
│   ├── constants.py           # Allowed players / teams / positions, target columns
│   └── models/                # Exported champion model families + features
│       ├── CHAMPION_FULL_ALL/
│       ├── CHAMPION_FULL_TIGHT/
│       ├── CHAMPION_REDUCED_ALL/
│       └── CHAMPION_REDUCED_TIGHT/
│
└── scraper_analysis/          # Scraping + pick generation + backtesting
    ├── scraper.py             # Scrapes lines → calls API → writes today_picks.csv
    ├── verify_picks.py        # Checks picks against real box scores (nba_api)
    ├── analyze_picks.py       # Historical accuracy / correlations / rerank reports
    ├── clean.py               # Cleans/repairs daily CSVs in place (nba_api)
    ├── today_picks.csv        # Latest generated picks (scraper output)
    ├── todays_picks_verified.csv  # Picks annotated with actual results
    ├── lines/                 # Raw input line files for the scraper
    │   ├── underdog_lines.txt
    │   └── prizepicks_lines.txt
    ├── csv/                   # Historical daily pick CSVs (analysis input)
    └── analysis_output/       # Generated reports from analyze_picks.py
```

> **Note:** the scripts in `scraper_analysis/` use paths relative to that folder
> (`lines/…`, `csv`, `analysis_output`, `today_picks.csv`), so run them from inside
> `scraper_analysis/`.

---

## Requirements

- **Python 3.9+**
- A running instance of the backend API for anything that produces predictions
  (the Streamlit app and `scraper.py` both call `http://127.0.0.1:8000/predict`).
- (Optional) An [Odds API](https://the-odds-api.com/) key for live sportsbook lines.

All Python dependencies are listed in `requirements.txt` (FastAPI, Uvicorn,
Streamlit, pandas, numpy, scikit-learn, xgboost, tensorflow, nba_api, altair, etc.).

---

## Installation

```bash
# From the repo root
pip3 install -r requirements.txt
```

---

## Running the Streamlit App

The app needs the backend running first.

**1. Start the backend API** (serves on `http://127.0.0.1:8000`):

```bash
cd backend
python3 main.py
```

**2. In a separate terminal, start the Streamlit UI:**

```bash
# from the repo root
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`), pick a player,
choose a stat and a line, and hit **Go**.

Useful API endpoints (once the backend is up):

- `GET  /`             — status + loaded stats
- `GET  /health`       — health check
- `GET  /models`       — which model families/stats are loaded
- `POST /predict`      — single prediction (`?stat=PTS&family=ALL`)
- `POST /predict_batch`— batch predictions

---

## Running the Scraper & Analysis Pipeline

All commands below are run from inside the `scraper_analysis/` folder. The scraper
requires the backend API to be running (see above).

```bash
cd scraper_analysis
```

**1. Generate today's picks** — scrapes the line files in `lines/`, requests a
prediction for each, and writes `today_picks.csv`:

```bash
python3 scraper.py
```

**2. Verify picks against real results** — reads `today_picks.csv`, pulls actual box
scores via `nba_api`, and writes `todays_picks_verified.csv`:

```bash
python3 verify_picks.py
```

**3. Analyze historical accuracy** — reads the daily CSVs in `csv/` and writes
backtest reports (accuracy by confidence/tier/stat/player, feature correlations,
rerank previews, etc.) to `analysis_output/`:

```bash
python3 analyze_picks.py --csv-dir csv --out-dir analysis_output
```

**(Optional) Clean/repair daily CSVs** in place before analysis:

```bash
python3 clean.py
```

---

## Model Training Notebook

`ProphetAI_NBA_Models.ipynb` (kept in the repo root) contains the full research and
training workflow: data collection, feature engineering, model training/tuning, and
export of the champion model families consumed by `backend/`. Re-run it whenever you
want to retrain or refresh the models the backend serves.

---

## Environment Variables

Create a `.env` file in the repo root for optional integrations:

```bash
ODDS_API_KEY=your_odds_api_key_here
```

- `ODDS_API_KEY` — used by the backend to fetch live sportsbook lines from The Odds
  API. The backend still runs without it; it simply won't fetch live odds.

`.env` files are git-ignored.
