import json
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from schema import NBAPredictionInput, Stat

import requests
import os
from dotenv import load_dotenv

load_dotenv()

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

BASE_URL = "https://api.the-odds-api.com/v4"

API_KEY = os.getenv("ODDS_API_KEY")

# ── Odds API: stat → market mapping ───────────────────────

STAT_TO_MARKET = {
  "PTS":  "player_points",
  "REB":  "player_rebounds",
  "AST":  "player_assists",
  "3PM":  "player_threes",
  "BLK":  "player_blocks",
  "STL":  "player_steals",
  "SB":   "player_blocks_steals",
  "TOV":  "player_turnovers",
  "PRA":  "player_points_rebounds_assists",
  "PR":   "player_points_rebounds",
  "PA":   "player_points_assists",
  "RA":   "player_rebounds_assists",
  "FGM":  "player_field_goals",
  "FTM":  "player_frees_made",
  "FTA":  "player_frees_attempts",
  # FGA and 3PA don't have standard Odds API markets
}

# ── Odds API: daily cache ─────────────────────────────────
#
# Cache structure:
#   _odds_cache = {
#     "events": {
#       "date": "2026-03-23",
#       "data": [ {id, home_team, away_team, ...}, ... ],
#       "team_to_event": {"Dallas Mavericks": "abc123", ...}
#     },
#     "event_market:{event_id}:{market}": {
#       "date": "2026-03-23",
#       "data": { ... single event odds response ... }
#     },
#   }
#
# Cost model:
#   - Fetching events: FREE (0 quota)
#   - Fetching one market for one event: 1 quota unit
#   - Only fetched on-demand when a player+stat is first requested
#   - Cached for the rest of the day
#

from datetime import date as _date_type, datetime as _datetime_type
from threading import Lock as _Lock
import unicodedata as _unicodedata

_odds_cache: Dict[str, dict] = {}
_odds_cache_lock = _Lock()


def _today_str() -> str:
  return _date_type.today().isoformat()


def _norm_name(s: str) -> str:
  """Normalize a name for matching: strip accents, lowercase, strip whitespace."""
  return ''.join(
    c for c in _unicodedata.normalize('NFD', s)
    if _unicodedata.category(c) != 'Mn'
  ).lower().strip()


def _get_events_cached() -> tuple[list, dict] | tuple[None, None]:
  """
  Get today's NBA events list + team→event_id lookup.
  Events endpoint is FREE (0 quota cost).
  Returns (events_list, team_to_event_id_dict) or (None, None).
  """
  today = _today_str()

  with _odds_cache_lock:
    entry = _odds_cache.get("events")
    if entry and entry.get("date") == today:
      return entry["data"], entry["team_to_event"]

  # Fetch fresh
  if not API_KEY:
    return None, None

  url = f"{BASE_URL}/sports/basketball_nba/events"
  params = {"apiKey": API_KEY}

  try:
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
      print(f"  ⚠️  Odds API events {resp.status_code}: {resp.text[:200]}")
      return None, None
    events = resp.json()
  except Exception as e:
    print(f"  ⚠️  Odds API events request failed: {e}")
    return None, None

  # Build team→event_id lookup
  team_to_event = {}
  for ev in events:
    eid = ev.get("id", "")
    home = ev.get("home_team", "")
    away = ev.get("away_team", "")
    if home:
      team_to_event[_norm_name(home)] = eid
    if away:
      team_to_event[_norm_name(away)] = eid

  print(f"  📡 Odds API: found {len(events)} NBA events, {len(team_to_event)} teams (0 quota cost)")

  with _odds_cache_lock:
    _odds_cache["events"] = {
      "date": today,
      "data": events,
      "team_to_event": team_to_event,
    }

  return events, team_to_event


def _get_event_market_cached(event_id: str, market: str) -> dict | None:
  """
  Get odds for a single event + single market.
  Costs 1 quota unit on cache miss.
  Returns the full event odds response dict or None.
  """
  today = _today_str()
  cache_key = f"event_market:{event_id}:{market}"

  with _odds_cache_lock:
    entry = _odds_cache.get(cache_key)
    if entry and entry.get("date") == today:
      return entry.get("data")

  # Fetch fresh — 1 quota unit
  if not API_KEY:
    return None

  url = f"{BASE_URL}/sports/basketball_nba/events/{event_id}/odds"
  params = {
    "apiKey": API_KEY,
    "regions": "us",
    "markets": market,
    "oddsFormat": "decimal",
  }

  try:
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
      print(f"    ⚠️  Odds API {event_id[:8]}:{market} → {resp.status_code}: {resp.text[:200]}")
      return None

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"    📡 Odds [{market}] event {event_id[:8]}... — used: {used}, remaining: {remaining}")

    data = resp.json()
  except Exception as e:
    print(f"    ⚠️  Odds API request failed ({market}, {event_id[:8]}): {e}")
    return None

  with _odds_cache_lock:
    _odds_cache[cache_key] = {
      "date": today,
      "data": data,
    }

  return data


def _find_player_event(player_name: str, team_to_event: dict, events: list) -> str | None:
  """
  Find which event a player is in.
  First tries to match via the team name embedded in the events.
  Falls back to searching all events if no team match.
  Returns event_id or None.
  """
  # We don't have a direct player→team mapping here,
  # so we'll return None and let the caller try all events.
  # The caller will search lazily.
  return None


def _decimal_to_american(decimal_odds: float) -> int:
  """Convert decimal odds to American odds."""
  if decimal_odds >= 2.0:
    return round((decimal_odds - 1) * 100)
  return round(-100 / (decimal_odds - 1))


def _clip_prob(p: float, floor: float = 0.01, ceil: float = 0.99) -> float:
  return max(floor, min(ceil, float(p)))


def _safe_round(value: float | None, digits: int = 4):
  return round(float(value), digits) if value is not None else None


def _apply_line_override_to_feature_frame(
  base_frame: pd.DataFrame,
  stat_name: str,
  old_line: float | None,
  new_line: float,
) -> pd.DataFrame:
  """Approximate line-sensitive feature updates for alternate sportsbook lines."""
  df = base_frame.copy()
  if old_line is None:
    old_line = new_line
  delta = float(new_line) - float(old_line)
  if abs(delta) < 1e-12:
    return df

  pl_col = f"PL_{stat_name}"
  if pl_col in df.columns:
    df[pl_col] = float(new_line)

  line_diff_col = f"{stat_name}_LINE_DIFF"
  z_line_col = f"{stat_name}_Z_LINE"
  line_diff_x_min_col = f"{stat_name}_LINE_DIFF_X_MIN"
  anchor_col = f"{stat_name}_ANCHOR"
  dist_anchor_col = f"{stat_name}_DIST_FROM_ANCHOR"

  old_line_diff = None
  new_line_diff = None
  if line_diff_col in df.columns:
    old_line_diff = pd.to_numeric(df.iloc[0][line_diff_col], errors="coerce")
    if pd.notna(old_line_diff):
      new_line_diff = float(old_line_diff) - delta
      df[line_diff_col] = new_line_diff

  if z_line_col in df.columns:
    old_z = pd.to_numeric(df.iloc[0][z_line_col], errors="coerce")
    if pd.notna(old_z):
      if new_line_diff is not None and old_line_diff is not None and abs(float(old_z)) > 1e-12:
        std_est = float(old_line_diff) / float(old_z)
        if abs(std_est) > 1e-12:
          df[z_line_col] = float(new_line_diff) / std_est
      elif new_line_diff is not None and old_line_diff is not None and abs(float(old_line_diff)) > 1e-12:
        df[z_line_col] = float(old_z) * (float(new_line_diff) / float(old_line_diff))

  if line_diff_x_min_col in df.columns:
    min_col = "MIN"
    if new_line_diff is not None and min_col in df.columns:
      min_val = pd.to_numeric(df.iloc[0][min_col], errors="coerce")
      if pd.notna(min_val):
        df[line_diff_x_min_col] = float(new_line_diff) * float(min_val)
    elif old_line_diff is not None:
      old_ldxm = pd.to_numeric(df.iloc[0][line_diff_x_min_col], errors="coerce")
      if pd.notna(old_ldxm) and abs(float(old_line_diff)) > 1e-12 and new_line_diff is not None:
        df[line_diff_x_min_col] = float(old_ldxm) * (float(new_line_diff) / float(old_line_diff))

  if anchor_col in df.columns and dist_anchor_col in df.columns:
    anchor_val = pd.to_numeric(df.iloc[0][anchor_col], errors="coerce")
    if pd.notna(anchor_val):
      df[dist_anchor_col] = abs(float(new_line) - float(anchor_val))

  return df


def _compute_family_prob_over_for_line(
  *,
  family_key: str,
  stat_name: str,
  old_line: float | None,
  new_line: float,
  mode_base_frames: Dict[str, pd.DataFrame],
) -> float:
  family_cfg = CHAMPION_FAMILIES[family_key]
  model = model_registry[family_key].get(stat_name)
  expected_features = feature_registry[family_key].get(stat_name)
  if model is None or expected_features is None:
    raise HTTPException(status_code=404, detail=f"No model/features for family '{family_key}' and stat '{stat_name}'")

  mode = family_cfg["feature_mode"]
  override_frame = _apply_line_override_to_feature_frame(
    mode_base_frames[mode], stat_name=stat_name, old_line=old_line, new_line=new_line
  )
  x_frame = override_frame.reindex(columns=expected_features, fill_value=0)
  x_frame = x_frame.apply(pd.to_numeric, errors="coerce").fillna(0)
  return float(model.predict_proba(x_frame)[0, 1])


def _enrich_bookmaker_entry(
  entry: dict,
  model_prob_over_dfs: float | None,
  model_prob_over_by_line: Dict[float, float] | None,
  dfs_line: float | None,
) -> dict:
  """Enrich sportsbook entry, translating off-line books to the DFS line."""
  over_dec = entry.get("over_decimal")
  under_dec = entry.get("under_decimal")
  sb_line = entry.get("line")

  if over_dec is not None:
    entry["over_american"] = _decimal_to_american(over_dec)
  if under_dec is not None:
    entry["under_american"] = _decimal_to_american(under_dec)

  over_implied = (1.0 / over_dec) if over_dec else None
  under_implied = (1.0 / under_dec) if under_dec else None

  if over_implied is not None:
    entry["over_implied_prob"] = _safe_round(over_implied)
  if under_implied is not None:
    entry["under_implied_prob"] = _safe_round(under_implied)

  total_implied = None
  over_no_vig_prob = None
  under_no_vig_prob = None
  if over_implied is not None and under_implied is not None:
    total_implied = over_implied + under_implied
    entry["vig"] = _safe_round(total_implied - 1.0)
    over_no_vig_prob = over_implied / total_implied
    under_no_vig_prob = under_implied / total_implied
    entry["over_no_vig_prob"] = _safe_round(over_no_vig_prob)
    entry["under_no_vig_prob"] = _safe_round(under_no_vig_prob)
  elif over_implied is not None:
    over_no_vig_prob = over_implied
    entry["over_no_vig_prob"] = _safe_round(over_no_vig_prob)
  elif under_implied is not None:
    under_no_vig_prob = under_implied
    entry["under_no_vig_prob"] = _safe_round(under_no_vig_prob)

  lines_match = False
  if dfs_line is not None and sb_line is not None:
    line_delta = float(sb_line) - float(dfs_line)
    entry["line_delta"] = _safe_round(line_delta, 1)
    lines_match = abs(line_delta) < 0.01
  else:
    entry["line_delta"] = None
  entry["lines_match"] = lines_match

  if model_prob_over_dfs is None or sb_line is None:
    return entry

  if model_prob_over_by_line is None:
    model_prob_over_by_line = {}

  model_prob_over_book = float(model_prob_over_by_line.get(float(sb_line), model_prob_over_dfs))
  entry["model_prob_over_at_book_line"] = _safe_round(model_prob_over_book)
  entry["model_prob_over_at_dfs_line"] = _safe_round(model_prob_over_dfs)

  if over_no_vig_prob is not None:
    translated_over = _clip_prob(over_no_vig_prob + (model_prob_over_dfs - model_prob_over_book))
    translated_under = 1.0 - translated_over
    entry["over_hit_prob"] = _safe_round(translated_over)
    entry["under_hit_prob"] = _safe_round(translated_under)

    anchor_over = 0.65 * float(model_prob_over_dfs) + 0.35 * translated_over
    anchor_under = 1.0 - anchor_over
    entry["anchor_prob_over"] = _safe_round(anchor_over)
    entry["anchor_prob_under"] = _safe_round(anchor_under)

    entry["edge_over"] = _safe_round(anchor_over - translated_over)
    entry["edge_under"] = _safe_round(anchor_under - translated_under)

    if total_implied is not None:
      priced_over_raw = translated_over * total_implied
      priced_under_raw = translated_under * total_implied
      priced_over_decimal = 1.0 / priced_over_raw
      priced_under_decimal = 1.0 / priced_under_raw
      entry["priced_over_decimal_at_dfs"] = _safe_round(priced_over_decimal)
      entry["priced_under_decimal_at_dfs"] = _safe_round(priced_under_decimal)
      entry["priced_over_american_at_dfs"] = _decimal_to_american(priced_over_decimal)
      entry["priced_under_american_at_dfs"] = _decimal_to_american(priced_under_decimal)
      entry["ev_over"] = _safe_round(anchor_over * (priced_over_decimal - 1.0) - anchor_under)
      entry["ev_under"] = _safe_round(anchor_under * (priced_under_decimal - 1.0) - anchor_over)

  return entry


def _lookup_player_odds_raw(player_name: str, stat: str, team_name: str | None = None) -> list | None:
  """Look up raw sportsbook odds for a player+stat without enrichment."""
  market = STAT_TO_MARKET.get(stat)
  if market is None:
    return None

  events, team_to_event = _get_events_cached()
  if not events:
    return None

  target = _norm_name(player_name)

  prioritized_event_ids = []
  if team_name:
    team_event_id = team_to_event.get(_norm_name(team_name))
    if team_event_id:
      prioritized_event_ids.append(team_event_id)

  for ev in events:
    eid = ev.get("id")
    if eid and eid not in prioritized_event_ids:
      prioritized_event_ids.append(eid)

  event_map = {ev.get("id"): ev for ev in events if ev.get("id")}
  results = []

  for event_id in prioritized_event_ids:
    event = event_map.get(event_id)
    if not event:
      continue

    event_data = _get_event_market_cached(event_id, market)
    if event_data is None:
      continue

    player_index = _get_player_index(event_id, market, event_data)
    found_in_event = False

    for bookmaker in event_data.get("bookmakers", []):
      book_key = bookmaker.get("key", "")
      book_title = bookmaker.get("title", "")

      for mkt in bookmaker.get("markets", []):
        if mkt.get("key") != market:
          continue

        idx_key = (book_key, target)
        player_outcomes = player_index.get(idx_key)
        if player_outcomes is None:
          continue

        over_price = player_outcomes.get("over_price")
        under_price = player_outcomes.get("under_price")
        line = player_outcomes.get("line")

        if line is not None and (over_price is not None or under_price is not None):
          found_in_event = True
          entry = {
            "bookmaker": book_key,
            "bookmaker_title": book_title,
            "line": float(line),
          }
          if over_price is not None:
            entry["over_decimal"] = float(over_price)
          if under_price is not None:
            entry["under_decimal"] = float(under_price)
          results.append(entry)

    if found_in_event:
      break

  return results if results else None


def _enrich_raw_sportsbook_entries(
  raw_entries: list | None,
  model_prob_over_dfs: float | None,
  model_prob_over_by_line: Dict[float, float] | None,
  dfs_line: float | None,
) -> list | None:
  if not raw_entries:
    return None
  return [
    _enrich_bookmaker_entry(
      entry=dict(raw_entry),
      model_prob_over_dfs=model_prob_over_dfs,
      model_prob_over_by_line=model_prob_over_by_line,
      dfs_line=dfs_line,
    )
    for raw_entry in raw_entries
  ]


# ── Player index for fast odds lookup ──────────────────────
# Avoids calling _norm_name on every outcome for every lookup.
# Built once per (event_id, market) when first accessed.
# Key: (bookmaker_key, normalized_player_name)
# Value: {"line": float, "over_price": float|None, "under_price": float|None}

_player_index_cache: Dict[str, dict] = {}


def _get_player_index(event_id: str, market: str, event_data: dict) -> dict:
  """
  Build or return cached player index for an (event, market) pair.
  Index maps (bookmaker_key, norm_player_name) → {line, over_price, under_price}.
  """
  index_key = f"idx:{event_id}:{market}"

  if index_key in _player_index_cache:
    return _player_index_cache[index_key]

  index = {}

  for bookmaker in event_data.get("bookmakers", []):
    book_key = bookmaker.get("key", "")

    for mkt in bookmaker.get("markets", []):
      if mkt.get("key") != market:
        continue

      # Group outcomes by player
      player_data: Dict[str, dict] = {}

      for outcome in mkt.get("outcomes", []):
        desc = outcome.get("description", "")
        norm = _norm_name(desc)
        point = outcome.get("point")
        price = outcome.get("price")
        name = outcome.get("name", "").lower()

        if norm not in player_data:
          player_data[norm] = {"line": None, "over_price": None, "under_price": None}

        if name == "over":
          player_data[norm]["over_price"] = price
          player_data[norm]["line"] = point
        elif name == "under":
          player_data[norm]["under_price"] = price
          player_data[norm]["line"] = point

      for norm_name, data in player_data.items():
        index[(book_key, norm_name)] = data

  _player_index_cache[index_key] = index
  return index

MODEL_ROOT = Path("./models")
CATEGORY_MAPPINGS_PATH = MODEL_ROOT / "category_mappings.json"

STAT_COLS = [
  "PTS", "REB", "AST", "STL", "BLK",
  "PRA", "PA", "PR", "RA", "SB",
  "TOV", "FTA", "FTM", "FGA", "FGM", "3PM", "3PA"
]

ALL_TARGETS = [f"TARGET_{s}" for s in STAT_COLS]

CATEGORY_COLS = ["PLAYER_NAME", "TEAM", "POSITION", "MATCHUP"]

DROP_BASE_COLS = [
  "PLAYER_NAME", "PLAYER_ID", "TEAM", "MATCHUP", "POSITION",
  "SEASON_YEAR", "SEASON_ID", "GAME_DATE", "GAME_ID",
  "PTS", "REB", "AST", "STL", "BLK", "PRA", "PA", "PR", "RA", "SB",
  "TOV", "FTM", "FGM", "3PM", "FGA", "3PA", "FTA",
  "MIN", "PLUS_MINUS", "TS%", "USG", "OFF_RATING"
]

CHAMPION_FAMILIES = {
  "FULL_ALL": {
    "folder": "CHAMPION_FULL_ALL",
    "feature_mode": "full",
    "calibration_set": "ALL",
    "features_file": "FEATURES_FULL_ALL.pkl",
  },
  "FULL_TIGHT": {
    "folder": "CHAMPION_FULL_TIGHT",
    "feature_mode": "full",
    "calibration_set": "TIGHT",
    "features_file": "FEATURES_FULL_TIGHT.pkl",
  },
  "REDUCED_ALL": {
    "folder": "CHAMPION_REDUCED_ALL",
    "feature_mode": "reduced",
    "calibration_set": "ALL",
    "features_file": "FEATURES_REDUCED_ALL.pkl",
  },
  "REDUCED_TIGHT": {
    "folder": "CHAMPION_REDUCED_TIGHT",
    "feature_mode": "reduced",
    "calibration_set": "TIGHT",
    "features_file": "FEATURES_REDUCED_TIGHT.pkl",
  },
}

FAMILY_ORDER = ["FULL_ALL", "FULL_TIGHT", "REDUCED_ALL", "REDUCED_TIGHT"]

model_registry: Dict[str, Dict[str, object]] = {k: {} for k in CHAMPION_FAMILIES}
feature_registry: Dict[str, Dict[str, List[str]]] = {k: {} for k in CHAMPION_FAMILIES}
category_mappings: Dict[str, Dict[str, int]] = {}

BETTING_THRESHOLDS = {
  "BLK": {"recommended_family": "REDUCED_TIGHT", "tier": "S", "min_conf": 0.55, "optimal_conf": 0.70, "base_acc": 0.7587, "realistic_acc": {0.55: 0.7799, 0.60: 0.8044, 0.65: 0.8132, 0.70: 0.8279}},
  "3PM": {"recommended_family": "FULL_ALL", "tier": "S", "min_conf": 0.60, "optimal_conf": 0.70, "base_acc": 0.6950, "realistic_acc": {0.55: 0.7049, 0.60: 0.7383, 0.65: 0.7663, 0.70: 0.8034}},
  "STL": {"recommended_family": "REDUCED_ALL", "tier": "A", "min_conf": 0.60, "optimal_conf": 0.70, "base_acc": 0.7095, "realistic_acc": {0.55: 0.7119, 0.60: 0.7327, 0.65: 0.7576, 0.70: 0.7768}},
  "TOV": {"recommended_family": "FULL_ALL", "tier": "A", "min_conf": 0.60, "optimal_conf": 0.70, "base_acc": 0.6966, "realistic_acc": {0.55: 0.6914, 0.60: 0.7220, 0.65: 0.7521, 0.70: 0.7732}},
  "SB": {"recommended_family": "REDUCED_ALL", "tier": "A", "min_conf": 0.60, "optimal_conf": 0.70, "base_acc": 0.6919, "realistic_acc": {0.55: 0.6929, 0.60: 0.7085, 0.65: 0.7322, 0.70: 0.7585}},
  "3PA": {"recommended_family": "FULL_ALL", "tier": "B", "min_conf": 0.60, "optimal_conf": 0.65, "base_acc": 0.6798, "realistic_acc": {0.55: 0.6569, 0.60: 0.6909, 0.65: 0.7275, 0.70: 0.7635}},
  "FTM": {"recommended_family": "FULL_ALL", "tier": "B", "min_conf": 0.60, "optimal_conf": 0.65, "base_acc": 0.6816, "realistic_acc": {0.55: 0.6576, 0.60: 0.6859, 0.65: 0.7137, 0.70: 0.7408}},
  "REB": {"recommended_family": "FULL_ALL", "tier": "B", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6773, "realistic_acc": {0.55: 0.6415, 0.60: 0.6748, 0.65: 0.7140, 0.70: 0.7339}},
  "FGM": {"recommended_family": "FULL_ALL", "tier": "C", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6857, "realistic_acc": {0.55: 0.6447, 0.60: 0.6794, 0.65: 0.7196, 0.70: 0.7536}},
  "FTA": {"recommended_family": "FULL_ALL", "tier": "C", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6726, "realistic_acc": {0.55: 0.6250, 0.60: 0.6577, 0.65: 0.6863, 0.70: 0.7213}},
  "AST": {"recommended_family": "FULL_ALL", "tier": "C", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6679, "realistic_acc": {0.55: 0.6362, 0.60: 0.6642, 0.65: 0.6876, 0.70: 0.7119}},
  "RA": {"recommended_family": "FULL_ALL", "tier": "C", "min_conf": 0.65, "optimal_conf": 0.70, "base_acc": 0.6750, "realistic_acc": {0.55: 0.6333, 0.60: 0.6642, 0.65: 0.6925, 0.70: 0.7398}},
  "PTS": {"recommended_family": "REDUCED_ALL", "tier": "D", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6747, "realistic_acc": {0.55: 0.6251, 0.60: 0.6619, 0.65: 0.6934, 0.70: 0.7313}},
  "PA": {"recommended_family": "FULL_ALL", "tier": "D", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6759, "realistic_acc": {0.55: 0.6299, 0.60: 0.6627, 0.65: 0.7050, 0.70: 0.7344}},
  "PR": {"recommended_family": "FULL_ALL", "tier": "D", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6747, "realistic_acc": {0.55: 0.6309, 0.60: 0.6625, 0.65: 0.6884, 0.70: 0.7089}},
  "PRA": {"recommended_family": "FULL_ALL", "tier": "D", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6730, "realistic_acc": {0.55: 0.6326, 0.60: 0.6749, 0.65: 0.6945, 0.70: 0.7314}},
  "FGA": {"recommended_family": "FULL_ALL", "tier": "D", "min_conf": 0.70, "optimal_conf": 0.70, "base_acc": 0.6797, "realistic_acc": {0.55: 0.6247, 0.60: 0.6643, 0.65: 0.7111, 0.70: 0.7493}}
}

TIER_POINTS = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0}

class ChampionFamily(str, Enum):
  ALL = "ALL"
  FULL_ALL = "FULL_ALL"
  FULL_TIGHT = "FULL_TIGHT"
  REDUCED_ALL = "REDUCED_ALL"
  REDUCED_TIGHT = "REDUCED_TIGHT"


def get_model_filename(stat_name: str, feature_mode: str, calibration_set: str) -> str:
  return f"{stat_name}_{feature_mode.upper()}_{calibration_set}.pkl"

def get_loaded_stats() -> List[str]:
  stats = set()
  for family_key in CHAMPION_FAMILIES:
    stats.update(model_registry[family_key].keys())
  return sorted(stats)

def get_confidence_bucket(confidence: float) -> float:
  if confidence >= 0.70:
    return 0.70
  elif confidence >= 0.65:
    return 0.65
  elif confidence >= 0.60:
    return 0.60
  else:
    return 0.55

def get_recommendation(confidence: float, min_conf: float, optimal_conf: float) -> str:
  midpoint = (min_conf + optimal_conf) / 2
  if confidence < min_conf:
    return "DO NOT BET"
  elif confidence < midpoint:
    return "BET WITH CAUTION"
  elif confidence < optimal_conf:
    return "BET"
  else:
    return "STRONG BET"

def _summarize_sportsbook_for_rank(prediction: str, sportsbook_odds: list | None) -> dict:
  summary = {
    "same_line_books": 0,
    "all_books": 0,
    "avg_same_line_market_prob_side": None,
    "avg_all_books_market_prob_side": None,
    "avg_edge_side": 0.0,
    "avg_ev_side": 0.0,
  }
  if not sportsbook_odds:
    return summary

  prob_key = "over_hit_prob" if prediction == "OVER" else "under_hit_prob"
  edge_key = "edge_over" if prediction == "OVER" else "edge_under"
  ev_key = "ev_over" if prediction == "OVER" else "ev_under"

  same_probs = []
  all_probs = []
  edges = []
  evs = []

  for entry in sportsbook_odds:
    prob = entry.get(prob_key)
    if prob is not None:
      all_probs.append(float(prob))
      if entry.get("lines_match"):
        same_probs.append(float(prob))
    edge = entry.get(edge_key)
    if edge is not None:
      edges.append(float(edge))
    ev = entry.get(ev_key)
    if ev is not None:
      evs.append(float(ev))

  summary["same_line_books"] = len(same_probs)
  summary["all_books"] = len(all_probs)
  summary["avg_same_line_market_prob_side"] = float(np.mean(same_probs)) if same_probs else None
  summary["avg_all_books_market_prob_side"] = float(np.mean(all_probs)) if all_probs else None
  summary["avg_edge_side"] = float(np.mean(edges)) if edges else 0.0
  summary["avg_ev_side"] = float(np.mean(evs)) if evs else 0.0
  return summary


def compute_rank_score(
  confidence: float,
  tier: str,
  z_vs_line: float,
  z_vs_recent: float,
  z_vs_matchup: float,
  line_diff: float,
  momentum: float,
  prediction: str,
  last10_rate: float,
  last5_rate: float,
  agreement_ratio: float,
  probability_std: float,
  sportsbook_odds: list | None = None,
) -> dict:
  direction = 1 if prediction == "OVER" else -1
  confidence_points = confidence * 100
  tier_points = TIER_POINTS.get(tier, 0)

  z_line_points = z_vs_line * direction
  z_recent_points = z_vs_recent * direction
  z_matchup_points = z_vs_matchup * direction
  line_diff_points = line_diff * direction
  momentum_points = momentum * direction

  if direction == 1:
    last10_points = 2 * last10_rate
    last5_points = 2 * last5_rate
  else:
    last10_points = 2 * (1 - last10_rate)
    last5_points = 2 * (1 - last5_rate)

  signal_total = (
    z_line_points +
    z_recent_points +
    z_matchup_points +
    line_diff_points +
    momentum_points +
    last10_points +
    last5_points
  )

  sportsbook_summary = _summarize_sportsbook_for_rank(prediction, sportsbook_odds)
  avg_same = sportsbook_summary["avg_same_line_market_prob_side"]
  avg_all = sportsbook_summary["avg_all_books_market_prob_side"]
  avg_edge = sportsbook_summary["avg_edge_side"]
  avg_ev = sportsbook_summary["avg_ev_side"]

  same_line_market_points = 100 * (avg_same - 0.5) if avg_same is not None else 0.0
  all_books_market_points = 100 * (avg_all - 0.5) if avg_all is not None else 0.0
  edge_points = 100 * avg_edge
  agreement_points = 5 * agreement_ratio - 10 * probability_std

  if sportsbook_summary["same_line_books"] > 0 and avg_same is not None:
    hard_penalty = -20.0 if avg_same < 0.50 else 0.0
    continuous_penalty = -25.0 * max(0.0, 0.50 - avg_same)
  elif avg_all is not None:
    hard_penalty = -8.0 if avg_all < 0.50 else 0.0
    continuous_penalty = -15.0 * max(0.0, 0.50 - avg_all)
  else:
    hard_penalty = 0.0
    continuous_penalty = 0.0

  penalty_points = hard_penalty + continuous_penalty

  final_score = (
    0.55 * confidence_points +
    1.00 * edge_points +
    0.50 * same_line_market_points +
    0.25 * all_books_market_points +
    1.00 * signal_total +
    agreement_points +
    tier_points +
    penalty_points
  )

  return {
    "rank_score": final_score,
    "rank_breakdown": {
      "confidence_points": confidence_points,
      "edge_points": edge_points,
      "same_line_market_points": same_line_market_points,
      "all_books_market_points": all_books_market_points,
      "agreement_points": agreement_points,
      "tier_points": tier_points,
      "penalty_points": penalty_points,
      "signal_points": {
        "z_line": z_line_points,
        "z_recent": z_recent_points,
        "z_matchup": z_matchup_points,
        "line_diff": line_diff_points,
        "momentum": momentum_points,
        "last10": last10_points,
        "last5": last5_points,
        "signal_total": signal_total,
      },
      "sportsbook_summary": {
        "same_line_books": sportsbook_summary["same_line_books"],
        "all_books": sportsbook_summary["all_books"],
        "avg_same_line_market_prob_side": _safe_round(avg_same) if avg_same is not None else None,
        "avg_all_books_market_prob_side": _safe_round(avg_all) if avg_all is not None else None,
        "avg_edge_side": _safe_round(avg_edge),
        "avg_ev_side": _safe_round(avg_ev),
      },
    }
  }

def apply_category_mappings(df: pd.DataFrame) -> pd.DataFrame:
  new_columns = {}

  for col in CATEGORY_COLS:
    if col in df.columns:
      new_columns[f"{col}_ID"] = (
        df[col]
        .astype(str)
        .map(category_mappings.get(col, {}))
        .fillna(-1)
        .astype(int)
      )

  if new_columns:
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

  return df

def build_feature_frame_inference(
  df: pd.DataFrame,
  target_stat: str,
  feature_mode: str
) -> pd.DataFrame:
  cols_to_drop = list(DROP_BASE_COLS) + list(ALL_TARGETS)

  for s in STAT_COLS:
    if s != target_stat:
      cols_to_drop.extend([
        f"PL_{s}",
        f"OVER_PL_RATE_{s}_L10",
        f"OVER_PL_RATE_{s}_L5",
        f"{s}_Z_LINE",
        f"{s}_Z_RECENT",
        f"{s}_Z_MATCHUP",
        f"{s}_LINE_DIFF_X_MIN",
        f"{s}_LINE_DIFF",
        f"{s}_DIST_FROM_ANCHOR",
        f"{s}_ANCHOR"
      ])

  X = df.drop(columns=cols_to_drop, errors="ignore")

  if feature_mode == "reduced":
    drop_more = []

    for s in STAT_COLS:
      if s == target_stat:
        continue

      prefixes = [
        f"CUM_AVG_{s}",
        f"L5_AVG_{s}",
        f"STD_CUM_AVG_{s}",
        f"STD_L5_AVG_{s}",
        f"{s}_",
        f"LAST_MATCHUP_{s}",
        f"MATCHUP_L4_AVG_{s}",
        f"MATCHUP_L4_STD_{s}",
        f"CUM_AVG_{s}_PER_MIN",
        f"L5_{s}_PER_MIN"
      ]

      for p in prefixes:
        drop_more.extend([c for c in X.columns if c.startswith(p)])

      drop_more.extend([c for c in X.columns if c.startswith(f"OPP_ALLOWED_{s}_")])
      drop_more.extend([c for c in X.columns if c.startswith(f"MATCHUP_OPP_ALLOWED_{s}_")])

    if drop_more:
      drop_more = list(dict.fromkeys(drop_more))
      X = X.drop(columns=drop_more, errors="ignore")

  return X


def safe_float_from_frame(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
  if col not in df.columns or df.empty:
    return default
  val = pd.to_numeric(df.iloc[0][col], errors="coerce")
  return float(val) if pd.notna(val) else default


def prepare_inference_frame(
  raw_df: pd.DataFrame,
  target_stat: str,
  feature_mode: str,
  expected_features: List[str]
) -> pd.DataFrame:
  df = raw_df.copy()
  df = apply_category_mappings(df)
  df = build_feature_frame_inference(df, target_stat=target_stat, feature_mode=feature_mode)
  df = df.reindex(columns=expected_features, fill_value=0)
  df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
  return df

def build_single_model_response(
  *,
  stat_name: str,
  player_name: str,
  parlay_line,
  family_key: str,
  x_frame: pd.DataFrame,
  prob: float,
  feature_count: int,
  sportsbook_odds: list | None = None,
  agreement_ratio: float = 1.0,
  probability_std: float = 0.0,
) -> dict:
  prediction = "OVER" if prob >= 0.5 else "UNDER"
  confidence = prob if prob >= 0.5 else (1.0 - prob)

  bet_info = BETTING_THRESHOLDS.get(stat_name)
  if bet_info is None:
    raise ValueError(f"No betting thresholds configured for stat '{stat_name}'")

  min_conf = bet_info["min_conf"]
  optimal_conf = bet_info["optimal_conf"]
  recommendation = get_recommendation(confidence, min_conf, optimal_conf)

  bucket = get_confidence_bucket(confidence)
  expected_acc_at_confidence = bet_info["realistic_acc"][bucket]

  z_line = safe_float_from_frame(x_frame, f"{stat_name}_Z_LINE")
  z_recent = safe_float_from_frame(x_frame, f"{stat_name}_Z_RECENT")
  z_matchup = safe_float_from_frame(x_frame, f"{stat_name}_Z_MATCHUP")
  line_diff = safe_float_from_frame(x_frame, f"{stat_name}_LINE_DIFF")
  momentum = safe_float_from_frame(x_frame, f"{stat_name}_MOMENTUM")
  last10_rate = safe_float_from_frame(x_frame, f"OVER_PL_RATE_{stat_name}_L10")
  last5_rate = safe_float_from_frame(x_frame, f"OVER_PL_RATE_{stat_name}_L5")

  rank_result = compute_rank_score(
    confidence=confidence,
    tier=bet_info["tier"],
    z_vs_line=z_line,
    z_vs_recent=z_recent,
    z_vs_matchup=z_matchup,
    line_diff=line_diff,
    momentum=momentum,
    prediction=prediction,
    last10_rate=last10_rate,
    last5_rate=last5_rate,
    agreement_ratio=agreement_ratio,
    probability_std=probability_std,
    sportsbook_odds=sportsbook_odds,
  )

  family_meta = CHAMPION_FAMILIES[family_key]

  return {
    "family": family_key,
    "feature_mode": family_meta["feature_mode"],
    "calibration_set": family_meta["calibration_set"],
    "feature_count": feature_count,
    "prediction": prediction,
    "model_output": round(float(prob), 4),
    "confidence": round(float(confidence), 4),
    "betting_analysis": {
      "recommendation": recommendation,
      "stat_tier": bet_info["tier"],
      "minimum_confidence": min_conf,
      "optimal_confidence": optimal_conf,
      "model_accuracy_at_confidence": f"Accuracy at {int(bucket * 100)}%+ confidence = {round(expected_acc_at_confidence * 100, 1)}%",
      "model_base_accuracy": f"{stat_name} Base Accuracy = {round(bet_info['base_acc'] * 100, 1)}%",
    },
    "Rank": rank_result
  }

def resolve_families_for_stat(stat_name: str, family: ChampionFamily) -> List[str]:
  if family == ChampionFamily.ALL:
    resolved = [fam for fam in FAMILY_ORDER if stat_name in model_registry[fam]]
  else:
    fam = family.value
    resolved = [fam] if stat_name in model_registry[fam] else []

  if not resolved:
    raise HTTPException(
      status_code=404,
      detail=f"No loaded model found for stat '{stat_name}' and family '{family.value}'"
    )

  return resolved

def run_prediction(input_data: NBAPredictionInput, stat_name: str, family: ChampionFamily) -> dict:
  parlay_line = getattr(input_data, f"PL_{stat_name}", None)
  raw_df = pd.DataFrame([input_data.model_dump()])

  families_to_run = resolve_families_for_stat(stat_name, family)
  mapped_df = apply_category_mappings(raw_df.copy())

  mode_base_frames = {}
  for fam in families_to_run:
    family_cfg = CHAMPION_FAMILIES[fam]
    mode = family_cfg["feature_mode"]
    if mode not in mode_base_frames:
      mode_base_frames[mode] = build_feature_frame_inference(
        mapped_df.copy(), target_stat=stat_name, feature_mode=mode
      )

  model_probs_by_family: Dict[str, float] = {}
  feature_count_by_family: Dict[str, int] = {}
  x_frames: Dict[str, pd.DataFrame] = {}

  for fam in families_to_run:
    family_cfg = CHAMPION_FAMILIES[fam]
    model = model_registry[fam].get(stat_name)
    expected_features = feature_registry[fam].get(stat_name)
    if model is None or expected_features is None:
      continue

    mode = family_cfg["feature_mode"]
    base_frame = mode_base_frames[mode]
    x_frame = base_frame.reindex(columns=expected_features, fill_value=0)
    x_frame = x_frame.apply(pd.to_numeric, errors="coerce").fillna(0)

    prob = float(model.predict_proba(x_frame)[0, 1])
    model_probs_by_family[fam] = prob
    feature_count_by_family[fam] = len(expected_features)
    x_frames[fam] = x_frame

  if not model_probs_by_family:
    raise HTTPException(
      status_code=404,
      detail=f"No usable model outputs for stat '{stat_name}'"
    )

  all_probs = list(model_probs_by_family.values())
  consensus_prob_over = float(np.mean(all_probs))
  std_prob = float(np.std(all_probs)) if len(all_probs) > 1 else 0.0
  consensus_prediction = "OVER" if consensus_prob_over >= 0.5 else "UNDER"
  consensus_confidence = consensus_prob_over if consensus_prob_over >= 0.5 else (1.0 - consensus_prob_over)
  agreement_ratio = float(np.mean([
    1.0 if ((model_probs_by_family[fam] >= 0.5 and consensus_prediction == "OVER") or (model_probs_by_family[fam] < 0.5 and consensus_prediction == "UNDER")) else 0.0
    for fam in model_probs_by_family
  ])) if len(model_probs_by_family) > 1 else 1.0

  dfs_line_value = float(parlay_line) if parlay_line is not None else None
  raw_sportsbook_odds = _lookup_player_odds_raw(
    input_data.PLAYER_NAME,
    stat_name,
    getattr(input_data, "TEAM", None),
  )

  consensus_sportsbook_odds = None
  family_sportsbook_odds: Dict[str, list | None] = {fam: None for fam in model_probs_by_family}

  if raw_sportsbook_odds and dfs_line_value is not None:
    unique_lines = sorted({float(entry["line"]) for entry in raw_sportsbook_odds if entry.get("line") is not None})

    family_prob_over_by_line: Dict[str, Dict[float, float]] = {fam: {dfs_line_value: model_probs_by_family[fam]} for fam in model_probs_by_family}
    for line_value in unique_lines:
      if abs(line_value - dfs_line_value) < 0.01:
        continue
      for fam in model_probs_by_family:
        family_prob_over_by_line[fam][line_value] = _compute_family_prob_over_for_line(
          family_key=fam,
          stat_name=stat_name,
          old_line=dfs_line_value,
          new_line=line_value,
          mode_base_frames=mode_base_frames,
        )

    consensus_prob_over_by_line = {}
    for line_value in set(unique_lines + [dfs_line_value]):
      probs_here = [family_prob_over_by_line[fam].get(line_value, model_probs_by_family[fam]) for fam in model_probs_by_family]
      consensus_prob_over_by_line[line_value] = float(np.mean(probs_here))

    consensus_sportsbook_odds = _enrich_raw_sportsbook_entries(
      raw_entries=raw_sportsbook_odds,
      model_prob_over_dfs=consensus_prob_over,
      model_prob_over_by_line=consensus_prob_over_by_line,
      dfs_line=dfs_line_value,
    )

    for fam in model_probs_by_family:
      family_sportsbook_odds[fam] = _enrich_raw_sportsbook_entries(
        raw_entries=raw_sportsbook_odds,
        model_prob_over_dfs=model_probs_by_family[fam],
        model_prob_over_by_line=family_prob_over_by_line[fam],
        dfs_line=dfs_line_value,
      )

  finalized_model_outputs = {}
  for fam in model_probs_by_family:
    finalized_model_outputs[fam] = build_single_model_response(
      stat_name=stat_name,
      player_name=input_data.PLAYER_NAME,
      parlay_line=parlay_line,
      family_key=fam,
      x_frame=x_frames[fam],
      prob=model_probs_by_family[fam],
      feature_count=feature_count_by_family[fam],
      sportsbook_odds=family_sportsbook_odds.get(fam),
      agreement_ratio=agreement_ratio,
      probability_std=std_prob,
    )

  if family != ChampionFamily.ALL and len(finalized_model_outputs) == 1:
    fam = list(finalized_model_outputs.keys())[0]
    out = finalized_model_outputs[fam]
    result = {
      "stat": stat_name,
      "player": input_data.PLAYER_NAME,
      "parlay_line": parlay_line,
      **out
    }
    if family_sportsbook_odds.get(fam) is not None:
      result["sportsbook_odds"] = family_sportsbook_odds[fam]
    return result

  bet_info = BETTING_THRESHOLDS[stat_name]
  min_conf = bet_info["min_conf"]
  optimal_conf = bet_info["optimal_conf"]
  recommendation = get_recommendation(consensus_confidence, min_conf, optimal_conf)

  bucket = get_confidence_bucket(consensus_confidence)
  expected_acc_at_confidence = bet_info["realistic_acc"][bucket]

  ref_family = list(x_frames.keys())[0]
  ref_frame = x_frames[ref_family]

  z_line = safe_float_from_frame(ref_frame, f"{stat_name}_Z_LINE")
  z_recent = safe_float_from_frame(ref_frame, f"{stat_name}_Z_RECENT")
  z_matchup = safe_float_from_frame(ref_frame, f"{stat_name}_Z_MATCHUP")
  line_diff = safe_float_from_frame(ref_frame, f"{stat_name}_LINE_DIFF")
  momentum = safe_float_from_frame(ref_frame, f"{stat_name}_MOMENTUM")
  last10_rate = safe_float_from_frame(ref_frame, f"OVER_PL_RATE_{stat_name}_L10")
  last5_rate = safe_float_from_frame(ref_frame, f"OVER_PL_RATE_{stat_name}_L5")

  rank_result = compute_rank_score(
    confidence=consensus_confidence,
    tier=bet_info["tier"],
    z_vs_line=z_line,
    z_vs_recent=z_recent,
    z_vs_matchup=z_matchup,
    line_diff=line_diff,
    momentum=momentum,
    prediction=consensus_prediction,
    last10_rate=last10_rate,
    last5_rate=last5_rate,
    agreement_ratio=agreement_ratio,
    probability_std=std_prob,
    sportsbook_odds=consensus_sportsbook_odds,
  )

  result = {
    "stat": stat_name,
    "player": input_data.PLAYER_NAME,
    "parlay_line": parlay_line,
    "consensus": {
      "prediction": consensus_prediction,
      "model_output_avg": round(consensus_prob_over, 4),
      "confidence": round(consensus_confidence, 4),
      "probability_std": round(std_prob, 4),
      "agreement_ratio": round(agreement_ratio, 4),
      "families_used": list(finalized_model_outputs.keys()),
      "betting_analysis": {
        "recommendation": recommendation,
        "stat_tier": bet_info["tier"],
        "minimum_confidence": min_conf,
        "optimal_confidence": optimal_conf,
        "model_accuracy_at_confidence": f"Accuracy at {int(bucket * 100)}%+ confidence = {round(expected_acc_at_confidence * 100, 1)}%",
        "model_base_accuracy": f"{stat_name} Base Accuracy = {round(bet_info['base_acc'] * 100, 1)}%"
      },
      "Rank": rank_result
    },
    "model_variants": finalized_model_outputs
  }

  if consensus_sportsbook_odds is not None:
    result["sportsbook_odds"] = consensus_sportsbook_odds

  return result

@asynccontextmanager
async def lifespan(app: FastAPI):
  global category_mappings, model_registry, feature_registry

  print("\n" + "=" * 70)
  print("LOADING STAGE 4 CHAMPION MODEL PACKAGE")
  print("=" * 70)

  try:
    with open(CATEGORY_MAPPINGS_PATH, "r") as f:
      category_mappings = json.load(f)
    print(f"Loaded category mappings from: {CATEGORY_MAPPINGS_PATH}")
    print(f"  Players:   {len(category_mappings.get('PLAYER_NAME', {}))}")
    print(f"  Teams:     {len(category_mappings.get('TEAM', {}))}")
    print(f"  Positions: {len(category_mappings.get('POSITION', {}))}")
    print(f"  Matchups:  {len(category_mappings.get('MATCHUP', {}))}")
  except Exception as e:
    print(f"Failed to load category mappings: {e}")
    raise

  total_models_loaded = 0

  for family_key, cfg in CHAMPION_FAMILIES.items():
    family_root = MODEL_ROOT / cfg["folder"]
    features_path = family_root / "features" / cfg["features_file"]
    models_dir = family_root / "models"

    try:
      feature_payload = joblib.load(features_path)
      by_stat = feature_payload.get("by_stat", {})
      feature_registry[family_key] = by_stat
    except Exception as e:
      print(f"Failed to load feature map for {family_key}: {e}")
      raise

    loaded_here = 0
    print(f"\n[{family_key}]")
    print(f"  feature map: {features_path}")

    for stat_name in sorted(by_stat.keys()):
      model_filename = get_model_filename(
        stat_name=stat_name,
        feature_mode=cfg["feature_mode"],
        calibration_set=cfg["calibration_set"]
      )
      model_path = models_dir / model_filename

      if not model_path.exists():
        continue

      try:
        model_registry[family_key][stat_name] = joblib.load(model_path)
        loaded_here += 1
        total_models_loaded += 1
        print(f"  loaded {stat_name:4s} -> {model_filename}")
      except Exception as e:
        print(f"  failed {stat_name:4s} -> {e}")

    print(f"  total loaded in {family_key}: {loaded_here}")

  print("\n" + "=" * 70)
  print("API READY")
  print(f"Total loaded models: {total_models_loaded}")
  print(f"Available stats: {get_loaded_stats()}")
  print("=" * 70 + "\n")

  yield

  print("\nShutting down API...")


app = FastAPI(
  title="NBA Player Prop Prediction API",
  description="Serves Stage 4 champion XGBoost model families for NBA player props.",
  version="2.0.0",
  lifespan=lifespan
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)


@app.get("/")
def root():
  stats = get_loaded_stats()
  return {
    "status": "API is live",
    "version": "2.0.0",
    "available_stats": stats,
    "champion_families": FAMILY_ORDER,
    "endpoints": {
      "predict": "/predict?stat={STAT}&family={ALL|FULL_ALL|FULL_TIGHT|REDUCED_ALL|REDUCED_TIGHT}",
      "predict_batch": "/predict_batch?stat={STAT}&family={ALL|FULL_ALL|FULL_TIGHT|REDUCED_ALL|REDUCED_TIGHT}",
      "models": "/models",
      "health": "/health"
    }
  }


@app.get("/health")
def health_check():
  total_models = sum(len(v) for v in model_registry.values())
  return {
    "status": "healthy",
    "models_loaded": total_models > 0,
    "total_models": total_models,
    "families_loaded": {
      fam: len(model_registry[fam]) for fam in FAMILY_ORDER
    }
  }


@app.get("/models")
def list_models():
  stats = get_loaded_stats()

  family_summary = {}
  for fam in FAMILY_ORDER:
    family_summary[fam] = {
      "loaded_stats": sorted(model_registry[fam].keys()),
      "total_loaded": len(model_registry[fam])
    }

  stat_summary = {}
  for stat_name in stats:
    stat_summary[stat_name] = {
      "available_families": [fam for fam in FAMILY_ORDER if stat_name in model_registry[fam]],
      "feature_counts": {
        fam: len(feature_registry[fam].get(stat_name, []))
        for fam in FAMILY_ORDER if stat_name in feature_registry[fam]
      }
    }

  return {
    "families": family_summary,
    "stats": stat_summary,
    "total_models": sum(len(v) for v in model_registry.values())
  }


@app.post("/predict")
def predict(
  input_data: NBAPredictionInput,
  stat: Stat = Query(..., description="Stat to predict"),
  family: ChampionFamily = Query(ChampionFamily.ALL, description="Which champion family to use")
):
  try:
    return run_prediction(
      input_data=input_data,
      stat_name=stat.value,
      family=family
    )
  except HTTPException:
    raise
  except Exception as e:
    import traceback
    raise HTTPException(
      status_code=500,
      detail={
        "error": str(e),
        "traceback": traceback.format_exc()
      }
    )


@app.post("/predict_batch")
def predict_batch(
  input_data: List[NBAPredictionInput],
  stat: Stat = Query(..., description="Stat to predict"),
  family: ChampionFamily = Query(ChampionFamily.ALL, description="Which champion family to use")
):
  try:
    predictions = [
      run_prediction(player_input, stat.value, family)
      for player_input in input_data
    ]

    return {
      "stat": stat.value,
      "family": family.value,
      "total_predictions": len(predictions),
      "predictions": predictions
    }
  except HTTPException:
    raise
  except Exception as e:
    import traceback
    raise HTTPException(
      status_code=500,
      detail={
        "error": str(e),
        "traceback": traceback.format_exc()
      }
    )


if __name__ == "__main__":
  uvicorn.run(
    "main:app",
    host="127.0.0.1",
    port=8000,
    reload=True,
    log_level="info"
  )