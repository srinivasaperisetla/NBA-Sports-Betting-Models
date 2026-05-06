"""Odds API integration: fetch and enrich sportsbook odds for player props."""
import json as _json
import logging
import math
import os
import time
import unicodedata as _unicodedata
from datetime import date as _date_type
from pathlib import Path as _Path
from threading import Lock as _Lock
from typing import Dict

import requests
from dotenv import load_dotenv
from scipy.stats import norm as _norm
from scipy.stats import poisson as _poisson

from config import STAT_TO_MARKET

logger = logging.getLogger(__name__)

load_dotenv()

BASE_URL = "https://api.the-odds-api.com/v4"
API_KEY = os.getenv("ODDS_API_KEY")

# ── Daily cache ────────────────────────────────────────────
# Structure:
#   "events" → {date, data: [...events...], team_to_event: {norm_team: event_id}}
#   "event_market:{event_id}:{market}" → {date, data: {...}}
#   "idx:{event_id}:{market}" → {(book_key, norm_player): {line, over_price, under_price}}

_odds_cache: Dict[str, dict] = {}
_odds_cache_lock = _Lock()
_player_index_cache: Dict[str, dict] = {}


_CACHE_FILE = _Path(__file__).parent / "odds_cache.json"
_disk_cache_loaded = False


def _today_str() -> str:
  return _date_type.today().isoformat()


def _load_cache_from_disk() -> None:
  """Load odds cache from disk if it exists and the date matches today."""
  global _odds_cache, _disk_cache_loaded
  if _disk_cache_loaded:
    return
  _disk_cache_loaded = True

  if not _CACHE_FILE.exists():
    return
  try:
    with open(_CACHE_FILE, "r") as f:
      disk = _json.load(f)
    if disk.get("date") != _today_str():
      logger.info("Odds disk cache stale (date=%s), ignoring", disk.get("date"))
      return
    with _odds_cache_lock:
      if "events" in disk:
        _odds_cache["events"] = disk["events"]
      for key, val in disk.get("event_markets", {}).items():
        _odds_cache[key] = val
    logger.info("Loaded odds cache from disk (%d entries)", len(disk.get("event_markets", {})) + (1 if "events" in disk else 0))
  except Exception as e:
    logger.warning("Failed to load odds disk cache: %s", e)


def _save_cache_to_disk() -> None:
  """Persist the in-memory odds cache to disk."""
  today = _today_str()
  with _odds_cache_lock:
    payload = {"date": today, "events": None, "event_markets": {}}
    for key, val in _odds_cache.items():
      if key == "events":
        payload["events"] = val
      elif key.startswith("event_market:"):
        payload["event_markets"][key] = val
  try:
    with open(_CACHE_FILE, "w") as f:
      _json.dump(payload, f)
  except Exception as e:
    logger.warning("Failed to save odds disk cache: %s", e)


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
  _load_cache_from_disk()
  today = _today_str()

  with _odds_cache_lock:
    entry = _odds_cache.get("events")
    if entry and entry.get("date") == today:
      return entry["data"], entry["team_to_event"]

  if not API_KEY:
    return None, None

  url = f"{BASE_URL}/sports/basketball_nba/events"
  params = {"apiKey": API_KEY}

  try:
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
      logger.warning("Odds API events %d: %s", resp.status_code, resp.text[:200])
      return None, None
    events = resp.json()
  except Exception as e:
    logger.warning("Odds API events request failed: %s", e)
    return None, None

  team_to_event = {}
  for ev in events:
    eid = ev.get("id", "")
    home = ev.get("home_team", "")
    away = ev.get("away_team", "")
    if home:
      team_to_event[_norm_name(home)] = eid
    if away:
      team_to_event[_norm_name(away)] = eid

  logger.info("Odds API: found %d NBA events, %d teams (0 quota cost)", len(events), len(team_to_event))

  with _odds_cache_lock:
    _odds_cache["events"] = {
      "date": today,
      "data": events,
      "team_to_event": team_to_event,
    }

  _save_cache_to_disk()
  return events, team_to_event


def _get_event_market_cached(event_id: str, market: str) -> dict | None:
  """
  Get odds for a single event + single market.
  Costs 1 quota unit on cache miss.
  Returns the full event odds response dict or None.
  """
  _load_cache_from_disk()
  today = _today_str()
  cache_key = f"event_market:{event_id}:{market}"

  with _odds_cache_lock:
    entry = _odds_cache.get(cache_key)
    if entry and entry.get("date") == today:
      return entry.get("data")

  if not API_KEY:
    return None

  url = f"{BASE_URL}/sports/basketball_nba/events/{event_id}/odds"
  params = {
    "apiKey": API_KEY,
    "regions": "us",
    "markets": market,
    "oddsFormat": "decimal",
  }

  data = None
  for attempt in range(3):
    try:
      resp = requests.get(url, params=params, timeout=30)
      if resp.status_code == 429:
        wait = 2 ** attempt
        logger.warning("Odds API rate limited, retrying in %ds", wait)
        time.sleep(wait)
        continue
      if resp.status_code >= 500:
        wait = 2 ** attempt
        logger.warning("Odds API %d for %s:%s, retrying in %ds", resp.status_code, event_id[:8], market, wait)
        time.sleep(wait)
        continue
      if resp.status_code != 200:
        logger.warning("Odds API %s:%s -> %d: %s", event_id[:8], market, resp.status_code, resp.text[:200])
        return None

      remaining = resp.headers.get("x-requests-remaining", "?")
      used = resp.headers.get("x-requests-used", "?")
      logger.info("Odds [%s] event %s... -- used: %s, remaining: %s", market, event_id[:8], used, remaining)

      data = resp.json()
      break
    except requests.exceptions.Timeout:
      logger.warning("Odds API timeout (attempt %d/3) for %s:%s", attempt + 1, event_id[:8], market)
      continue
    except requests.exceptions.ConnectionError:
      logger.error("Odds API connection error for %s:%s, skipping", event_id[:8], market)
      return None
    except Exception as e:
      logger.warning("Odds API request failed (%s, %s): %s", market, event_id[:8], e)
      return None

  if data is None:
    return None

  with _odds_cache_lock:
    _odds_cache[cache_key] = {
      "date": today,
      "data": data,
    }

  _save_cache_to_disk()
  return data


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


def _decimal_to_american(decimal_odds: float) -> int:
  if decimal_odds >= 2.0:
    return round((decimal_odds - 1) * 100)
  return round(-100 / (decimal_odds - 1))


def _clip_prob(p: float, floor: float = 0.01, ceil: float = 0.99) -> float:
  return max(floor, min(ceil, float(p)))


def _safe_round(value, digits: int = 4):
  return round(float(value), digits) if value is not None else None


POISSON_STATS = {"STL", "BLK", "SB"}
NORMAL_STATS = {
  "PTS", "REB", "AST", "PRA", "PA", "PR", "RA",
  "FGA", "FGM", "3PA", "FTA", "FTM", "3PM", "TOV",
}


def _solve_poisson_lambda(line: float, p_over: float, tol: float = 1e-6) -> float:
  """Find λ such that P(X > line) ≈ p_over for Poisson(λ)."""
  k = int(math.floor(line))
  target_cdf = 1.0 - p_over
  lo, hi = 0.01, max(50.0, line * 5)
  for _ in range(200):
    mid = (lo + hi) / 2.0
    if _poisson.cdf(k, mid) > target_cdf:
      lo = mid
    else:
      hi = mid
    if hi - lo < tol:
      break
  return (lo + hi) / 2.0


def _translate_no_vig_to_dfs(
  sb_line: float,
  dfs_line: float,
  over_no_vig: float,
  stat_name: str,
  player_std: float,
) -> tuple[float, float]:
  """Translate sportsbook no-vig probability to the DFS line via CDF math.
  Returns (p_over_at_dfs, p_under_at_dfs)."""
  if stat_name in POISSON_STATS:
    lam = _solve_poisson_lambda(sb_line, over_no_vig)
    p_over_dfs = 1.0 - _poisson.cdf(int(math.floor(dfs_line)), lam)
  else:
    z = _norm.ppf(1.0 - over_no_vig)
    mu_adjusted = sb_line - z * player_std
    z_dfs = (dfs_line - mu_adjusted) / player_std
    p_over_dfs = 1.0 - _norm.cdf(z_dfs)

  p_over_dfs = _clip_prob(p_over_dfs)
  return p_over_dfs, 1.0 - p_over_dfs


def _enrich_bookmaker_entry(
  entry: dict,
  model_prob_over_dfs: float | None,
  dfs_line: float | None,
  stat_name: str | None = None,
  player_mean: float | None = None,
  player_std: float | None = None,
) -> dict:
  """Enrich sportsbook entry with edge vs 0.54 breakeven. CDF-translates no-vig prob when lines differ."""
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

  entry["model_prob_over_at_dfs_line"] = _safe_round(model_prob_over_dfs)

  over_nv_at_dfs = over_no_vig_prob
  under_nv_at_dfs = under_no_vig_prob

  if not lines_match and over_no_vig_prob is not None and stat_name and player_std and player_std > 0:
    over_nv_at_dfs, under_nv_at_dfs = _translate_no_vig_to_dfs(
      sb_line=float(sb_line),
      dfs_line=float(dfs_line),
      over_no_vig=float(over_no_vig_prob),
      stat_name=stat_name,
      player_std=float(player_std),
    )

  entry["over_no_vig_prob_at_dfs"] = _safe_round(over_nv_at_dfs)
  entry["under_no_vig_prob_at_dfs"] = _safe_round(under_nv_at_dfs)

  entry["edge_over"] = _safe_round(float(model_prob_over_dfs) - 0.54)
  entry["edge_under"] = _safe_round((1.0 - float(model_prob_over_dfs)) - 0.54)

  return entry


def lookup_raw_odds(player_name: str, stat: str, team_name: str | None = None) -> list | None:
  """
  Look up raw sportsbook odds for a player+stat without enrichment.

  Uses team_name to prioritize the player's event (team-aware routing),
  reducing API calls from ~4 per player to ~1.

  Returns list of raw entry dicts or None.
  """
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


def enrich_sportsbook_entries(
  raw_entries: list | None,
  model_prob_over_dfs: float | None,
  dfs_line: float | None,
  stat_name: str | None = None,
  player_mean: float | None = None,
  player_std: float | None = None,
) -> list | None:
  """Enrich a list of raw sportsbook entries with model probabilities and edge."""
  if not raw_entries:
    return None
  return [
    _enrich_bookmaker_entry(
      entry=dict(raw_entry),
      model_prob_over_dfs=model_prob_over_dfs,
      dfs_line=dfs_line,
      stat_name=stat_name,
      player_mean=player_mean,
      player_std=player_std,
    )
    for raw_entry in raw_entries
  ]
