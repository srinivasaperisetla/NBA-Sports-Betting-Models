"""Odds API integration: fetch and enrich sportsbook odds for player props."""
import logging
import os
import time
import unicodedata as _unicodedata
from datetime import date as _date_type
from threading import Lock as _Lock
from typing import Dict

import requests
from dotenv import load_dotenv

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
  model_prob_over_by_line: Dict[float, float] | None,
  dfs_line: float | None,
) -> list | None:
  """Enrich a list of raw sportsbook entries with model probabilities and edge/EV."""
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
