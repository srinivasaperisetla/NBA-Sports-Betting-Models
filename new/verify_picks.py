"""Verify predictions against actual game results."""
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players

from config import SEASONS, LOG_FORMAT

logger = logging.getLogger(__name__)

INPUT_CSV = "today_picks.csv"
OUTPUT_CSV = "todays_picks_verified.csv"
CURRENT_SEASON = SEASONS[-1]

# ── Date filter ──────────────────────────────────────────────
# Set to a specific date string to only check games from that date.
# Set to None to use the most recent game (today/yesterday).
DATE = None  # e.g. "2026-03-20"
# ─────────────────────────────────────────────────────────────

TARGET_DATE = datetime.strptime(DATE, "%Y-%m-%d").date() if DATE else None

all_nba_players = players.get_players()
PLAYER_LOOKUP = {p["full_name"]: p for p in all_nba_players}

df_picks = pd.read_csv(INPUT_CSV)

logger.info("Loaded %d predictions", len(df_picks))
logger.info("Unique players: %d", df_picks['Player'].nunique())
if TARGET_DATE:
  logger.info("Filtering for games on: %s", TARGET_DATE)
else:
  logger.info("Using most recent game (today/yesterday)")

required_output_cols = [
  "Result",
  "Actual value",
  "FULL_ALL Result",
  "FULL_TIGHT Result",
  "REDUCED_ALL Result",
  "REDUCED_TIGHT Result",
]

for col in required_output_cols:
  if col not in df_picks.columns:
    df_picks[col] = ""

unique_players = df_picks["Player"].dropna().unique()
player_games = {}

MAX_RETRIES = 3

for player_name in unique_players:
  logger.info("Fetching data for %s...", player_name)

  player_obj = PLAYER_LOOKUP.get(player_name)
  if player_obj is None:
    logger.warning("Player not found: %s", player_name)
    player_games[player_name] = None
    continue

  player_id = player_obj["id"]

  for attempt in range(MAX_RETRIES):
    try:
      time.sleep(0.6 + attempt * 0.5)

      game_log = PlayerGameLog(
        player_id=player_id,
        season=CURRENT_SEASON,
        season_type_all_star="Regular Season"
      ).get_data_frames()[0]

      if game_log.empty:
        logger.warning("  No games found for %s", player_name)
        player_games[player_name] = None
        break

      game_log["GAME_DATE"] = pd.to_datetime(game_log["GAME_DATE"], errors="coerce")
      game_log = game_log.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

      if TARGET_DATE:
        match = game_log[game_log["GAME_DATE"].dt.date == TARGET_DATE]
        if match.empty:
          logger.warning("  No game found on %s for %s", TARGET_DATE, player_name)
          player_games[player_name] = None
        else:
          logger.info("  Found game from %s", TARGET_DATE)
          player_games[player_name] = match.iloc[0]
      else:
        most_recent = game_log.iloc[0]
        game_date = pd.to_datetime(most_recent["GAME_DATE"]).date()
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)

        if game_date in [today, yesterday]:
          logger.info("  Found game from %s", game_date)
          player_games[player_name] = most_recent
        else:
          logger.warning("  Most recent game is from %s (not today/yesterday)", game_date)
          player_games[player_name] = None
      break

    except Exception as e:
      if attempt == MAX_RETRIES - 1:
        logger.error("Failed to fetch %s after %d attempts: %s", player_name, MAX_RETRIES, e)
        player_games[player_name] = None
      else:
        logger.warning("Attempt %d failed for %s: %s, retrying...", attempt + 1, player_name, e)


def get_actual_stat_value(game_row, stat):
  if game_row is None:
    return None

  stat_map = {
    "PTS": "PTS", "REB": "REB", "AST": "AST",
    "STL": "STL", "BLK": "BLK", "TOV": "TOV",
    "FGM": "FGM", "FGA": "FGA", "FTM": "FTM", "FTA": "FTA",
    "3PM": "FG3M", "3PA": "FG3A",
  }

  if stat == "PRA":
    return float(game_row["PTS"]) + float(game_row["REB"]) + float(game_row["AST"])
  if stat == "PA":
    return float(game_row["PTS"]) + float(game_row["AST"])
  if stat == "PR":
    return float(game_row["PTS"]) + float(game_row["REB"])
  if stat == "RA":
    return float(game_row["REB"]) + float(game_row["AST"])
  if stat == "SB":
    return float(game_row["STL"]) + float(game_row["BLK"])

  col = stat_map.get(stat)
  if col is None or col not in game_row.index:
    return None
  return float(game_row[col])


def evaluate_pick(pick, actual_value, line):
  if actual_value is None or pd.isna(actual_value):
    return "STAT NOT FOUND"
  pick = str(pick).strip().upper()
  line = float(line)
  if pick == "OVER":
    return "HIT" if actual_value > line else "MISS"
  if pick == "UNDER":
    return "HIT" if actual_value < line else "MISS"
  return "BAD PICK"


logger.info("=" * 60)
logger.info("VERIFYING PREDICTIONS")
logger.info("=" * 60)

hits = 0
misses = 0
no_games = 0

for idx, row in df_picks.iterrows():
  player = row["Player"]
  stat = row["Stat"]
  line = row["Line"]

  overall_pick = row.get("Pick", "")
  full_all_pick = row.get("FULL_ALL Pick", overall_pick)
  full_tight_pick = row.get("FULL_TIGHT Pick", overall_pick)
  reduced_all_pick = row.get("REDUCED_ALL Pick", overall_pick)
  reduced_tight_pick = row.get("REDUCED_TIGHT Pick", overall_pick)

  if pd.isna(full_all_pick) or str(full_all_pick).strip() == "":
    full_all_pick = overall_pick
  if pd.isna(full_tight_pick) or str(full_tight_pick).strip() == "":
    full_tight_pick = overall_pick
  if pd.isna(reduced_all_pick) or str(reduced_all_pick).strip() == "":
    reduced_all_pick = overall_pick
  if pd.isna(reduced_tight_pick) or str(reduced_tight_pick).strip() == "":
    reduced_tight_pick = overall_pick

  game = player_games.get(player)
  actual_value = get_actual_stat_value(game, stat)

  if game is None:
    overall_result = "NO GAME"
    full_all_result = "NO GAME"
    full_tight_result = "NO GAME"
    reduced_all_result = "NO GAME"
    reduced_tight_result = "NO GAME"
    no_games += 1
  elif actual_value is None:
    overall_result = "STAT NOT FOUND"
    full_all_result = "STAT NOT FOUND"
    full_tight_result = "STAT NOT FOUND"
    reduced_all_result = "STAT NOT FOUND"
    reduced_tight_result = "STAT NOT FOUND"
  else:
    overall_result = evaluate_pick(overall_pick, actual_value, line)
    full_all_result = evaluate_pick(full_all_pick, actual_value, line)
    full_tight_result = evaluate_pick(full_tight_pick, actual_value, line)
    reduced_all_result = evaluate_pick(reduced_all_pick, actual_value, line)
    reduced_tight_result = evaluate_pick(reduced_tight_pick, actual_value, line)

    if overall_result == "HIT":
      hits += 1
    elif overall_result == "MISS":
      misses += 1

  df_picks.at[idx, "Result"] = overall_result
  df_picks.at[idx, "Actual value"] = "" if actual_value is None else round(float(actual_value), 2)
  df_picks.at[idx, "FULL_ALL Result"] = full_all_result
  df_picks.at[idx, "FULL_TIGHT Result"] = full_tight_result
  df_picks.at[idx, "REDUCED_ALL Result"] = reduced_all_result
  df_picks.at[idx, "REDUCED_TIGHT Result"] = reduced_tight_result

df_picks.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

graded_total = hits + misses
total = len(df_picks)

print(f"Total Predictions: {total}")
print(f"Hits: {hits}")
print(f"Misses: {misses}")
print(f"No Games: {no_games}")

if graded_total > 0:
  accuracy = (hits / graded_total) * 100
  print(f"Accuracy: {accuracy:.1f}%")

  print("\nBY RECOMMENDATION:")
  for rec in df_picks["Recommendation"].dropna().unique():
    subset = df_picks[df_picks["Recommendation"] == rec]
    subset_hits = (subset["Result"] == "HIT").sum()
    subset_misses = (subset["Result"] == "MISS").sum()
    subset_total = subset_hits + subset_misses
    if subset_total > 0:
      subset_acc = (subset_hits / subset_total) * 100
      print(f"  {rec}: {subset_hits}/{subset_total} ({subset_acc:.1f}%)")

  print("\nBY TIER:")
  for tier in ["S", "A", "B", "C", "D"]:
    subset = df_picks[df_picks["Tier"] == tier]
    subset_hits = (subset["Result"] == "HIT").sum()
    subset_misses = (subset["Result"] == "MISS").sum()
    subset_total = subset_hits + subset_misses
    if subset_total > 0:
      subset_acc = (subset_hits / subset_total) * 100
      print(f"  Tier {tier}: {subset_hits}/{subset_total} ({subset_acc:.1f}%)")

  # ── New: Accuracy by Rank Score bucket ──
  rank_col = "Rank Score"
  if rank_col in df_picks.columns:
    numeric_rank = pd.to_numeric(df_picks[rank_col], errors="coerce")
    df_picks["_rank_bucket"] = pd.cut(
      numeric_rank,
      bins=[0, 25, 50, 75, 100],
      labels=["0-25", "25-50", "50-75", "75-100"],
      include_lowest=True,
    )
    print("\nBY RANK SCORE (new 0-100):")
    for bucket in ["75-100", "50-75", "25-50", "0-25"]:
      subset = df_picks[df_picks["_rank_bucket"] == bucket]
      subset_hits = (subset["Result"] == "HIT").sum()
      subset_misses = (subset["Result"] == "MISS").sum()
      subset_total = subset_hits + subset_misses
      if subset_total > 0:
        subset_acc = (subset_hits / subset_total) * 100
        print(f"  Rank {bucket}: {subset_hits}/{subset_total} ({subset_acc:.1f}%)")
    df_picks.drop(columns=["_rank_bucket"], inplace=True, errors="ignore")

  # ── New: Accuracy by Legacy Rank Score bucket ──
  legacy_col = "Legacy Rank Score"
  if legacy_col in df_picks.columns:
    numeric_legacy = pd.to_numeric(df_picks[legacy_col], errors="coerce")
    df_picks["_legacy_bucket"] = pd.cut(
      numeric_legacy,
      bins=[-100, 30, 45, 60, 200],
      labels=["<30", "30-45", "45-60", "60+"],
      include_lowest=True,
    )
    print("\nBY LEGACY RANK SCORE:")
    for bucket in ["60+", "45-60", "30-45", "<30"]:
      subset = df_picks[df_picks["_legacy_bucket"] == bucket]
      subset_hits = (subset["Result"] == "HIT").sum()
      subset_misses = (subset["Result"] == "MISS").sum()
      subset_total = subset_hits + subset_misses
      if subset_total > 0:
        subset_acc = (subset_hits / subset_total) * 100
        print(f"  Legacy {bucket}: {subset_hits}/{subset_total} ({subset_acc:.1f}%)")
    df_picks.drop(columns=["_legacy_bucket"], inplace=True, errors="ignore")

else:
  print("No completed games found")

print(f"\nResults saved to: {OUTPUT_CSV}")
