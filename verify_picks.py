import time
from datetime import datetime, timedelta

import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players

INPUT_CSV = "today_picks.csv"
OUTPUT_CSV = "todays_picks_verified.csv"
CURRENT_SEASON = "2025-26"

all_players = players.get_players()
PLAYER_LOOKUP = {p["full_name"]: p for p in all_players}

df_picks = pd.read_csv(INPUT_CSV)

print(f"📊 Loaded {len(df_picks)} predictions")
print(f"🏀 Unique players: {df_picks['Player'].nunique()}")

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

for player_name in unique_players:
  print(f"\n🔄 Fetching data for {player_name}...")

  player_obj = PLAYER_LOOKUP.get(player_name)
  if player_obj is None:
    print(f"  ⚠️ Player not found: {player_name}")
    player_games[player_name] = None
    continue

  player_id = player_obj["id"]

  try:
    time.sleep(0.6)

    game_log = PlayerGameLog(
      player_id=player_id,
      season=CURRENT_SEASON,
      season_type_all_star="Regular Season"
    ).get_data_frames()[0]

    if game_log.empty:
      print("  ⚠️ No games found")
      player_games[player_name] = None
      continue

    game_log["GAME_DATE"] = pd.to_datetime(game_log["GAME_DATE"], errors="coerce")
    game_log = game_log.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    most_recent = game_log.iloc[0]
    game_date = pd.to_datetime(most_recent["GAME_DATE"]).date()

    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    if game_date in [today, yesterday]:
      print(f"  ✅ Found game from {game_date}")
      player_games[player_name] = most_recent
    else:
      print(f"  ⚠️ Most recent game is from {game_date} (not today/yesterday)")
      player_games[player_name] = None

  except Exception as e:
    print(f"  ❌ Error: {e}")
    player_games[player_name] = None


def get_actual_stat_value(game_row, stat):
  if game_row is None:
    return None

  stat_map = {
    "PTS": "PTS",
    "REB": "REB",
    "AST": "AST",
    "STL": "STL",
    "BLK": "BLK",
    "TOV": "TOV",
    "FGM": "FGM",
    "FGA": "FGA",
    "FTM": "FTM",
    "FTA": "FTA",
    "3PM": "FG3M",
    "3PA": "FG3A",
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
    return "✅ HIT" if actual_value > line else "❌ MISS"
  if pick == "UNDER":
    return "✅ HIT" if actual_value < line else "❌ MISS"
  return "BAD PICK"


print("\n" + "=" * 60)
print("🎯 VERIFYING PREDICTIONS")
print("=" * 60)

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

    if overall_result == "✅ HIT":
      hits += 1
    elif overall_result == "❌ MISS":
      misses += 1

  df_picks.at[idx, "Result"] = overall_result
  df_picks.at[idx, "Actual value"] = "" if actual_value is None else round(float(actual_value), 2)

  df_picks.at[idx, "FULL_ALL Result"] = full_all_result
  df_picks.at[idx, "FULL_TIGHT Result"] = full_tight_result
  df_picks.at[idx, "REDUCED_ALL Result"] = reduced_all_result
  df_picks.at[idx, "REDUCED_TIGHT Result"] = reduced_tight_result

df_picks.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("📊 RESULTS SUMMARY")
print("=" * 60)

graded_total = hits + misses
total = len(df_picks)

print(f"📈 Total Predictions: {total}")
print(f"✅ Hits: {hits}")
print(f"❌ Misses: {misses}")
print(f"⚠️ No Games: {no_games}")

if graded_total > 0:
  accuracy = (hits / graded_total) * 100
  print(f"🎯 Accuracy: {accuracy:.1f}%")

  print("\n📊 BY RECOMMENDATION:")
  for rec in df_picks["Recommendation"].dropna().unique():
    subset = df_picks[df_picks["Recommendation"] == rec]
    subset_hits = (subset["Result"] == "✅ HIT").sum()
    subset_misses = (subset["Result"] == "❌ MISS").sum()
    subset_total = subset_hits + subset_misses
    if subset_total > 0:
      subset_acc = (subset_hits / subset_total) * 100
      print(f"  {rec}: {subset_hits}/{subset_total} ({subset_acc:.1f}%)")

  print("\n🏆 BY TIER:")
  for tier in ["S", "A", "B", "C", "D"]:
    subset = df_picks[df_picks["Tier"] == tier]
    subset_hits = (subset["Result"] == "✅ HIT").sum()
    subset_misses = (subset["Result"] == "❌ MISS").sum()
    subset_total = subset_hits + subset_misses
    if subset_total > 0:
      subset_acc = (subset_hits / subset_total) * 100
      print(f"  Tier {tier}: {subset_hits}/{subset_total} ({subset_acc:.1f}%)")
else:
  print("⚠️ No completed games found")

print(f"\n✅ Results saved to: {OUTPUT_CSV}")