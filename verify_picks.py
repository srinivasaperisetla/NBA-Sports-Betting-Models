"""Verify predictions against actual game results."""
import time
from datetime import datetime, timedelta

import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players

from config import SEASONS

INPUT_CSV = "todays_picks.csv"
OUTPUT_CSV = "todays_picks_verified.csv"
CURRENT_SEASON = SEASONS[-1]

# ── Date filter ──────────────────────────────────────────────
# Set to a specific date string to only check games from that date.
# Set to None to use the most recent game (today/yesterday).
# Examples: "2026-03-20", "2026-03-15", None
DATE = None  # "2026-03-20"
# ─────────────────────────────────────────────────────────────

TARGET_DATE = datetime.strptime(DATE, "%Y-%m-%d").date() if DATE else None

all_nba_players = players.get_players()
PLAYER_LOOKUP = {p["full_name"]: p for p in all_nba_players}

df_picks = pd.read_csv(INPUT_CSV)

print(f"📊 Loaded {len(df_picks)} predictions")
print(f"🏀 Unique players: {df_picks['Player'].nunique()}")
if TARGET_DATE:
	print(f"📅 Filtering for games on: {TARGET_DATE}")
else:
	print(f"📅 Using most recent game (today/yesterday)")

for col in ["Result", "Actual Value"]:
	if col not in df_picks.columns:
		df_picks[col] = ""

unique_players = df_picks["Player"].dropna().unique()
player_games = {}

MAX_RETRIES = 3

for player_name in unique_players:
	print(f"\n🔄 Fetching data for {player_name}...")

	player_obj = PLAYER_LOOKUP.get(player_name)
	if player_obj is None:
		print(f"  ⚠️ Player not found: {player_name}")
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
				print(f"  ⚠️ No games found for {player_name}")
				player_games[player_name] = None
				break

			game_log["GAME_DATE"] = pd.to_datetime(game_log["GAME_DATE"], errors="coerce")
			game_log = game_log.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

			if TARGET_DATE:
				match = game_log[game_log["GAME_DATE"].dt.date == TARGET_DATE]
				if match.empty:
					print(f"  ⚠️ No game found on {TARGET_DATE}")
					player_games[player_name] = None
				else:
					print(f"  ✅ Found game from {TARGET_DATE}")
					player_games[player_name] = match.iloc[0]
			else:
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
			break

		except Exception as e:
			if attempt == MAX_RETRIES - 1:
				print(f"  ❌ Failed to fetch {player_name} after {MAX_RETRIES} attempts: {e}")
				player_games[player_name] = None
			else:
				print(f"  ⚠️ Attempt {attempt + 1} failed for {player_name}: {e}, retrying...")


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

	game = player_games.get(player)
	actual_value = get_actual_stat_value(game, stat)

	if game is None:
		overall_result = "NO GAME"
		no_games += 1
	elif actual_value is None:
		overall_result = "STAT NOT FOUND"
	else:
		overall_result = evaluate_pick(overall_pick, actual_value, line)
		if "HIT" in overall_result:
			hits += 1
		elif "MISS" in overall_result:
			misses += 1

	df_picks.at[idx, "Result"] = overall_result
	df_picks.at[idx, "Actual Value"] = "" if actual_value is None else round(float(actual_value), 2)

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

	rank_col = "Rank Score"
	if rank_col in df_picks.columns:
		numeric_rank = pd.to_numeric(df_picks[rank_col], errors="coerce")
		df_picks["_rank_bucket"] = pd.cut(
			numeric_rank,
			bins=[-1000, 0, 25, 50, 75, 100, 1000],
			labels=["<0", "0-25", "25-50", "50-75", "75-100", "100+"],
			include_lowest=True,
		)
		print("\n🏆 BY RANK SCORE:")
		for bucket in ["100+", "75-100", "50-75", "25-50", "0-25", "<0"]:
			subset = df_picks[df_picks["_rank_bucket"] == bucket]
			subset_hits = subset["Result"].str.contains("HIT", na=False).sum()
			subset_misses = subset["Result"].str.contains("MISS", na=False).sum()
			subset_total = subset_hits + subset_misses
			if subset_total > 0:
				subset_acc = (subset_hits / subset_total) * 100
				print(f"  Rank {bucket}: {subset_hits}/{subset_total} ({subset_acc:.1f}%)")
		df_picks.drop(columns=["_rank_bucket"], inplace=True, errors="ignore")

else:
	print("⚠️ No completed games found")

print(f"\n✅ Results saved to: {OUTPUT_CSV}")
