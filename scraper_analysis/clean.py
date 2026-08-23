import os
import re
import time
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players

CSV_FOLDER = "csv"

all_players = players.get_players()
PLAYER_LOOKUP = {p["full_name"]: p for p in all_players}

REQUIRED_OUTPUT_COLS = [
    "Result",
    "Actual value",
    "FULL_ALL Result",
    "FULL_TIGHT Result",
    "REDUCED_ALL Result",
    "REDUCED_TIGHT Result",
]


def parse_date_from_filename(filename: str):
    base = os.path.basename(filename)
    match = re.search(r"([A-Za-z]+ \d{1,2} \d{4})", base)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%B %d %Y").date()


def season_from_date(game_date):
    year = game_date.year
    if game_date.month >= 10:
        start_year = year
    else:
        start_year = year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def ensure_columns(df):
    for col in REQUIRED_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""
    return df


def normalize_pick(pick):
    if pd.isna(pick):
        return ""
    return str(pick).strip().upper()


def safe_pick(row, primary_col, fallback_col="Pick"):
    if primary_col in row and not pd.isna(row[primary_col]) and str(row[primary_col]).strip() != "":
        return row[primary_col]
    if fallback_col in row and not pd.isna(row[fallback_col]):
        return row[fallback_col]
    return ""


def fetch_player_gamelog(player_name, season, cache):
    """
    Exactly one API call per (player, season), then reused from cache.
    """
    cache_key = (player_name, season)
    if cache_key in cache:
        return cache[cache_key]

    player_obj = PLAYER_LOOKUP.get(player_name)
    if player_obj is None:
        print(f"  ⚠️ Player not found: {player_name}")
        cache[cache_key] = None
        return None

    try:
        time.sleep(0.6)
        game_log = PlayerGameLog(
            player_id=player_obj["id"],
            season=season,
            season_type_all_star="Regular Season",
        ).get_data_frames()[0]

        if game_log.empty:
            cache[cache_key] = None
            return None

        game_log["GAME_DATE"] = pd.to_datetime(game_log["GAME_DATE"], errors="coerce").dt.date
        game_log = game_log.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

        cache[cache_key] = game_log
        return game_log

    except Exception as e:
        print(f"  ❌ Error fetching {player_name}: {e}")
        cache[cache_key] = None
        return None


def get_game_on_date(player_name, target_date, cache):
    season = season_from_date(target_date)
    game_log = fetch_player_gamelog(player_name, season, cache)

    if game_log is None or game_log.empty:
        return None

    match = game_log[game_log["GAME_DATE"] == target_date]
    if match.empty:
        return None

    return match.iloc[0]


def get_actual_stat_value(game_row, stat):
    if game_row is None:
        return None

    stat = str(stat).strip().upper()

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

    pick = normalize_pick(pick)
    line = float(line)

    if pick == "OVER":
        return "✅ HIT" if actual_value > line else "❌ MISS"
    if pick == "UNDER":
        return "✅ HIT" if actual_value < line else "❌ MISS"
    return "BAD PICK"


def fill_row_from_game(df, idx, row, game):
    stat = row["Stat"]
    line = row["Line"]

    overall_pick = safe_pick(row, "Pick", "Pick")
    full_all_pick = safe_pick(row, "FULL_ALL Pick", "Pick")
    full_tight_pick = safe_pick(row, "FULL_TIGHT Pick", "Pick")
    reduced_all_pick = safe_pick(row, "REDUCED_ALL Pick", "Pick")
    reduced_tight_pick = safe_pick(row, "REDUCED_TIGHT Pick", "Pick")

    actual_value = get_actual_stat_value(game, stat)

    if actual_value is None:
        overall_result = "STAT NOT FOUND"
        full_all_result = "STAT NOT FOUND"
        full_tight_result = "STAT NOT FOUND"
        reduced_all_result = "STAT NOT FOUND"
        reduced_tight_result = "STAT NOT FOUND"
        actual_out = ""
    else:
        overall_result = evaluate_pick(overall_pick, actual_value, line)
        full_all_result = evaluate_pick(full_all_pick, actual_value, line)
        full_tight_result = evaluate_pick(full_tight_pick, actual_value, line)
        reduced_all_result = evaluate_pick(reduced_all_pick, actual_value, line)
        reduced_tight_result = evaluate_pick(reduced_tight_pick, actual_value, line)
        actual_out = round(float(actual_value), 2)

    df.at[idx, "Result"] = overall_result
    df.at[idx, "Actual value"] = actual_out
    df.at[idx, "FULL_ALL Result"] = full_all_result
    df.at[idx, "FULL_TIGHT Result"] = full_tight_result
    df.at[idx, "REDUCED_ALL Result"] = reduced_all_result
    df.at[idx, "REDUCED_TIGHT Result"] = reduced_tight_result


def process_file(file_path, cache):
    print("\n" + "=" * 80)
    print(f"📄 Processing: {os.path.basename(file_path)}")

    target_date = parse_date_from_filename(file_path)
    if target_date is None:
        print("  ⚠️ Could not parse date from filename. Skipping.")
        return

    season = season_from_date(target_date)
    print(f"  📅 File date: {target_date}")
    print(f"  🏀 Season: {season}")

    df = pd.read_csv(file_path)
    df = ensure_columns(df)

    if "Player" not in df.columns or "Stat" not in df.columns or "Line" not in df.columns:
        print("  ⚠️ Missing required columns like Player, Stat, or Line. Skipping.")
        return

    # Only look at rows currently marked NO GAME
    no_game_mask = df["Result"].astype(str).str.strip().str.upper() == "NO GAME"
    no_game_rows = df[no_game_mask].copy()

    print(f"  📊 Total rows: {len(df)}")
    print(f"  ⚠️ Rows currently marked NO GAME: {len(no_game_rows)}")

    if no_game_rows.empty:
        print("  ✅ No NO GAME rows to verify.")
        return

    unique_players_to_check = no_game_rows["Player"].dropna().unique()
    print(f"  🧍 Unique NO GAME players to check: {len(unique_players_to_check)}")

    corrected_rows = 0
    true_no_game_players = 0

    for player_name in unique_players_to_check:
        player_indices = no_game_rows[no_game_rows["Player"] == player_name].index.tolist()

        print(f"  🔄 Checking {player_name} ({len(player_indices)} row(s))")

        game = get_game_on_date(player_name, target_date, cache)

        if game is None:
            print(f"     ➜ True NO GAME on {target_date}")
            true_no_game_players += 1
            continue

        print(f"     ➜ Game found on {target_date}, filling rows")

        for idx in player_indices:
            row = df.loc[idx]
            fill_row_from_game(df, idx, row, game)
            corrected_rows += 1

    df.to_csv(file_path, index=False)

    print(f"  🔧 Corrected rows: {corrected_rows}")
    print(f"  ✅ Players with actual true NO GAME: {true_no_game_players}")
    print(f"  💾 Updated in place: {file_path}")


def main():
    files = [
        os.path.join(CSV_FOLDER, f)
        for f in os.listdir(CSV_FOLDER)
        if f.lower().endswith(".csv")
    ]
    files.sort()

    if not files:
        print(f"⚠️ No CSV files found in folder: {CSV_FOLDER}")
        return

    print(f"📁 Found {len(files)} file(s)")

    cache = {}

    for file_path in files:
        process_file(file_path, cache)

    print("\n✅ Done. All original CSVs were edited in place.")


if __name__ == "__main__":
    main()