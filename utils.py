from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players
from nba_api.stats.endpoints import CommonPlayerInfo
from nba_api.stats.endpoints import ScheduleLeagueV2
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.endpoints import LeagueGameLog
import numpy as np
import pandas as pd
from unidecode import unidecode
from config import ALLOWED_PLAYERS_LIST
from stat_utils import STAT_COLS, ADVANCED_COLS, MATCHUP_ALLOWED_METRICS, MATCHUP_ALLOWED_METRICS_W_PACE_DEF
import concurrent.futures
import concurrent.futures
import numpy as np
import pandas as pd
import time

all_players = players.get_players()
active_players = [p for p in all_players if p.get("is_active")]
PLAYER_LOOKUP = {p["full_name"]: p for p in active_players}

STAT_COLS = list(STAT_COLS)
ADVANCED_COLS = list(ADVANCED_COLS)
MATCHUP_ALLOWED_METRICS = list(MATCHUP_ALLOWED_METRICS)
MATCHUP_ALLOWED_METRICS_W_PACE_DEF = list(MATCHUP_ALLOWED_METRICS_W_PACE_DEF)

_PLAYER_INFO_CACHE = {}
_PLAYER_LOG_CACHE = {}
_SCHEDULE_CACHE = {}
_TEAM_LOG_CACHE = {}
_TEAM_LOG_RAW_CACHE = {}

def _copy_df(df):
  return df.copy(deep=True)

def _get_player_info_cached(player_id):
  if player_id not in _PLAYER_INFO_CACHE:
    _PLAYER_INFO_CACHE[player_id] = pd.DataFrame(
      CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    )
  return _copy_df(_PLAYER_INFO_CACHE[player_id])

def _get_player_log_cached(player_id, season_type):
  key = (player_id, season_type)
  if key not in _PLAYER_LOG_CACHE:
    _PLAYER_LOG_CACHE[key] = pd.concat(
      PlayerGameLog(
        player_id=player_id,
        season=SeasonAll.all,
        season_type_all_star=season_type
      ).get_data_frames(),
      ignore_index=True
    )
  return _copy_df(_PLAYER_LOG_CACHE[key])

def _get_schedule_cached(season):
  if season not in _SCHEDULE_CACHE:
    _SCHEDULE_CACHE[season] = pd.DataFrame(
      ScheduleLeagueV2(season=season).get_data_frames()[0]
    )
  return _copy_df(_SCHEDULE_CACHE[season])

def _get_team_log_raw_cached(season, season_type):
  key = (season, season_type)
  if key not in _TEAM_LOG_RAW_CACHE:
    _TEAM_LOG_RAW_CACHE[key] = pd.DataFrame(
      LeagueGameLog(
        player_or_team_abbreviation="T",
        season=season,
        season_type_all_star=season_type
      ).get_data_frames()[0]
    )
  return _copy_df(_TEAM_LOG_RAW_CACHE[key])

def count_games_before_date_sorted(series, days, target_date):
  dates = pd.to_datetime(series, errors="coerce").values.astype("datetime64[ns]")
  dates = dates[~np.isnat(dates)]
  if dates.size == 0:
    return 0

  target = np.datetime64(pd.to_datetime(target_date), "ns")
  lower = target - np.timedelta64(days, "D")

  right = np.searchsorted(dates, target, side="left")
  left = np.searchsorted(dates, lower, side="left")
  return int(right - left)

def _safe_mean(series, default=np.nan):
  if series is None or len(series) == 0:
    return default
  v = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
  v = v[np.isfinite(v)]
  return float(v.mean()) if v.size else default

def _safe_ratio(num, den, default=1.0):
  num = float(num) if np.isfinite(num) else np.nan
  den = float(den) if np.isfinite(den) else np.nan
  if (not np.isfinite(num)) or (not np.isfinite(den)) or den == 0.0:
    return float(default)
  return float(num / den)

def _round_line_to_training_grid(x):
  if x is None:
    return np.nan
  try:
    x = float(x)
  except Exception:
    return np.nan
  if not np.isfinite(x):
    return np.nan
  x = np.round(x * 2.0) / 2.0
  return float(max(x, 0.5))

def _over_rate_last_n(prior_values_1d, line, n, default_if_empty=0.0):
  v = pd.to_numeric(prior_values_1d, errors="coerce").to_numpy(dtype=np.float64)
  v = v[np.isfinite(v)]
  if v.size == 0:
    return float(default_if_empty)
  v = v[-n:]
  return float((v > float(line)).mean())

def _build_team_merged_for_season(season_str):
  if season_str in _TEAM_LOG_CACHE:
    return _copy_df(_TEAM_LOG_CACHE[season_str])

  reg_df = _get_team_log_raw_cached(season_str, "Regular Season")
  po_df = _get_team_log_raw_cached(season_str, "Playoffs")

  frames = [df for df in [reg_df, po_df] if df is not None and not df.empty]
  if not frames:
    merged = pd.DataFrame()
    _TEAM_LOG_CACHE[season_str] = merged
    return _copy_df(merged)

  full_log = pd.concat(frames, ignore_index=True)
  full_log["GAME_DATE"] = pd.to_datetime(full_log["GAME_DATE"], errors="coerce")
  full_log = full_log.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

  full_log["POSS"] = (
    full_log["FGA"] +
    0.44 * full_log["FTA"] -
    full_log["OREB"] +
    full_log["TOV"]
  )

  opp_stats = full_log[[
    "GAME_ID", "TEAM_ID", "PTS", "REB", "AST", "STL", "BLK", "TOV",
    "FG3M", "FGM", "FGA", "FG3A", "FTM", "FTA", "POSS", "MIN"
  ]].copy()

  opp_stats.columns = [
    "GAME_ID", "OPP_TEAM_ID", "PTS_ALLOWED", "REB_ALLOWED", "AST_ALLOWED", "STL_ALLOWED",
    "BLK_ALLOWED", "TOV_ALLOWED", "3PM_ALLOWED", "FGM_ALLOWED", "FGA_ALLOWED", "3PA_ALLOWED",
    "FTM_ALLOWED", "FTA_ALLOWED", "OPP_POSS", "OPP_MIN"
  ]

  merged = full_log.merge(opp_stats, on="GAME_ID")
  merged = merged[merged["TEAM_ID"] != merged["OPP_TEAM_ID"]].copy()

  merged["PACE"] = 48.0 * (
    (merged["POSS"] + merged["OPP_POSS"]) /
    (2.0 * (merged["MIN"] / 5.0).replace(0, np.nan))
  )
  merged["DEF_RATING"] = 100.0 * (
    merged["PTS_ALLOWED"] / merged["POSS"].replace(0, np.nan)
  )

  merged = merged.sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

  _TEAM_LOG_CACHE[season_str] = merged
  return _copy_df(merged)

def _opponent_allowed_l5_cum(merged_season_df, opp_team_abbr):
  df_opp = merged_season_df[merged_season_df["TEAM_ABBREVIATION"] == opp_team_abbr].copy()
  df_opp = df_opp.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
  if df_opp.empty:
    return None, None, df_opp

  last5 = df_opp.tail(5)

  allowed_col = {
    "PTS": "PTS_ALLOWED",
    "REB": "REB_ALLOWED",
    "AST": "AST_ALLOWED",
    "STL": "STL_ALLOWED",
    "BLK": "BLK_ALLOWED",
    "TOV": "TOV_ALLOWED",
    "FTA": "FTA_ALLOWED",
    "FTM": "FTM_ALLOWED",
    "FGA": "FGA_ALLOWED",
    "FGM": "FGM_ALLOWED",
    "3PM": "3PM_ALLOWED",
    "3PA": "3PA_ALLOWED",
  }

  out_l5 = {}
  out_cum = {}

  for m, c in allowed_col.items():
    out_l5[m] = float(pd.to_numeric(last5[c], errors="coerce").mean())
    out_cum[m] = float(pd.to_numeric(df_opp[c], errors="coerce").mean())

  out_l5["PACE"] = float(pd.to_numeric(last5["PACE"], errors="coerce").mean())
  out_cum["PACE"] = float(pd.to_numeric(df_opp["PACE"], errors="coerce").mean())

  out_l5["DEF_RATING"] = float(pd.to_numeric(last5["DEF_RATING"], errors="coerce").mean())
  out_cum["DEF_RATING"] = float(pd.to_numeric(df_opp["DEF_RATING"], errors="coerce").mean())

  return out_l5, out_cum, df_opp

def _entering_game_l5_for_game(df_opp_sorted, game_id):
  if df_opp_sorted.empty:
    return None

  idx = df_opp_sorted.index[df_opp_sorted["GAME_ID"].astype(str) == str(game_id)]
  if len(idx) == 0:
    prev = df_opp_sorted
  else:
    pos = int(idx[0])
    prev = df_opp_sorted.iloc[:pos]

  prev5 = prev.tail(5)
  if prev5.empty:
    return None

  allowed_col = {
    "PTS": "PTS_ALLOWED",
    "REB": "REB_ALLOWED",
    "AST": "AST_ALLOWED",
    "STL": "STL_ALLOWED",
    "BLK": "BLK_ALLOWED",
    "TOV": "TOV_ALLOWED",
    "FTA": "FTA_ALLOWED",
    "FTM": "FTM_ALLOWED",
    "FGA": "FGA_ALLOWED",
    "FGM": "FGM_ALLOWED",
    "3PM": "3PM_ALLOWED",
    "3PA": "3PA_ALLOWED",
  }

  out = {}
  for m, c in allowed_col.items():
    out[m] = float(pd.to_numeric(prev5[c], errors="coerce").mean())
  out["PACE"] = float(pd.to_numeric(prev5["PACE"], errors="coerce").mean())
  out["DEF_RATING"] = float(pd.to_numeric(prev5["DEF_RATING"], errors="coerce").mean())
  return out

def get_input(player_name: str, parlays: dict, seasons: list):
  time.sleep(1.0)
  player = PLAYER_LOOKUP.get(player_name)
  if player is None:
    raise ValueError(f"Player '{player_name}' not found")
  player_id = player["id"]

  current_season = seasons[-1]

  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    f_schedule = executor.submit(_get_schedule_cached, current_season)
    f_info = executor.submit(_get_player_info_cached, player_id)
    f_inseason = executor.submit(_get_player_log_cached, player_id, "Regular Season")
    f_postseason = executor.submit(_get_player_log_cached, player_id, "Playoffs")

    df_schedule = f_schedule.result()
    df_player_info = f_info.result()
    df_inseason_full = f_inseason.result()
    df_playoffs_full = f_postseason.result()

  regular_season_ids = [f"2{s[:4]}" for s in seasons]
  playoff_season_ids = [f"4{s[:4]}" for s in seasons]

  df_inseason_full = df_inseason_full[df_inseason_full["SEASON_ID"].isin(regular_season_ids)].copy()
  df_playoffs_full = df_playoffs_full[df_playoffs_full["SEASON_ID"].isin(playoff_season_ids)].copy()

  df_inseason_full["TEAM"] = df_inseason_full["MATCHUP"].str.split().str[0]
  df_inseason_full["HOME"] = df_inseason_full["MATCHUP"].str.contains("vs.").astype(int)
  df_inseason_full["MATCHUP"] = df_inseason_full["MATCHUP"].str.split().str[-1]
  df_inseason_full = df_inseason_full.rename(columns={
    "FG3M": "3PM",
    "FG3A": "3PA",
    "Player_ID": "PLAYER_ID",
    "Game_ID": "GAME_ID"
  })
  df_inseason_full["POSTSEASON"] = 0

  df_playoffs_full["TEAM"] = df_playoffs_full["MATCHUP"].str.split().str[0]
  df_playoffs_full["HOME"] = df_playoffs_full["MATCHUP"].str.contains("vs.").astype(int)
  df_playoffs_full["MATCHUP"] = df_playoffs_full["MATCHUP"].str.split().str[-1]
  df_playoffs_full = df_playoffs_full.rename(columns={
    "FG3M": "3PM",
    "FG3A": "3PA",
    "Player_ID": "PLAYER_ID",
    "Game_ID": "GAME_ID"
  })
  df_playoffs_full["POSTSEASON"] = 1

  player_position = df_player_info["POSITION"].iloc[0]
  player_height = int(df_player_info["HEIGHT"].iloc[0].split("-")[0]) * 12 + int(df_player_info["HEIGHT"].iloc[0].split("-")[1])
  player_weight = pd.to_numeric(df_player_info["WEIGHT"].iloc[0], errors="coerce")

  yearly_frames = []

  for i, season in enumerate(seasons):
    df_inseason = df_inseason_full[df_inseason_full["SEASON_ID"] == regular_season_ids[i]].copy()
    df_playoffs = df_playoffs_full[df_playoffs_full["SEASON_ID"] == playoff_season_ids[i]].copy()

    if df_inseason.empty and df_playoffs.empty:
      continue

    df_year = pd.concat([d for d in [df_inseason, df_playoffs] if not d.empty], ignore_index=True)

    df_year["GAME_DATE"] = pd.to_datetime(df_year["GAME_DATE"], format="mixed", errors="coerce")
    df_year = df_year.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    df_year["GAME_DIFF"] = df_year["GAME_DATE"].diff().dt.days
    df_year["BACK_TO_BACK"] = (df_year["GAME_DIFF"] == 1).astype(int)
    df_year["PLAYER_REST_DAYS"] = df_year["GAME_DIFF"].fillna(3).clip(0, 5)

    df_year["PRA"] = df_year["PTS"] + df_year["REB"] + df_year["AST"]
    df_year["PA"] = df_year["PTS"] + df_year["AST"]
    df_year["PR"] = df_year["PTS"] + df_year["REB"]
    df_year["RA"] = df_year["REB"] + df_year["AST"]
    df_year["SB"] = df_year["STL"] + df_year["BLK"]

    ts_denom = (2.0 * (df_year["FGA"] + 0.44 * df_year["FTA"])).replace(0, np.nan)
    df_year["TS%"] = 100.0 * (df_year["PTS"] / ts_denom)

    usg_denom = df_year["MIN"].replace(0, np.nan)
    df_year["USG"] = (df_year["FGA"] + 0.44 * df_year["FTA"] + df_year["TOV"]) / usg_denom

    df_year["PTS_PRODUCED"] = df_year["PTS"] + (df_year["AST"] * 2.0)
    df_year["PLAYER_POSSESSIONS"] = df_year["FGA"] + 0.44 * df_year["FTA"] + df_year["TOV"]
    off_denom = df_year["PLAYER_POSSESSIONS"].replace(0, np.nan)
    df_year["OFF_RATING"] = 100.0 * (df_year["PTS_PRODUCED"] / off_denom)

    df_year["SEASON_YEAR"] = season
    df_year["POSITION"] = player_position
    df_year["HEIGHT"] = player_height
    df_year["WEIGHT"] = player_weight
    df_year["PLAYER_NAME"] = player_name
    df_year["PLAYER_ID"] = player_id

    yearly_frames.append(df_year)

  if not yearly_frames:
    raise ValueError(f"No usable games for {player_name} in seasons={seasons}")

  df_player = pd.concat(yearly_frames, ignore_index=True)
  df_player["GAME_DATE"] = pd.to_datetime(df_player["GAME_DATE"], errors="coerce")
  df_player = df_player.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

  team = df_player_info["TEAM_ABBREVIATION"].iloc[0]

  df_previous_games = df_schedule[df_schedule["gameStatus"] == 3]
  df_upcoming = df_schedule[df_schedule["gameStatus"] != 3]

  next_game = df_upcoming[
    (df_upcoming["awayTeam_teamTricode"] == team) |
    (df_upcoming["homeTeam_teamTricode"] == team)
  ].iloc[0]

  try:
    prev_game = df_previous_games[
      (df_previous_games["awayTeam_teamTricode"] == team) |
      (df_previous_games["homeTeam_teamTricode"] == team)
    ].iloc[-1]
    prev_game_date = pd.to_datetime(prev_game["gameDate"])
  except Exception:
    prev_game_date = pd.to_datetime(df_player["GAME_DATE"].max())

  next_game_date = pd.to_datetime(next_game["gameDate"])
  days_diff = int((next_game_date - prev_game_date).days) if pd.notna(next_game_date) and pd.notna(prev_game_date) else 3
  rest_days = float(np.clip(days_diff, 0, 5))

  away = next_game["awayTeam_teamTricode"]
  home_team = next_game["homeTeam_teamTricode"]

  if team == home_team:
    matchup = away
    home = 1
  else:
    matchup = home_team
    home = 0

  postseason = 1 if next_game.get("gameLabel") in [
    "SoFi Play-In Tournament",
    "East First Round",
    "West First Round",
    "East Conf. Semifinals",
    "West Conf. Semifinals",
    "West Conf. Finals",
    "East Conf. Finals",
    "NBA Finals"
  ] else 0

  df_current_season = df_player[df_player["SEASON_YEAR"] == current_season].sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
  df_vs_team = df_player[df_player["MATCHUP"] == matchup].sort_values(["GAME_DATE", "GAME_ID"]).tail(4).reset_index(drop=True)
  gp_against_team = int(len(df_vs_team))

  with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(seasons), 4)) as executor:
    merged_results = list(executor.map(_build_team_merged_for_season, seasons))
  merged_by_season = {season: merged_results[i] for i, season in enumerate(seasons)}

  features = {}

  features["PLAYER_NAME"] = player_name
  features["PLAYER_ID"] = int(player_id)
  features["GAME_ID"] = next_game.get("gameId", "TBD")
  features["POSITION"] = player_position
  features["HEIGHT"] = int(player_height)
  features["WEIGHT"] = int(player_weight) if pd.notna(player_weight) else np.nan
  features["SEASON_YEAR"] = current_season
  features["SEASON_ID"] = f"2{current_season[:4]}"
  features["GAME_DATE"] = next_game_date
  features["TEAM"] = team
  features["MATCHUP"] = matchup
  features["HOME"] = int(home)
  features["POSTSEASON"] = int(postseason)
  features["BACK_TO_BACK"] = int(rest_days == 1.0)
  features["PLAYER_REST_DAYS"] = float(rest_days)

  for col in STAT_COLS:
    features[f"PL_{col}"] = _round_line_to_training_grid(parlays.get(f"PL_{col}", np.nan))

  features["GP_AGAINST_TEAM"] = int(min(gp_against_team, 4))

  if gp_against_team > 0:
    cols_for_per = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FTM", "FGA", "FTA"]
    per_sums = df_vs_team[cols_for_per].sum(numeric_only=True)

    features["PER_GROUPED"] = (
      (per_sums.get("PTS", 0) + per_sums.get("REB", 0) + per_sums.get("AST", 0) +
       per_sums.get("STL", 0) + per_sums.get("BLK", 0)) -
      ((per_sums.get("FGA", 0) - per_sums.get("FGM", 0)) +
       (per_sums.get("FTA", 0) - per_sums.get("FTM", 0)) +
       per_sums.get("TOV", 0))
    ) / float(gp_against_team)

    features["PLUS_MINUS_GROUPED"] = _safe_mean(df_vs_team.get("PLUS_MINUS"), default=0.0)
    features["OFF_RATING_GROUPED"] = _safe_mean(df_vs_team.get("OFF_RATING"), default=0.0)
  else:
    features["PER_GROUPED"] = 0.0
    features["PLUS_MINUS_GROUPED"] = 0.0
    features["OFF_RATING_GROUPED"] = 0.0

  features["GAMES_L3_DAYS"] = int(count_games_before_date_sorted(df_current_season["GAME_DATE"], 3, next_game_date))
  features["GAMES_L7_DAYS"] = int(count_games_before_date_sorted(df_current_season["GAME_DATE"], 7, next_game_date))

  opp_l5_map, opp_cum_map, df_opp_current = _opponent_allowed_l5_cum(merged_by_season[current_season], matchup)
  if opp_l5_map is None:
    raise ValueError(f"No team logs found for opponent {matchup} in season {current_season}")

  for m in MATCHUP_ALLOWED_METRICS:
    features[f"OPP_ALLOWED_{m}_L5_AVG"] = float(opp_l5_map.get(m, np.nan))
    features[f"OPP_ALLOWED_{m}_CUM_AVG"] = float(opp_cum_map.get(m, np.nan))

  features["OPP_PACE_L5_AVG"] = float(opp_l5_map.get("PACE", np.nan))
  features["OPP_PACE_CUM_AVG"] = float(opp_cum_map.get("PACE", np.nan))
  features["OPP_DEF_RATING_L5_AVG"] = float(opp_l5_map.get("DEF_RATING", np.nan))
  features["OPP_DEF_RATING_CUM_AVG"] = float(opp_cum_map.get("DEF_RATING", np.nan))

  features["FATIGUE_FACTOR"] = float(features["OPP_PACE_L5_AVG"] * features["PLAYER_REST_DAYS"])

  features["AST_VULN_RATIO"] = _safe_ratio(
    features.get("OPP_ALLOWED_AST_L5_AVG", np.nan),
    features.get("OPP_ALLOWED_AST_CUM_AVG", np.nan),
    default=1.0
  )
  features["REB_VULN_RATIO"] = _safe_ratio(
    features.get("OPP_ALLOWED_REB_L5_AVG", np.nan),
    features.get("OPP_ALLOWED_REB_CUM_AVG", np.nan),
    default=1.0
  )

  matchup_entering_vals = {m: [] for m in MATCHUP_ALLOWED_METRICS_W_PACE_DEF}

  if gp_against_team > 0:
    for _, row in df_vs_team.iterrows():
      season_y = str(row.get("SEASON_YEAR"))
      gid = str(row.get("GAME_ID"))

      df_merged_season = merged_by_season.get(season_y)
      if df_merged_season is None or df_merged_season.empty:
        continue

      df_opp_season = df_merged_season[
        df_merged_season["TEAM_ABBREVIATION"] == matchup
      ].sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

      entering = _entering_game_l5_for_game(df_opp_season, gid)
      if entering is None:
        continue

      for m in MATCHUP_ALLOWED_METRICS_W_PACE_DEF:
        v = entering.get(m, np.nan)
        if np.isfinite(v):
          matchup_entering_vals[m].append(float(v))

  for m in MATCHUP_ALLOWED_METRICS_W_PACE_DEF:
    feat_key = f"MATCHUP_OPP_ALLOWED_{m}_L4"
    vals = matchup_entering_vals.get(m, [])
    features[feat_key] = float(np.mean(vals)) if len(vals) else 0.0

  cumulative_avg = df_current_season[ADVANCED_COLS].mean(numeric_only=True)
  last5_avg = df_current_season[ADVANCED_COLS].tail(5).mean(numeric_only=True)
  cum_std = df_current_season[ADVANCED_COLS].std(ddof=0, numeric_only=True)
  std_last5 = df_current_season[ADVANCED_COLS].tail(5).std(ddof=0, numeric_only=True)

  if gp_against_team > 0:
    last_matchup = df_vs_team[ADVANCED_COLS].iloc[-1]
    matchup_avg = df_vs_team[ADVANCED_COLS].mean(numeric_only=True)
    matchup_std = df_vs_team[ADVANCED_COLS].std(ddof=0, numeric_only=True)
  else:
    last_matchup = pd.Series({c: np.nan for c in ADVANCED_COLS})
    matchup_avg = pd.Series({c: np.nan for c in ADVANCED_COLS})
    matchup_std = pd.Series({c: np.nan for c in ADVANCED_COLS})

  for col in ADVANCED_COLS:
    features[f"CUM_AVG_{col}"] = float(cumulative_avg.get(col, np.nan))
    features[f"L5_AVG_{col}"] = float(last5_avg.get(col, np.nan))
    features[f"STD_CUM_AVG_{col}"] = float(cum_std.get(col, np.nan))
    features[f"STD_L5_AVG_{col}"] = float(std_last5.get(col, np.nan))
    features[f"LAST_MATCHUP_{col}"] = float(last_matchup.get(col, np.nan))
    features[f"MATCHUP_L4_AVG_{col}"] = float(matchup_avg.get(col, np.nan))
    features[f"MATCHUP_L4_STD_{col}"] = float(matchup_std.get(col, np.nan))
    features[f"{col}_MOMENTUM"] = float(features[f"L5_AVG_{col}"] - features[f"CUM_AVG_{col}"])

  pre_min = features.get("L5_AVG_MIN", np.nan)
  if not np.isfinite(pre_min):
    pre_min = features.get("CUM_AVG_MIN", np.nan)
  if not np.isfinite(pre_min):
    pre_min = 30.0

  pre_usg = features.get("L5_AVG_USG", np.nan)
  if not np.isfinite(pre_usg):
    pre_usg = features.get("CUM_AVG_USG", np.nan)
  if not np.isfinite(pre_usg):
    pre_usg = 0.2

  features["PACE_IMPACT_POSS"] = float((pre_min / 48.0) * (features["OPP_PACE_L5_AVG"] - features["OPP_PACE_CUM_AVG"]))
  features["DEF_VS_VOL"] = float(features["OPP_DEF_RATING_L5_AVG"] * pre_usg)

  for col in STAT_COLS:
    line = features[f"PL_{col}"]
    if not np.isfinite(line):
      features[f"OVER_PL_RATE_{col}_L10"] = 0.0
      features[f"OVER_PL_RATE_{col}_L5"] = 0.0
      continue

    series_all = df_player[col] if col in df_player.columns else pd.Series([], dtype=float)
    features[f"OVER_PL_RATE_{col}_L10"] = _over_rate_last_n(series_all, line, n=10, default_if_empty=0.0)
    features[f"OVER_PL_RATE_{col}_L5"] = _over_rate_last_n(series_all, line, n=5, default_if_empty=0.0)

  matchup_weight_cap = 0.25
  matchup_games_scale = 8.0
  recent_w = 0.35
  matchup_w = min(float(gp_against_team) / matchup_games_scale, matchup_weight_cap)
  season_w = 1.0 - recent_w - matchup_w

  for col in STAT_COLS:
    line = features.get(f"PL_{col}", np.nan)
    l5_avg_val = features.get(f"L5_AVG_{col}", np.nan)
    cum_avg_val = features.get(f"CUM_AVG_{col}", np.nan)
    std5 = features.get(f"STD_L5_AVG_{col}", np.nan)
    matchup_avg_val = features.get(f"MATCHUP_L4_AVG_{col}", np.nan)
    matchup_std_val = features.get(f"MATCHUP_L4_STD_{col}", np.nan)

    has_matchup = (gp_against_team > 0) and np.isfinite(matchup_avg_val)

    if np.isfinite(line) and np.isfinite(l5_avg_val) and np.isfinite(cum_avg_val):
      if has_matchup:
        anchor = recent_w * l5_avg_val + season_w * cum_avg_val + matchup_w * matchup_avg_val
      else:
        anchor = 0.35 * l5_avg_val + 0.65 * cum_avg_val
    else:
      anchor = np.nan

    features[f"{col}_ANCHOR"] = float(anchor) if np.isfinite(anchor) else np.nan
    features[f"{col}_DIST_FROM_ANCHOR"] = float(abs(line - anchor)) if np.isfinite(line) and np.isfinite(anchor) else np.nan
    features[f"{col}_LINE_DIFF"] = float(line - l5_avg_val) if np.isfinite(line) and np.isfinite(l5_avg_val) else np.nan

    denom = (std5 + 1e-6) if np.isfinite(std5) else np.nan
    features[f"{col}_Z_LINE"] = float(np.clip((line - cum_avg_val) / denom, -6, 6)) if np.isfinite(line) and np.isfinite(cum_avg_val) and np.isfinite(denom) else np.nan
    features[f"{col}_Z_RECENT"] = float(np.clip((line - l5_avg_val) / denom, -6, 6)) if np.isfinite(line) and np.isfinite(l5_avg_val) and np.isfinite(denom) else np.nan

    denom_m = (matchup_std_val + 1e-6) if np.isfinite(matchup_std_val) else np.nan
    z_m = float(np.clip((line - matchup_avg_val) / denom_m, -6, 6)) if np.isfinite(line) and np.isfinite(matchup_avg_val) and np.isfinite(denom_m) else np.nan
    gp_weight = min(float(gp_against_team) / 4.0, 1.0)
    features[f"{col}_Z_MATCHUP"] = (z_m * gp_weight) if np.isfinite(z_m) else np.nan

    cum_avg_min = features.get("CUM_AVG_MIN", np.nan)
    l5_avg_min = features.get("L5_AVG_MIN", np.nan)

    features[f"CUM_AVG_{col}_PER_MIN"] = (
      cum_avg_val / cum_avg_min
      if np.isfinite(cum_avg_val) and np.isfinite(cum_avg_min) and cum_avg_min != 0
      else np.nan
    )
    features[f"L5_{col}_PER_MIN"] = (
      l5_avg_val / l5_avg_min
      if np.isfinite(l5_avg_val) and np.isfinite(l5_avg_min) and l5_avg_min != 0
      else np.nan
    )

    features[f"{col}_LINE_DIFF_X_MIN"] = (
      float(features[f"{col}_LINE_DIFF"] * cum_avg_min)
      if np.isfinite(features.get(f"{col}_LINE_DIFF", np.nan)) and np.isfinite(cum_avg_min)
      else np.nan
    )

    mom = features.get(f"{col}_MOMENTUM", np.nan)
    features[f"{col}_MOMENTUM_X_VOL"] = float(mom * std5) if np.isfinite(mom) and np.isfinite(std5) else np.nan

  input_df = pd.DataFrame([features])

  for col in input_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
    input_df[col] = input_df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")

  return input_df, df_player

COLOR = {
  "no_bet": "#DBC415",        # yellow
  "over": "#25C335",          # green
  "over_strong": "#0F5E19",   # deep green
  "under": "#D61C1CFF",         # red
  "under_strong": "#801E16",  # deep red
  "neutral": "#AEB2B5",
  "text": "#FAFAFA",
}

def advice_and_color(pred: int, conf_pct: float):
  """
  pred: 1=OVER, 0=UNDER
  conf_pct: 0..100 (model confidence in its pick)
  """
  if conf_pct < 55:
    return "Do Not Bet", COLOR["no_bet"]
  if pred == 1:
    if conf_pct >= 60:
      return "Strong Bet Over", COLOR["over_strong"]
    return "Bet Over", COLOR["over"]
  else:
    if conf_pct >= 60:
      return "Strongly Bet Under", COLOR["under_strong"]
    return "Bet Under", COLOR["under"]


def normalize_player_name(name):
  """
  Remove accents from player names and match against ALLOWED_PLAYERS_LIST.
  Returns the EXACT name from ALLOWED_PLAYERS_LIST (with original formatting/accents) or None if not found.
  """
  # Normalize input for comparison
  normalized_input = unidecode(name).lower()
  
  # Always search through ALLOWED_PLAYERS_LIST to find exact match
  for allowed_player in ALLOWED_PLAYERS_LIST:
    # Normalize allowed player name for comparison
    normalized_allowed = unidecode(allowed_player).lower()
    
    # If normalized versions match, return the EXACT name from the list
    if normalized_allowed == normalized_input:
      return allowed_player
  
  return None




