from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players
from nba_api.stats.endpoints import CommonPlayerInfo
from nba_api.stats.endpoints import ScheduleLeagueV2
from nba_api.stats.library.parameters import SeasonAll
import numpy as np
import pandas as pd

all_players = players.get_players()
active_players = [player for player in all_players if player['is_active']]
PLAYER_LOOKUP = {p["full_name"]: p for p in active_players}

def count_games_before_date_sorted(series, days, target_date):
    # force clean datetime64[ns] numpy array
    dates = pd.to_datetime(series, errors='coerce').values.astype('datetime64[ns]')
    dates = dates[~np.isnat(dates)]  # remove NaT safely

    target = np.datetime64(target_date, 'ns')
    lower = target - np.timedelta64(days, 'D')

    right = np.searchsorted(dates, target, side='left')
    left  = np.searchsorted(dates, lower, side='left')

    return right - left

def get_input(player_name: str, parlays: dict, seasons: list):

  player = next((p for p in active_players if p['full_name'] == player_name), None)
  if player == None:
    raise ValueError(f"Player '{player_name}' not found")
  player_id = player['id']
  player_name = player['full_name']
  df_player = pd.DataFrame()

  df_inseason_full = pd.concat(PlayerGameLog(player_id=player_id, season=SeasonAll.all, season_type_all_star = "Regular Season").get_data_frames())
  df_playoffs_full = pd.concat(PlayerGameLog(player_id=player_id, season=SeasonAll.all, season_type_all_star = "Playoffs").get_data_frames())

  regular_season_ids = []
  playoff_season_ids = []
  for s in seasons:
    start_year = s[:4]
    regular_season_ids.append(f"2{start_year}")
    playoff_season_ids.append(f"4{start_year}")

  df_inseason_full = df_inseason_full[df_inseason_full["SEASON_ID"].isin(regular_season_ids)]
  df_playoffs_full = df_playoffs_full[df_playoffs_full["SEASON_ID"].isin(playoff_season_ids)]

  df_inseason_full['MATCHUP'] = df_inseason_full['MATCHUP'].str.split().str[-1]
  df_inseason_full = df_inseason_full.rename(columns={'FG3M':'3PM', 'FG3A':'3PA','Player_ID': 'PLAYER_ID'})

  df_playoffs_full['MATCHUP'] = df_playoffs_full['MATCHUP'].str.split().str[-1]
  df_playoffs_full = df_playoffs_full.rename(columns={'FG3M':'3PM', 'FG3A':'3PA', 'Player_ID': 'PLAYER_ID'})

  for i in range(len(regular_season_ids)):
    df_inseason = df_inseason_full[df_inseason_full["SEASON_ID"] == regular_season_ids[i]]
    df_playoffs = df_playoffs_full[df_playoffs_full["SEASON_ID"] == playoff_season_ids[i]]

    try:
      df_year = pd.concat([df for df in [df_inseason, df_playoffs] if not df.empty])
    except ValueError:
      print(f"Skipped {player_name} {seasons[i]} (no games)")
      continue

    df_year['GAME_DATE'] = pd.to_datetime(df_year['GAME_DATE'], format='mixed', errors='raise')
    df_year = df_year.sort_values("GAME_DATE").reset_index(drop=True)
    df_year['PRA']  = df_year['PTS'] + df_year['REB'] + df_year['AST']
    df_year['PA']  = df_year['PTS'] + df_year['AST']
    df_year['PR']  = df_year['PTS'] + df_year['REB']
    df_year['RA']  = df_year['REB'] + df_year['AST']
    df_year['SB']  = df_year['STL'] + df_year['BLK']
    df_year['TS%'] = df_year['PTS'] / (2 * (df_year['FGA'] + 0.44 * df_year['FTA'])) * 100
    df_year["USG"] = (df_year["FGA"] + 0.44 * df_year["FTA"] + df_year["TOV"]) / df_year["MIN"]
    df_year['PTS_PRODUCED'] = df_year['PTS'] + (df_year['AST'] * 2)
    df_year['PLAYER_POSSESSIONS'] = df_year['FGA'] + 0.44 * df_year['FTA'] + df_year['TOV']
    df_year['ORtg'] = 100 * (df_year['PTS_PRODUCED'] / df_year['PLAYER_POSSESSIONS'])
    df_year['SEASON_YEAR'] = seasons[i]
    print("Adding", player_name, seasons[i])

    try:
      df_player = pd.concat([df_player, df_year])
    except ValueError:
      print(f"Skipped {player_name} {seasons[i]} (no games)")

  current_season = seasons[-1]
  prev_season = seasons[-2]

  df_player_info = pd.DataFrame(CommonPlayerInfo(player_id=player_id).get_data_frames()[0])
  df_schedule = pd.DataFrame(ScheduleLeagueV2(season = current_season).get_data_frames()[0])
  df_previous_games = df_schedule[df_schedule['gameStatus'] == 3]
  df_schedule = df_schedule[df_schedule['gameStatus'] != 3]

  team = df_player_info['TEAM_ABBREVIATION'].iloc[0]
  next_game = df_schedule[(df_schedule["awayTeam_teamTricode"] == team) | (df_schedule["homeTeam_teamTricode"] == team)].iloc[0]
  prev_game = df_previous_games[(df_previous_games["awayTeam_teamTricode"] == team) | (df_previous_games["homeTeam_teamTricode"] == team)].iloc[-1]
  next_game_date = next_game["gameDate"]
  prev_game_date = prev_game["gameDate"]
  days_diff = (pd.to_datetime(next_game_date) - pd.to_datetime(prev_game_date)).days

  away = next_game["awayTeam_teamTricode"]
  home = next_game["homeTeam_teamTricode"]

  if team == home:
    matchup = away
    home = 1
  elif team == away:
    matchup = home
    home = 0

  df_player = df_player.reset_index(drop=True)

  df_grouped_by_matchup = df_player[df_player['MATCHUP'] == matchup].tail(4)
  gp_against_team = df_grouped_by_matchup.shape[0]

  if next_game['gameLabel'] in ['SoFi Play-In Tournament', 'East First Round', 'West First Round', 'East Conf. Semifinals', 'West Conf. Semifinals', 'West Conf. Finals', 'East Conf. Finals', 'NBA Finals']:
    postseason = 1
  else:
    postseason = 0

  input = pd.DataFrame()
  features = {}

  features['PLAYER_NAME'] = player_name
  features['POSITION'] = df_player_info['POSITION'].iloc[0]
  features['HEIGHT'] = int(df_player_info['HEIGHT'].iloc[0].split('-')[0]) * 12 + int(df_player_info['HEIGHT'].iloc[0].split('-')[1])
  features['WEIGHT'] = df_player_info['WEIGHT'].iloc[0]

  features['SEASON_YEAR'] = current_season
  features['GAME_DATE'] = next_game_date

  for parlay in parlays:
    features[parlay] = parlays[parlay]

  features['TEAM'] = team
  features['MATCHUP'] = matchup

  per_features = pd.DataFrame(index=[0])

  for col in ['PTS', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'FGA', 'FGM', '3PM', 'FTA', 'FTM']:
    per_features[f'CUMULATIVE_VS_TEAM_{col}'] = df_grouped_by_matchup[col].sum()

  features['PER_GROUPED'] = (
      (per_features['CUMULATIVE_VS_TEAM_PTS'] + per_features['CUMULATIVE_VS_TEAM_REB'] + per_features['CUMULATIVE_VS_TEAM_AST'] +
      per_features['CUMULATIVE_VS_TEAM_STL'] + per_features['CUMULATIVE_VS_TEAM_BLK']) -
      ((per_features['CUMULATIVE_VS_TEAM_FGA'] - per_features['CUMULATIVE_VS_TEAM_FGM']) +
      (per_features['CUMULATIVE_VS_TEAM_FTA'] - per_features['CUMULATIVE_VS_TEAM_FTM']) +
      per_features['CUMULATIVE_VS_TEAM_TOV'])
  ) / gp_against_team

  features['PLUS_MINUS_GROUPED'] = df_grouped_by_matchup['PLUS_MINUS'].mean()
  features['ORtg_GROUPED'] = df_grouped_by_matchup['ORtg'].mean()

  features['HOME'] = home
  features['POSTSEASON'] = postseason
  features['BACK_TO_BACK'] = 1 if days_diff == 1 else 0

  df_current_season = df_player[df_player['SEASON_YEAR'] == current_season]
  df_current_season = df_current_season.sort_values('GAME_DATE')

  features['GAMES_LAST3_DAYS'] = count_games_before_date_sorted(df_current_season['GAME_DATE'], 3, pd.to_datetime(next_game_date))
  features['GAMES_LAST7_DAYS'] = count_games_before_date_sorted(df_current_season['GAME_DATE'], 7, pd.to_datetime(next_game_date))
  features['OPP_DEF_EASY'] = -features['PLUS_MINUS_GROUPED']

  STAT_COLS = [
      'PTS', 'REB', 'AST', 'STL', 'BLK',
      'PRA', 'PA', 'PR', 'RA', 'SB',
      'TOV', 'FTM', 'FGM', '3PM', 'FGA', '3PA', 'FTA',
  ]
  ADVANCED_COLS = [
      'PTS', 'REB', 'AST', 'STL', 'BLK',
      'PRA', 'PA', 'PR', 'RA', 'SB',
      'TOV', 'FTM', 'FGM', '3PM', 'FGA', '3PA', 'FTA',
      'PLUS_MINUS', 'TS%', 'USG', 'MIN'
  ]

  cumulative_avg = df_current_season[ADVANCED_COLS].mean()
  last5_avg = df_current_season[ADVANCED_COLS].tail(5).mean()
  last_matchup = df_grouped_by_matchup[ADVANCED_COLS].iloc[-1]
  last_matchup_avg = df_grouped_by_matchup[ADVANCED_COLS].mean()
  matchup_delta = last_matchup - cumulative_avg
  momentums = last5_avg - cumulative_avg

  for col in ADVANCED_COLS:
    features[f'CUMULATIVE_AVG_{col}'] = cumulative_avg[col]
    features[f'LAST5_AVG_{col}'] = last5_avg[col]
    features[f'LAST_MATCHUP_{col}'] = last_matchup[col]
    features[f'LAST_MATCHUP_AVG_{col}'] = last_matchup_avg[col]
    features[f'MATCHUP_{col}_DELTA'] = matchup_delta[col]
    features[f'{col}_MOMENTUM'] = momentums[col]


  std_last5 = df_current_season[STAT_COLS].tail(5).std(ddof=0)
  cum_std = df_current_season[STAT_COLS].std(ddof=0)
  std_spike_ratio = std_last5 / (cum_std + 1e-6)

  last10_vals = df_player[STAT_COLS].tail(10)
  last5_vals  = df_player[STAT_COLS].tail(5)

  for col in STAT_COLS:
    features[f'STD_LAST5_{col}'] = std_last5[col]
    features[f'CUMULATIVE_STD_{col}'] = cum_std[col]
    features[f'STD_SPIKE_RATIO_{col}'] = std_spike_ratio[col]

    line = float(features[f'PL_{col}'])
    last5_avg = features[f'LAST5_AVG_{col}']
    cum_avg = features[f'CUMULATIVE_AVG_{col}']
    std5 = std_last5[col]
    line_diff = line - last5_avg

    features[f'{col}_Z_LINE'] = np.clip((line - cum_avg) / (std5 + 1e-6), -6, 6)
    features[f'{col}_Z_RECENT'] = np.clip((line - last5_avg) / (std5 + 1e-6), -6, 6)
    features[f'CUM_AVG_{col}_PER_MIN'] = cum_avg / features['CUMULATIVE_AVG_MIN']
    features[f'LAST5_{col}_PER_MIN'] = last5_avg / features['LAST5_AVG_MIN']
    features[f'{col}_LINE_DIFF_X_MIN'] = line_diff * features['CUMULATIVE_AVG_MIN']
    features[f'{col}_MOMENTUM_X_VOL'] = features[f'{col}_MOMENTUM'] * std5
    features[f'MATCHUP_EASINESS_{col}'] = features[f'LAST_MATCHUP_AVG_{col}'] - cum_avg
    features[f'OVER_PL_RATE_{col}_LAST10'] = (last10_vals[col] > line).mean()
    features[f'OVER_PL_RATE_{col}_LAST5'] = (last5_vals[col] > line).mean()
    features[f'LINE_EDGE_{col}'] = abs(line - last5_avg) / (std5 + 1e-6)
    features[f'LINE_AMBIGUITY_{col}'] = (abs(features[f'{col}_Z_LINE']) < 0.25).astype(int)

  input = pd.DataFrame([features])
  input = input.applymap(
    lambda x:
        x.iloc[0] if isinstance(x, pd.Series)
        else x.isoformat() if isinstance(x, pd.Timestamp)
        else x
  )
  return input, df_player

