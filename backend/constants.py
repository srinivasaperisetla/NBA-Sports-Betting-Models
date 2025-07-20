from typing import Literal

ALLOWED_POSITIONS = Literal[
  'Guard',
  'Forward',
]

ALLOWED_TEAMS = Literal[
  'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET',
  'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN',
  'NOP', 'NYK', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS',
  'TOR', 'UTA', 'WAS'
]

ALLOWED_PLAYERS = Literal[
  'LeBron James',
  'Stephen Curry',
  'Kevin Durant',
  'Kyrie Irving',
  'Anthony Edwards',
  'Shai Gilgeous-Alexander',
  'Giannis Antetokounmpo'
]

TARGET_COLUMNS = [
  'PTS', 'REB', 'AST', 'STL', 'BLK',
  'TOV', 'FTM', 'FGM', '3PM', 'FS',
  'PRA', 'PA', 'PR', 'RA', 'SB'
]

FEATURES_TO_DUMMY = [
  'PLAYER_NAME',
  'POSITION',
  'TEAM',
  'MATCHUP',
]