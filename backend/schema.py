import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic import BaseModel, Field, create_model, ConfigDict
from enum import Enum
from constants import ALLOWED_PLAYERS, ALLOWED_POSITIONS, ALLOWED_TEAMS
from stat_utils import ADVANCED_COLS, STAT_COLS, MATCHUP_ALLOWED_METRICS, MATCHUP_ALLOWED_METRICS_W_PACE_DEF
from typing import List

class Stat(str, Enum):
  PTS = "PTS"
  REB = "REB"
  AST = "AST"
  STL = "STL"
  BLK = "BLK"
  PRA = "PRA"
  PA = "PA" 
  PR = "PR"
  RA = "RA"
  SB = "SB"
  TOV = "TOV"
  FGM = "FGM"
  FGA = "FGA"
  FTM = "FTM"
  FTA = "FTA"
  PM3 = "3PM"
  PA3 = "3PA"

def generate_field_columns(prefix: str, columns: List[str]):
  """Generate field definitions for pydantic model"""
  return {f"{prefix}_{col}": (float, Field(...)) for col in columns}

BASE_FIELDS = {
  'PLAYER_NAME': (ALLOWED_PLAYERS, Field(...)),
  'POSITION': (ALLOWED_POSITIONS, Field(...)),
  'HEIGHT': (int, Field(...)),
  'WEIGHT': (int, Field(...)),
  'HOME': (int, Field(..., ge=0, le=1)),
  'POSTSEASON': (int, Field(..., ge=0, le=1)),
  'BACK_TO_BACK': (int, Field(..., ge=0, le=1)),
  'PLAYER_REST_DAYS': (float, Field(...)),
  'GAMES_L3_DAYS': (int, Field(...)),
  'GAMES_L7_DAYS': (int, Field(...)),
  'TEAM': (ALLOWED_TEAMS, Field(...)),
  'MATCHUP': (ALLOWED_TEAMS, Field(...)),
  'PER_GROUPED': (float, Field(...)),
  'OFF_RATING_GROUPED': (float, Field(...)),
  'PLUS_MINUS_GROUPED': (float, Field(...)),
  'GP_AGAINST_TEAM': (int, Field(...)),
  'FATIGUE_FACTOR': (float, Field(...)),
}

OPPONENT_BASIC_FIELDS = {
  'PACE_IMPACT_POSS': (float, Field(...)),
  'AST_VULN_RATIO': (float, Field(...)),
  'REB_VULN_RATIO': (float, Field(...)),
  'DEF_VS_VOL': (float, Field(...)),
}

OPP_ALLOWED_L5_FIELDS = {f'OPP_ALLOWED_{m}_L5_AVG': (float, Field(...)) for m in MATCHUP_ALLOWED_METRICS}
OPP_ALLOWED_CUM_FIELDS = {f'OPP_ALLOWED_{m}_CUM_AVG': (float, Field(...)) for m in MATCHUP_ALLOWED_METRICS}

OPP_PACE_DEF_FIELDS = {
  'OPP_PACE_L5_AVG': (float, Field(...)),
  'OPP_PACE_CUM_AVG': (float, Field(...)),
  'OPP_DEF_RATING_L5_AVG': (float, Field(...)),
  'OPP_DEF_RATING_CUM_AVG': (float, Field(...)),
}

MATCHUP_OPP_ALLOWED_FIELDS = {f'MATCHUP_OPP_ALLOWED_{m}_L4': (float, Field(...)) for m in MATCHUP_ALLOWED_METRICS_W_PACE_DEF}

PL_FIELDS = generate_field_columns('PL', STAT_COLS)

CUM_AVG_FIELDS = generate_field_columns('CUM_AVG', ADVANCED_COLS)
L5_AVG_FIELDS = generate_field_columns('L5_AVG', ADVANCED_COLS)
STD_CUM_AVG_FIELDS = generate_field_columns('STD_CUM_AVG', ADVANCED_COLS)
STD_L5_AVG_FIELDS = generate_field_columns('STD_L5_AVG', ADVANCED_COLS)
MOMENTUM_FIELDS = {f'{col}_MOMENTUM': (float, Field(...)) for col in ADVANCED_COLS}
LAST_MATCHUP_FIELDS = generate_field_columns('LAST_MATCHUP', ADVANCED_COLS)
MATCHUP_L4_AVG_FIELDS = generate_field_columns('MATCHUP_L4_AVG', ADVANCED_COLS)
MATCHUP_L4_STD_FIELDS = generate_field_columns('MATCHUP_L4_STD', ADVANCED_COLS)

LINE_DIFF_FIELDS = {f'{col}_LINE_DIFF': (float, Field(...)) for col in STAT_COLS}
Z_LINE_FIELDS = {f'{col}_Z_LINE': (float, Field(...)) for col in STAT_COLS}
Z_RECENT_FIELDS = {f'{col}_Z_RECENT': (float, Field(...)) for col in STAT_COLS}
Z_MATCHUP_FIELDS = {f'{col}_Z_MATCHUP': (float, Field(...)) for col in STAT_COLS}
ANCHOR_FIELDS = {f'{col}_ANCHOR': (float, Field(...)) for col in STAT_COLS}
DIST_FROM_ANCHOR_FIELDS = {f'{col}_DIST_FROM_ANCHOR': (float, Field(...)) for col in STAT_COLS}

CUM_AVG_PER_MIN_FIELDS = {f'CUM_AVG_{col}_PER_MIN': (float, Field(...)) for col in STAT_COLS}
L5_PER_MIN_FIELDS = {f'L5_{col}_PER_MIN': (float, Field(...)) for col in STAT_COLS}

LINE_DIFF_X_MIN_FIELDS = {f'{col}_LINE_DIFF_X_MIN': (float, Field(...)) for col in STAT_COLS}
MOMENTUM_X_VOL_FIELDS = {f'{col}_MOMENTUM_X_VOL': (float, Field(...)) for col in STAT_COLS}

OVER_PL_RATE_L10_FIELDS = {f'OVER_PL_RATE_{col}_L10': (float, Field(...)) for col in STAT_COLS}
OVER_PL_RATE_L5_FIELDS = {f'OVER_PL_RATE_{col}_L5': (float, Field(...)) for col in STAT_COLS}


ALL_FIELDS = {
  **BASE_FIELDS,
  **OPPONENT_BASIC_FIELDS,
  **OPP_ALLOWED_L5_FIELDS,
  **OPP_ALLOWED_CUM_FIELDS,
  **OPP_PACE_DEF_FIELDS,
  **MATCHUP_OPP_ALLOWED_FIELDS,
  **PL_FIELDS,
  **CUM_AVG_FIELDS,
  **L5_AVG_FIELDS,
  **STD_CUM_AVG_FIELDS,
  **STD_L5_AVG_FIELDS,
  **MOMENTUM_FIELDS,
  **LAST_MATCHUP_FIELDS,
  **MATCHUP_L4_AVG_FIELDS,
  **MATCHUP_L4_STD_FIELDS,
  **LINE_DIFF_FIELDS,
  **Z_LINE_FIELDS,
  **Z_RECENT_FIELDS,
  **Z_MATCHUP_FIELDS,
  **ANCHOR_FIELDS,
  **DIST_FROM_ANCHOR_FIELDS,
  **CUM_AVG_PER_MIN_FIELDS,
  **L5_PER_MIN_FIELDS,
  **LINE_DIFF_X_MIN_FIELDS,
  **MOMENTUM_X_VOL_FIELDS,
  **OVER_PL_RATE_L10_FIELDS,
  **OVER_PL_RATE_L5_FIELDS,
}


NBAPredictionInput = create_model(
  'NBAPredictionInput',
  __config__=ConfigDict(extra='allow'),
  **ALL_FIELDS
)

print(NBAPredictionInput.model_json_schema())


