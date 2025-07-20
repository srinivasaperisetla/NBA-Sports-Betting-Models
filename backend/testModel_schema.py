from pydantic import BaseModel, Field
from constants import ALLOWED_PLAYERS, ALLOWED_POSITIONS, ALLOWED_TEAMS

class TestModelPredictionInput(BaseModel):
  PLAYER_NAME: ALLOWED_PLAYERS = Field(...)
  POSITION: ALLOWED_POSITIONS = Field(...)
  HEIGHT: int = Field(..., description="Players height in inches")
  WEIGHT: int = Field(..., description="Players weight in pounds")

  HOME: int = Field(..., description="1 if home game, 0 if away game", ge=0, le=1)
  POSTSEASON: int = Field(...,  description="1 if playoff game, 0 if in season game", ge=0, le=1)
  BACK_TO_BACK: int = Field(..., description="1 playing back to back, 0 other wise", ge=0, le=1)

  TEAM: ALLOWED_TEAMS = Field(...)
  MATCHUP: ALLOWED_TEAMS = Field(...)

  PER_GROUPED: float = Field(
    ...,
    description = "Player Efficiency Rating grouped by MATCHUP. " \
    "Calculated as: ((CUMULATIVE_PTS + REB + AST + STL + BLK) - (missed FGs + missed FTs + TOV)) / (Total Games Played against Matchup)" \
    "Calculate the cumulative stats required for every game against the specific matchup in the current season" \
    "Cumulative stats are aggregated per matchup."\
    "If it's the first matchup of the season, forward-fill with the last PER_GROUPED value from the previous season." \
  )
  ORtg_GROUPED: float = Field(
    ...,
    description = "Offensive Rating grouped by MATCHUP. " \
    "Calculated as the cumulative mean of: 100 * (PTS_PRODUCED / PLAYER_POSSESSIONS), " \
    "where PTS_PRODUCED = PTS + (AST * 2). " \
    "Stats are aggregated per matchup. " \
    "Forward-fill with the last ORtg_GROUPED value from the previous season if this is the first game against that team."
  )
  PLUS_MINUS_GROUPED: float = Field(
    ...,
    description = "Rolling average of PLUS_MINUS grouped by MATCHUP. " \
    "Calculated as the expanding mean of PLUS_MINUS from past games against the same team. " \
    "If it's the first matchup of the season, forward-fill with the last PLUS_MINUS_GROUPED from the previous season."
  )

  PL_PTS: float = Field(0, description="Player's parlay for points")
  PL_REB: float = Field(0, description="Player's parlay for rebounds")
  PL_AST: float = Field(0, description="Player's parlay for assists")
  PL_STL: float = Field(0, description="Player's parlay for steals")
  PL_BLK: float = Field(0, description="Player's parlay for blocks")
  PL_TOV: float = Field(0, description="Player's parlay for turnovers")
  PL_FTM: float = Field(0, description="Player's parlay for free throws made")
  PL_FGM: float = Field(0, description="Player's parlay for field goals made")
  PL_3PM: float = Field(0, description="Player's parlay for 3-point shots made")
  PL_FS: float = Field(0, description="Player's parlay for fantasy score")
  PL_PRA: float = Field(0, description="Player's parlay for points + rebounds + assists")
  PL_PA: float = Field(0, description="Player's parlay for points + assists")
  PL_PR: float = Field(0, description="Player's parlay for points + rebounds")
  PL_RA: float = Field(0, description="Player's parlay for rebounds + assists")
  PL_SB: float = Field(0, description="Player's parlay for steals + blocks")

  CUMULATIVE_AVG_MIN: float = Field(...)
  CUMULATIVE_AVG_PTS: float = Field(...)
  CUMULATIVE_AVG_REB: float = Field(...)
  CUMULATIVE_AVG_AST: float = Field(...)
  CUMULATIVE_AVG_STL: float = Field(...)
  CUMULATIVE_AVG_BLK: float = Field(...)
  CUMULATIVE_AVG_TOV: float = Field(...)
  CUMULATIVE_AVG_FTM: float = Field(...)
  CUMULATIVE_AVG_FGM: float = Field(...)
  CUMULATIVE_AVG_3PM: float = Field(...)
  CUMULATIVE_AVG_FS: float = Field(...)
  CUMULATIVE_AVG_PRA: float = Field(...)
  CUMULATIVE_AVG_PA: float = Field(...)
  CUMULATIVE_AVG_PR: float = Field(...)
  CUMULATIVE_AVG_RA: float = Field(...)
  CUMULATIVE_AVG_SB: float = Field(...)
  CUMULATIVE_AVG_PLUS_MINUS: float = Field(...)
  CUMULATIVE_AVG_TS: float = Field(..., alias="CUMULATIVE_AVG_TS%")
  CUMULATIVE_AVG_USG: float = Field(..., alias="CUMULATIVE_AVG_USG%")

  LAST5_AVG_MIN: float = Field(...)
  LAST5_AVG_PTS: float = Field(...)
  LAST5_AVG_REB: float = Field(...)
  LAST5_AVG_AST: float = Field(...)
  LAST5_AVG_STL: float = Field(...)
  LAST5_AVG_BLK: float = Field(...)
  LAST5_AVG_TOV: float = Field(...)
  LAST5_AVG_FTM: float = Field(...)
  LAST5_AVG_FGM: float = Field(...)
  LAST5_AVG_3PM: float = Field(...)
  LAST5_AVG_FS: float = Field(...)
  LAST5_AVG_PRA: float = Field(...)
  LAST5_AVG_PA: float = Field(...)
  LAST5_AVG_PR: float = Field(...)
  LAST5_AVG_RA: float = Field(...)
  LAST5_AVG_SB: float = Field(...)
  LAST5_AVG_PLUS_MINUS: float = Field(...)
  LAST5_AVG_TS: float = Field(..., alias="LAST5_AVG_TS%")
  LAST5_AVG_USG: float = Field(..., alias="LAST5_AVG_USG%")

  LAST_MATCHUP_MIN: float = Field(...)
  LAST_MATCHUP_PTS: float = Field(...)
  LAST_MATCHUP_REB: float = Field(...)
  LAST_MATCHUP_AST: float = Field(...)
  LAST_MATCHUP_STL: float = Field(...)
  LAST_MATCHUP_BLK: float = Field(...)
  LAST_MATCHUP_TOV: float = Field(...)
  LAST_MATCHUP_FTM: float = Field(...)
  LAST_MATCHUP_FGM: float = Field(...)
  LAST_MATCHUP_3PM: float = Field(...)
  LAST_MATCHUP_FS: float = Field(...)
  LAST_MATCHUP_PRA: float = Field(...)
  LAST_MATCHUP_PA: float = Field(...)
  LAST_MATCHUP_PR: float = Field(...)
  LAST_MATCHUP_RA: float = Field(...)
  LAST_MATCHUP_SB: float = Field(...)
  LAST_MATCHUP_PLUS_MINUS: float = Field(...)
  LAST_MATCHUP_TS: float = Field(..., alias="LAST_MATCHUP_TS%")
  LAST_MATCHUP_USG: float = Field(..., alias="LAST_MATCHUP_USG%")

  LAST_MATCHUP_AVG_MIN: float = Field(...)
  LAST_MATCHUP_AVG_PTS: float = Field(...)
  LAST_MATCHUP_AVG_REB: float = Field(...)
  LAST_MATCHUP_AVG_AST: float = Field(...)
  LAST_MATCHUP_AVG_STL: float = Field(...)
  LAST_MATCHUP_AVG_BLK: float = Field(...)
  LAST_MATCHUP_AVG_TOV: float = Field(...)
  LAST_MATCHUP_AVG_FTM: float = Field(...)
  LAST_MATCHUP_AVG_FGM: float = Field(...)
  LAST_MATCHUP_AVG_3PM: float = Field(...)
  LAST_MATCHUP_AVG_FS: float = Field(...)
  LAST_MATCHUP_AVG_PRA: float = Field(...)
  LAST_MATCHUP_AVG_PA: float = Field(...)
  LAST_MATCHUP_AVG_PR: float = Field(...)
  LAST_MATCHUP_AVG_RA: float = Field(...)
  LAST_MATCHUP_AVG_SB: float = Field(...)
  LAST_MATCHUP_AVG_PLUS_MINUS: float = Field(...)
  LAST_MATCHUP_AVG_TS: float = Field(..., alias="LAST_MATCHUP_AVG_TS%")
  LAST_MATCHUP_AVG_USG: float = Field(..., alias="LAST_MATCHUP_AVG_USG%")

  OVER_PL_PTS_LAST10: float = Field(...)
  OVER_PL_REB_LAST10: float = Field(...)
  OVER_PL_AST_LAST10: float = Field(...)
  OVER_PL_STL_LAST10: float = Field(...)
  OVER_PL_BLK_LAST10: float = Field(...)
  OVER_PL_TOV_LAST10: float = Field(...)
  OVER_PL_FTM_LAST10: float = Field(...)
  OVER_PL_FGM_LAST10: float = Field(...)
  OVER_PL_3PM_LAST10: float = Field(...)
  OVER_PL_FS_LAST10: float = Field(...)
  OVER_PL_PRA_LAST10: float = Field(...)
  OVER_PL_PA_LAST10: float = Field(...)
  OVER_PL_PR_LAST10: float = Field(...)
  OVER_PL_RA_LAST10: float = Field(...)
  OVER_PL_SB_LAST10: float = Field(...)