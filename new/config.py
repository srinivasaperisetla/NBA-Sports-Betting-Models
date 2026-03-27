"""Single source of truth for all constants used across the pipeline."""
from pathlib import Path

# ── Seasons ────────────────────────────────────────────────
SEASONS = ["2024-25", "2025-26"]

CHAMPION_API_URL = "http://127.0.0.1:8000/predict"

# ── Model root (relative to this file) ────────────────────
MODEL_ROOT = Path(__file__).parent / "models"
CATEGORY_MAPPINGS_PATH = MODEL_ROOT / "category_mappings.json"

# ── Stat columns ───────────────────────────────────────────
STAT_COLS = [
  "PTS", "REB", "AST", "STL", "BLK",
  "PRA", "PA", "PR", "RA", "SB",
  "TOV", "FTA", "FTM", "FGA", "FGM", "3PM", "3PA"
]

TARGET_COLUMNS = [
  'PTS', 'REB', 'AST', 'STL', 'BLK',
  'PRA', 'PA', 'PR', 'RA', 'SB',
  'TOV', 'FGM', '3PM', 'FTM', 'FGA', '3PA', 'FTA'
]

ALL_TARGETS = [f"TARGET_{s}" for s in STAT_COLS]

CATEGORY_COLS = ["PLAYER_NAME", "TEAM", "POSITION", "MATCHUP"]

DROP_BASE_COLS = [
  "PLAYER_NAME", "PLAYER_ID", "TEAM", "MATCHUP", "POSITION",
  "SEASON_YEAR", "SEASON_ID", "GAME_DATE", "GAME_ID",
  "PTS", "REB", "AST", "STL", "BLK", "PRA", "PA", "PR", "RA", "SB",
  "TOV", "FTM", "FGM", "3PM", "FGA", "3PA", "FTA",
  "MIN", "PLUS_MINUS", "TS%", "USG", "OFF_RATING"
]

# ── Model families ─────────────────────────────────────────
CHAMPION_FAMILIES = {
  "FULL_ALL": {
    "folder": "CHAMPION_FULL_ALL",
    "feature_mode": "full",
    "calibration_set": "ALL",
    "features_file": "FEATURES_FULL_ALL.pkl",
  },
  "FULL_TIGHT": {
    "folder": "CHAMPION_FULL_TIGHT",
    "feature_mode": "full",
    "calibration_set": "TIGHT",
    "features_file": "FEATURES_FULL_TIGHT.pkl",
  },
  "REDUCED_ALL": {
    "folder": "CHAMPION_REDUCED_ALL",
    "feature_mode": "reduced",
    "calibration_set": "ALL",
    "features_file": "FEATURES_REDUCED_ALL.pkl",
  },
  "REDUCED_TIGHT": {
    "folder": "CHAMPION_REDUCED_TIGHT",
    "feature_mode": "reduced",
    "calibration_set": "TIGHT",
    "features_file": "FEATURES_REDUCED_TIGHT.pkl",
  },
}

FAMILY_ORDER = ["FULL_ALL", "FULL_TIGHT", "REDUCED_ALL", "REDUCED_TIGHT"]

# ── Betting thresholds ─────────────────────────────────────
BETTING_THRESHOLDS = {
  "BLK": {"recommended_family": "REDUCED_TIGHT", "tier": "S", "min_conf": 0.55, "optimal_conf": 0.70, "base_acc": 0.7587, "realistic_acc": {0.55: 0.7799, 0.60: 0.8044, 0.65: 0.8132, 0.70: 0.8279}},
  "3PM": {"recommended_family": "FULL_ALL", "tier": "S", "min_conf": 0.60, "optimal_conf": 0.70, "base_acc": 0.6950, "realistic_acc": {0.55: 0.7049, 0.60: 0.7383, 0.65: 0.7663, 0.70: 0.8034}},
  "STL": {"recommended_family": "REDUCED_ALL", "tier": "A", "min_conf": 0.60, "optimal_conf": 0.70, "base_acc": 0.7095, "realistic_acc": {0.55: 0.7119, 0.60: 0.7327, 0.65: 0.7576, 0.70: 0.7768}},
  "TOV": {"recommended_family": "FULL_ALL", "tier": "A", "min_conf": 0.60, "optimal_conf": 0.70, "base_acc": 0.6966, "realistic_acc": {0.55: 0.6914, 0.60: 0.7220, 0.65: 0.7521, 0.70: 0.7732}},
  "SB": {"recommended_family": "REDUCED_ALL", "tier": "A", "min_conf": 0.60, "optimal_conf": 0.70, "base_acc": 0.6919, "realistic_acc": {0.55: 0.6929, 0.60: 0.7085, 0.65: 0.7322, 0.70: 0.7585}},
  "3PA": {"recommended_family": "FULL_ALL", "tier": "B", "min_conf": 0.60, "optimal_conf": 0.65, "base_acc": 0.6798, "realistic_acc": {0.55: 0.6569, 0.60: 0.6909, 0.65: 0.7275, 0.70: 0.7635}},
  "FTM": {"recommended_family": "FULL_ALL", "tier": "B", "min_conf": 0.60, "optimal_conf": 0.65, "base_acc": 0.6816, "realistic_acc": {0.55: 0.6576, 0.60: 0.6859, 0.65: 0.7137, 0.70: 0.7408}},
  "REB": {"recommended_family": "FULL_ALL", "tier": "B", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6773, "realistic_acc": {0.55: 0.6415, 0.60: 0.6748, 0.65: 0.7140, 0.70: 0.7339}},
  "FGM": {"recommended_family": "FULL_ALL", "tier": "C", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6857, "realistic_acc": {0.55: 0.6447, 0.60: 0.6794, 0.65: 0.7196, 0.70: 0.7536}},
  "FTA": {"recommended_family": "FULL_ALL", "tier": "C", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6726, "realistic_acc": {0.55: 0.6250, 0.60: 0.6577, 0.65: 0.6863, 0.70: 0.7213}},
  "AST": {"recommended_family": "FULL_ALL", "tier": "C", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6679, "realistic_acc": {0.55: 0.6362, 0.60: 0.6642, 0.65: 0.6876, 0.70: 0.7119}},
  "RA": {"recommended_family": "FULL_ALL", "tier": "C", "min_conf": 0.65, "optimal_conf": 0.70, "base_acc": 0.6750, "realistic_acc": {0.55: 0.6333, 0.60: 0.6642, 0.65: 0.6925, 0.70: 0.7398}},
  "PTS": {"recommended_family": "REDUCED_ALL", "tier": "D", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6747, "realistic_acc": {0.55: 0.6251, 0.60: 0.6619, 0.65: 0.6934, 0.70: 0.7313}},
  "PA": {"recommended_family": "FULL_ALL", "tier": "D", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6759, "realistic_acc": {0.55: 0.6299, 0.60: 0.6627, 0.65: 0.7050, 0.70: 0.7344}},
  "PR": {"recommended_family": "FULL_ALL", "tier": "D", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6747, "realistic_acc": {0.55: 0.6309, 0.60: 0.6625, 0.65: 0.6884, 0.70: 0.7089}},
  "PRA": {"recommended_family": "FULL_ALL", "tier": "D", "min_conf": 0.65, "optimal_conf": 0.65, "base_acc": 0.6730, "realistic_acc": {0.55: 0.6326, 0.60: 0.6749, 0.65: 0.6945, 0.70: 0.7314}},
  "FGA": {"recommended_family": "FULL_ALL", "tier": "D", "min_conf": 0.70, "optimal_conf": 0.70, "base_acc": 0.6797, "realistic_acc": {0.55: 0.6247, 0.60: 0.6643, 0.65: 0.7111, 0.70: 0.7493}}
}

TIER_POINTS = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0}

# ── New ranking weights (0–100 composite) ──────────────────
BREAKEVEN_PROB = 0.54  # breakeven for standard -110 bet
RANK_WEIGHTS = {
  "model_confidence": 0.35,
  "market_edge": 0.30,
  "model_alpha": 0.25,
  "family_agreement": 0.05,
  "statistical_signals": 0.05,
}

# ── Logging ────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"

# ── Odds API: stat → market mapping ───────────────────────
STAT_TO_MARKET = {
  "PTS":  "player_points",
  "REB":  "player_rebounds",
  "AST":  "player_assists",
  "3PM":  "player_threes",
  "BLK":  "player_blocks",
  "STL":  "player_steals",
  "SB":   "player_blocks_steals",
  "TOV":  "player_turnovers",
  "PRA":  "player_points_rebounds_assists",
  "PR":   "player_points_rebounds",
  "PA":   "player_points_assists",
  "RA":   "player_rebounds_assists",
  "FGM":  "player_field_goals",
  "FTM":  "player_frees_made",
  "FTA":  "player_frees_attempts",
  # FGA and 3PA don't have standard Odds API markets
}

# ── Platform stat maps ─────────────────────────────────────
UNDERDOG_STAT_MAP = {
  "Points": "PTS",
  "Rebounds": "REB",
  "Assists": "AST",
  "Steals": "STL",
  "Blocks": "BLK",
  "Pts + Rebs + Asts": "PRA",
  "Pts+Rebs+Asts": "PRA",
  "Pts+Reb+Ast": "PRA",
  "Points + Assists": "PA",
  "Points+Assists": "PA",
  "Assists+Points": "PA",
  "Assists + Points": "PA",
  "Points + Rebounds": "PR",
  "Points+Rebounds": "PR",
  "Rebounds + Points": "PR",
  "Rebounds+Points": "PR",
  "Rebounds + Assists": "RA",
  "Rebounds+Assists": "RA",
  "Assists + Rebounds": "RA",
  "Assists+Rebounds": "RA",
  "Blocks + Steals": "SB",
  "Blocks+Steals": "SB",
  "Steals + Blocks": "SB",
  "Steals+Blocks": "SB",
  "Turnovers": "TOV",
  "FG Made": "FGM",
  "3-Pointers Made": "3PM",
  "3PM": "3PM",
  "FT Made": "FTM",
  "FG Attempted": "FGA",
  "3s Attempted": "3PA",
  "3PT Attempted": "3PA",
  "FT Attempted": "FTA",
}

PRIZEPICKS_STAT_MAP = {
  "Points":         "PTS",
  "Rebounds":       "REB",
  "Assists":        "AST",
  "Steals":         "STL",
  "Blocked Shots":  "BLK",
  "PRA":            "PRA",
  "Pts+Asts":       "PA",
  "Pts+Rebs":       "PR",
  "Rebs+Asts":      "RA",
  "Blks+Stls":      "SB",
  "Turnovers":      "TOV",
  "FG Made":        "FGM",
  "3PTM":           "3PM",
  "FTM":            "FTM",
  "FG Attempted":   "FGA",
  "3PTA":           "3PA",
  "FTA":            "FTA",
}
