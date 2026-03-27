"""Model loading and prediction for the NBA player prop pipeline.

Replaces the FastAPI server: models are loaded once in-process via
load_models() and predictions are made via predict().
"""
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from config import (
  CHAMPION_FAMILIES,
  FAMILY_ORDER,
  CATEGORY_COLS,
  DROP_BASE_COLS,
  ALL_TARGETS,
  STAT_COLS,
  BETTING_THRESHOLDS,
  MODEL_ROOT,
  CATEGORY_MAPPINGS_PATH,
)
from odds import lookup_raw_odds, enrich_sportsbook_entries
from ranking import compute_rank_score

# ── Global registries (populated by load_models) ───────────
model_registry: Dict[str, Dict[str, object]] = {k: {} for k in CHAMPION_FAMILIES}
feature_registry: Dict[str, Dict[str, List[str]]] = {k: {} for k in CHAMPION_FAMILIES}
category_mappings: Dict[str, Dict[str, int]] = {}


# ── Model loading ──────────────────────────────────────────

def _get_model_filename(stat_name: str, feature_mode: str, calibration_set: str) -> str:
  return f"{stat_name}_{feature_mode.upper()}_{calibration_set}.pkl"


def get_loaded_stats() -> List[str]:
  stats = set()
  for family_key in CHAMPION_FAMILIES:
    stats.update(model_registry[family_key].keys())
  return sorted(stats)


def load_models(model_root: Path | None = None) -> None:
  """Load all model families and category mappings into global registries."""
  global category_mappings, model_registry, feature_registry

  root = model_root or MODEL_ROOT
  cat_path = root / "category_mappings.json"

  logger.info("=" * 70)
  logger.info("LOADING STAGE 4 CHAMPION MODEL PACKAGE")
  logger.info("=" * 70)

  with open(cat_path, "r") as f:
    category_mappings = json.load(f)

  logger.info("Loaded category mappings from: %s", cat_path)
  logger.info("  Players: %d  Teams: %d  Positions: %d  Matchups: %d",
    len(category_mappings.get('PLAYER_NAME', {})),
    len(category_mappings.get('TEAM', {})),
    len(category_mappings.get('POSITION', {})),
    len(category_mappings.get('MATCHUP', {})),
  )

  total_models_loaded = 0

  for family_key, cfg in CHAMPION_FAMILIES.items():
    family_root = root / cfg["folder"]
    features_path = family_root / "features" / cfg["features_file"]
    models_dir = family_root / "models"

    feature_payload = joblib.load(features_path)
    by_stat = feature_payload.get("by_stat", {})
    feature_registry[family_key] = by_stat

    loaded_here = 0
    logger.info("[%s] feature map: %s", family_key, features_path)

    for stat_name in sorted(by_stat.keys()):
      model_filename = _get_model_filename(
        stat_name=stat_name,
        feature_mode=cfg["feature_mode"],
        calibration_set=cfg["calibration_set"]
      )
      model_path = models_dir / model_filename

      if not model_path.exists():
        continue

      try:
        model_registry[family_key][stat_name] = joblib.load(model_path)
        loaded_here += 1
        total_models_loaded += 1
        logger.debug("  loaded %s -> %s", stat_name, model_filename)
      except Exception as e:
        logger.error("  failed %s -> %s", stat_name, e)

    logger.info("  total loaded in %s: %d", family_key, loaded_here)

  logger.info("=" * 70)
  logger.info("Total loaded models: %d", total_models_loaded)
  logger.info("Available stats: %s", get_loaded_stats())
  logger.info("=" * 70)


# ── Feature preparation ────────────────────────────────────

def apply_category_mappings(df: pd.DataFrame) -> pd.DataFrame:
  new_columns = {}
  for col in CATEGORY_COLS:
    if col in df.columns:
      new_columns[f"{col}_ID"] = (
        df[col]
        .astype(str)
        .map(category_mappings.get(col, {}))
        .fillna(-1)
        .astype(int)
      )
  if new_columns:
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
  return df


def build_feature_frame_inference(
  df: pd.DataFrame,
  target_stat: str,
  feature_mode: str,
) -> pd.DataFrame:
  cols_to_drop = list(DROP_BASE_COLS) + list(ALL_TARGETS)

  for s in STAT_COLS:
    if s != target_stat:
      cols_to_drop.extend([
        f"PL_{s}",
        f"OVER_PL_RATE_{s}_L10",
        f"OVER_PL_RATE_{s}_L5",
        f"{s}_Z_LINE",
        f"{s}_Z_RECENT",
        f"{s}_Z_MATCHUP",
        f"{s}_LINE_DIFF_X_MIN",
        f"{s}_LINE_DIFF",
        f"{s}_DIST_FROM_ANCHOR",
        f"{s}_ANCHOR"
      ])

  X = df.drop(columns=cols_to_drop, errors="ignore")

  if feature_mode == "reduced":
    drop_more = []
    for s in STAT_COLS:
      if s == target_stat:
        continue
      prefixes = [
        f"CUM_AVG_{s}",
        f"L5_AVG_{s}",
        f"STD_CUM_AVG_{s}",
        f"STD_L5_AVG_{s}",
        f"{s}_",
        f"LAST_MATCHUP_{s}",
        f"MATCHUP_L4_AVG_{s}",
        f"MATCHUP_L4_STD_{s}",
        f"CUM_AVG_{s}_PER_MIN",
        f"L5_{s}_PER_MIN"
      ]
      for p in prefixes:
        drop_more.extend([c for c in X.columns if c.startswith(p)])
      drop_more.extend([c for c in X.columns if c.startswith(f"OPP_ALLOWED_{s}_")])
      drop_more.extend([c for c in X.columns if c.startswith(f"MATCHUP_OPP_ALLOWED_{s}_")])

    if drop_more:
      drop_more = list(dict.fromkeys(drop_more))
      X = X.drop(columns=drop_more, errors="ignore")

  return X


def prepare_inference_frame(
  raw_df: pd.DataFrame,
  target_stat: str,
  feature_mode: str,
  expected_features: List[str],
) -> pd.DataFrame:
  df = raw_df.copy()
  df = apply_category_mappings(df)
  df = build_feature_frame_inference(df, target_stat=target_stat, feature_mode=feature_mode)
  df = df.reindex(columns=expected_features, fill_value=0)
  df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
  return df


def safe_float_from_frame(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
  if col not in df.columns or df.empty:
    return default
  val = pd.to_numeric(df.iloc[0][col], errors="coerce")
  return float(val) if pd.notna(val) else default


# ── Line override for alternate sportsbook lines ───────────

def _apply_line_override_to_feature_frame(
  base_frame: pd.DataFrame,
  stat_name: str,
  old_line: float | None,
  new_line: float,
) -> pd.DataFrame:
  """Approximate line-sensitive feature updates for alternate sportsbook lines."""
  df = base_frame.copy()
  if old_line is None:
    old_line = new_line
  delta = float(new_line) - float(old_line)
  if abs(delta) < 1e-12:
    return df

  pl_col = f"PL_{stat_name}"
  if pl_col in df.columns:
    df[pl_col] = float(new_line)

  line_diff_col = f"{stat_name}_LINE_DIFF"
  z_line_col = f"{stat_name}_Z_LINE"
  line_diff_x_min_col = f"{stat_name}_LINE_DIFF_X_MIN"
  anchor_col = f"{stat_name}_ANCHOR"
  dist_anchor_col = f"{stat_name}_DIST_FROM_ANCHOR"

  old_line_diff = None
  new_line_diff = None
  if line_diff_col in df.columns:
    old_line_diff = pd.to_numeric(df.iloc[0][line_diff_col], errors="coerce")
    if pd.notna(old_line_diff):
      new_line_diff = float(old_line_diff) - delta
      df[line_diff_col] = new_line_diff

  if z_line_col in df.columns:
    old_z = pd.to_numeric(df.iloc[0][z_line_col], errors="coerce")
    if pd.notna(old_z):
      if new_line_diff is not None and old_line_diff is not None and abs(float(old_z)) > 1e-12:
        std_est = float(old_line_diff) / float(old_z)
        if abs(std_est) > 1e-12:
          df[z_line_col] = float(new_line_diff) / std_est
      elif new_line_diff is not None and old_line_diff is not None and abs(float(old_line_diff)) > 1e-12:
        df[z_line_col] = float(old_z) * (float(new_line_diff) / float(old_line_diff))

  if line_diff_x_min_col in df.columns:
    min_col = "MIN"
    if new_line_diff is not None and min_col in df.columns:
      min_val = pd.to_numeric(df.iloc[0][min_col], errors="coerce")
      if pd.notna(min_val):
        df[line_diff_x_min_col] = float(new_line_diff) * float(min_val)
    elif old_line_diff is not None:
      old_ldxm = pd.to_numeric(df.iloc[0][line_diff_x_min_col], errors="coerce")
      if pd.notna(old_ldxm) and abs(float(old_line_diff)) > 1e-12 and new_line_diff is not None:
        df[line_diff_x_min_col] = float(old_ldxm) * (float(new_line_diff) / float(old_line_diff))

  if anchor_col in df.columns and dist_anchor_col in df.columns:
    anchor_val = pd.to_numeric(df.iloc[0][anchor_col], errors="coerce")
    if pd.notna(anchor_val):
      df[dist_anchor_col] = abs(float(new_line) - float(anchor_val))

  return df


def _compute_family_prob_over_for_line(
  *,
  family_key: str,
  stat_name: str,
  old_line: float | None,
  new_line: float,
  mode_base_frames: Dict[str, pd.DataFrame],
) -> float:
  family_cfg = CHAMPION_FAMILIES[family_key]
  model = model_registry[family_key].get(stat_name)
  expected_features = feature_registry[family_key].get(stat_name)
  if model is None or expected_features is None:
    raise ValueError(f"No model/features for family '{family_key}' and stat '{stat_name}'")

  mode = family_cfg["feature_mode"]
  override_frame = _apply_line_override_to_feature_frame(
    mode_base_frames[mode], stat_name=stat_name, old_line=old_line, new_line=new_line
  )
  x_frame = override_frame.reindex(columns=expected_features, fill_value=0)
  x_frame = x_frame.apply(pd.to_numeric, errors="coerce").fillna(0)
  return float(model.predict_proba(x_frame)[0, 1])


# ── Betting analysis helpers ───────────────────────────────

def _get_confidence_bucket(confidence: float) -> float:
  if confidence >= 0.70:
    return 0.70
  elif confidence >= 0.65:
    return 0.65
  elif confidence >= 0.60:
    return 0.60
  else:
    return 0.55


def _get_recommendation(confidence: float, min_conf: float, optimal_conf: float) -> str:
  midpoint = (min_conf + optimal_conf) / 2
  if confidence < min_conf:
    return "DO NOT BET"
  elif confidence < midpoint:
    return "BET WITH CAUTION"
  elif confidence < optimal_conf:
    return "BET"
  else:
    return "STRONG BET"


def _build_single_model_response(
  *,
  stat_name: str,
  player_name: str,
  parlay_line,
  family_key: str,
  x_frame: pd.DataFrame,
  prob: float,
  feature_count: int,
  sportsbook_odds: list | None = None,
  agreement_ratio: float = 1.0,
  probability_std: float = 0.0,
) -> dict:
  prediction = "OVER" if prob >= 0.5 else "UNDER"
  confidence = prob if prob >= 0.5 else (1.0 - prob)

  bet_info = BETTING_THRESHOLDS.get(stat_name)
  if bet_info is None:
    raise ValueError(f"No betting thresholds configured for stat '{stat_name}'")

  min_conf = bet_info["min_conf"]
  optimal_conf = bet_info["optimal_conf"]
  recommendation = _get_recommendation(confidence, min_conf, optimal_conf)

  bucket = _get_confidence_bucket(confidence)
  expected_acc_at_confidence = bet_info["realistic_acc"][bucket]

  z_line = safe_float_from_frame(x_frame, f"{stat_name}_Z_LINE")
  z_recent = safe_float_from_frame(x_frame, f"{stat_name}_Z_RECENT")
  z_matchup = safe_float_from_frame(x_frame, f"{stat_name}_Z_MATCHUP")
  line_diff = safe_float_from_frame(x_frame, f"{stat_name}_LINE_DIFF")
  momentum = safe_float_from_frame(x_frame, f"{stat_name}_MOMENTUM")
  last10_rate = safe_float_from_frame(x_frame, f"OVER_PL_RATE_{stat_name}_L10")
  last5_rate = safe_float_from_frame(x_frame, f"OVER_PL_RATE_{stat_name}_L5")

  rank_result = compute_rank_score(
    confidence=confidence,
    tier=bet_info["tier"],
    z_vs_line=z_line,
    z_vs_recent=z_recent,
    z_vs_matchup=z_matchup,
    line_diff=line_diff,
    momentum=momentum,
    prediction=prediction,
    last10_rate=last10_rate,
    last5_rate=last5_rate,
    agreement_ratio=agreement_ratio,
    probability_std=probability_std,
    sportsbook_odds=sportsbook_odds,
  )

  family_meta = CHAMPION_FAMILIES[family_key]

  return {
    "family": family_key,
    "feature_mode": family_meta["feature_mode"],
    "calibration_set": family_meta["calibration_set"],
    "feature_count": feature_count,
    "prediction": prediction,
    "model_output": round(float(prob), 4),
    "confidence": round(float(confidence), 4),
    "betting_analysis": {
      "recommendation": recommendation,
      "stat_tier": bet_info["tier"],
      "minimum_confidence": min_conf,
      "optimal_confidence": optimal_conf,
      "model_accuracy_at_confidence": f"Accuracy at {int(bucket * 100)}%+ confidence = {round(expected_acc_at_confidence * 100, 1)}%",
      "model_base_accuracy": f"{stat_name} Base Accuracy = {round(bet_info['base_acc'] * 100, 1)}%",
    },
    "Rank": rank_result
  }


# ── Market summary builder ─────────────────────────────────

def _build_market_summary(
  sportsbook_odds: list | None,
  prediction: str,
  dfs_line: float | None,
) -> dict:
  """Build a market consensus summary from enriched sportsbook entries."""
  summary = {
    "consensus_line": None,
    "consensus_prob_over": None,
    "consensus_prob_under": None,
    "best_over_odds": None,
    "best_over_book": None,
    "best_under_odds": None,
    "best_under_book": None,
    "dfs_vs_market": None,
    "books_with_data": 0,
  }
  if not sportsbook_odds:
    return summary

  lines = [e["line"] for e in sportsbook_odds if e.get("line") is not None]
  over_probs = [e["over_no_vig_prob"] for e in sportsbook_odds if e.get("over_no_vig_prob") is not None]
  under_probs = [e["under_no_vig_prob"] for e in sportsbook_odds if e.get("under_no_vig_prob") is not None]

  summary["books_with_data"] = len(sportsbook_odds)
  summary["consensus_line"] = round(float(np.median(lines)), 1) if lines else None
  summary["consensus_prob_over"] = round(float(np.mean(over_probs)), 4) if over_probs else None
  summary["consensus_prob_under"] = round(float(np.mean(under_probs)), 4) if under_probs else None

  best_over = max(sportsbook_odds, key=lambda e: e.get("over_decimal", 0), default=None)
  best_under = max(sportsbook_odds, key=lambda e: e.get("under_decimal", 0), default=None)
  if best_over and best_over.get("over_decimal"):
    summary["best_over_odds"] = best_over.get("over_american")
    summary["best_over_book"] = best_over.get("bookmaker_title")
  if best_under and best_under.get("under_decimal"):
    summary["best_under_odds"] = best_under.get("under_american")
    summary["best_under_book"] = best_under.get("bookmaker_title")

  if dfs_line is not None and summary["consensus_line"] is not None:
    delta = float(dfs_line) - summary["consensus_line"]
    if prediction == "OVER":
      if delta < -0.5:
        summary["dfs_vs_market"] = "FAVORABLE"
      elif delta > 0.5:
        summary["dfs_vs_market"] = "UNFAVORABLE"
      else:
        summary["dfs_vs_market"] = "NEUTRAL"
    else:
      if delta > 0.5:
        summary["dfs_vs_market"] = "FAVORABLE"
      elif delta < -0.5:
        summary["dfs_vs_market"] = "UNFAVORABLE"
      else:
        summary["dfs_vs_market"] = "NEUTRAL"

  return summary


# ── Main prediction function ───────────────────────────────

def predict(
  payload_dict: dict,
  stat_name: str,
  raw_sportsbook_odds: list | None = None,
  dfs_line: float | None = None,
) -> dict:
  """
  Run prediction for a single player+stat.

  Args:
    payload_dict: dict from get_input() (feature row)
    stat_name: e.g. "PTS", "REB"
    raw_sportsbook_odds: unenriched odds entries from odds.lookup_raw_odds()
    dfs_line: the DFS platform line value (for odds enrichment)

  Returns:
    Full prediction result dict (same structure as old /predict endpoint).
  """
  player_name = payload_dict.get("PLAYER_NAME", "Unknown")
  parlay_line = payload_dict.get(f"PL_{stat_name}")

  raw_df = pd.DataFrame([payload_dict])

  families_to_run = [fam for fam in FAMILY_ORDER if stat_name in model_registry[fam]]
  if not families_to_run:
    raise ValueError(f"No loaded model found for stat '{stat_name}'")

  mapped_df = apply_category_mappings(raw_df.copy())

  mode_base_frames = {}
  for fam in families_to_run:
    family_cfg = CHAMPION_FAMILIES[fam]
    mode = family_cfg["feature_mode"]
    if mode not in mode_base_frames:
      mode_base_frames[mode] = build_feature_frame_inference(
        mapped_df.copy(), target_stat=stat_name, feature_mode=mode
      )

  model_probs_by_family: Dict[str, float] = {}
  feature_count_by_family: Dict[str, int] = {}
  x_frames: Dict[str, pd.DataFrame] = {}

  for fam in families_to_run:
    family_cfg = CHAMPION_FAMILIES[fam]
    model = model_registry[fam].get(stat_name)
    expected_features = feature_registry[fam].get(stat_name)
    if model is None or expected_features is None:
      continue

    mode = family_cfg["feature_mode"]
    base_frame = mode_base_frames[mode]
    x_frame = base_frame.reindex(columns=expected_features, fill_value=0)
    x_frame = x_frame.apply(pd.to_numeric, errors="coerce").fillna(0)

    prob = float(model.predict_proba(x_frame)[0, 1])
    model_probs_by_family[fam] = prob
    feature_count_by_family[fam] = len(expected_features)
    x_frames[fam] = x_frame

  if not model_probs_by_family:
    raise ValueError(f"No usable model outputs for stat '{stat_name}'")

  all_probs = list(model_probs_by_family.values())
  consensus_prob_over = float(np.mean(all_probs))
  std_prob = float(np.std(all_probs)) if len(all_probs) > 1 else 0.0
  consensus_prediction = "OVER" if consensus_prob_over >= 0.5 else "UNDER"
  consensus_confidence = consensus_prob_over if consensus_prob_over >= 0.5 else (1.0 - consensus_prob_over)
  agreement_ratio = float(np.mean([
    1.0 if (
      (model_probs_by_family[fam] >= 0.5 and consensus_prediction == "OVER") or
      (model_probs_by_family[fam] < 0.5 and consensus_prediction == "UNDER")
    ) else 0.0
    for fam in model_probs_by_family
  ])) if len(model_probs_by_family) > 1 else 1.0

  dfs_line_value = float(dfs_line) if dfs_line is not None else (
    float(parlay_line) if parlay_line is not None else None
  )

  consensus_sportsbook_odds = None
  family_sportsbook_odds: Dict[str, list | None] = {fam: None for fam in model_probs_by_family}

  if raw_sportsbook_odds and dfs_line_value is not None:
    unique_lines = sorted({
      float(entry["line"]) for entry in raw_sportsbook_odds if entry.get("line") is not None
    })

    family_prob_over_by_line: Dict[str, Dict[float, float]] = {
      fam: {dfs_line_value: model_probs_by_family[fam]}
      for fam in model_probs_by_family
    }
    for line_value in unique_lines:
      if abs(line_value - dfs_line_value) < 0.01:
        continue
      for fam in model_probs_by_family:
        family_prob_over_by_line[fam][line_value] = _compute_family_prob_over_for_line(
          family_key=fam,
          stat_name=stat_name,
          old_line=dfs_line_value,
          new_line=line_value,
          mode_base_frames=mode_base_frames,
        )

    consensus_prob_over_by_line = {}
    for line_value in set(unique_lines + [dfs_line_value]):
      probs_here = [
        family_prob_over_by_line[fam].get(line_value, model_probs_by_family[fam])
        for fam in model_probs_by_family
      ]
      consensus_prob_over_by_line[line_value] = float(np.mean(probs_here))

    consensus_sportsbook_odds = enrich_sportsbook_entries(
      raw_entries=raw_sportsbook_odds,
      model_prob_over_dfs=consensus_prob_over,
      model_prob_over_by_line=consensus_prob_over_by_line,
      dfs_line=dfs_line_value,
    )

    for fam in model_probs_by_family:
      family_sportsbook_odds[fam] = enrich_sportsbook_entries(
        raw_entries=raw_sportsbook_odds,
        model_prob_over_dfs=model_probs_by_family[fam],
        model_prob_over_by_line=family_prob_over_by_line[fam],
        dfs_line=dfs_line_value,
      )

  finalized_model_outputs = {}
  for fam in model_probs_by_family:
    finalized_model_outputs[fam] = _build_single_model_response(
      stat_name=stat_name,
      player_name=player_name,
      parlay_line=parlay_line,
      family_key=fam,
      x_frame=x_frames[fam],
      prob=model_probs_by_family[fam],
      feature_count=feature_count_by_family[fam],
      sportsbook_odds=family_sportsbook_odds.get(fam),
      agreement_ratio=agreement_ratio,
      probability_std=std_prob,
    )

  bet_info = BETTING_THRESHOLDS[stat_name]
  min_conf = bet_info["min_conf"]
  optimal_conf = bet_info["optimal_conf"]
  recommendation = _get_recommendation(consensus_confidence, min_conf, optimal_conf)

  bucket = _get_confidence_bucket(consensus_confidence)
  expected_acc_at_confidence = bet_info["realistic_acc"][bucket]

  ref_family = list(x_frames.keys())[0]
  ref_frame = x_frames[ref_family]

  z_line = safe_float_from_frame(ref_frame, f"{stat_name}_Z_LINE")
  z_recent = safe_float_from_frame(ref_frame, f"{stat_name}_Z_RECENT")
  z_matchup = safe_float_from_frame(ref_frame, f"{stat_name}_Z_MATCHUP")
  line_diff = safe_float_from_frame(ref_frame, f"{stat_name}_LINE_DIFF")
  momentum = safe_float_from_frame(ref_frame, f"{stat_name}_MOMENTUM")
  last10_rate = safe_float_from_frame(ref_frame, f"OVER_PL_RATE_{stat_name}_L10")
  last5_rate = safe_float_from_frame(ref_frame, f"OVER_PL_RATE_{stat_name}_L5")

  rank_result = compute_rank_score(
    confidence=consensus_confidence,
    tier=bet_info["tier"],
    z_vs_line=z_line,
    z_vs_recent=z_recent,
    z_vs_matchup=z_matchup,
    line_diff=line_diff,
    momentum=momentum,
    prediction=consensus_prediction,
    last10_rate=last10_rate,
    last5_rate=last5_rate,
    agreement_ratio=agreement_ratio,
    probability_std=std_prob,
    sportsbook_odds=consensus_sportsbook_odds,
  )

  result = {
    "stat": stat_name,
    "player": player_name,
    "parlay_line": parlay_line,
    "consensus": {
      "prediction": consensus_prediction,
      "model_output_avg": round(consensus_prob_over, 4),
      "confidence": round(consensus_confidence, 4),
      "probability_std": round(std_prob, 4),
      "agreement_ratio": round(agreement_ratio, 4),
      "families_used": list(finalized_model_outputs.keys()),
      "betting_analysis": {
        "recommendation": recommendation,
        "stat_tier": bet_info["tier"],
        "minimum_confidence": min_conf,
        "optimal_confidence": optimal_conf,
        "model_accuracy_at_confidence": f"Accuracy at {int(bucket * 100)}%+ confidence = {round(expected_acc_at_confidence * 100, 1)}%",
        "model_base_accuracy": f"{stat_name} Base Accuracy = {round(bet_info['base_acc'] * 100, 1)}%"
      },
      "Rank": rank_result
    },
    "model_variants": finalized_model_outputs
  }

  if consensus_sportsbook_odds is not None:
    result["sportsbook_odds"] = consensus_sportsbook_odds
    result["market_summary"] = _build_market_summary(
      consensus_sportsbook_odds, consensus_prediction, dfs_line_value
    )

  return result
