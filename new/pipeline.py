"""Main pipeline: parse lines -> get features -> predict -> rank -> CSV.

Replaces scraper.py's HTTP-based run_pipeline(). Models are loaded
in-process via models.load_models(); no FastAPI server required.
"""
import logging
import sys

import pandas as pd

import models
import odds as odds_module
from config import SEASONS, TARGET_COLUMNS, MODEL_ROOT, LOG_FORMAT

logger = logging.getLogger(__name__)
from parsers import (
  parse_underdog_txt,
  parse_prizepicks_txt,
  classify_player,
  build_combined_parlays,
  build_platform_parlays,
)


# ── Formatting helpers ─────────────────────────────────────

def _pct_or_blank(x):
  if x is None or (isinstance(x, float) and pd.isna(x)):
    return ""
  return f"{float(x) * 100:.1f}%"


def _float_or_blank(x, digits=2):
  if x is None or (isinstance(x, float) and pd.isna(x)):
    return ""
  return round(float(x), digits)


def _extract_variant_metrics(model_variants: dict, family_name: str):
  family_result = model_variants.get(family_name, {})
  pred = family_result.get("prediction", "")
  conf = family_result.get("confidence", None)
  rank = family_result.get("Rank", {}).get("rank_score", None)
  return pred, conf, rank


# ── Feature preparation ────────────────────────────────────

def _prepare_payload(get_input, player_name, parlays):
  """Call get_input, sanitise, return payload dict or None on failure."""
  try:
    input_df, _ = get_input(
      player_name=player_name,
      parlays=parlays,
      seasons=SEASONS,
    )
  except Exception as e:
    logger.error("get_input failed for %s: %s", player_name, e)
    return None

  safe_df = input_df.copy()
  safe_df = safe_df.replace([float("inf"), float("-inf")], pd.NA)
  for col in safe_df.columns:
    if pd.api.types.is_numeric_dtype(safe_df[col]):
      safe_df[col] = pd.to_numeric(safe_df[col], errors="coerce").fillna(0.0)
  return safe_df.iloc[0].to_dict()


# ── Row builder ────────────────────────────────────────────

def _build_row(player_name, entry, result, model_variants):
  """Build a CSV row dict from a line entry and model result."""
  consensus = result.get("consensus", {})

  if not consensus:
    logger.warning("missing consensus for %s", player_name)
    return None

  prediction = consensus.get("prediction", "N/A")
  confidence = float(consensus.get("confidence", 0.0))
  probability_std = float(consensus.get("probability_std", 0.0))
  agreement_ratio = float(consensus.get("agreement_ratio", 0.0))
  betting = consensus.get("betting_analysis", {})
  tier = betting.get("stat_tier", "?")
  recommendation = betting.get("recommendation", "DO NOT BET")
  rank_data = consensus.get("Rank", {})
  rank_score = float(rank_data.get("rank_score", 0.0))
  legacy_rank_score = rank_data.get("legacy_rank_score")

  platform = entry['platform']
  goblin_demon = entry['goblin_demon']

  # PrizePicks-only, no Less option, model says UNDER → skip
  if platform == 'PrizePicks' and not entry['pp_has_less'] and prediction == "UNDER":
    logger.info("SKIP (PrizePicks %s -- no Less option, model says UNDER)", goblin_demon)
    return None

  # Shared platform but PP has only More and model says UNDER →
  # downgrade to Underdog only
  if platform == 'Underdog and PrizePicks' and not entry['pp_has_less'] and prediction == "UNDER":
    platform = 'Underdog'
    goblin_demon = ''
    logger.info("PP no Less -> Underdog only")

  if prediction == "OVER":
    multiplier = entry['higher_multiplier']
  elif prediction == "UNDER":
    multiplier = entry['lower_multiplier']
  else:
    multiplier = 1.0

  full_all_pick, full_all_conf, full_all_rank = _extract_variant_metrics(model_variants, "FULL_ALL")
  full_tight_pick, full_tight_conf, full_tight_rank = _extract_variant_metrics(model_variants, "FULL_TIGHT")
  reduced_all_pick, reduced_all_conf, reduced_all_rank = _extract_variant_metrics(model_variants, "REDUCED_ALL")
  reduced_tight_pick, reduced_tight_conf, reduced_tight_rank = _extract_variant_metrics(model_variants, "REDUCED_TIGHT")

  logger.info(
    "%s %5.1f%%  %.2fx  Tier %s  Rank %.2f  Std %.4f  Agr %.1f%%",
    prediction, confidence*100, multiplier, tier, rank_score,
    probability_std, agreement_ratio*100,
  )

  rank_breakdown = consensus.get("Rank", {}).get("rank_breakdown", {})
  signal_points = rank_breakdown.get("signal_points", {})

  sportsbook_entries = {e.get("bookmaker", ""): e for e in result.get("sportsbook_odds", [])}
  sportsbook_column_map = [
    ("fanduel", "FanDuel"),
    ("draftkings", "DraftKings"),
    ("betrivers", "BetRivers"),
    ("betonlineag", "BetOnline"),
    ("bovada", "Bovada"),
    ("betmgm", "BetMGM"),
  ]
  chosen_edge_key = "edge_over" if prediction == "OVER" else "edge_under"

  row = {
    "Player": player_name,
    "Stat": entry['stat'],
    "Line": entry['line'],
    "Pick": prediction,
    "Platform": platform,
    "Goblin/Demon": goblin_demon,
    "Multiplier": f"{multiplier:.2f}x" if multiplier != 1.0 else "",
    "Confidence": _pct_or_blank(confidence),
    "Tier": tier,
    "Recommendation": recommendation,
    "Rank Score": _float_or_blank(rank_score),
    "Legacy Rank Score": _float_or_blank(legacy_rank_score),
    "Rank: Confidence": _float_or_blank(rank_breakdown.get("model_confidence_score")),
    "Rank: Market Edge": _float_or_blank(rank_breakdown.get("market_edge_score")),
    "Rank: Model Alpha": _float_or_blank(rank_breakdown.get("model_alpha_score")),
    "Rank: Agreement": _float_or_blank(rank_breakdown.get("family_agreement_score")),
    "Rank: Signals": _float_or_blank(rank_breakdown.get("statistical_signals_score")),
    "Result": "",
    "Actual value": "",

    "FULL_ALL Pick": full_all_pick,
    "FULL_ALL Confidence": _pct_or_blank(full_all_conf),
    "FULL_ALL Rank score": _float_or_blank(full_all_rank),
    "FULL_ALL Result": "",

    "FULL_TIGHT Pick": full_tight_pick,
    "FULL_TIGHT Confidence": _pct_or_blank(full_tight_conf),
    "FULL_TIGHT Rank score": _float_or_blank(full_tight_rank),
    "FULL_TIGHT Result": "",

    "REDUCED_ALL Pick": reduced_all_pick,
    "REDUCED_ALL Confidence": _pct_or_blank(reduced_all_conf),
    "REDUCED_ALL Rank score": _float_or_blank(reduced_all_rank),
    "REDUCED_ALL Result": "",

    "REDUCED_TIGHT Pick": reduced_tight_pick,
    "REDUCED_TIGHT Confidence": _pct_or_blank(reduced_tight_conf),
    "REDUCED_TIGHT Rank score": _float_or_blank(reduced_tight_rank),
    "REDUCED_TIGHT Result": "",

    "z_line": _float_or_blank(signal_points.get("z_line")),
    "z_recent": _float_or_blank(signal_points.get("z_recent")),
    "z_matchup": _float_or_blank(signal_points.get("z_matchup")),
    "line_diff": _float_or_blank(signal_points.get("line_diff")),
    "momentum": _float_or_blank(signal_points.get("momentum")),
    "last10": _float_or_blank(signal_points.get("last10")),
    "last5": _float_or_blank(signal_points.get("last5")),
    "probability_std": _float_or_blank(probability_std, digits=4),
    "Agreement ratio": _pct_or_blank(agreement_ratio),
  }

  for book_key, book_label in sportsbook_column_map:
    book = sportsbook_entries.get(book_key, {})
    row[f"{book_label} line"] = _float_or_blank(book.get("line"), digits=1)
    row[f"{book_label} line_delta"] = _float_or_blank(book.get("line_delta"), digits=1)
    row[f"{book_label} over_hit_prob"] = _pct_or_blank(book.get("over_hit_prob"))
    row[f"{book_label} under_hit_prob"] = _pct_or_blank(book.get("under_hit_prob"))
    row[f"{book_label} edge"] = _float_or_blank(book.get(chosen_edge_key), digits=4)

  # Market summary columns
  market = result.get("market_summary", {})
  market_prob_key = "consensus_prob_over" if prediction == "OVER" else "consensus_prob_under"
  row["Market Consensus Line"] = _float_or_blank(market.get("consensus_line"), digits=1)
  row["Market Consensus Prob"] = _pct_or_blank(market.get(market_prob_key))
  row["Best Over Odds"] = market.get("best_over_odds", "")
  row["Best Over Book"] = market.get("best_over_book", "")
  row["Best Under Odds"] = market.get("best_under_odds", "")
  row["Best Under Book"] = market.get("best_under_book", "")
  row["DFS vs Market"] = market.get("dfs_vs_market", "")

  return row


# ── Main pipeline ──────────────────────────────────────────
#
# Payload strategy (minimises get_input calls):
#
#   Player on BOTH platforms, NO conflicts:
#     → 1 get_input with combined parlays (UD lines + PP-only lines)
#     → 1 predict() call per unique stat
#
#   Player on BOTH platforms, HAS conflicts:
#     → 1 get_input with combined parlays (for combined + conflict_ud stats)
#     → 1 get_input with PP-only parlays  (for conflict_pp stats)
#
#   Player on ONE platform only:
#     → 1 get_input
#     → 1 predict() per stat
#

def run_pipeline(underdog_path: str, prizepicks_path: str, output_csv: str):
  try:
    from utils import get_input
  except ImportError:
    logger.error("Cannot import get_input from utils.py.")
    sys.exit(1)

  logger.info("=" * 60)
  logger.info("Parsing Underdog:   %s", underdog_path)
  logger.info("Parsing PrizePicks: %s", prizepicks_path)
  logger.info("=" * 60)

  ud_parsed = parse_underdog_txt(underdog_path)
  pp_parsed = parse_prizepicks_txt(prizepicks_path)

  logger.info("Underdog players: %d", len(ud_parsed))
  logger.info("PrizePicks players: %d", len(pp_parsed))

  all_players = sorted(set(ud_parsed.keys()) | set(pp_parsed.keys()))
  logger.info("Total unique players: %d", len(all_players))

  total_get_input = 0
  total_predict_calls = 0
  rows = []

  for player_name in all_players:
    ud_stats = ud_parsed.get(player_name)
    pp_stats = pp_parsed.get(player_name)

    entries, has_conflicts = classify_player(ud_stats, pp_stats)
    if not entries:
      continue

    logger.info("-" * 60)
    has_ud = ud_stats is not None
    has_pp = pp_stats is not None

    if has_ud and has_pp:
      combined_parlays = build_combined_parlays(ud_stats, pp_stats, TARGET_COLUMNS)
      combined_payload = _prepare_payload(get_input, player_name, combined_parlays)
      if combined_payload is None:
        continue
      total_get_input += 1

      pp_payload = None
      if has_conflicts:
        pp_parlays = build_platform_parlays(pp_stats, TARGET_COLUMNS)
        pp_payload = _prepare_payload(get_input, player_name, pp_parlays)
        if pp_payload is not None:
          total_get_input += 1

      n_gi = 2 if has_conflicts and pp_payload else 1
      label = "lines differ" if has_conflicts else "all lines match"
      logger.info("%s  (%d lines, %d get_input -- %s)", player_name, len(entries), n_gi, label)

      combined_cache = {}
      pp_cache = {}

      for entry in entries:
        stat = entry['stat']
        group = entry['payload_group']
        logger.info("  %s  %s  [%s]", stat, entry['line'], entry['platform'])

        if group == 'conflict_pp':
          payload = pp_payload
          cache = pp_cache
        else:
          payload = combined_payload
          cache = combined_cache

        if payload is None:
          logger.info("  payload unavailable, skipping")
          continue

        if stat in cache:
          result = cache[stat]
          if result is not None:
            logger.debug("  (cached)")
        else:
          # Fetch raw odds once per stat+player (team-aware)
          team_name = payload.get("TEAM")
          dfs_line = entry['line']
          raw_odds = odds_module.lookup_raw_odds(player_name, stat, team_name)

          result = models.predict(payload, stat, raw_sportsbook_odds=raw_odds, dfs_line=dfs_line)
          cache[stat] = result
          total_predict_calls += 1

        if result is None:
          continue

        model_variants = result.get("model_variants", {})
        row = _build_row(player_name, entry, result, model_variants)
        if row:
          rows.append(row)

    else:
      raw_stats = ud_stats or pp_stats
      parlays = build_platform_parlays(raw_stats, TARGET_COLUMNS)
      payload = _prepare_payload(get_input, player_name, parlays)
      if payload is None:
        continue
      total_get_input += 1

      src = "Underdog" if has_ud else "PrizePicks"
      logger.info("%s  (%d lines, 1 get_input -- %s only)", player_name, len(entries), src)

      model_cache = {}
      for entry in entries:
        stat = entry['stat']
        logger.info("  %s  %s  [%s]", stat, entry['line'], entry['platform'])

        if stat in model_cache:
          result = model_cache[stat]
          if result is not None:
            logger.debug("  (cached)")
        else:
          team_name = payload.get("TEAM")
          dfs_line = entry['line']
          raw_odds = odds_module.lookup_raw_odds(player_name, stat, team_name)

          result = models.predict(payload, stat, raw_sportsbook_odds=raw_odds, dfs_line=dfs_line)
          model_cache[stat] = result
          total_predict_calls += 1

        if result is None:
          continue

        model_variants = result.get("model_variants", {})
        row = _build_row(player_name, entry, result, model_variants)
        if row:
          rows.append(row)

  if not rows:
    logger.warning("No predictions generated.")
    return

  logger.info("-" * 60)
  logger.info("Pipeline summary:")
  logger.info("  get_input calls:  %d", total_get_input)
  logger.info("  predict() calls:  %d", total_predict_calls)

  df_out = pd.DataFrame(rows).sort_values("Rank Score", ascending=False).reset_index(drop=True)
  df_out.index += 1
  df_out.index.name = "Rank"

  df_out.to_csv(output_csv)

  logger.info("=" * 60)
  logger.info("Saved -> %s  (%d predictions)", output_csv, len(df_out))
  logger.info("=" * 60)
  print(
    df_out[[
      "Player", "Stat", "Line", "Pick", "Platform", "Goblin/Demon",
      "Multiplier", "Confidence", "Tier", "Recommendation", "Rank Score",
      "Legacy Rank Score", "probability_std", "Agreement ratio"
    ]].to_string()
  )


if __name__ == "__main__":
  logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
  models.load_models(MODEL_ROOT)
  run_pipeline(
    "lines/underdog_lines.txt",
    "lines/prizepicks_lines.txt",
    "today_picks.csv",
  )
