"""Rank score computation for NBA player prop picks.

New formula: normalized 0–100 composite score.
Legacy formula preserved for comparison.
"""
import numpy as np
from config import TIER_POINTS, BREAKEVEN_PROB, RANK_WEIGHTS


def _summarize_sportsbook_for_rank(prediction: str, sportsbook_odds: list | None) -> dict:
  summary = {
    "same_line_books": 0,
    "all_books": 0,
    "avg_same_line_market_prob_side": None,
    "avg_all_books_market_prob_side": None,
    "avg_edge_side": 0.0,
    "avg_ev_side": 0.0,
  }
  if not sportsbook_odds:
    return summary

  prob_key = "over_hit_prob" if prediction == "OVER" else "under_hit_prob"
  edge_key = "edge_over" if prediction == "OVER" else "edge_under"
  ev_key = "ev_over" if prediction == "OVER" else "ev_under"

  same_probs = []
  all_probs = []
  edges = []
  evs = []

  for entry in sportsbook_odds:
    prob = entry.get(prob_key)
    if prob is not None:
      all_probs.append(float(prob))
      if entry.get("lines_match"):
        same_probs.append(float(prob))
    edge = entry.get(edge_key)
    if edge is not None:
      edges.append(float(edge))
    ev = entry.get(ev_key)
    if ev is not None:
      evs.append(float(ev))

  summary["same_line_books"] = len(same_probs)
  summary["all_books"] = len(all_probs)
  summary["avg_same_line_market_prob_side"] = float(np.mean(same_probs)) if same_probs else None
  summary["avg_all_books_market_prob_side"] = float(np.mean(all_probs)) if all_probs else None
  summary["avg_edge_side"] = float(np.mean(edges)) if edges else 0.0
  summary["avg_ev_side"] = float(np.mean(evs)) if evs else 0.0
  return summary


def _safe_round(value, digits: int = 4):
  return round(float(value), digits) if value is not None else None


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
  return max(lo, min(hi, value))


# ── Signal total (shared by both formulas) ─────────────────

def _compute_signal_total(
  z_vs_line: float,
  z_vs_recent: float,
  z_vs_matchup: float,
  line_diff: float,
  momentum: float,
  last10_rate: float,
  last5_rate: float,
  prediction: str,
) -> tuple[float, dict]:
  """Compute directional signal total and individual points."""
  direction = 1 if prediction == "OVER" else -1

  z_line_points = z_vs_line * direction
  z_recent_points = z_vs_recent * direction
  z_matchup_points = z_vs_matchup * direction
  line_diff_points = line_diff * direction
  momentum_points = momentum * direction

  if direction == 1:
    last10_points = 2 * last10_rate
    last5_points = 2 * last5_rate
  else:
    last10_points = 2 * (1 - last10_rate)
    last5_points = 2 * (1 - last5_rate)

  signal_total = (
    z_line_points +
    z_recent_points +
    z_matchup_points +
    line_diff_points +
    momentum_points +
    last10_points +
    last5_points
  )

  signal_breakdown = {
    "z_line": z_line_points,
    "z_recent": z_recent_points,
    "z_matchup": z_matchup_points,
    "line_diff": line_diff_points,
    "momentum": momentum_points,
    "last10": last10_points,
    "last5": last5_points,
    "signal_total": signal_total,
  }

  return signal_total, signal_breakdown


# ── Legacy ranking (preserved for comparison) ──────────────

def compute_legacy_rank_score(
  confidence: float,
  tier: str,
  z_vs_line: float,
  z_vs_recent: float,
  z_vs_matchup: float,
  line_diff: float,
  momentum: float,
  prediction: str,
  last10_rate: float,
  last5_rate: float,
  agreement_ratio: float,
  probability_std: float,
  sportsbook_odds: list | None = None,
) -> dict:
  confidence_points = confidence * 100
  tier_points = TIER_POINTS.get(tier, 0)

  signal_total, signal_breakdown = _compute_signal_total(
    z_vs_line, z_vs_recent, z_vs_matchup,
    line_diff, momentum, last10_rate, last5_rate, prediction,
  )

  sportsbook_summary = _summarize_sportsbook_for_rank(prediction, sportsbook_odds)
  avg_same = sportsbook_summary["avg_same_line_market_prob_side"]
  avg_all = sportsbook_summary["avg_all_books_market_prob_side"]
  avg_edge = sportsbook_summary["avg_edge_side"]
  avg_ev = sportsbook_summary["avg_ev_side"]

  same_line_market_points = 100 * (avg_same - 0.5) if avg_same is not None else 0.0
  all_books_market_points = 100 * (avg_all - 0.5) if avg_all is not None else 0.0
  edge_points = 100 * avg_edge
  agreement_points = 5 * agreement_ratio - 10 * probability_std

  if sportsbook_summary["same_line_books"] > 0 and avg_same is not None:
    hard_penalty = -20.0 if avg_same < 0.50 else 0.0
    continuous_penalty = -25.0 * max(0.0, 0.50 - avg_same)
  elif avg_all is not None:
    hard_penalty = -8.0 if avg_all < 0.50 else 0.0
    continuous_penalty = -15.0 * max(0.0, 0.50 - avg_all)
  else:
    hard_penalty = 0.0
    continuous_penalty = 0.0

  penalty_points = hard_penalty + continuous_penalty

  final_score = (
    0.55 * confidence_points +
    1.00 * edge_points +
    0.50 * same_line_market_points +
    0.25 * all_books_market_points +
    1.00 * signal_total +
    agreement_points +
    tier_points +
    penalty_points
  )

  return {
    "rank_score": final_score,
    "rank_breakdown": {
      "confidence_points": confidence_points,
      "edge_points": edge_points,
      "same_line_market_points": same_line_market_points,
      "all_books_market_points": all_books_market_points,
      "agreement_points": agreement_points,
      "tier_points": tier_points,
      "penalty_points": penalty_points,
      "signal_points": signal_breakdown,
      "sportsbook_summary": {
        "same_line_books": sportsbook_summary["same_line_books"],
        "all_books": sportsbook_summary["all_books"],
        "avg_same_line_market_prob_side": _safe_round(avg_same) if avg_same is not None else None,
        "avg_all_books_market_prob_side": _safe_round(avg_all) if avg_all is not None else None,
        "avg_edge_side": _safe_round(avg_edge),
        "avg_ev_side": _safe_round(avg_ev),
      },
    }
  }


# ── New ranking (0–100 normalized composite) ───────────────

def compute_rank_score(
  confidence: float,
  tier: str,
  z_vs_line: float,
  z_vs_recent: float,
  z_vs_matchup: float,
  line_diff: float,
  momentum: float,
  prediction: str,
  last10_rate: float,
  last5_rate: float,
  agreement_ratio: float,
  probability_std: float,
  sportsbook_odds: list | None = None,
) -> dict:
  """Compute new 0–100 rank score plus legacy score for comparison."""

  # ── Shared signal computation ──
  signal_total, signal_breakdown = _compute_signal_total(
    z_vs_line, z_vs_recent, z_vs_matchup,
    line_diff, momentum, last10_rate, last5_rate, prediction,
  )

  # ── Sportsbook summary ──
  sportsbook_summary = _summarize_sportsbook_for_rank(prediction, sportsbook_odds)
  avg_same = sportsbook_summary["avg_same_line_market_prob_side"]
  avg_all = sportsbook_summary["avg_all_books_market_prob_side"]

  # Best available market probability on the predicted side
  market_prob = avg_same if avg_same is not None else avg_all

  # ── Component 1: Model Confidence (35%) ──
  # Maps [0.50, 1.00] → [0, 100]
  model_confidence_score = _clamp((confidence - 0.50) * 200.0)

  # ── Component 2: Market Edge vs Breakeven (30%) ──
  # How far above 54% breakeven the market consensus is
  if market_prob is not None:
    market_edge_score = _clamp((market_prob - BREAKEVEN_PROB) / 0.20 * 100.0)
  else:
    market_edge_score = 0.0

  # ── Component 3: Model Alpha vs Market (25%) ──
  # Model's excess prediction over market probability
  if market_prob is not None:
    alpha = confidence - market_prob
    model_alpha_score = _clamp(alpha / 0.20 * 100.0)
  else:
    # No market data: scaled confidence as fallback (half sensitivity)
    model_alpha_score = _clamp((confidence - 0.50) * 100.0)

  # ── Component 4: Family Agreement (5%) ──
  agreement_base = agreement_ratio * 100.0
  std_penalty = min(50.0, probability_std * 500.0)
  family_agreement_score = _clamp(agreement_base - std_penalty)

  # ── Component 5: Statistical Signals (5%) ──
  statistical_signals_score = _clamp(signal_total / 10.0 * 100.0)

  # ── Weighted composite ──
  w = RANK_WEIGHTS
  new_score = (
    w["model_confidence"] * model_confidence_score +
    w["market_edge"] * market_edge_score +
    w["model_alpha"] * model_alpha_score +
    w["family_agreement"] * family_agreement_score +
    w["statistical_signals"] * statistical_signals_score
  )

  # ── Legacy score (for comparison) ──
  legacy_result = compute_legacy_rank_score(
    confidence=confidence,
    tier=tier,
    z_vs_line=z_vs_line,
    z_vs_recent=z_vs_recent,
    z_vs_matchup=z_vs_matchup,
    line_diff=line_diff,
    momentum=momentum,
    prediction=prediction,
    last10_rate=last10_rate,
    last5_rate=last5_rate,
    agreement_ratio=agreement_ratio,
    probability_std=probability_std,
    sportsbook_odds=sportsbook_odds,
  )

  return {
    "rank_score": round(new_score, 2),
    "legacy_rank_score": legacy_result["rank_score"],
    "rank_breakdown": {
      "model_confidence_score": round(model_confidence_score, 2),
      "market_edge_score": round(market_edge_score, 2),
      "model_alpha_score": round(model_alpha_score, 2),
      "family_agreement_score": round(family_agreement_score, 2),
      "statistical_signals_score": round(statistical_signals_score, 2),
      "signal_points": signal_breakdown,
      "sportsbook_summary": {
        "same_line_books": sportsbook_summary["same_line_books"],
        "all_books": sportsbook_summary["all_books"],
        "avg_same_line_market_prob_side": _safe_round(avg_same) if avg_same is not None else None,
        "avg_all_books_market_prob_side": _safe_round(avg_all) if avg_all is not None else None,
        "avg_edge_side": _safe_round(sportsbook_summary["avg_edge_side"]),
        "avg_ev_side": _safe_round(sportsbook_summary["avg_ev_side"]),
      },
    },
    "legacy_rank_breakdown": legacy_result["rank_breakdown"],
  }
