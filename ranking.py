"""Rank score computation for NBA player prop picks.

Composite score (unclamped) using weighted components.
"""
import numpy as np
from config import BREAKEVEN_PROB, RANK_WEIGHTS


def _summarize_sportsbook_for_rank(prediction: str, sportsbook_odds: list | None) -> dict:
	summary = {
		"same_line_books": 0,
		"all_books": 0,
		"avg_same_line_market_prob_side": None,
		"avg_all_books_market_prob_side": None,
		"avg_edge_side": 0.0,
	}
	if not sportsbook_odds:
		return summary

	prob_key = "over_no_vig_prob_at_dfs" if prediction == "OVER" else "under_no_vig_prob_at_dfs"

	same_probs = []
	all_probs = []
	edges = []

	for entry in sportsbook_odds:
		prob = entry.get(prob_key)
		if prob is not None:
			all_probs.append(float(prob))
			edges.append(float(prob) - BREAKEVEN_PROB)
			if entry.get("lines_match"):
				same_probs.append(float(prob))

	summary["same_line_books"] = len(same_probs)
	summary["all_books"] = len(all_probs)
	summary["avg_same_line_market_prob_side"] = float(np.mean(same_probs)) if same_probs else None
	summary["avg_all_books_market_prob_side"] = float(np.mean(all_probs)) if all_probs else None
	summary["avg_edge_side"] = float(np.mean(edges)) if edges else 0.0
	return summary


def _safe_round(value, digits: int = 4):
	return round(float(value), digits) if value is not None else None


# ── Signal total ────────────────────────────────────────────

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


# ── Rank score (unclamped composite) ────────────────────────

def compute_rank_score(
	confidence: float,
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
	"""Compute rank score -- unclamped weighted composite."""

	signal_total, signal_breakdown = _compute_signal_total(
		z_vs_line, z_vs_recent, z_vs_matchup,
		line_diff, momentum, last10_rate, last5_rate, prediction,
	)

	sportsbook_summary = _summarize_sportsbook_for_rank(prediction, sportsbook_odds)
	avg_same = sportsbook_summary["avg_same_line_market_prob_side"]
	avg_all = sportsbook_summary["avg_all_books_market_prob_side"]

	market_prob = avg_same if avg_same is not None else avg_all

	# ── Component 1: Model Confidence ──
	model_confidence_score = (confidence - 0.50) * 200.0

	# ── Component 2: Market Edge vs Breakeven ──
	if market_prob is not None:
		edge_raw = market_prob - BREAKEVEN_PROB
		market_edge_score = edge_raw / 0.20 * 100.0
	else:
		edge_raw = None
		market_edge_score = 0.0

	# ── Component 3: Model Alpha vs Market ──
	if market_prob is not None:
		alpha_raw = confidence - market_prob
		model_alpha_score = alpha_raw / 0.20 * 100.0
	else:
		alpha_raw = confidence - BREAKEVEN_PROB
		model_alpha_score = alpha_raw / 0.20 * 100.0

	# ── Component 4: Family Agreement ──
	agreement_base = agreement_ratio * 100.0
	std_penalty = min(50.0, probability_std * 500.0)
	family_agreement_score = agreement_base - std_penalty

	# ── Component 5: Statistical Signals ──
	statistical_signals_score = signal_total / 10.0 * 100.0

	# ── Weighted composite (unclamped) ──
	w = RANK_WEIGHTS
	rank_score = (
		w["model_confidence"] * model_confidence_score +
		w["market_edge"] * market_edge_score +
		w["model_alpha"] * model_alpha_score +
		w["family_agreement"] * family_agreement_score +
		w["statistical_signals"] * statistical_signals_score
	)

	return {
		"rank_score": round(rank_score, 2),
		"rank_breakdown": {
			"model_confidence_score": round(model_confidence_score, 2),
			"market_edge_score": round(market_edge_score, 2),
			"model_alpha_score": round(model_alpha_score, 2),
			"family_agreement_score": round(family_agreement_score, 2),
			"statistical_signals_score": round(statistical_signals_score, 2),
			"edge_raw": _safe_round(edge_raw) if edge_raw is not None else None,
			"alpha_raw": _safe_round(alpha_raw),
			"agreement_raw": _safe_round(agreement_ratio),
			"signal_total_raw": _safe_round(signal_total),
			"signal_points": signal_breakdown,
			"sportsbook_summary": {
				"same_line_books": sportsbook_summary["same_line_books"],
				"all_books": sportsbook_summary["all_books"],
				"avg_same_line_market_prob_side": _safe_round(avg_same) if avg_same is not None else None,
				"avg_all_books_market_prob_side": _safe_round(avg_all) if avg_all is not None else None,
				"avg_edge_side": _safe_round(sportsbook_summary["avg_edge_side"]),
			},
		},
	}
