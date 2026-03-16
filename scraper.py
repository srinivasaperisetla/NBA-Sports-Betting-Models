import re
import sys
import unicodedata
import requests
import pandas as pd

try:
  from unidecode import unidecode as _norm
except ImportError:
  def _norm(s: str) -> str:
    return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
    )

CHAMPION_API_URL = "http://127.0.0.1:8000/predict"
SEASONS = ["2024-25", "2025-26"]

TARGET_COLUMNS = [
  "PTS", "REB", "AST", "STL", "BLK",
  "PRA", "PA", "PR", "RA", "SB",
  "TOV", "FGM", "3PM", "FTM", "FGA", "3PA", "FTA"
]

required_names = {
  "Aaron Gordon",
  "Ace Bailey",
  "Alex Caruso",
  "Alex Sarr",
  "Alperen Sengun",
  "Amen Thompson",
  "Andrew Nembhard",
  "Anthony Davis",
  "Anthony Edwards",
  "Anfernee Simons",
  "Ausar Thompson",
  "Austin Reaves",
  "Bam Adebayo",
  "Bennedict Mathurin",
  "Brandin Podziemski",
  "Brandon Ingram",
  "Brandon Miller",
  "CJ McCollum",
  "Cade Cunningham",
  "Cam Thomas",
  "Chet Holmgren",
  "Coby White",
  "Collin Sexton",
  "Cooper Flagg",
  "D'Angelo Russell",
  "Darius Garland",
  "De'Aaron Fox",
  "DeMar DeRozan",
  "Deandre Ayton",
  "Dejounte Murray",
  "Deni Avdija",
  "Derrick White",
  "Desmond Bane",
  "Devin Booker",
  "Dillon Brooks",
  "Domantas Sabonis",
  "Donovan Mitchell",
  "Donte DiVincenzo",
  "Draymond Green",
  "Dyson Daniels",
  "Evan Mobley",
  "Franz Wagner",
  "Fred VanVleet",
  "Giannis Antetokounmpo",
  "Immanuel Quickley",
  "Isaiah Hartenstein",
  "Ivica Zubac",
  "Ja Morant",
  "Jaden McDaniels",
  "Jaden Ivey",
  "Jaime Jaquez Jr.",
  "Jalen Brunson",
  "Jalen Duren",
  "Jalen Green",
  "Jalen Johnson",
  "Jalen Suggs",
  "Jalen Williams",
  "Jamal Murray",
  "James Harden",
  "Jaren Jackson Jr.",
  "Jarrett Allen",
  "Jaylen Brown",
  "Jaylen Wells",
  "Jayson Tatum",
  "Jerami Grant",
  "Jimmy Butler III",
  "Joel Embiid",
  "Jonathan Kuminga",
  "Jordan Poole",
  "Josh Giddey",
  "Josh Hart",
  "Jrue Holiday",
  "Julius Randle",
  "Jusuf Nurkić",
  "Karl-Anthony Towns",
  "Kawhi Leonard",
  "Keegan Murray",
  "Kelly Oubre Jr.",
  "Kentavious Caldwell-Pope",
  "Kevin Durant",
  "Kevin Love",
  "Kevin Porter Jr.",
  "Keyonte George",
  "Khris Middleton",
  "Klay Thompson",
  "Kon Knueppel",
  "Kristaps Porziņģis",
  "Kyle Kuzma",
  "Kyrie Irving",
  "Kyshawn George",
  "LaMelo Ball",
  "Lauri Markkanen",
  "LeBron James",
  "Lonzo Ball",
  "Luka Dončić",
  "Marcus Smart",
  "Michael Porter Jr.",
  "Mikal Bridges",
  "Miles Bridges",
  "Myles Turner",
  "Naz Reid",
  "Nikola Jokić",
  "Nikola Vučević",
  "Norman Powell",
  "OG Anunoby",
  "Onyeka Okongwu",
  "P.J. Washington",
  "Paolo Banchero",
  "Pascal Siakam",
  "Paul George",
  "Payton Pritchard",
  "Quentin Grimes",
  "RJ Barrett",
  "Reed Sheppard",
  "Rudy Gobert",
  "Russell Westbrook",
  "Scottie Barnes",
  "Scoot Henderson",
  "Shaedon Sharpe",
  "Shai Gilgeous-Alexander",
  "Stephen Curry",
  "Stephon Castle",
  "Steven Adams",
  "Tim Hardaway Jr.",
  "Tobias Harris",
  "Trae Young",
  "Trey Murphy III",
  "Tyler Herro",
  "Tyrese Haliburton",
  "Tyrese Maxey",
  "VJ Edgecombe",
  "Victor Wembanyama",
  "Zach LaVine",
  "Zion Williamson",
}

_ORIGINAL_NAMES = sorted(required_names)

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

_PLAYER_LOOKUP = {_norm(p).lower(): p for p in _ORIGINAL_NAMES}


def normalize_player(raw_name: str):
  return _PLAYER_LOOKUP.get(_norm(raw_name.strip()).lower())


def parse_underdog_txt(filepath: str) -> dict:
  with open(filepath, "r", encoding="utf-8") as f:
    lines = [line.rstrip() for line in f]

  players = {}
  current_player = None
  i = 0

  while i < len(lines):
    line = lines[i]

    candidate = normalize_player(line)
    if candidate:
      current_player = candidate
      players.setdefault(current_player, {})
      i += 1
      continue

    if current_player is None:
      i += 1
      continue

    num_match = re.match(r'^(\d+\.?\d*)$', line)
    if num_match and i + 1 < len(lines):
      parlay_line = float(num_match.group(1))
      stat_label = lines[i + 1].strip()
      stat_key = UNDERDOG_STAT_MAP.get(stat_label)

      if stat_key and stat_key not in players[current_player]:
        players[current_player][stat_key] = parlay_line
        i += 2
        continue

    i += 1

  return players


def build_parlays_dict(stat_lines: dict) -> dict:
  parlays = {f"PL_{col}": 0.0 for col in TARGET_COLUMNS}
  for stat, val in stat_lines.items():
    parlays[f"PL_{stat}"] = float(val)
  return parlays


def call_model(payload: dict, stat: str) -> dict | None:
  try:
    resp = requests.post(
      CHAMPION_API_URL,
      params={"stat": stat, "family": "ALL"},
      json=payload,
      timeout=60,
    )
    if resp.status_code == 200:
      return resp.json()
    print(f"    ⚠️  API {resp.status_code} for {stat}: {resp.text[:160]}")
    return None
  except Exception as e:
    print(f"    ⚠️  Request failed for {stat}: {e}")
    return None


def _pct_or_blank(x):
  if x is None or pd.isna(x):
    return ""
  return f"{float(x) * 100:.1f}%"


def _float_or_blank(x, digits=2):
  if x is None or pd.isna(x):
    return ""
  return round(float(x), digits)


def _extract_variant_metrics(model_variants: dict, family_name: str):
  family_result = model_variants.get(family_name, {})
  pred = family_result.get("prediction", "")
  conf = family_result.get("confidence", None)
  rank = family_result.get("Rank", {}).get("rank_score", None)
  return pred, conf, rank


def run_pipeline(txt_path: str, output_csv: str):
  try:
    from utils import get_input
  except ImportError:
    print("ERROR: Cannot import get_input from utils.py.")
    print("       Place this script in the same directory as utils.py.")
    sys.exit(1)

  print(f"\n{'='*60}")
  print(f"  Parsing: {txt_path}")
  print(f"{'='*60}")

  parsed = parse_underdog_txt(txt_path)

  if not parsed:
    print("No players found. Check the txt file format.")
    return

  print(f"\n✅ Found {len(parsed)} player(s):")
  for p, stats in parsed.items():
    print(f"   {p:35s} → {list(stats.keys())}")

  rows = []

  for player_name, stat_lines in parsed.items():
    if not stat_lines:
      print(f"\n⏭  Skipping {player_name} — no supported stats found")
      continue

    print(f"\n{'─'*60}")
    print(f"  {player_name}  ({len(stat_lines)} stats)")

    parlays = build_parlays_dict(stat_lines)

    try:
      input_df, _ = get_input(
        player_name=player_name,
        parlays=parlays,
        seasons=SEASONS,
      )
    except Exception as e:
      print(f"  ❌ get_input failed: {e}")
      continue

    safe_df = input_df.copy()
    safe_df = safe_df.replace([float("inf"), float("-inf")], pd.NA)

    for col in safe_df.columns:
      if pd.api.types.is_numeric_dtype(safe_df[col]):
        safe_df[col] = pd.to_numeric(safe_df[col], errors="coerce").fillna(0.0)

    payload = safe_df.iloc[0].to_dict()

    for stat, parlay_line in stat_lines.items():
      print(f"    {stat:<6} {parlay_line:<7} ... ", end="", flush=True)

      result = call_model(payload, stat)
      if result is None:
        continue

      consensus = result.get("consensus", {})
      model_variants = result.get("model_variants", {})

      if not consensus:
        print("⚠️  missing consensus")
        continue

      prediction = consensus.get("prediction", "N/A")
      confidence = float(consensus.get("confidence", 0.0))
      probability_std = float(consensus.get("probability_std", 0.0))
      agreement_ratio = float(consensus.get("agreement_ratio", 0.0))
      betting = consensus.get("betting_analysis", {})
      tier = betting.get("stat_tier", "?")
      recommendation = betting.get("recommendation", "DO NOT BET")
      rank_score = float(consensus.get("Rank", {}).get("rank_score", 0.0))

      full_all_pick, full_all_conf, full_all_rank = _extract_variant_metrics(model_variants, "FULL_ALL")
      full_tight_pick, full_tight_conf, full_tight_rank = _extract_variant_metrics(model_variants, "FULL_TIGHT")
      reduced_all_pick, reduced_all_conf, reduced_all_rank = _extract_variant_metrics(model_variants, "REDUCED_ALL")
      reduced_tight_pick, reduced_tight_conf, reduced_tight_rank = _extract_variant_metrics(model_variants, "REDUCED_TIGHT")

      print(
        f"{prediction:<6} "
        f"{confidence*100:5.1f}%  "
        f"Tier {tier}  "
        f"Rank {rank_score:.2f}  "
        f"Std {probability_std:.4f}  "
        f"Agr {agreement_ratio*100:5.1f}%"
      )

      rank_breakdown = consensus.get("Rank", {}).get("rank_breakdown", {})
      signal_points = rank_breakdown.get("signal_points", {})

      rows.append({
        "Player": player_name,
        "Stat": stat,
        "Line": parlay_line,
        "Pick": prediction,
        "Confidence": _pct_or_blank(confidence),
        "Tier": tier,
        "Recommendation": recommendation,
        "Rank Score": _float_or_blank(rank_score),
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

        "z_line": _float_or_blank(signal_points.get("z_line", None)),
        "z_recent": _float_or_blank(signal_points.get("z_recent", None)),
        "z_matchup": _float_or_blank(signal_points.get("z_matchup", None)),
        "line_diff": _float_or_blank(signal_points.get("line_diff", None)),
        "momentum": _float_or_blank(signal_points.get("momentum", None)),
        "last10": _float_or_blank(signal_points.get("last10", None)),
        "last5": _float_or_blank(signal_points.get("last5", None)),
        "probability_std": _float_or_blank(probability_std, digits=4),
        "Agreement ratio": _pct_or_blank(agreement_ratio),
      })

  if not rows:
    print("\n⚠️  No predictions generated.")
    return

  df_out = pd.DataFrame(rows).sort_values("Rank Score", ascending=False).reset_index(drop=True)
  df_out.index += 1
  df_out.index.name = "Rank"

  df_out.to_csv(output_csv)

  print(f"\n{'='*60}")
  print(f"  ✅ Saved → {output_csv}  ({len(df_out)} predictions)")
  print(f"{'='*60}\n")
  print(
    df_out[[
      "Player", "Line", "Pick", "Confidence",
      "Tier", "Recommendation", "Rank Score",
      "probability_std", "Agreement ratio"
    ]].to_string()
  )


if __name__ == "__main__":
  run_pipeline("lines/underdog_lines.txt", "today_picks.csv")