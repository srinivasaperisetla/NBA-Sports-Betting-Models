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

# ── Stat maps ──────────────────────────────────────────────

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

_PLAYER_LOOKUP = {_norm(p).lower(): p for p in _ORIGINAL_NAMES}


def normalize_player(raw_name: str):
  return _PLAYER_LOOKUP.get(_norm(raw_name.strip()).lower())


# ── Underdog parser ────────────────────────────────────────

def _is_underdog_player_header(lines: list, i: int) -> bool:
  """
  Check if line i is the start of a player block in Underdog format:
    Line i:   Player Name (or unknown name)
    Line i+1: (blank)
    Line i+2: Team Name (e.g. "Houston Rockets")
    Line i+3: (blank)
    Line i+4: "vs." or "@" game info
  """
  if i + 4 >= len(lines):
    return False
  name = lines[i].strip()
  if not name or re.match(r'^\d', name):
    return False
  # Must not be a known stat label or control word
  if name in UNDERDOG_STAT_MAP or name.lower() in ('higher', 'lower', 'higher/lower'):
    return False
  if lines[i + 1].strip() != '':
    return False
  team = lines[i + 2].strip()
  if not team or re.match(r'^\d', team):
    return False
  if lines[i + 3].strip() != '':
    return False
  game = lines[i + 4].strip()
  if not (game.startswith('vs.') or game.startswith('@')):
    return False
  return True


def parse_underdog_txt(filepath: str) -> dict:
  with open(filepath, "r", encoding="utf-8") as f:
    lines = [line.rstrip() for line in f]

  players = {}
  current_player = None
  i = 0

  while i < len(lines):
    line = lines[i].strip()

    # ── Check for player header (known or unknown) ──
    if _is_underdog_player_header(lines, i):
      candidate = normalize_player(line)
      if candidate:
        current_player = candidate
        players.setdefault(current_player, {})
      else:
        # Unknown player — reset current_player so their stats don't
        # leak into the previous player
        current_player = None
      # Skip past the header block (name, blank, team, blank, game, blank, Higher/Lower)
      i += 5
      # Skip any remaining header lines (blank, "Higher/Lower", etc.)
      while i < len(lines):
        peek = lines[i].strip()
        if peek == '' or peek == 'Higher/Lower':
          i += 1
          continue
        break
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
        players[current_player][stat_key] = {
          'line': parlay_line,
          'higher_multiplier': 1.0,
          'lower_multiplier': 1.0
        }

        # Look ahead for "Higher"/"Lower" with multipliers,
        # but STOP if we hit a new number+stat pair (next line boundary)
        j = i + 2
        while j < len(lines) and j < i + 10:
          current_line = lines[j].strip()

          # Stop conditions: hit a new stat line (number followed by stat label)
          if re.match(r'^(\d+\.?\d*)$', current_line):
            if j + 1 < len(lines):
              next_stat = lines[j + 1].strip()
              # If next line is ANY stat label (supported or not), this is
              # a new stat boundary — stop look-ahead
              if (next_stat in UNDERDOG_STAT_MAP or
                  re.match(r'^(Double Doubles|Triple Doubles|Fantasy Points|'
                           r'1[QH] .+)$', next_stat)):
                break
            # Also stop if it's just a bare number that isn't Higher/Lower context
            break

          if current_line.lower() in ['higher', 'lower']:
            direction = current_line.lower()

            if j + 1 < len(lines):
              next_line = lines[j + 1].strip()
              mult_match = re.match(r'^(\d+\.?\d*)x$', next_line, re.IGNORECASE)

              if mult_match:
                mult_value = float(mult_match.group(1))
              else:
                mult_value = 1.0

              if direction == 'higher':
                players[current_player][stat_key]['higher_multiplier'] = mult_value
              elif direction == 'lower':
                players[current_player][stat_key]['lower_multiplier'] = mult_value

          j += 1

        i += 2
        continue

    i += 1

  return players


# ── PrizePicks parser ──────────────────────────────────────

def _strip_goblin_demon(name: str):
  """Strip trailing 'Goblin' or 'Demon' tag from a player name line."""
  if name.endswith("Goblin"):
    return name[:-6], "Goblin"
  if name.endswith("Demon"):
    return name[:-5], "Demon"
  return name, ""


def parse_prizepicks_txt(filepath: str) -> dict:
  """
  Parse PrizePicks lines file.

  Returns:
    { canonical_player_name: {
        stat_key: {
          'line': float,
          'has_less': bool,
          'has_more': bool,
          'goblin_demon': str   # 'Goblin', 'Demon', or ''
        }, ...
      }, ...
    }

  Only players in required_names are included.
  """
  with open(filepath, "r", encoding="utf-8") as f:
    lines = [line.rstrip() for line in f]

  players = {}
  i = 0

  while i < len(lines):
    raw_line = lines[i].strip()

    if not raw_line:
      i += 1
      continue

    clean_name, goblin_demon_tag = _strip_goblin_demon(raw_line)
    candidate = normalize_player(clean_name)

    if candidate is None:
      i += 1
      continue

    # Verify player header: next line should be team-position
    if i + 1 >= len(lines):
      i += 1
      continue

    team_line = lines[i + 1].strip()
    if not re.match(r'^[A-Z]{2,3}\s*-\s*[A-Z]', team_line):
      i += 1
      continue

    # Skip header block (player name, team, repeated name, game info)
    j = i + 2
    while j < len(lines) and j < i + 5:
      check = lines[j].strip()
      if not check or re.match(r'^(vs |@ )', check):
        j += 1
        continue
      if normalize_player(check) == candidate:
        j += 1
        continue
      break

    # Skip blanks
    while j < len(lines) and not lines[j].strip():
      j += 1

    if j >= len(lines):
      i = j
      continue

    num_match = re.match(r'^(\d+\.?\d*)$', lines[j].strip())
    if not num_match:
      i = j
      continue

    parlay_line = float(num_match.group(1))
    j += 1

    if j >= len(lines):
      i = j
      continue

    stat_label = lines[j].strip()
    stat_key = PRIZEPICKS_STAT_MAP.get(stat_label)
    j += 1

    if stat_key is None:
      i = j
      continue

    # Look for "Less" and/or "More"
    has_less = False
    has_more = False
    k = j
    while k < len(lines) and k < j + 6:
      option = lines[k].strip()
      if option == "Less":
        has_less = True
      elif option == "More":
        has_more = True
        k += 1
        break
      elif option == "" or option.startswith("Trending") or re.match(r'^\d+\.?\d*K?$', option):
        pass
      else:
        break
      k += 1

    players.setdefault(candidate, {})
    if stat_key not in players[candidate]:
      players[candidate][stat_key] = {
        'line': parlay_line,
        'has_less': has_less,
        'has_more': has_more,
        'goblin_demon': goblin_demon_tag,
      }

    i = k
    continue

  return players


# ── Classify & merge logic ─────────────────────────────────

def classify_player(ud_stats: dict | None, pp_stats: dict | None):
  """
  Classify a player's lines and determine payload strategy.

  Payload groups:
    'combined'     — non-conflicting stats (shared, UD-only, PP-only):
                     use one combined payload
    'conflict_ud'  — conflicting stat, UD line: use combined payload
                     (UD line is already in there)
    'conflict_pp'  — conflicting stat, PP line: needs PP-only payload

  Returns:
    entries: list of line entry dicts
    has_conflicts: bool — True if any overlapping stat has different lines
  """
  ud_stats = ud_stats or {}
  pp_stats = pp_stats or {}

  all_stats = set(ud_stats.keys()) | set(pp_stats.keys())
  entries = []
  has_conflicts = False

  for stat in all_stats:
    ud = ud_stats.get(stat)
    pp = pp_stats.get(stat)

    if ud and pp:
      if ud['line'] == pp['line']:
        entries.append({
          'stat': stat,
          'line': ud['line'],
          'platform': 'Underdog and PrizePicks',
          'higher_multiplier': ud['higher_multiplier'],
          'lower_multiplier': ud['lower_multiplier'],
          'goblin_demon': pp['goblin_demon'],
          'pp_has_less': pp['has_less'],
          'pp_has_more': pp['has_more'],
          'payload_group': 'combined',
        })
      else:
        has_conflicts = True
        entries.append({
          'stat': stat,
          'line': ud['line'],
          'platform': 'Underdog',
          'higher_multiplier': ud['higher_multiplier'],
          'lower_multiplier': ud['lower_multiplier'],
          'goblin_demon': '',
          'pp_has_less': True,
          'pp_has_more': True,
          'payload_group': 'conflict_ud',
        })
        entries.append({
          'stat': stat,
          'line': pp['line'],
          'platform': 'PrizePicks',
          'higher_multiplier': 1.0,
          'lower_multiplier': 1.0,
          'goblin_demon': pp['goblin_demon'],
          'pp_has_less': pp['has_less'],
          'pp_has_more': pp['has_more'],
          'payload_group': 'conflict_pp',
        })
    elif ud:
      entries.append({
        'stat': stat,
        'line': ud['line'],
        'platform': 'Underdog',
        'higher_multiplier': ud['higher_multiplier'],
        'lower_multiplier': ud['lower_multiplier'],
        'goblin_demon': '',
        'pp_has_less': True,
        'pp_has_more': True,
        'payload_group': 'combined',
      })
    else:
      entries.append({
        'stat': stat,
        'line': pp['line'],
        'platform': 'PrizePicks',
        'higher_multiplier': 1.0,
        'lower_multiplier': 1.0,
        'goblin_demon': pp['goblin_demon'],
        'pp_has_less': pp['has_less'],
        'pp_has_more': pp['has_more'],
        'payload_group': 'combined',
      })

  return entries, has_conflicts


# ── Parlays builders ───────────────────────────────────────

def _build_combined_parlays(ud_stats: dict | None, pp_stats: dict | None) -> dict:
  """
  Build a single combined parlays dict for the combined payload:
    - UD line for UD stats (shared or UD-only — also covers conflict_ud)
    - PP line for PP-only stats
  """
  parlays = {f"PL_{col}": 0.0 for col in TARGET_COLUMNS}
  ud_stats = ud_stats or {}
  pp_stats = pp_stats or {}

  # UD lines first (takes priority for overlapping stats)
  for stat, data in ud_stats.items():
    pl_key = f"PL_{stat}"
    if pl_key in parlays:
      parlays[pl_key] = float(data['line'])

  # PP lines — only fill stats not already set by UD
  for stat, data in pp_stats.items():
    pl_key = f"PL_{stat}"
    if pl_key in parlays and parlays[pl_key] == 0.0:
      parlays[pl_key] = float(data['line'])

  return parlays


def _build_platform_parlays(stat_dict: dict) -> dict:
  """Build parlays dict from a single platform's raw stat dict."""
  parlays = {f"PL_{col}": 0.0 for col in TARGET_COLUMNS}
  for stat, data in stat_dict.items():
    pl_key = f"PL_{stat}"
    if pl_key in parlays:
      if isinstance(data, dict):
        parlays[pl_key] = float(data.get('line', 0.0))
      else:
        parlays[pl_key] = float(data)
  return parlays


# ── API helpers ────────────────────────────────────────────

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


def _prepare_payload(get_input, player_name, parlays):
  """Call get_input, sanitise, return payload dict or None on failure."""
  try:
    input_df, _ = get_input(
      player_name=player_name,
      parlays=parlays,
      seasons=SEASONS,
    )
  except Exception as e:
    print(f"  ❌ get_input failed: {e}")
    return None

  safe_df = input_df.copy()
  safe_df = safe_df.replace([float("inf"), float("-inf")], pd.NA)
  for col in safe_df.columns:
    if pd.api.types.is_numeric_dtype(safe_df[col]):
      safe_df[col] = pd.to_numeric(safe_df[col], errors="coerce").fillna(0.0)
  return safe_df.iloc[0].to_dict()


def _build_row(player_name, entry, result, model_variants):
  """Build a CSV row dict from a line entry and model result."""
  consensus = result.get("consensus", {})

  if not consensus:
    print("⚠️  missing consensus")
    return None

  prediction = consensus.get("prediction", "N/A")
  confidence = float(consensus.get("confidence", 0.0))
  probability_std = float(consensus.get("probability_std", 0.0))
  agreement_ratio = float(consensus.get("agreement_ratio", 0.0))
  betting = consensus.get("betting_analysis", {})
  tier = betting.get("stat_tier", "?")
  recommendation = betting.get("recommendation", "DO NOT BET")
  rank_score = float(consensus.get("Rank", {}).get("rank_score", 0.0))

  platform = entry['platform']
  goblin_demon = entry['goblin_demon']

  # PrizePicks-only, no Less option, model says UNDER → skip
  if platform == 'PrizePicks' and not entry['pp_has_less'] and prediction == "UNDER":
    print(f"⏭  SKIP (PrizePicks {goblin_demon} — no Less option, model says UNDER)")
    return None

  # Shared platform but PP has only More and model says UNDER →
  # downgrade to Underdog only
  if platform == 'Underdog and PrizePicks' and not entry['pp_has_less'] and prediction == "UNDER":
    platform = 'Underdog'
    goblin_demon = ''
    print(f"(PP no Less → Underdog only) ", end="")

  # Determine multiplier
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

  print(
    f"{prediction:<6} "
    f"{confidence*100:5.1f}%  "
    f"{multiplier:.2f}x  "
    f"Tier {tier}  "
    f"Rank {rank_score:.2f}  "
    f"Std {probability_std:.4f}  "
    f"Agr {agreement_ratio*100:5.1f}%"
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
  }

  for book_key, book_label in sportsbook_column_map:
    book = sportsbook_entries.get(book_key, {})
    row[f"{book_label} line"] = _float_or_blank(book.get("line"), digits=1)
    row[f"{book_label} line_delta"] = _float_or_blank(book.get("line_delta"), digits=1)
    row[f"{book_label} over_hit_prob"] = _pct_or_blank(book.get("over_hit_prob"))
    row[f"{book_label} under_hit_prob"] = _pct_or_blank(book.get("under_hit_prob"))
    row[f"{book_label} edge"] = _float_or_blank(book.get(chosen_edge_key), digits=4)

  return row


# ── Main pipeline ──────────────────────────────────────────
#
# Payload strategy (minimises get_input calls):
#
#   Player on BOTH platforms, NO conflicts (all overlapping lines match):
#     → 1 get_input  with combined parlays (UD lines + PP-only lines)
#     → 1 model call per unique stat
#
#   Player on BOTH platforms, HAS conflicts (≥1 overlapping stat differs):
#     → 1 get_input  with combined parlays  (for combined + conflict_ud stats)
#     → 1 get_input  with PP-only parlays   (for conflict_pp stats only)
#     → combined + conflict_ud entries use combined payload
#     → conflict_pp entries use PP payload
#     → model results cached within each payload so no duplicate calls
#
#   Player on ONE platform only:
#     → 1 get_input
#     → 1 model call per stat
#

def run_pipeline(underdog_path: str, prizepicks_path: str, output_csv: str):
  try:
    from utils import get_input
  except ImportError:
    print("ERROR: Cannot import get_input from utils.py.")
    print("       Place this script in the same directory as utils.py.")
    sys.exit(1)

  # ── Parse both files ──
  print(f"\n{'='*60}")
  print(f"  Parsing Underdog:  {underdog_path}")
  print(f"  Parsing PrizePicks: {prizepicks_path}")
  print(f"{'='*60}")

  ud_parsed = parse_underdog_txt(underdog_path)
  pp_parsed = parse_prizepicks_txt(prizepicks_path)

  print(f"\n  Underdog players:   {len(ud_parsed)}")
  print(f"  PrizePicks players: {len(pp_parsed)}")

  all_players = sorted(set(ud_parsed.keys()) | set(pp_parsed.keys()))
  print(f"  Total unique players: {len(all_players)}")

  total_get_input = 0
  total_model_calls = 0
  rows = []

  for player_name in all_players:
    ud_stats = ud_parsed.get(player_name)
    pp_stats = pp_parsed.get(player_name)

    entries, has_conflicts = classify_player(ud_stats, pp_stats)
    if not entries:
      continue

    print(f"\n{'─'*60}")
    has_ud = ud_stats is not None
    has_pp = pp_stats is not None

    if has_ud and has_pp:
      # ── Both platforms ──
      # Always build a combined payload (UD lines + PP-only lines)
      combined_parlays = _build_combined_parlays(ud_stats, pp_stats)
      combined_payload = _prepare_payload(get_input, player_name, combined_parlays)
      if combined_payload is None:
        continue
      total_get_input += 1

      # Only build a PP payload if there are actual conflicts
      pp_payload = None
      if has_conflicts:
        pp_parlays = _build_platform_parlays(pp_stats)
        pp_payload = _prepare_payload(get_input, player_name, pp_parlays)
        if pp_payload is not None:
          total_get_input += 1

      n_gi = 2 if has_conflicts and pp_payload else 1
      label = "lines differ" if has_conflicts else "all lines match"
      print(f"  {player_name}  ({len(entries)} lines, {n_gi} get_input — {label})")

      # Model caches: one per payload
      combined_cache = {}
      pp_cache = {}

      for entry in entries:
        stat = entry['stat']
        group = entry['payload_group']
        print(f"    {stat:<6} {entry['line']:<7} [{entry['platform']}] ", end="", flush=True)

        if group == 'conflict_pp':
          payload = pp_payload
          cache = pp_cache
        else:
          # 'combined' or 'conflict_ud' — both use combined payload
          payload = combined_payload
          cache = combined_cache

        if payload is None:
          print("⏭  payload unavailable")
          continue

        if stat in cache:
          result = cache[stat]
          if result is not None:
            print("(cached) ", end="", flush=True)
        else:
          result = call_model(payload, stat)
          cache[stat] = result
          total_model_calls += 1

        if result is None:
          continue

        model_variants = result.get("model_variants", {})
        row = _build_row(player_name, entry, result, model_variants)
        if row:
          rows.append(row)

    else:
      # ── Single platform only ──
      raw_stats = ud_stats or pp_stats
      parlays = _build_platform_parlays(raw_stats)
      payload = _prepare_payload(get_input, player_name, parlays)
      if payload is None:
        continue
      total_get_input += 1

      src = "Underdog" if has_ud else "PrizePicks"
      print(f"  {player_name}  ({len(entries)} lines, 1 get_input — {src} only)")

      model_cache = {}
      for entry in entries:
        stat = entry['stat']
        print(f"    {stat:<6} {entry['line']:<7} [{entry['platform']}] ", end="", flush=True)

        if stat in model_cache:
          result = model_cache[stat]
          if result is not None:
            print("(cached) ", end="", flush=True)
        else:
          result = call_model(payload, stat)
          model_cache[stat] = result
          total_model_calls += 1

        if result is None:
          continue

        model_variants = result.get("model_variants", {})
        row = _build_row(player_name, entry, result, model_variants)
        if row:
          rows.append(row)

  if not rows:
    print("\n⚠️  No predictions generated.")
    return

  print(f"\n{'─'*60}")
  print(f"  API summary:")
  print(f"    get_input calls: {total_get_input}")
  print(f"    call_model calls: {total_model_calls}")

  df_out = pd.DataFrame(rows).sort_values("Rank Score", ascending=False).reset_index(drop=True)
  df_out.index += 1
  df_out.index.name = "Rank"

  df_out.to_csv(output_csv)

  print(f"\n{'='*60}")
  print(f"  ✅ Saved → {output_csv}  ({len(df_out)} predictions)")
  print(f"{'='*60}\n")
  print(
    df_out[[
      "Player", "Stat", "Line", "Pick", "Platform", "Goblin/Demon",
      "Multiplier", "Confidence", "Tier", "Recommendation", "Rank Score",
      "probability_std", "Agreement ratio"
    ]].to_string()
  )


if __name__ == "__main__":
  run_pipeline(
    "lines/underdog_lines.txt",
    "lines/prizepicks_lines.txt",
    "today_picks.csv",
  )