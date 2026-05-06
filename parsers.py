"""Parse Underdog and PrizePicks text line files into plain dicts.

No dependencies on models, utils, or the NBA API.
"""
import re
import unicodedata

try:
  from unidecode import unidecode as _norm
except ImportError:
  def _norm(s: str) -> str:
    return ''.join(
      c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn'
    )

from config import ALLOWED_PLAYERS_LIST, UNDERDOG_STAT_MAP, PRIZEPICKS_STAT_MAP

_ORIGINAL_NAMES = sorted(ALLOWED_PLAYERS_LIST)
_PLAYER_LOOKUP = {_norm(p).lower(): p for p in _ORIGINAL_NAMES}

required_names = set(ALLOWED_PLAYERS_LIST)


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
  """
  Parse Underdog text file.

  Returns:
    { canonical_player_name: {
        stat_key: {
          'line': float,
          'higher_multiplier': float,
          'lower_multiplier': float
        }, ...
      }, ...
    }
  """
  with open(filepath, "r", encoding="utf-8") as f:
    lines = [line.rstrip() for line in f]

  players = {}
  current_player = None
  i = 0

  while i < len(lines):
    line = lines[i].strip()

    if _is_underdog_player_header(lines, i):
      candidate = normalize_player(line)
      if candidate:
        current_player = candidate
        players.setdefault(current_player, {})
      else:
        current_player = None
      i += 5
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

        j = i + 2
        while j < len(lines) and j < i + 10:
          current_line = lines[j].strip()

          if re.match(r'^(\d+\.?\d*)$', current_line):
            if j + 1 < len(lines):
              next_stat = lines[j + 1].strip()
              if (next_stat in UNDERDOG_STAT_MAP or
                  re.match(r'^(Double Doubles|Triple Doubles|Fantasy Points|'
                           r'1[QH] .+)$', next_stat)):
                break
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

    if i + 1 >= len(lines):
      i += 1
      continue

    team_line = lines[i + 1].strip()
    if not re.match(r'^[A-Z]{2,3}\s*-\s*[A-Z]', team_line):
      i += 1
      continue

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
    'combined'     — non-conflicting stats (shared, UD-only, PP-only)
    'conflict_ud'  — conflicting stat, UD line
    'conflict_pp'  — conflicting stat, PP line: needs PP-only payload

  Returns:
    entries: list of line entry dicts
    has_conflicts: bool
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

def build_combined_parlays(ud_stats: dict | None, pp_stats: dict | None, target_columns: list) -> dict:
  """
  Build a combined parlays dict:
    - UD line for UD stats (takes priority for overlapping stats)
    - PP line for PP-only stats
  """
  parlays = {f"PL_{col}": 0.0 for col in target_columns}
  ud_stats = ud_stats or {}
  pp_stats = pp_stats or {}

  for stat, data in ud_stats.items():
    pl_key = f"PL_{stat}"
    if pl_key in parlays:
      parlays[pl_key] = float(data['line'])

  for stat, data in pp_stats.items():
    pl_key = f"PL_{stat}"
    if pl_key in parlays and parlays[pl_key] == 0.0:
      parlays[pl_key] = float(data['line'])

  return parlays


def build_platform_parlays(stat_dict: dict, target_columns: list) -> dict:
  """Build parlays dict from a single platform's raw stat dict."""
  parlays = {f"PL_{col}": 0.0 for col in target_columns}
  for stat, data in stat_dict.items():
    pl_key = f"PL_{stat}"
    if pl_key in parlays:
      if isinstance(data, dict):
        parlays[pl_key] = float(data.get('line', 0.0))
      else:
        parlays[pl_key] = float(data)
  return parlays


if __name__ == "__main__":
  import json

  ud = parse_underdog_txt("lines/underdog_lines.txt")
  pp = parse_prizepicks_txt("lines/prizepicks_lines.txt")

  print(f"Underdog players: {len(ud)}")
  for player, stats in list(ud.items())[:3]:
    print(f"  {player}: {list(stats.keys())}")

  print(f"\nPrizePicks players: {len(pp)}")
  for player, stats in list(pp.items())[:3]:
    print(f"  {player}: {list(stats.keys())}")

  all_players = sorted(set(ud.keys()) | set(pp.keys()))
  print(f"\nTotal unique players: {len(all_players)}")
