import requests
import pandas as pd
from utils import (
	normalize_player_name,
	map_stat_name,
	extract_payout,
	get_input
)
from backend.constants import TARGET_COLUMNS

CHAMPION_API_URL = "http://127.0.0.1:8000/predict_champion"


def get_stat_full_name(stat_code):
	"""
	Convert stat code to full readable name.
	Returns full stat name (e.g., "PTS" -> "Points", "3PM" -> "Three Pointers Made")
	"""
	stat_mapping = {
		'PTS': 'Points',
		'REB': 'Rebounds',
		'AST': 'Assists',
		'STL': 'Steals',
		'BLK': 'Blocks',
		'PRA': 'Points + Rebounds + Assists',
		'PA': 'Points + Assists',
		'PR': 'Points + Rebounds',
		'RA': 'Rebounds + Assists',
		'SB': 'Steals + Blocks',
		'TOV': 'Turnovers',
		'FGM': 'Field Goals Made',
		'3PM': 'Three Pointers Made',
		'FTM': 'Free Throws Made',
		'FGA': 'Field Goals Attempted',
		'3PA': 'Three Pointers Attempted',
		'FTA': 'Free Throws Attempted'
	}
	return stat_mapping.get(stat_code, stat_code)


def parse_underdog_lines(file_path):
	"""
	Parse Underdog format text file.
	Returns list of dicts with player, stat, line, payout_higher, payout_lower, platform.
	"""
	lines = []
	with open(file_path, 'r', encoding='utf-8') as f:
		all_lines = [line.rstrip('\n') for line in f.readlines()]
	
	i = 0
	current_player = None
	
	while i < len(all_lines):
		line = all_lines[i].strip()
		
		# Check if this is a player name (followed by empty line, then team)
		if line and i + 2 < len(all_lines):
			next_line = all_lines[i + 1].strip()
			next_next_line = all_lines[i + 2].strip()
			
			# Pattern: player name, empty, team name (team names usually end with common suffixes)
			if not next_line and next_next_line:
				# Check if it looks like a team name (contains common team name words or is all caps)
				team_indicators = ['Timberwolves', 'Warriors', 'Lakers', 'Celtics', 'Nets', 'Bulls', 
								 'Mavericks', 'Nuggets', 'Pistons', 'Rockets', 'Pacers', 'Clippers',
								 'Heat', 'Bucks', 'Pelicans', 'Knicks', 'Thunder', 'Magic', '76ers',
								 'Suns', 'Trail Blazers', 'Kings', 'Spurs', 'Raptors', 'Jazz', 'Wizards',
								 'Hornets', 'Cavaliers', 'Hawks', 'Grizzlies']
				if any(indicator in next_next_line for indicator in team_indicators):
					current_player = line
					# Skip to after "Higher/Lower" line
					j = i + 1
					while j < len(all_lines) and all_lines[j].strip() != "Higher/Lower":
						j += 1
					i = j + 1
					continue
		
		# If we have a current player, look for stat patterns
		if current_player:
			# Pattern: stat value, stat name, empty, "Higher", [empty/payout], "Lower", [empty/payout]
			try:
				stat_value = float(line)
				# Next should be stat name
				if i + 1 < len(all_lines):
					stat_name = all_lines[i + 1].strip()
					payout_higher = 1.0
					payout_lower = 1.0
					
					# Look ahead for "Higher" and "Lower" with payouts
					j = i + 2
					higher_idx = None
					lower_idx = None
					
					# Find "Higher" and "Lower" positions
					while j < len(all_lines) and j < i + 12:
						check_line = all_lines[j].strip()
						if check_line == "Higher":
							higher_idx = j
						elif check_line == "Lower":
							lower_idx = j
							break  # Found both, stop
						j += 1
					
					# Extract payout for Higher
					if higher_idx is not None:
						# Check next 2 lines after "Higher"
						for k in range(higher_idx + 1, min(higher_idx + 3, len(all_lines))):
							check_payout = all_lines[k].strip()
							if check_payout and 'x' in check_payout:
								payout_higher = extract_payout(check_payout)
								break
							elif check_payout == "Lower":
								break
					
					# Extract payout for Lower
					if lower_idx is not None:
						# Check next 2 lines after "Lower"
						for k in range(lower_idx + 1, min(lower_idx + 3, len(all_lines))):
							check_payout = all_lines[k].strip()
							if check_payout and 'x' in check_payout:
								payout_lower = extract_payout(check_payout)
								break
							# If we hit another stat value (number), we're done
							try:
								float(check_payout)
								break
							except ValueError:
								pass
					
					lines.append({
						'player': current_player,
						'stat': stat_name,
						'line': stat_value,
						'payout_higher': payout_higher,
						'payout_lower': payout_lower,
						'platform': 'underdog'
					})
					
					# Skip ahead past this stat block (to after Lower payout or next stat)
					if lower_idx is not None:
						# Find where this stat block ends (next stat value or next player)
						i = lower_idx + 1
						while i < len(all_lines):
							check_line = all_lines[i].strip()
							# If it's a number (next stat) or a player name, stop
							try:
								float(check_line)
								break
							except ValueError:
								# Check if this might be a new player name (followed by empty then team)
								if check_line and not check_line.startswith('@') and i + 2 < len(all_lines):
									next_check = all_lines[i + 1].strip()
									next_next_check = all_lines[i + 2].strip()
									team_indicators = ['Timberwolves', 'Warriors', 'Lakers', 'Celtics', 'Nets', 'Bulls']
									if not next_check and any(indicator in next_next_check for indicator in team_indicators):
										break
							i += 1
						continue
			except ValueError:
				pass
		
		i += 1
	
	return lines


def parse_prizepicks_lines(file_path):
	"""
	Parse PrizePicks format text file.
	Returns list of dicts with player, stat, line, payout, platform, is_goblin, is_demon.
	"""
	lines = []
	with open(file_path, 'r', encoding='utf-8') as f:
		all_lines = [line.rstrip('\n') for line in f.readlines()]
	
	i = 0
	while i < len(all_lines):
		line = all_lines[i].strip()
		
		# Skip empty lines
		if not line:
			i += 1
			continue
		
		# Check if this is a player name (may have Goblin/Demon suffix)
		# Skip combined players (contain "+")
		if '+' in line:
			# Skip ahead to next player (look for pattern: player, team-pos, player, matchup)
			i += 1
			continue
		
		is_goblin = 'Goblin' in line
		is_demon = 'Demon' in line
		
		# Extract player name (remove Goblin/Demon suffix)
		player_name = line.replace('Goblin', '').replace('Demon', '').strip()
		
		# Pattern: player name, team-position, player name again, matchup, empty, stat value, stat name, Less, More
		# Check if we have enough lines ahead
		if i + 8 < len(all_lines):
			# Verify pattern: i+1 should be team-position, i+2 should be player name, i+3 should be matchup
			team_pos = all_lines[i + 1].strip()
			player_again = all_lines[i + 2].strip()
			matchup = all_lines[i + 3].strip()
			empty_line = all_lines[i + 4].strip()
			
			# Check if pattern matches
			if not empty_line and ('@' in matchup or 'vs' in matchup):
				# Stat value should be at i+5
				stat_value_line = all_lines[i + 5].strip()
				
				try:
					stat_value = float(stat_value_line)
					
					# Stat name should be at i+6
					if i + 6 < len(all_lines):
						stat_name = all_lines[i + 6].strip()
						
						# Default payout is 1.0x (PrizePicks doesn't show payout in text for regular lines)
						payout = 1.0
						
						lines.append({
							'player': player_name,
							'stat': stat_name,
							'line': stat_value,
							'payout': payout,
							'platform': 'prizepicks',
							'is_goblin': is_goblin,
							'is_demon': is_demon
						})
						
						# Skip ahead to next player (after More line)
						i += 9
						continue
				except (ValueError, IndexError):
					pass
		
		i += 1
	
	return lines


def get_prediction_from_input(input_df, stat, line_value):
	"""
	Get prediction from champion model API using pre-prepared input_df.
	Updates the PL_{stat} value in the payload before calling API.
	Returns dict with prediction, probability, confidence, and stat_params or None on error.
	"""
	try:
		# Create a copy of the input_df to avoid modifying the original
		payload = input_df.iloc[0].to_dict().copy()
		
		# Update the specific stat line value in the payload
		payload[f"PL_{stat}"] = float(line_value)
		
		# Extract only the stat being predicted for stat params
		stat_params = {stat: float(line_value)}
		
		# Call API
		resp = requests.post(
			CHAMPION_API_URL,
			params={"stat": stat},
			json=payload,
			timeout=30,
		)
		
		if resp.status_code != 200:
			print(f"API error for {stat}: {resp.text}")
			return None
		
		result = resp.json()
		
		pred = int(result["prediction"])  # 1=OVER, 0=UNDER
		prob_over = float(result["probability"])
		conf_pick = prob_over if pred == 1 else (1 - prob_over)
		confidence = conf_pick * 100.0
		
		return {
			'prediction': pred,
			'probability': prob_over,
			'confidence': confidence,
			'stat_params': stat_params
		}
	
	except Exception as e:
		print(f"Error getting prediction for {stat}: {str(e)}")
		return None


def get_prediction(player, stat, line):
	"""
	Get prediction from champion model API (legacy function for backward compatibility).
	Returns dict with prediction, probability, confidence or None on error.
	"""
	try:
		# Prepare input using existing get_input function
		parlays = {f"PL_{c}": 0.0 for c in TARGET_COLUMNS}
		parlays[f"PL_{stat}"] = float(line)
		
		input_df, _ = get_input(
			player_name=player,
			parlays=parlays,
			seasons=["2024-25", "2025-26"],
		)
		
		return get_prediction_from_input(input_df, stat, line)
	
	except Exception as e:
		print(f"Error getting prediction for {player} {stat}: {str(e)}")
		return None


def get_predictions_for_player(player_name, all_lines_for_player):
	"""
	Process all lines for a single player.
	Calls get_input once per player with all their stats, then reuses input_df for all predictions.
	Returns list of result dicts.
	"""
	results = []
	
	# Normalize player name
	normalized_player = normalize_player_name(player_name)
	if not normalized_player:
		return results
	
	# Collect all unique stats and their line values
	# Use first encountered value for each stat if duplicates exist
	parlays = {f"PL_{c}": 0.0 for c in TARGET_COLUMNS}
	processed_lines = []
	
	for line_data in all_lines_for_player:
		platform = line_data.get('platform', 'unknown')
		stat_name = line_data['stat']
		
		# Map stat name based on platform
		mapped_stat = map_stat_name(stat_name, platform)
		if not mapped_stat:
			continue
		
		line_value = line_data['line']
		
		# Store line value for this stat (use first encountered if duplicate)
		if f"PL_{mapped_stat}" not in parlays or parlays[f"PL_{mapped_stat}"] == 0.0:
			parlays[f"PL_{mapped_stat}"] = float(line_value)
		
		# Store processed line data for later use
		processed_lines.append({
			'line_data': line_data,
			'mapped_stat': mapped_stat,
			'line_value': line_value,
			'platform': platform
		})
	
	# Filter out BLK if SB exists for this player
	# When a player has both "Blocks" and "Blocks + Steals" lines, 
	# we should only process "Blocks + Steals" (SB) since it's more comprehensive
	has_sb = any(pl['mapped_stat'] == 'SB' for pl in processed_lines)
	if has_sb:
		processed_lines = [pl for pl in processed_lines if pl['mapped_stat'] != 'BLK']
		# Also remove BLK from parlays if it was added
		if 'PL_BLK' in parlays:
			parlays['PL_BLK'] = 0.0
	
	# If no valid stats found, return empty results
	if not processed_lines:
		return results
	
	# Call get_input once for this player with all stats
	try:
		input_df, _ = get_input(
			player_name=normalized_player,
			parlays=parlays,
			seasons=["2024-25", "2025-26"],
		)
	except Exception as e:
		print(f"Error getting input for {normalized_player}: {str(e)}")
		return results
	
	# Process each line using the same input_df
	for processed_line in processed_lines:
		line_data = processed_line['line_data']
		mapped_stat = processed_line['mapped_stat']
		line_value = processed_line['line_value']
		platform = processed_line['platform']
		
		# Get prediction using the pre-prepared input_df
		prediction_data = get_prediction_from_input(input_df, mapped_stat, line_value)
		if not prediction_data:
			continue
		
		pred = prediction_data['prediction']
		confidence = prediction_data['confidence']
		stat_params = prediction_data.get('stat_params', {})
		
		# Handle platform-specific logic
		if platform == 'underdog':
			payout_higher = line_data.get('payout_higher', 1.0)
			payout_lower = line_data.get('payout_lower', 1.0)
			
			# Determine payout based on prediction direction
			if pred == 1:  # OVER
				payout = payout_higher
				# Check if Higher direction is available
				if payout_higher is None or payout_higher == 0:
					continue
			else:  # UNDER
				payout = payout_lower
				# Check if Lower direction is available
				if payout_lower is None or payout_lower == 0:
					continue
			
			result_entry = {
				'player': normalized_player,
				'stat': mapped_stat,
				'line': line_value,
				'payout': payout,
				'confidence': confidence,
				'prediction': 'OVER' if pred == 1 else 'UNDER',
				'platform': 'underdog',
				'is_goblin': False,
				'is_demon': False,
				'stat_params': stat_params
			}
			results.append(result_entry)
		
		elif platform == 'prizepicks':
			payout = line_data.get('payout', 1.0)
			is_goblin = line_data.get('is_goblin', False)
			is_demon = line_data.get('is_demon', False)
			
			# Filter out if direction constraint conflicts with prediction
			# Goblins and demons must pick OVER (More)
			if (is_goblin or is_demon) and pred == 0:
				continue  # Model says UNDER but must pick OVER
			
			result_entry = {
				'player': normalized_player,
				'stat': mapped_stat,
				'line': line_value,
				'payout': payout,
				'confidence': confidence,
				'prediction': 'OVER' if pred == 1 else 'UNDER',
				'platform': 'prizepicks',
				'is_goblin': is_goblin,
				'is_demon': is_demon,
				'stat_params': stat_params
			}
			results.append(result_entry)
	
	return results


def analyze_all_lines():
	"""
	Main function that parses both files, validates, gets predictions, filters, and outputs to CSV.
	Optimized to call get_input once per player instead of once per line.
	"""
	print("Parsing Underdog lines...")
	underdog_lines = parse_underdog_lines('underdog_lines.txt')
	print(f"Found {len(underdog_lines)} Underdog lines")
	
	print("Parsing PrizePicks lines...")
	prizepicks_lines = parse_prizepicks_lines('prizepicks_lines.txt')
	print(f"Found {len(prizepicks_lines)} PrizePicks lines")
	
	# Group lines by player
	print("\nGrouping lines by player...")
	lines_by_player = {}
	
	# Add platform info to each line_data
	for line_data in underdog_lines:
		line_data['platform'] = 'underdog'
		player = line_data['player']
		if player not in lines_by_player:
			lines_by_player[player] = []
		lines_by_player[player].append(line_data)
	
	for line_data in prizepicks_lines:
		# Skip combined players (should already be filtered, but double-check)
		if '+' in line_data.get('player', ''):
			continue
		line_data['platform'] = 'prizepicks'
		player = line_data['player']
		if player not in lines_by_player:
			lines_by_player[player] = []
		lines_by_player[player].append(line_data)
	
	print(f"Found {len(lines_by_player)} unique players")
	
	# Process each player
	results = []
	print("\nProcessing players...")
	for player_name, all_lines in lines_by_player.items():
		player_results = get_predictions_for_player(player_name, all_lines)
		results.extend(player_results)
	
	# Sort results: primary by confidence (descending), secondary by payout (descending)
	results.sort(key=lambda x: (-x['confidence'], -x['payout']))
	
	# Reformat results for CSV output
	reformatted_results = []
	for result in results:
		# Get full stat name
		stat_full_name = get_stat_full_name(result['stat'])
		
		# Combine line and stat: "3.5 Three Pointers Made"
		line_stat = f"{result['line']} {stat_full_name}"
		
		# Format demon/goblin column
		if result.get('is_demon', False):
			demon_goblin = 'Demon'
		elif result.get('is_goblin', False):
			demon_goblin = 'Goblin'
		else:
			demon_goblin = ''
		
		# Format stat params as a string showing only the predicted stat and its parlay line
		# Format: "SB:0.5" (only the stat being predicted)
		stat_params = result.get('stat_params', {})
		stat_params_str = ','.join([f"{stat}:{value}" for stat, value in stat_params.items()])
		
		reformatted_results.append({
			'player': result['player'],
			'prediction': result['prediction'],
			'line_stat': line_stat,
			'confidence': result['confidence'],
			'payout': result['payout'],
			'platform': result['platform'],
			'demon_goblin': demon_goblin,
			'stat_params': stat_params_str
		})
	
	# Output to CSV
	df = pd.DataFrame(reformatted_results)
	output_file = 'parlay_analysis_results.csv'
	df.to_csv(output_file, index=False)
	print(f"\nResults saved to {output_file}")
	print(f"Total valid lines: {len(reformatted_results)}")
	
	return results


if __name__ == "__main__":
	analyze_all_lines()
