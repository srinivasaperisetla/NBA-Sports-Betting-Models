import streamlit as st
import requests
from backend.constants import TARGET_COLUMNS, ALLOWED_PLAYERS_LIST
from utils import get_input, COLOR
import pandas as pd
import numpy as np
import altair as alt

CHAMPION_API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="NBA Parlay Predictor", layout="centered")
st.title("🏀 NBA Parlay Predictor")

mode = st.radio("Prediction Mode", ["Champion (single stat)"], horizontal=True)

player = st.selectbox("Select a Player", [""] + ALLOWED_PLAYERS_LIST)
if player == "":
  player = None

if mode == "Champion (single stat)":
  stat = st.selectbox("Select Stat", TARGET_COLUMNS)
  parlay_line = st.number_input(f"Enter {stat} Parlay Line", min_value=0.0, step=0.5)
  parlays = {f"PL_{c}": 0.0 for c in TARGET_COLUMNS}
  parlays[f"PL_{stat}"] = float(parlay_line)

run = st.button("Go")

if run:
  if player is None:
    st.error("Please select a player.")
    st.stop()

  with st.spinner("Running model..."):
    input_df, df_player = get_input(
      player_name=player,
      parlays=parlays,
      seasons=["2024-25", "2025-26"],
    )

    payload = input_df.iloc[0].to_dict()

    resp = requests.post(
      CHAMPION_API_URL,
      params={"stat": stat, "family": "ALL"},
      json=payload,
      timeout=30
    )

    if resp.status_code != 200:
      st.error(f"API Error: {resp.status_code}")
      st.error(resp.text)
      st.stop()

    result = resp.json()

  consensus = result.get("consensus", {})
  model_variants = result.get("model_variants", {})

  if not consensus:
    st.error("No consensus output returned from API.")
    st.stop()

  # ================================================================
  # PREDICTION SUMMARY
  # ================================================================
  prediction = consensus["prediction"]
  model_output = float(consensus["model_output_avg"])
  confidence = float(consensus["confidence"])
  probability_std = float(consensus.get("probability_std", 0.0))
  agreement_ratio = float(consensus.get("agreement_ratio", 0.0))
  betting = consensus.get("betting_analysis", {})
  overall_rank = float(consensus.get("Rank", {}).get("rank_score", 0.0))

  pred = 1 if prediction == "OVER" else 0
  conf_pct = confidence * 100.0
  agreement_pct = agreement_ratio * 100.0
  player_name = input_df["PLAYER_NAME"].iloc[0]
  pick_text = "↑ OVER" if pred == 1 else "↓ UNDER"
  pick_color = COLOR["over"] if pred == 1 else COLOR["under"]
  conf_norm = (conf_pct - 50.0) / 20.0 * 100.0
  conf_norm = max(0.0, min(conf_norm, 100.0))

  rec = betting.get("recommendation", "DO NOT BET")
  min_conf_val = float(betting.get("minimum_confidence", 0.55))
  opt_conf_val = float(betting.get("optimal_confidence", 0.60))
  midpoint_val = (min_conf_val + opt_conf_val) / 2

  if rec == "STRONG BET":
    rec_color = COLOR["over_strong"]
  elif rec == "BET":
    rec_color = COLOR["over"]
  elif rec == "BET WITH CAUTION":
    rec_color = COLOR["no_bet"]
  else:
    rec_color = COLOR["under"]

  st.markdown(f"## {player_name} {parlay_line:g} {stat}")

  r1c1, r1c2 = st.columns([1, 1.5])
  with r1c1:
    st.markdown(
      f"""
      <div style="font-size:0.85rem; margin:0 0 2px 0; color:{COLOR["text"]};">Pick</div>
      <div style="font-weight:1000; font-size:1.8rem; margin:0; color:{pick_color};">{pick_text}</div>
      """,
      unsafe_allow_html=True
    )
  with r1c2:
    st.markdown(
      f"""
      <div style="font-size:0.85rem; margin:0 0 2px 0; color:{COLOR["text"]};">Recommendation</div>
      <div style="font-weight:1000; font-size:1.8rem; margin:0; color:{rec_color};">{rec}</div>
      """,
      unsafe_allow_html=True
    )

  st.caption(
    f"Consensus avg P(OVER) = {model_output:.4f} | Std = {probability_std:.4f} | "
    f"Agreement = {agreement_pct:.1f}% | "
    f"{betting.get('model_accuracy_at_confidence', '')}"
  )

  st.subheader(f"Consensus Confidence = {conf_pct:.1f}%")
  st.markdown(
    f"""
    <div style="position:relative; width:100%; height:18px; border-radius:10px; overflow:hidden;">
      <div style="position:absolute; top:0; left:0; width:100%; height:100%;
        background:linear-gradient(
          to right,
          {COLOR['under']} 0%,
          {COLOR['no_bet']} 2%,
          {COLOR['no_bet']} 25%,
          {COLOR['over']} 50%,
          {COLOR['over_strong']} 100%
        );
      "></div>
      <div style="position:absolute; top:0; right:0; width:{100 - conf_norm:.1f}%; height:100%; background:{COLOR['neutral']};"></div>
    </div>
    """,
    unsafe_allow_html=True
  )

  # ================================================================
  # CONSENSUS METRICS
  # ================================================================
  st.markdown("## Consensus Metrics")

  c1, c2, c3 = st.columns(3)
  c1.metric("Probability Std", f"{probability_std:.4f}")
  c2.metric("Agreement Ratio", f"{agreement_pct:.1f}%")
  c3.metric("Overall Rank", f"{overall_rank:.2f}")

  # ================================================================
  # MODEL FAMILY BREAKDOWN
  # ================================================================
  st.markdown("## Model Family Breakdown")

  if model_variants:
    family_cols = st.columns(len(model_variants))
    for i, (family_name, family_result) in enumerate(model_variants.items()):
      fam_conf = float(family_result.get("confidence", 0.0)) * 100.0
      fam_rank = float(family_result.get("Rank", {}).get("rank_score", 0.0))
      fam_pred = family_result.get("prediction", "-")

      with family_cols[i]:
        st.markdown(f"**{family_name}**")
        st.metric("Prediction", fam_pred)
        st.metric("Confidence", f"{fam_conf:.1f}%")
        st.metric("Rank", f"{fam_rank:.2f}")

  # ================================================================
  # BETTING ANALYSIS
  # ================================================================
  if betting:
    st.markdown("## Betting Analysis")

    b1, b2, b3 = st.columns(3)
    b1.metric("Stat Tier", betting["stat_tier"])
    b2.metric("Min Confidence", f"{min_conf_val:.0%}")
    b3.metric("Optimal Confidence", f"{opt_conf_val:.0%}")

    st.caption(f"Min confidence: {min_conf_val:.0%} | Optimal confidence: {opt_conf_val:.0%} | Midpoint: {midpoint_val:.0%}")
    st.caption(betting.get("model_base_accuracy", ""))

  # ================================================================
  # SIGNAL STRENGTH
  # ================================================================
  st.markdown("## Signal Strength")

  momentum = float(input_df.get(f"{stat}_MOMENTUM", pd.Series([0.0])).iloc[0])
  z_line = float(input_df.get(f"{stat}_Z_LINE", pd.Series([0.0])).iloc[0])
  z_recent = float(input_df.get(f"{stat}_Z_RECENT", pd.Series([0.0])).iloc[0])
  z_matchup = float(input_df.get(f"{stat}_Z_MATCHUP", pd.Series([0.0])).iloc[0])
  line_diff = float(input_df.get(f"{stat}_LINE_DIFF", pd.Series([0.0])).iloc[0])
  cum_avg = float(input_df.get(f"CUM_AVG_{stat}", pd.Series([0.0])).iloc[0])
  l5_avg = float(input_df.get(f"L5_AVG_{stat}", pd.Series([0.0])).iloc[0])
  matchup_avg = float(input_df.get(f"MATCHUP_L4_AVG_{stat}", pd.Series([0.0])).iloc[0])

  s1, s2, s3, s4 = st.columns(4)
  s1.metric(f"{stat} Momentum", f"{momentum:+.2f}")
  s2.metric("Line Diff", f"{line_diff:+.2f}")
  s3.metric("Z vs Line", f"{z_line:+.2f}")
  s4.metric("Z vs Recent", f"{z_recent:+.2f}")

  s5, s6, s7, s8 = st.columns(4)
  s5.metric("Z vs Matchup", f"{z_matchup:+.2f}")
  s6.metric(f"Season Avg {stat}", f"{cum_avg:.2f}")
  s7.metric(f"Last 5 Avg {stat}", f"{l5_avg:.2f}")
  s8.metric(f"Matchup Avg {stat}", f"{matchup_avg:.2f}")

  with st.expander("How to interpret these (quick guide)"):
    st.markdown(
      """
- **Momentum**: `last5_avg − season_avg`. Positive = trending up; negative = trending down.
- **Line Diff**: `line − last5_avg`. Positive = line is above recent form; negative = below.
- **Z vs Line**: `(line − season_avg) / last5_std`. How far the line is from the season average scaled by recent volatility.
- **Z vs Recent**: `(line − last5_avg) / last5_std`. Same but anchored to recent form.
- **Z vs Matchup**: Z-score relative to historical performance vs this opponent.
- **Large positive Z** → line is above typical output (harder to hit OVER).
- **Large negative Z** → line is below typical output (easier to hit OVER).
- **Season Avg / Last 5 Avg / Matchup Avg**: Raw average values for context.
      """.strip()
    )

  # ================================================================
  # GAME CONTEXT
  # ================================================================
  st.markdown("## Game Context")

  matchup = input_df["MATCHUP"].iloc[0]
  is_home = int(input_df["HOME"].iloc[0]) == 1
  postseason = int(input_df["POSTSEASON"].iloc[0]) == 1
  b2b = int(input_df["BACK_TO_BACK"].iloc[0]) == 1
  game_date = input_df["GAME_DATE"].iloc[0]
  rest_days = int(input_df["PLAYER_REST_DAYS"].iloc[0])

  g1, g2, g3, g4 = st.columns(4)
  g1.metric("Matchup", f"{input_df['TEAM'].iloc[0]} vs {matchup}" if is_home else f"{input_df['TEAM'].iloc[0]} @ {matchup}")
  g2.metric("Season Type", "Postseason" if postseason else "Regular Season")
  g3.metric("Back-to-Back", "Yes" if b2b else "No")
  g4.metric("Rest Days", rest_days)
  st.caption(f"Game Date: {game_date}")

  # ================================================================
  # OVER RATE + LAST 10 BAR CHART
  # ================================================================
  st.markdown(f"## {stat} Over Rate — Last 10 Games")

  last10 = df_player.tail(10)[["GAME_DATE", stat]].copy()
  last10["GAME_DATE"] = pd.to_datetime(last10["GAME_DATE"], errors="coerce")
  last10[stat] = pd.to_numeric(last10[stat], errors="coerce")
  last10 = last10.dropna(subset=["GAME_DATE", stat]).reset_index(drop=True)
  last10["Hit"] = np.where(last10[stat] >= parlay_line, "Over", "Under")
  over_rate_l10 = (last10["Hit"] == "Over").mean()

  if len(last10) > 0:
    last10_plot = last10.copy()
    last10_plot["Game"] = last10_plot["GAME_DATE"].dt.strftime("%m/%d")

    bars = (
      alt.Chart(last10_plot)
      .mark_bar()
      .encode(
        x=alt.X("Game:N", title="Game"),
        y=alt.Y(f"{stat}:Q", title=stat),
        color=alt.Color(
          "Hit:N",
          scale=alt.Scale(domain=["Over", "Under"], range=[COLOR["over"], COLOR["under"]]),
          legend=alt.Legend(title="Result vs Line"),
        ),
        tooltip=[
          alt.Tooltip("GAME_DATE:T", title="Date"),
          alt.Tooltip(f"{stat}:Q", title=stat, format=".1f"),
          alt.Tooltip("Hit:N", title="Vs Line"),
        ],
      )
    )
    rule = (
      alt.Chart(pd.DataFrame({"y": [float(parlay_line)]}))
      .mark_rule(color="#F7DC6F")
      .encode(y="y:Q")
    )
    st.altair_chart((bars + rule).properties(height=260), use_container_width=True)
    st.caption(f"Over rate last 10 games: {over_rate_l10:.0%}")
  else:
    st.warning("Not enough recent games to chart last 10.")

  # ================================================================
  # OVER RATE VS MATCHUP
  # ================================================================
  st.markdown(f"## {stat} Over Rate vs {matchup} (Recent Matchups)")

  matchups = df_player[df_player["MATCHUP"] == matchup].copy()
  matchups = matchups.sort_values("GAME_DATE").tail(10)
  matchups["GAME_DATE"] = pd.to_datetime(matchups["GAME_DATE"], errors="coerce")
  matchups[stat] = pd.to_numeric(matchups[stat], errors="coerce")
  matchups = matchups.dropna(subset=["GAME_DATE", stat]).reset_index(drop=True)
  matchups["Hit"] = np.where(matchups[stat] >= parlay_line, "Over", "Under")

  if len(matchups) > 0:
    over_rate_matchup = (matchups["Hit"] == "Over").mean()
    matchups_plot = matchups.copy()
    matchups_plot["Game"] = matchups_plot["GAME_DATE"].dt.strftime("%m/%d")

    bars = (
      alt.Chart(matchups_plot)
      .mark_bar()
      .encode(
        x=alt.X("Game:N", title="Game vs Opponent"),
        y=alt.Y(f"{stat}:Q", title=stat),
        color=alt.Color(
          "Hit:N",
          scale=alt.Scale(domain=["Over", "Under"], range=[COLOR["over"], COLOR["under"]]),
          legend=alt.Legend(title="Result vs Line"),
        ),
        tooltip=[
          alt.Tooltip("GAME_DATE:T", title="Date"),
          alt.Tooltip(f"{stat}:Q", title=stat, format=".1f"),
          alt.Tooltip("Hit:N", title="Vs Line"),
        ],
      )
    )
    rule = (
      alt.Chart(pd.DataFrame({"y": [float(parlay_line)]}))
      .mark_rule(color=COLOR["no_bet"])
      .encode(y="y:Q")
    )
    st.altair_chart((bars + rule).properties(height=260), use_container_width=True)
    st.caption(f"Over rate vs {matchup}: {over_rate_matchup:.0%} ({len(matchups)} games)")
  else:
    st.warning("No recent matchups against this opponent.")

  # ================================================================
  # SEASON AVERAGES (CUM vs L5)
  # ================================================================
  st.markdown(f"## 2025-26 {stat} Averages (Cumulative vs Last 5)")

  season_df = df_player[df_player["SEASON_YEAR"] == "2025-26"].copy()
  season_df["GAME_DATE"] = pd.to_datetime(season_df["GAME_DATE"], errors="coerce")
  season_df[stat] = pd.to_numeric(season_df[stat], errors="coerce")
  season_df = season_df.dropna(subset=["GAME_DATE", stat]).sort_values("GAME_DATE")

  if len(season_df) >= 2:
    season_df["CUM_AVG"] = season_df[stat].expanding().mean()
    season_df["LAST5_AVG"] = season_df[stat].rolling(5, min_periods=1).mean()

    avg_plot = (
      alt.Chart(season_df)
      .transform_fold(["CUM_AVG", "LAST5_AVG"], as_=["series", "value"])
      .mark_line()
      .encode(
        x=alt.X("GAME_DATE:T", title="Game Date"),
        y=alt.Y("value:Q", title=f"{stat} Average"),
        color=alt.Color(
          "series:N",
          scale=alt.Scale(domain=["CUM_AVG", "LAST5_AVG"], range=["#5DADE2", "#AF7AC5"]),
          legend=alt.Legend(title=""),
        ),
        tooltip=[
          alt.Tooltip("GAME_DATE:T", title="Date"),
          alt.Tooltip("series:N", title="Series"),
          alt.Tooltip("value:Q", title="Avg", format=".2f"),
        ],
      )
      .properties(height=240)
    )
    st.altair_chart(avg_plot, use_container_width=True)
  else:
    st.info("Not enough 2025-26 games to show cumulative/last5 averages.")

  # ================================================================
  # SEASON VOLATILITY (CUM vs L5 STD)
  # ================================================================
  st.markdown(f"## 2025-26 {stat} Volatility (Cumulative vs Last 5 STD)")

  if len(season_df) >= 2:
    season_df["CUM_STD"] = season_df[stat].expanding().std(ddof=0).fillna(0.0)
    season_df["LAST5_STD"] = season_df[stat].rolling(5, min_periods=2).std(ddof=0).fillna(0.0)

    std_plot = (
      alt.Chart(season_df)
      .transform_fold(["CUM_STD", "LAST5_STD"], as_=["series", "value"])
      .mark_line()
      .encode(
        x=alt.X("GAME_DATE:T", title="Game Date"),
        y=alt.Y("value:Q", title=f"{stat} STD"),
        color=alt.Color(
          "series:N",
          scale=alt.Scale(domain=["CUM_STD", "LAST5_STD"], range=["#48C9B0", "#F5B041"]),
          legend=alt.Legend(title=""),
        ),
        tooltip=[
          alt.Tooltip("GAME_DATE:T", title="Date"),
          alt.Tooltip("series:N", title="Series"),
          alt.Tooltip("value:Q", title="STD", format=".2f"),
        ],
      )
      .properties(height=240)
    )
    st.altair_chart(std_plot, use_container_width=True)
  else:
    st.info("Not enough 2025-26 games to show volatility trends.")

  # ================================================================
  # SEASON MOMENTUM
  # ================================================================
  st.markdown(f"## 2025-26 {stat} Momentum (Last 5 vs Season Baseline)")

  if len(season_df) >= 5:
    season_df["MOMENTUM"] = season_df["LAST5_AVG"] - season_df["CUM_AVG"]

    momentum_plot = (
      alt.Chart(season_df)
      .mark_line()
      .encode(
        x=alt.X("GAME_DATE:T", title="Game Date"),
        y=alt.Y("MOMENTUM:Q", title=f"{stat} Momentum"),
        color=alt.value(COLOR["over"]),
        tooltip=[
          alt.Tooltip("GAME_DATE:T", title="Date"),
          alt.Tooltip("MOMENTUM:Q", title="Momentum", format="+.2f"),
        ],
      )
    )
    zero_rule = (
      alt.Chart(pd.DataFrame({"y": [0]}))
      .mark_rule(strokeDash=[4, 4], color=COLOR["neutral"])
      .encode(y="y:Q")
    )
    st.altair_chart((momentum_plot + zero_rule).properties(height=220), use_container_width=True)
  else:
    st.info("Not enough games to compute momentum reliably.")