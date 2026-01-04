import streamlit as st
import requests
from backend.constants import TARGET_COLUMNS, ALLOWED_PLAYERS_LIST
from utils import get_input, advice_and_color, COLOR
import pandas as pd
import numpy as np
import altair as alt

CHAMPION_API_URL = "http://127.0.0.1:8000/predict_champion"
ALLSTAR_API_URL = "http://127.0.0.1:8000/predict_allstar"

st.set_page_config(page_title="NBA Parlay Predictor", layout="centered")
st.title("🏀 NBA Parlay Predictor")

mode = st.radio(
  "Prediction Mode",
  ["Champion (single stat)", #"AllStar (all stats)"
   
   ],
  horizontal=True
)

player = st.selectbox("Select a Player",  [""] + ALLOWED_PLAYERS_LIST)
if player == "":
  player = None

if mode == "Champion (single stat)":
  stat = st.selectbox("Select Stat", TARGET_COLUMNS)
  parlay_line = st.number_input(f"Enter {stat} Parlay Line", min_value=0.0, step=0.5)

  parlays = {f"PL_{c}": 0.0 for c in TARGET_COLUMNS}
  parlays[f"PL_{stat}"] = float(parlay_line)

else:
  st.subheader("Enter Parlay Lines (All Stats)")
  # Simple layout: 3 columns of inputs
  cols = st.columns(3)
  parlays = {}

  for i, c in enumerate(TARGET_COLUMNS):
    with cols[i % 3]:
      parlays[f"PL_{c}"] = float(
        st.number_input(f"{c} Line", min_value=0.0, step=0.5, key=f"line_{c}")
      )


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
      params={"stat": stat},
      json=payload,
      timeout=30,
    )

    if resp.status_code != 200:
      st.error(resp.text)
      st.stop()

    result = resp.json()

  # ------------------------
  # Prediction Summary
  # ------------------------
  pred = int(result["prediction"])                  # 1=OVER, 0=UNDER
  prob_over = float(result["probability"])
  conf_pick = prob_over if pred == 1 else (1 - prob_over)
  conf_pct = conf_pick * 100.0
  advice_text, advice_color = advice_and_color(pred, conf_pct)
  player_name = input_df["PLAYER_NAME"].iloc[0]
  pick_text = "↑ OVER" if pred == 1 else "↓ UNDER"
  pick_color = COLOR["over"] if pred == 1 else COLOR["under"]
  conf_norm = (conf_pct - 50.0) / 20.0 * 100.0
  conf_norm = max(0.0, min(conf_norm, 100.0))
  ambiguity = int(input_df.get(f"LINE_AMBIGUITY_{stat}", pd.Series([0])).iloc[0])

  st.markdown(f"## {player_name} {parlay_line:g} {stat}")

  r1c1, r1c2 = st.columns([1, 1.5])
  with r1c1:
    st.markdown(
      f"""
      <div style="font-size:0.85rem; margin:0 0 2px 0; color:{COLOR["text"]};">Pick</div>
      <div style="font-weight:1000; font-size:1.8rem; margin:0; color:{pick_color};">
        {pick_text}
      </div>
      """,
      unsafe_allow_html=True
    )
  with r1c2:
    st.markdown(
      f"""
      <div style="font-size:0.85rem; margin:0 0 2px 0; color:{COLOR["text"]};">Advice</div>
      <div style="font-weight:1000; font-size:1.8rem; margin:0; color:{advice_color};">
        {advice_text}
      </div>
      """,
      unsafe_allow_html=True 
    )
  st.caption(f"Model output P(OVER) = {prob_over}")

  st.subheader(f"Model Confidence = {conf_pct:.1f}%")
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
  st.caption("Do not bet under 55% Confidence | Pick is considered a strong bet with > 60% Confidence")

  st.markdown("## Signal Strength")
  if ambiguity == 1:
    st.warning("High ambiguity: the line is very close to expectation (more coin-flippy).")
  else:
    st.success("Low ambiguity: the line is not near expectation (clearer edge).")


  momentum = float(input_df.get(f"{stat}_MOMENTUM", pd.Series([0.0])).iloc[0])
  edge = float(input_df.get(f"LINE_EDGE_{stat}", pd.Series([0.0])).iloc[0])
  z_line = float(input_df.get(f"{stat}_Z_LINE", pd.Series([0.0])).iloc[0])
  z_recent = float(input_df.get(f"{stat}_Z_RECENT", pd.Series([0.0])).iloc[0])

  s1, s2, s3, s4 = st.columns(4)

  # Momentum: positive is "hot"

  s1.metric(
    f"{stat} Momentum",
    f"{momentum:+.2f}",
  )

  # Edge: higher is better (your heuristic)
  s2.metric(
    "Line Edge",
    f"{edge:+.2f}",
  )


  # Z-scores: absolute magnitude big => line far from expectation; small => close/ambiguous-ish
  s3.metric("Z vs Line", f"{z_line:+.2f}")
  s4.metric("Z vs Recent", f"{z_recent:+.2f}")

  with st.expander("How to interpret these (quick guide)"):
    st.markdown(
        """
- **Probability (OVER)**: Model-estimated chance the player goes **over** the line.
- **Confidence**: Confidence in the model’s **final pick**  
  (OVER uses `P(OVER)`; UNDER uses `1 − P(OVER)`).

- **Z vs Line**: `(line − season_avg) / last5_std` (clipped).  
  Measures how far the line is from the player’s **season-long average**, scaled by **recent volatility**.

- **Z vs Recent**: `(line − last5_avg) / last5_std`.  
  Same idea, but anchored to **recent form** instead of the season.

- **Z interpretation**:
  - Large **positive** Z → line is **above** typical output (harder to hit).
  - Large **negative** Z → line is **below** typical output (easier to hit).

- **Line Edge**: `abs(line − last5_avg) / last5_std`.  
  Bigger = line is **farther from recent mean** → clearer edge.

- **Line Ambiguity**: `1` if `|Z vs Line| < 0.25`.  
  Line is very close to expectation → **coin-flippy**.

- **Momentum**: `last5_avg − season_avg`.  
  Positive = trending up; negative = trending down.
        """.strip()
    )
  


  # ------------------------
  # Game Context
  # ------------------------
  st.markdown("## Game Context")

  matchup = input_df["MATCHUP"].iloc[0]
  is_home = int(input_df["HOME"].iloc[0]) == 1
  postseason = int(input_df["POSTSEASON"].iloc[0]) == 1
  b2b = int(input_df["BACK_TO_BACK"].iloc[0]) == 1
  game_date = input_df["GAME_DATE"].iloc[0]

  g1, g2, g3 = st.columns([1, 1.5, 1])
  g1.metric("Matchup", f"{input_df['TEAM'].iloc[0]} vs {matchup}" if is_home else f"{input_df['TEAM'].iloc[0]} @ {matchup}")
  g2.metric("Season Type", "Postseason" if postseason else "Regular Season")
  g3.metric("Back-to-Back", "Yes" if b2b else "No")

  st.caption(f"Game Date: {game_date}")

  # ------------------------
  # Over Rate + Last 10 Bar Chart (hit vs miss)
  # ------------------------
  st.markdown("## Over Rate (Last 10 Games)")

  # safer: compute from df_player instead of trusting feature (still show feature too)
  last10 = df_player.tail(10)[["GAME_DATE", stat]].copy()
  last10["GAME_DATE"] = pd.to_datetime(last10["GAME_DATE"], errors="coerce")
  last10[stat] = pd.to_numeric(last10[stat], errors="coerce")
  last10 = last10.dropna(subset=["GAME_DATE", stat]).reset_index(drop=True)

  last10["Hit"] = np.where(last10[stat] >= parlay_line, "Over", "Under")

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
          .mark_rule(color="#F7DC6F")  # soft yellow line
          .encode(y="y:Q")
      )

      st.altair_chart((bars + rule).properties(height=260), use_container_width=True)
  else:
      st.warning("Not enough recent games to chart last 10.")

  st.markdown(f"## Over Rate vs {matchup} (Recent Matchups)")

  matchups = df_player[df_player["MATCHUP"] == input_df["MATCHUP"].iloc[0]].copy()
  matchups = matchups.sort_values("GAME_DATE").tail(10)  # cap at last 10 matchups

  matchups["GAME_DATE"] = pd.to_datetime(matchups["GAME_DATE"], errors="coerce")
  matchups[stat] = pd.to_numeric(matchups[stat], errors="coerce")
  matchups = matchups.dropna(subset=["GAME_DATE", stat]).reset_index(drop=True)

  matchups["Hit"] = np.where(matchups[stat] >= parlay_line, "Over", "Under")
  
  if len(matchups) > 0:
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
          scale=alt.Scale(
            domain=["Over", "Under"],
            range=[COLOR["over"], COLOR["under"]]
          ),
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
  else:
    st.warning("No recent matchups against this opponent.")

  # ------------------------
  # Trend vs Line (already fixed)
  # ------------------------

  df_chart = df_player.tail(20)[["GAME_DATE", stat]].copy()
  df_chart["GAME_DATE"] = pd.to_datetime(df_chart["GAME_DATE"], errors="coerce")
  df_chart[stat] = pd.to_numeric(df_chart[stat], errors="coerce")
  df_chart = df_chart.dropna(subset=["GAME_DATE", stat]).sort_values("GAME_DATE")
  df_chart["Line"] = float(parlay_line)

  # ------------------------
  # Season (2025-26) cumulative avg vs last5 avg
  # ------------------------
  st.markdown("## 2025-26 Averages (Cumulative vs Last 5)")

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
              y=alt.Y("value:Q", title=f"{stat} average"),
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

  # ------------------------
  # Season (2025-26) cumulative std vs last5 std
  # ------------------------
  st.markdown("## 2025-26 Volatility (Cumulative vs Last 5 STD)")

  if len(season_df) >= 2:
      # expanding std isn't in pandas as a direct method; do it via expanding().apply
      season_df["CUM_STD"] = season_df[stat].expanding().std(ddof=0)
      season_df["LAST5_STD"] = season_df[stat].rolling(5, min_periods=2).std(ddof=0)
      season_df["CUM_STD"] = season_df["CUM_STD"].fillna(0.0)
      season_df["LAST5_STD"] = season_df["LAST5_STD"].fillna(0.0)

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

  # ------------------------
  # Season (2025-26) Momentum (Last5 Avg - Cumulative Avg)
  # ------------------------
  st.markdown(f"## 2025-26 {stat} Momentum (Last 5 vs Season Baseline)")

  if len(season_df) >= 5:
    season_df["MOMENTUM"] = season_df["LAST5_AVG"] - season_df["CUM_AVG"]

    momentum_plot = (
      alt.Chart(season_df)
      .mark_line()
      .encode(
        x=alt.X("GAME_DATE:T", title="Game Date"),
        y=alt.Y("MOMENTUM:Q", title=f"{stat} Momentum"),
        color=alt.value(COLOR["over"]),  # green baseline
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

    st.altair_chart(
      (momentum_plot + zero_rule).properties(height=220),
      use_container_width=True
    )
  else:
    st.info("Not enough games to compute momentum reliably.")






