import streamlit as st
import requests
from backend.constants import TARGET_COLUMNS, ALLOWED_PLAYERS_LIST
from utils import get_input
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

COLOR = {
    "no_bet": "#F1C40F",        # yellow
    "over": "#2ECC71",          # green
    "over_strong": "#1E8449",   # deep green
    "under": "#E74C3C",         # red
    "under_strong": "#922B21",  # deep red
    "neutral": "#BDC3C7",
}

def parse_percent(s: str) -> float:
    # "64.28%" -> 0.6428
    return float(str(s).replace("%", "")) / 100.0

def advice_and_color(pred: int, conf_pct: float):
    """
    pred: 1=OVER, 0=UNDER
    conf_pct: 0..100 (model confidence in its pick)
    """
    if conf_pct < 55:
        return "DO NOT BET", COLOR["no_bet"]
    if pred == 1:
        if conf_pct >= 60:
            return "STRONG OVER", COLOR["over_strong"]
        return "OVER", COLOR["over"]
    else:
        if conf_pct >= 60:
            return "STRONG UNDER", COLOR["under_strong"]
        return "UNDER", COLOR["under"]

def metric_delta_label(value: float, good_if_high: bool = True, strong_thr: float = 0.6):
    # Used for "Line Edge" and similar
    if good_if_high:
        return ("↑ Strong", "normal") if value >= strong_thr else ("↓ Weak", "inverse")
    else:
        return ("↓ Strong", "inverse") if value <= strong_thr else ("↑ Weak", "normal")

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
    prob_over = float(result["probability"])          # probability of OVER (as your endpoint returns)
    conf_pick = parse_percent(result["confidence"])   # confidence in the predicted class

    conf_pct = conf_pick * 100.0
    advice_text, advice_color = advice_and_color(pred, conf_pct)

    st.markdown("## Prediction")

    a, b, c, d = st.columns(4)
    a.metric("Stat", stat)
    b.metric("Line", f"{parlay_line:g}")
    c.metric("Pick", "OVER" if pred == 1 else "UNDER")
    d.metric("Advice", advice_text)

    # bar for probability of OVER, colored by the advice/pick strength
    st.markdown(
        f"""
        <div style="margin-top:8px; margin-bottom:4px;">
          <div style="display:flex; justify-content:space-between; font-size:0.9rem;">
            <span>Probability (OVER)</span>
            <span><b>{prob_over*100:.1f}%</b></span>
          </div>
          <div style="width:100%; background:{COLOR["neutral"]}; border-radius:10px; height:14px; overflow:hidden;">
            <div style="width:{min(max(prob_over,0),1)*100:.1f}%; background:{advice_color}; height:14px;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        f"Confidence in pick: {conf_pct:.1f}% (if model picked UNDER, this is ~ 1 - P(OVER))."
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

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Matchup", f"{input_df['TEAM'].iloc[0]} vs {matchup}" if is_home else f"{input_df['TEAM'].iloc[0]} @ {matchup}")
    g2.metric("Home/Away", "Home" if is_home else "Away")
    g3.metric("Season Type", "Postseason" if postseason else "Regular Season")
    g4.metric("Back-to-Back", "Yes" if b2b else "No")
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
    empirical_over_rate = float((last10[stat] >= parlay_line).mean()) if len(last10) else 0.0

    r1, r2 = st.columns([1, 1])
    with r1:
        st.metric("Empirical Over Rate (Last 10)", f"{empirical_over_rate*100:.1f}%")
    with r2:
        if f"OVER_PL_RATE_{stat}_LAST10" in input_df.columns:
            feat_over_rate = float(input_df[f"OVER_PL_RATE_{stat}_LAST10"].iloc[0])
            st.metric("Model Feature Over Rate (Last 10)", f"{feat_over_rate*100:.1f}%")

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

    # ------------------------
    # Trend vs Line (already fixed)
    # ------------------------
    st.markdown("## Recent Trend vs Line")

    df_chart = df_player.tail(20)[["GAME_DATE", stat]].copy()
    df_chart["GAME_DATE"] = pd.to_datetime(df_chart["GAME_DATE"], errors="coerce")
    df_chart[stat] = pd.to_numeric(df_chart[stat], errors="coerce")
    df_chart = df_chart.dropna(subset=["GAME_DATE", stat]).sort_values("GAME_DATE")
    df_chart["Line"] = float(parlay_line)

    trend = (
        alt.Chart(df_chart)
        .transform_fold(fold=[stat, "Line"], as_=["series", "value"])
        .mark_line(point=True)
        .encode(
            x=alt.X("GAME_DATE:T", title="Game Date"),
            y=alt.Y("value:Q", title=stat),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=[stat, "Line"], range=[advice_color, "#F7DC6F"]),
                legend=alt.Legend(title=""),
            ),
            tooltip=[
                alt.Tooltip("GAME_DATE:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(trend, use_container_width=True)

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
    # Signal Strength + Context (with color + arrows)
    # ------------------------
    st.markdown("## Signal Strength")

    momentum = float(input_df.get(f"{stat}_MOMENTUM", pd.Series([0.0])).iloc[0])
    edge = float(input_df.get(f"LINE_EDGE_{stat}", pd.Series([0.0])).iloc[0])
    ambiguity = int(input_df.get(f"LINE_AMBIGUITY_{stat}", pd.Series([0])).iloc[0])
    z_line = float(input_df.get(f"{stat}_Z_LINE", pd.Series([0.0])).iloc[0])
    z_recent = float(input_df.get(f"{stat}_Z_RECENT", pd.Series([0.0])).iloc[0])

    s1, s2, s3, s4 = st.columns(4)

    # Momentum: positive is "hot"
    s1.metric(
        "Momentum",
        f"{momentum:+.2f}",
        delta="↑ Hot" if momentum > 0 else "↓ Cold",
        delta_color="normal" if momentum > 0 else "inverse",
    )

    # Edge: higher is better (your heuristic)
    edge_label, edge_delta_color = metric_delta_label(edge, good_if_high=True, strong_thr=0.6)
    s2.metric("Line Edge", f"{edge:.2f}", delta=edge_label, delta_color=edge_delta_color)

    # Z-scores: absolute magnitude big => line far from expectation; small => close/ambiguous-ish
    s3.metric("Z vs Line", f"{z_line:+.2f}")
    s4.metric("Z vs Recent", f"{z_recent:+.2f}")

    with st.expander("How to interpret these (quick guide)"):
        st.markdown(
            """
- **Probability (OVER)**: model-estimated chance the player goes **over** the line.  
- **Confidence**: confidence in the model's **final pick** (OVER uses P(OVER); UNDER uses 1 - P(OVER)).  
- **Over rate (last 10)**: fraction of the last 10 games where stat ≥ line.
- **Z vs Line**: `(line - season_avg) / last5_std` (clipped).  
  - Large positive Z means the line is **above** typical output (harder to hit).  
  - Large negative Z means the line is **below** typical output (easier to hit).
- **Z vs Recent**: `(line - last5_avg) / last5_std`. Same idea but anchored to recent form.
- **Line Edge**: `abs(line - last5_avg) / last5_std`. Bigger means the line is **far** from recent mean (clearer edge).
- **Line Ambiguity**: 1 if `|Z vs Line| < 0.25` → line is close to expectation (coin-flippy).
- **Momentum**: `last5_avg - season_avg`. Positive = trending up; negative = trending down.
            """.strip()
        )

    st.markdown("## Line Ambiguity")
    if ambiguity == 1:
        st.warning("High ambiguity: the line is very close to expectation (more coin-flippy).")
    else:
        st.success("Low ambiguity: the line is not near expectation (clearer edge).")







