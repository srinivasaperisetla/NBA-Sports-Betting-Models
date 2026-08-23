from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

MODEL_FAMILIES = ["FULL_ALL", "FULL_TIGHT", "REDUCED_ALL", "REDUCED_TIGHT"]
DEFAULT_FEATURES = [
    "Confidence_num",
    "Rank Score",
    "z_line",
    "z_recent",
    "z_matchup",
    "line_diff",
    "momentum",
    "last10",
    "last5",
    "probability_std",
    "Agreement_ratio_num",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze betting pick CSVs, summarize hit/miss metrics, compute correlations, "
            "and build a historical rerank score to push likely hits toward the top."
        )
    )
    parser.add_argument("--csv-dir", default="csv", help="Folder containing daily CSV files. Default: csv")
    parser.add_argument("--out-dir", default="analysis_output", help="Folder where reports will be written. Default: analysis_output")
    parser.add_argument("--bin-start", type=float, default=0.50, help="Starting confidence bin edge. Default: 0.50")
    parser.add_argument("--bin-end", type=float, default=1.00, help="Ending confidence bin edge. Default: 1.00")
    parser.add_argument("--bin-width", type=float, default=0.05, help="Confidence bin width. Default: 0.05")
    parser.add_argument("--min-group-size", type=int, default=8, help="Minimum sample size before a group is trusted. Default: 8")
    parser.add_argument("--top-n-checks", default="10,25,50,100", help="Comma-separated top-N cutoffs. Default: 10,25,50,100")
    return parser.parse_args()


def pct_to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    ) / 100.0


def classify_result(value: object) -> str:
    text = str(value).upper().strip()
    if "HIT" in text:
        return "HIT"
    if "MISS" in text:
        return "MISS"
    if "NO GAME" in text:
        return "NO_GAME"
    if text in {"NAN", "NONE", ""}:
        return "UNKNOWN"
    return text


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_stat(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip().str.upper()
    replacements = {
        "3:00 PM": "3PM",
        "3 PM": "3PM",
        "3PTM": "3PM",
        "THREES": "3PM",
    }
    return out.replace(replacements)


def load_csvs(csv_dir: Path) -> pd.DataFrame:
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir.resolve()}")

    frames: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        if "Unnamed: 0" in df.columns and "Rank" not in df.columns:
            df = df.rename(columns={"Unnamed: 0": "Rank"})
        df["source_file"] = path.name
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    return standardize_columns(all_df)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    percent_columns = ["Confidence", "Agreement ratio", *[f"{m} Confidence" for m in MODEL_FAMILIES]]
    for col in percent_columns:
        if col in df.columns:
            out_col = col.replace(" ", "_") + "_num"
            df[out_col] = pct_to_float(df[col])

    numeric_cols = [
        "Rank", "Line", "Rank Score", "Actual value", "z_line", "z_recent", "z_matchup",
        "line_diff", "momentum", "last10", "last5", "probability_std",
        *[f"{m} Rank score" for m in MODEL_FAMILIES],
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    result_columns = ["Result", *[f"{m} Result" for m in MODEL_FAMILIES]]
    for col in result_columns:
        if col in df.columns:
            df[col.replace(" ", "_") + "_class"] = df[col].map(classify_result)

    pick_columns = ["Pick", *[f"{m} Pick" for m in MODEL_FAMILIES]]
    for col in pick_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().str.strip()

    for cat_col in ["Player", "Tier", "Recommendation"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str).str.strip()
    if "Stat" in df.columns:
        df["Stat"] = normalize_stat(df["Stat"])
        df["Stat_is_suspicious"] = df["Stat"].astype(str).str.contains(r":", regex=True)

    if "Result_class" not in df.columns and "Result" in df.columns:
        df["Result_class"] = df["Result"].map(classify_result)

    df["is_hit"] = df.get("Result_class", pd.Series(index=df.index)).eq("HIT")
    df["is_miss"] = df.get("Result_class", pd.Series(index=df.index)).eq("MISS")
    df["is_no_game"] = df.get("Result_class", pd.Series(index=df.index)).eq("NO_GAME")
    df["is_graded"] = df["is_hit"] | df["is_miss"]
    return df


def build_confidence_bins(start: float, end: float, width: float) -> List[float]:
    bins = [round(start, 10)]
    current = start
    while current < end:
        current = round(current + width, 10)
        bins.append(min(current, end))
    if bins[-1] < end:
        bins.append(end)
    return bins


def summarize_outcomes(df: pd.DataFrame, result_col: str) -> Dict[str, float]:
    series = df[result_col]
    hits = int((series == "HIT").sum())
    misses = int((series == "MISS").sum())
    no_games = int((series == "NO_GAME").sum())
    graded = hits + misses
    total = hits + misses + no_games
    return {
        "total_rows": int(total),
        "graded_rows": int(graded),
        "hits": hits,
        "misses": misses,
        "no_games": no_games,
        "accuracy": (hits / graded) if graded else np.nan,
        "coverage": (graded / total) if total else np.nan,
    }


def accuracy_by_confidence(df: pd.DataFrame, confidence_col: str, result_col: str, bins: List[float]) -> pd.DataFrame:
    work = df[[confidence_col, result_col]].copy()
    work = work[work[result_col].isin(["HIT", "MISS"])].dropna(subset=[confidence_col])
    if work.empty:
        return pd.DataFrame(columns=["confidence_bin", "hits", "misses", "count", "accuracy"])

    labels = [f"{bins[i]:.2f}-{bins[i + 1]:.2f}" for i in range(len(bins) - 1)]
    work["confidence_bin"] = pd.cut(work[confidence_col], bins=bins, labels=labels, include_lowest=True, right=False)
    grouped = work.groupby("confidence_bin", dropna=False, observed=False)[result_col].agg(
        hits=lambda s: int((s == "HIT").sum()),
        misses=lambda s: int((s == "MISS").sum()),
        count="count",
    ).reset_index()
    grouped["accuracy"] = grouped["hits"] / grouped["count"]
    return grouped


def grouped_accuracy(df: pd.DataFrame, group_col: str, result_col: str = "Result_class") -> pd.DataFrame:
    work = df[df[result_col].isin(["HIT", "MISS", "NO_GAME"])].copy()
    grouped = work.groupby(group_col)[result_col].agg(
        hits=lambda s: int((s == "HIT").sum()),
        misses=lambda s: int((s == "MISS").sum()),
        no_games=lambda s: int((s == "NO_GAME").sum()),
        total="count",
    ).reset_index()
    grouped["graded"] = grouped["hits"] + grouped["misses"]
    grouped["accuracy"] = grouped["hits"] / grouped["graded"]
    grouped["coverage"] = grouped["graded"] / grouped["total"]
    return grouped.sort_values(["accuracy", "graded"], ascending=[False, False])


def correlation_table(df: pd.DataFrame, feature_cols: List[str], target_col: str = "is_hit") -> pd.DataFrame:
    work = df[df["is_graded"]].copy()
    y = work[target_col].astype(float)
    rows = []
    for col in feature_cols:
        if col not in work.columns:
            continue
        x = safe_numeric(work[col])
        valid = x.notna() & y.notna()
        if valid.sum() < 3:
            continue
        rows.append(
            {
                "feature": col,
                "pearson_corr_with_hit": x[valid].corr(y[valid], method="pearson"),
                "spearman_corr_with_hit": x[valid].corr(y[valid], method="spearman"),
                "sample_size": int(valid.sum()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("pearson_corr_with_hit", ascending=False)
    return out


def pairwise_feature_corr(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    return df[available].apply(safe_numeric).corr(method="pearson")


def top_n_lift_table(df: pd.DataFrame, score_col: str, top_ns: List[int]) -> pd.DataFrame:
    work = df[df["is_graded"] & df[score_col].notna()].copy().sort_values(score_col, ascending=False)
    if work.empty:
        return pd.DataFrame(columns=["score", "top_n", "top_n_accuracy", "overall_accuracy", "lift", "hits", "graded"])
    overall_acc = work["is_hit"].mean()
    rows = []
    for n in top_ns:
        subset = work.head(n)
        if subset.empty:
            continue
        top_acc = subset["is_hit"].mean()
        rows.append(
            {
                "score": score_col,
                "top_n": n,
                "top_n_accuracy": top_acc,
                "overall_accuracy": overall_acc,
                "lift": (top_acc / overall_acc) if overall_acc else np.nan,
                "hits": int(subset["is_hit"].sum()),
                "graded": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def smoothed_group_hit_rate(df: pd.DataFrame, group_col: str, global_rate: float, alpha: int = 12) -> pd.DataFrame:
    work = df[df["is_graded"]].copy()
    grp = work.groupby(group_col)["is_hit"].agg(["sum", "count"]).reset_index()
    grp = grp.rename(columns={"sum": "hits", "count": "graded"})
    grp["raw_hit_rate"] = grp["hits"] / grp["graded"]
    grp["smoothed_hit_rate"] = (grp["hits"] + alpha * global_rate) / (grp["graded"] + alpha)
    return grp


def add_historical_lift_features(df: pd.DataFrame, min_group_size: int) -> pd.DataFrame:
    work = df.copy()
    graded = work[work["is_graded"]].copy()
    global_rate = graded["is_hit"].mean()

    for group_col, prefix in [("Stat", "stat"), ("Tier", "tier"), ("Recommendation", "rec"), ("Player", "player")]:
        if group_col not in work.columns:
            continue
        grp = smoothed_group_hit_rate(graded, group_col, global_rate)
        grp = grp.rename(columns={
            group_col: group_col,
            "graded": f"{prefix}_graded_hist",
            "raw_hit_rate": f"{prefix}_raw_hit_rate",
            "smoothed_hit_rate": f"{prefix}_smoothed_hit_rate",
        })[[group_col, f"{prefix}_graded_hist", f"{prefix}_raw_hit_rate", f"{prefix}_smoothed_hit_rate"]]
        work = work.merge(grp, on=group_col, how="left")
        enough_hist = work[f"{prefix}_graded_hist"].fillna(0) >= min_group_size
        work[f"{prefix}_trusted_hit_rate"] = np.where(enough_hist, work[f"{prefix}_smoothed_hit_rate"], global_rate)

    work["agreement_bonus"] = work.get("Agreement_ratio_num", pd.Series(0.0, index=work.index)).fillna(0)
    work["stability_bonus"] = 1 - work.get("probability_std", pd.Series(0.0, index=work.index)).fillna(0)
    return work


def build_rerank_score(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    global_rate = work[work["is_graded"]]["is_hit"].mean()

    work["base_conf_component"] = work.get("Confidence_num", pd.Series(global_rate, index=work.index)).fillna(global_rate)
    work["stat_component"] = work.get("stat_trusted_hit_rate", pd.Series(global_rate, index=work.index)).fillna(global_rate)
    work["tier_component"] = work.get("tier_trusted_hit_rate", pd.Series(global_rate, index=work.index)).fillna(global_rate)
    work["rec_component"] = work.get("rec_trusted_hit_rate", pd.Series(global_rate, index=work.index)).fillna(global_rate)
    work["player_component"] = work.get("player_trusted_hit_rate", pd.Series(global_rate, index=work.index)).fillna(global_rate)
    work["agreement_component"] = work.get("agreement_bonus", pd.Series(0.0, index=work.index)).fillna(0)
    work["stability_component"] = work.get("stability_bonus", pd.Series(1.0, index=work.index)).fillna(1)

    work["historical_rerank_score"] = (
        0.34 * work["base_conf_component"] +
        0.20 * work["stat_component"] +
        0.12 * work["tier_component"] +
        0.12 * work["rec_component"] +
        0.10 * work["player_component"] +
        0.07 * work["agreement_component"] +
        0.05 * work["stability_component"]
    )

    if "Rank Score" in work.columns:
        rank_score_norm = work["Rank Score"]
        rank_score_norm = (rank_score_norm - rank_score_norm.min()) / (rank_score_norm.max() - rank_score_norm.min() + 1e-9)
        work["hybrid_rerank_score"] = 0.65 * work["historical_rerank_score"] + 0.35 * rank_score_norm
    else:
        work["hybrid_rerank_score"] = work["historical_rerank_score"]

    work["rerank_position"] = work["hybrid_rerank_score"].rank(method="first", ascending=False).astype(int)
    return work


def model_family_summary(df: pd.DataFrame, bins: List[float]) -> Dict[str, pd.DataFrame]:
    rows = []
    conf_tables: Dict[str, pd.DataFrame] = {}
    for model in MODEL_FAMILIES:
        result_col = f"{model}_Result_class"
        conf_col = f"{model}_Confidence_num"
        if result_col not in df.columns:
            continue
        rows.append({"model_family": model, **summarize_outcomes(df, result_col)})
        if conf_col in df.columns:
            conf_tables[model] = accuracy_by_confidence(df, conf_col, result_col, bins)
    return {
        "summary": (pd.DataFrame(rows).sort_values("accuracy", ascending=False) if rows else pd.DataFrame()),
        **{f"confidence_{model}": table for model, table in conf_tables.items()},
    }


def recommendation_transition_table(df: pd.DataFrame) -> pd.DataFrame:
    if "Recommendation" not in df.columns:
        return pd.DataFrame()
    work = df[df["is_graded"]].copy()
    pivot = work.pivot_table(index="Recommendation", values="is_hit", aggfunc=["mean", "count"])
    pivot.columns = ["accuracy", "graded"]
    return pivot.reset_index().sort_values("accuracy", ascending=False)


def stat_tier_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if not {"Stat", "Tier"}.issubset(df.columns):
        return pd.DataFrame()
    work = df[df["is_graded"]].copy()
    return work.pivot_table(index="Stat", columns="Tier", values="is_hit", aggfunc="mean")


def save_outputs(out_dir: Path, outputs: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in outputs.items():
        path_base = out_dir / name
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path_base.with_suffix(".csv"), index=False)
        elif isinstance(obj, dict):
            with open(path_base.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=lambda x: None if pd.isna(x) else float(x) if isinstance(x, (np.floating, np.integer)) else x)
        else:
            with open(path_base.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write(str(obj))


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    bins = build_confidence_bins(args.bin_start, args.bin_end, args.bin_width)
    top_ns = [int(x.strip()) for x in args.top_n_checks.split(",") if x.strip()]

    df = load_csvs(csv_dir)
    df = add_historical_lift_features(df, min_group_size=args.min_group_size)
    df = build_rerank_score(df)

    overall_summary = summarize_outcomes(df, "Result_class")
    overall_conf = accuracy_by_confidence(df, "Confidence_num", "Result_class", bins)
    by_recommendation = grouped_accuracy(df, "Recommendation") if "Recommendation" in df.columns else pd.DataFrame()
    by_stat = grouped_accuracy(df, "Stat") if "Stat" in df.columns else pd.DataFrame()
    by_player = grouped_accuracy(df, "Player") if "Player" in df.columns else pd.DataFrame()
    by_tier = grouped_accuracy(df, "Tier") if "Tier" in df.columns else pd.DataFrame()
    feature_corr = correlation_table(df, DEFAULT_FEATURES)
    pairwise_corr = pairwise_feature_corr(df, DEFAULT_FEATURES)
    top_n_original = top_n_lift_table(df, "Confidence_num", top_ns)
    top_n_rerank = top_n_lift_table(df, "hybrid_rerank_score", top_ns)
    model_outputs = model_family_summary(df, bins)
    rec_matrix = recommendation_transition_table(df)
    stat_tier = stat_tier_matrix(df)

    rerank_preview_cols = [
        c for c in [
            "source_file", "Rank", "rerank_position", "Player", "Stat", "Pick", "Confidence", "Recommendation", "Tier",
            "Result", "Rank Score", "historical_rerank_score", "hybrid_rerank_score",
            "stat_component", "tier_component", "rec_component", "player_component",
            "agreement_component", "stability_component",
        ] if c in df.columns
    ]
    rerank_preview = df.sort_values("hybrid_rerank_score", ascending=False)[rerank_preview_cols]

    outputs: Dict[str, object] = {
        "overall_summary": overall_summary,
        "overall_accuracy_by_confidence": overall_conf,
        "overall_by_recommendation": by_recommendation,
        "overall_by_stat": by_stat,
        "overall_by_player": by_player,
        "overall_by_tier": by_tier,
        "feature_hit_correlations": feature_corr,
        "pairwise_feature_correlations": pairwise_corr,
        "top_n_lift_original_confidence": top_n_original,
        "top_n_lift_rerank": top_n_rerank,
        "recommendation_summary": rec_matrix,
        "stat_tier_accuracy_matrix": stat_tier,
        "reranked_picks_preview": rerank_preview,
        "suspicious_stat_rows": df[df.get("Stat_is_suspicious", False)].copy(),
    }
    outputs.update(model_outputs)
    save_outputs(out_dir, outputs)

    print("Analysis complete.")
    print(f"Input folder: {csv_dir.resolve()}")
    print(f"Output folder: {out_dir.resolve()}")
    print("\nOverall summary")
    print(json.dumps(overall_summary, indent=2))
    print("\nTop-N accuracy lift using original confidence")
    print(top_n_original.to_string(index=False))
    print("\nTop-N accuracy lift using hybrid rerank score")
    print(top_n_rerank.to_string(index=False))
    if not feature_corr.empty:
        print("\nTop feature correlations with hits")
        print(feature_corr.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
