import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

st.set_page_config(page_title="Seasonal Explorer", layout="wide")
st.title("Seasonal Data Explorer")

st.markdown(
    """
    **Views**: Global • By Season • Summary  
    **Aggregation**: Sum • Mean • Raw (daily)  
    - Optional: Interpolate to **daily inside season months** (breaks off-season gaps)  
    - Optional: **Rolling average** overlay (days or seasons)
    """
)

# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.header("Season window")
    start_month = st.selectbox(
        "Start month",
        options=list(range(1, 13)),
        format_func=lambda m: datetime(2000, m, 1).strftime("%b"),
        index=9  # Oct
    )
    end_month = st.selectbox(
        "End month",
        options=list(range(1, 13)),
        format_func=lambda m: datetime(2000, m, 1).strftime("%b"),
        index=3  # Apr
    )
    st.caption("If start > end, the season crosses calendar years (e.g., Oct→Apr).")

    st.header("Upload data")
    file = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

# ---------------------------
# Helpers
# ---------------------------
def load_table(file):
    if file is None: return None
    name = file.name.lower()
    return pd.read_csv(file) if name.endswith(".csv") else pd.read_excel(file)

def coerce_datetime(series: pd.Series):
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=False)
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def assign_season_year(d: pd.Timestamp, start_m: int, end_m: int):
    if pd.isna(d): return np.nan
    m, y = d.month, d.year
    if start_m > end_m:  # cross-year (e.g., Oct–Apr)
        if m >= start_m:      return y
        elif m <= end_m:      return y - 1
        else:                 return np.nan
    else:
        return y if start_m <= m <= end_m else np.nan

def season_label_from_year(y: int, start_m: int, end_m: int):
    mon = lambda m: datetime(2000, m, 1).strftime("%b")
    return f"{mon(start_m)}-{mon(end_m)} {y}/{y+1}" if start_m > end_m else f"{mon(start_m)}-{mon(end_m)} {y}"

def months_in_window(start_m: int, end_m: int):
    return list(range(start_m, 13)) + list(range(1, end_m+1)) if start_m > end_m else list(range(start_m, end_m+1))

def rolling_days_series(df, date_col, value_col, days, min_periods=1, out_region=None):
    """Return [date_col,'roll','region'] rolling-mean over a true days window; safe reset_index."""
    tmp = df[[date_col, value_col]].dropna().sort_values(date_col).copy()
    if tmp.empty:
        out = pd.DataFrame(columns=[date_col, "roll"])
        if out_region is not None: out["region"] = out_region
        return out
    tmp = tmp.set_index(pd.to_datetime(tmp[date_col])).drop(columns=[date_col])
    tmp.index.name = "_dt"
    tmp["roll"] = tmp[value_col].rolling(f"{days}D", min_periods=min_periods).mean()
    out = tmp.reset_index()[["_dt", "roll"]].rename(columns={"_dt": date_col})
    if out_region is not None: out = out.assign(region=out_region)
    return out

def interpolate_to_daily_within_season(df, date_col, value_col, start_m, end_m, out_region=None):
    """
    Expand sparse monthly season-points to DAILY and interpolate ONLY inside season months.
    Off-season remains NaN -> Altair breaks the line (no diagonal bridges).
    """
    s = df[[date_col, value_col]].dropna().sort_values(date_col).copy()
    if s.empty:
        out = pd.DataFrame(columns=[date_col, "value"])
        if out_region is not None: out["region"] = out_region
        return out
    start, end = pd.to_datetime(s[date_col].min()), pd.to_datetime(s[date_col].max())
    days = pd.date_range(start, end, freq="D")
    ts = s.set_index(pd.to_datetime(s[date_col]))[value_col].reindex(days)
    mask = ts.index.month.isin(months_in_window(start_m, end_m))
    ts = ts.where(mask)
    ts = ts.interpolate(method="time", limit_area="inside")
    out = pd.DataFrame({date_col: ts.index, "value": ts.values}).dropna(subset=["value"])
    if out_region is not None: out["region"] = out_region
    return out

# ---------------------------
# Load + prep
# ---------------------------
df = load_table(file)
if df is None:
    st.info("Upload a file to begin. Your table should have **one date column** and **numeric region columns**.")
    st.stop()

df.columns = df.columns.map(lambda s: str(s).strip())

def find_date_col(df: pd.DataFrame) -> str:
    for cand in ["date","Date","DATE","timestamp","Timestamp","datetime","Datetime"]:
        if cand in df.columns: return cand
    hit = [c for c in df.columns if "date" in str(c).lower()]
    if hit: return hit[0]
    dt = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt: return dt[0]
    raise ValueError("No date column found. Add a column named like 'Date'.")

date_col = find_date_col(df)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
if df[date_col].isna().all():
    st.error(f"Failed to parse any dates in '{date_col}'. Check raw values.")

_df = df.copy()
_df[date_col] = coerce_datetime(_df[date_col])
_df = _df.sort_values(by=date_col)

numeric_candidates = [c for c in _df.columns if c != date_col]
for c in numeric_candidates:
    _df[c] = pd.to_numeric(_df[c], errors='coerce')

with st.sidebar:
    value_cols = st.multiselect("Region columns", options=numeric_candidates,
                                default=[c for c in numeric_candidates if not c.endswith("_LTA")])

if not value_cols:
    st.warning("Select at least one region column.")
    st.stop()

_df["season_year"] = _df[date_col].apply(lambda d: assign_season_year(d, start_month, end_month))
filtered = _df.dropna(subset=["season_year"]).copy()
if filtered.empty:
    st.error("No rows fall inside the selected season window. Try different months.")
    st.stop()

filtered["season_label"] = filtered["season_year"].astype(int).apply(
    lambda y: season_label_from_year(y, start_month, end_month)
)

# Aggregations (seasonal)
agg_method = st.radio("Aggregate by", ["Sum","Mean","Raw (daily)"], horizontal=True)
_grouped = filtered[["season_year","season_label"]+value_cols].groupby(["season_year","season_label"], as_index=False)
seasonal_sum  = _grouped.aggregate(np.nansum).sort_values(["season_year"])
seasonal_mean = _grouped.aggregate(np.nanmean).sort_values(["season_year"])
seasonal = seasonal_sum if agg_method=="Sum" else seasonal_mean
agg_func = np.nansum if agg_method=="Sum" else np.nanmean

# Seasons list
season_options = (
    filtered[["season_year"]]
    .assign(season_label=lambda d: d["season_year"].astype(int).apply(lambda y: season_label_from_year(y,start_month,end_month)))
    .drop_duplicates().sort_values("season_year")
)
season_labels = season_options["season_label"].tolist()
label_to_year = dict(zip(season_options["season_label"], season_options["season_year"]))

st.subheader("📈 Seasons detected")
st.dataframe(season_options[["season_label"]].reset_index(drop=True))

base_regions = [c for c in value_cols if not c.endswith("_LTA")]

st.markdown("---")
view_mode = st.radio("View / Analysis", ["Global view","By Season","Summary (all regions)"])

# ---------- chart helpers (no more schema error) ----------
def enc_line(m, x_field: str, x_type: str, y_title: str):
    return (
        alt.Chart(m)
        .mark_line(interpolate="monotone", point=False, strokeWidth=2)
        .encode(
            x=alt.X(x_field, type=x_type, title="Date" if x_type=="temporal" else "Season",
                    axis=alt.Axis(labelAngle=-30 if x_type=="temporal" else -35, labelOverlap=True, ticks=(x_type!="temporal"))),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("region:N", title="Region", scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip(x_field, type=x_type, title="Date" if x_type=="temporal" else "Season"),
                alt.Tooltip("value:Q", title="Value", format=",.3f"),
            ],
        )
    )

def enc_bar(m, x_field: str, x_type: str, y_title: str):
    return (
        alt.Chart(m)
        .mark_bar()
        .encode(
            x=alt.X(x_field, type=x_type, title="Date" if x_type=="temporal" else "Season",
                    axis=alt.Axis(labelAngle=-30 if x_type=="temporal" else -35, labelOverlap=True, ticks=(x_type!="temporal"))),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("region:N", title="Region", scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip(x_field, type=x_type, title="Date" if x_type=="temporal" else "Season"),
                alt.Tooltip("value:Q", title="Value", format=",.3f"),
            ],
        )
    )

# ----------------------------
# GLOBAL VIEW
# ----------------------------
if view_mode == "Global view":
    sel_regions = st.multiselect("Regions to plot", options=base_regions, default=base_regions)
    add_ma = st.checkbox("Add rolling average overlay", value=True)

    if agg_method == "Raw (daily)":
        interp_daily = st.checkbox("Interpolate to daily within season months", value=True)
        ma_win_days = st.slider("Rolling window (days)", 30, 365, 180, step=15)
    else:
        window_unit = st.radio("Overlay window", ["Seasons","Days"], horizontal=True)
        if window_unit=="Seasons":
            ma_win_seasons = st.slider("Rolling window (seasons)", 2, 10, 5)
        else:
            ma_win_days = st.slider("Rolling window (days)", 30, 365, 90, step=30)

    if not sel_regions:
        st.warning("Choose at least one region to plot.")
    else:
        left, right = st.columns(2)

        # ----- RAW (daily) -----
        if agg_method == "Raw (daily)":
            if interp_daily:
                pieces = [interpolate_to_daily_within_season(filtered, date_col, r, start_month, end_month, r) for r in sel_regions]
                m = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=[date_col,"value","region"])
            else:
                m = filtered[[date_col]+sel_regions].melt(id_vars=[date_col], var_name="region", value_name="value").sort_values(date_col)

            # overlay (dashed)
            if add_ma and not m.empty:
                parts = []
                for r in sel_regions:
                    df_reg = m[m["region"]==r][[date_col,"value"]].rename(columns={"value": r})
                    rolled = rolling_days_series(df_reg, date_col, r, ma_win_days, out_region=r)
                    parts.append(rolled)
                ma_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

            # Line (left)
            line_chart = enc_line(m, date_col, "temporal", "Value")
            if add_ma and not ma_df.empty:
                line_chart = alt.layer(
                    line_chart,
                    alt.Chart(ma_df).mark_line(strokeDash=[6,4], strokeWidth=2, opacity=0.9)
                        .encode(x=alt.X(date_col, type="temporal"), y="roll:Q", color="region:N")
                )
            left.altair_chart(line_chart.properties(title="Daily — Line"), use_container_width=True)

            # Bar (right)
            bar_chart = enc_bar(m, date_col, "temporal", "Value")
            if add_ma and not ma_df.empty:
                bar_chart = alt.layer(
                    bar_chart,
                    alt.Chart(ma_df).mark_line(strokeDash=[6,4], strokeWidth=2, opacity=0.9)
                        .encode(x=alt.X(date_col, type="temporal"), y="roll:Q", color="region:N")
                )
            right.altair_chart(bar_chart.properties(title="Daily — Bar"), use_container_width=True)

        # ----- Aggregated by season (Sum/Mean) -----
        else:
            m = seasonal[["season_year","season_label"]+sel_regions] \
                    .melt(id_vars=["season_year","season_label"], var_name="region", value_name="value") \
                    .sort_values(["region","season_year"])

            # overlay
            ma_for_plot = pd.DataFrame()
            if add_ma:
                if window_unit=="Seasons":
                    m["roll"] = m.groupby("region", group_keys=False)["value"].apply(lambda s: s.rolling(ma_win_seasons, min_periods=1).mean())
                    ma_for_plot = m[["season_label","region","roll"]].dropna()
                else:
                    parts = []
                    for r in sel_regions:
                        rolled = rolling_days_series(filtered[[date_col,"season_label",r]].dropna(), date_col, r, ma_win_days)
                        parts.append(rolled.groupby("season_label", as_index=False)["roll"].agg(agg_func).assign(region=r))
                    ma_for_plot = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

            # Line (left)
            line_chart = enc_line(m, "season_label", "nominal", f"{agg_method} over regions")
            if add_ma and not ma_for_plot.empty:
                line_chart = alt.layer(
                    line_chart,
                    alt.Chart(ma_for_plot).mark_line(strokeDash=[6,4], strokeWidth=2, opacity=0.9)
                        .encode(x=alt.X("season_label:N"), y="roll:Q", color="region:N")
                )
            left.altair_chart(line_chart.properties(title="By Season — Line"), use_container_width=True)

            # Bar (right)
            bar_chart = enc_bar(m, "season_label", "nominal", f"{agg_method} over regions")
            if add_ma and not ma_for_plot.empty:
                bar_chart = alt.layer(
                    bar_chart,
                    alt.Chart(ma_for_plot).mark_line(strokeDash=[6,4], strokeWidth=2, opacity=0.9)
                        .encode(x=alt.X("season_label:N"), y="roll:Q", color="region:N")
                )
            right.altair_chart(bar_chart.properties(title="By Season — Bar"), use_container_width=True)

# ---------------------------
# BY SEASON (date-level within one season)
# ---------------------------
elif view_mode == "By Season":
    sel_season = st.selectbox("Season", options=season_labels,
                              index=len(season_labels)-1 if season_labels else 0)
    sel_regions = st.multiselect("Regions to plot", options=base_regions, default=base_regions)
    add_ma_season = st.checkbox("Add rolling average (days) in this season", value=False)
    if add_ma_season:
        ma_win_days_s = st.slider("Rolling window (days)", 7, 120, 30, step=7)

    if not sel_regions:
        st.warning("Choose at least one region to plot.")
    else:
        y = label_to_year[sel_season]
        within = filtered[filtered["season_year"]==y].copy()
        interp_here = st.checkbox("Interpolate to daily inside this season", value=True)

        if interp_here:
            parts = [interpolate_to_daily_within_season(within, date_col, r, start_month, end_month, r) for r in sel_regions]
            m = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[date_col,"value","region"])
        else:
            m = within[[date_col]+sel_regions].melt(id_vars=[date_col], var_name="region", value_name="value")

        if add_ma_season and not m.empty:
            parts = []
            for r in sel_regions:
                df_reg = m[m["region"]==r][[date_col,"value"]].rename(columns={"value": r})
                parts.append(rolling_days_series(df_reg, date_col, r, ma_win_days_s, out_region=r))
            ma_df = pd.concat(parts, ignore_index=True)
        else:
            ma_df = pd.DataFrame()

        left, right = st.columns(2)

        line_chart = enc_line(m, date_col, "temporal", "Value")
        if add_ma_season and not ma_df.empty:
            line_chart = alt.layer(
                line_chart,
                alt.Chart(ma_df).mark_line(strokeDash=[6,4], strokeWidth=2, opacity=0.9)
                    .encode(x=alt.X(date_col, type="temporal"), y="roll:Q", color="region:N")
            )
        left.altair_chart(line_chart.properties(title=f"{sel_season} — Line"), use_container_width=True)

        bar_chart = enc_bar(m, date_col, "temporal", "Value")
        if add_ma_season and not ma_df.empty:
            bar_chart = alt.layer(
                bar_chart,
                alt.Chart(ma_df).mark_line(strokeDash=[6,4], strokeWidth=2, opacity=0.9)
                    .encode(x=alt.X(date_col, type="temporal"), y="roll:Q", color="region:N")
            )
        right.altair_chart(bar_chart.properties(title=f"{sel_season} — Bar"), use_container_width=True)

# ---------------------------
# SUMMARY
# ---------------------------
elif view_mode == "Summary (all regions)":
    st.markdown("### Season × Region summary")
    pivot = seasonal[["season_label"]+base_regions].set_index("season_label").sort_index()
    st.dataframe(pivot)
    st.download_button("Download Season × Region (CSV)",
        pivot.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="season_region_summary.csv", mime="text/csv")

    st.markdown("### Overall stats per region (across seasons)")
    stats = seasonal[base_regions].agg(["count","mean","std","min","max"]).T
    stats = stats.rename(columns={"count":"n_seasons","mean":f"mean_{'sum' if agg_method=='Sum' else 'mean'}"})
    st.dataframe(stats)
    st.download_button("Download Region Stats (CSV)",
        stats.reset_index().rename(columns={"index":"region"}).to_csv(index=False).encode("utf-8"),
        file_name="region_stats.csv", mime="text/csv")

st.markdown("---")
st.subheader("⬇️ Downloads")
c1, c2 = st.columns(2)
with c1:
    st.download_button("Download seasonal summary (CSV)",
        seasonal.to_csv(index=False).encode("utf-8"),
        file_name="seasonal_summary.csv", mime="text/csv")
with c2:
    st.download_button("Download filtered rows (CSV)",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_rows.csv", mime="text/csv")

st.caption("Tip: If the file is huge, pre-aggregate (daily→monthly) before upload to speed things up.")
