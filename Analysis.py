import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
import plotly.express as px

st.set_page_config(page_title="Seasonal Explorer", layout="wide")
st.title("Seasonal Data Explorer")

st.markdown(
    """
    **New in v2**
    - Toggle **By Season**, **By Region**, or **Summary (all regions)** views.
    - Optional **LTA overlay** if you include columns named like `REGION_LTA`.
    - Downloads for pivot and per-region stats.
    - **Chart Type** toggle (Line ↔ Bar) for line charts.
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
        index=9  # default Oct
    )
    end_month = st.selectbox(
        "End month",
        options=list(range(1, 13)),
        format_func=lambda m: datetime(2000, m, 1).strftime("%b"),
        index=3  # default Apr
    )

    st.caption("If start > end, the season crosses calendar years (e.g., Oct→Apr).")

    st.header("Upload data")
    file = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])

# ---------------------------
# Helpers
# ---------------------------

def load_table(file):
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def coerce_datetime(series: pd.Series):
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=False)
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def assign_season_year(d: pd.Timestamp, start_m: int, end_m: int):
    if pd.isna(d):
        return np.nan
    m = d.month
    y = d.year
    if start_m > end_m:  # cross-year
        if m >= start_m:
            return y
        elif m <= end_m:
            return y - 1
        else:
            return np.nan
    else:  # same-year window
        if start_m <= m <= end_m:
            return y
        else:
            return np.nan

def season_label_from_year(y: int, start_m: int, end_m: int):
    mon = lambda m: datetime(2000, m, 1).strftime("%b")
    if start_m > end_m:
        return f"{mon(start_m)}-{mon(end_m)} {y}/{y+1}"
    else:
        return f"{mon(start_m)}-{mon(end_m)} {y}"

# ---------------------------
# Main Flow
# ---------------------------

df = load_table(file)
if df is None:
    st.info("Upload a file to begin. Your table should have **one date column** and **numeric region columns**.")
    st.stop()

# Normalize headers (trims spaces to avoid Altair shorthand pitfalls)
df.columns = df.columns.map(lambda s: str(s).strip())

# pick date column
# --- find the date column (no UI) ---
def find_date_col(df: pd.DataFrame) -> str:
    # 1) exact/common names first
    for cand in ["date", "Date", "DATE", "timestamp", "Timestamp", "datetime", "Datetime"]:
        if cand in df.columns:
            return cand
    # 2) fuzzy match on name
    name_hits = [c for c in df.columns if "date" in str(c).lower()]
    if name_hits:
        return name_hits[0]
    # 3) dtype-based guess
    dt_hits = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_hits:
        return dt_hits[0]
    # 4) last resort: raise so you notice
    raise ValueError("Could not find a date column. Please add one named like 'Date' or 'date'.")

date_col = find_date_col(df)

# --- coerce dates (handles messy strings) ---
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
if df[date_col].isna().all():
    st.error(f"Failed to parse any dates in column '{date_col}'. Check the raw values.")


# coerce dates
_df = df.copy()
_df[date_col] = coerce_datetime(_df[date_col])
_df = _df.sort_values(by=date_col)

# choose value columns (regions)
numeric_candidates = [c for c in _df.columns if c != date_col]
for c in numeric_candidates:
    _df[c] = pd.to_numeric(_df[c], errors='coerce')

with st.sidebar:
    value_cols = st.multiselect("Region columns", options=numeric_candidates,
                                default=[c for c in numeric_candidates if not c.endswith("_LTA")])

if not value_cols:
    st.warning("Select at least one region column.")
    st.stop()

# Assign season-year key and filter to season window
_df["season_year"] = _df[date_col].apply(lambda d: assign_season_year(d, start_month, end_month))
filtered = _df.dropna(subset=["season_year"]).copy()
if filtered.empty:
    st.error("No rows fall inside the selected season window. Try different months.")
    st.stop()

# Label seasons
filtered["season_label"] = filtered["season_year"].astype(int).apply(
    lambda y: season_label_from_year(y, start_month, end_month)
)

# Aggregations per season-year, per region
agg_method = st.radio("Aggregate by", ["Sum", "Mean"], horizontal=True)
agg_func = np.nansum if agg_method == "Sum" else np.nanmean
seasonal = (
    filtered[["season_year", "season_label"] + value_cols]
    .groupby(["season_year", "season_label"], as_index=False)
    .agg(agg_func)
    .sort_values(["season_year"])
)

st.subheader("📈 Seasons detected")
st.dataframe(seasonal[["season_label"]].drop_duplicates().reset_index(drop=True))

# Option sets
season_options = seasonal[["season_year", "season_label"]].drop_duplicates().sort_values("season_year")
season_labels = season_options["season_label"].tolist()
label_to_year = dict(zip(season_options["season_label"], season_options["season_year"]))

# Detect LTA columns
base_regions = []
lta_map = {}
for c in value_cols:
    if c.endswith("_LTA"):
        continue
    base_regions.append(c)
    if f"{c}_LTA" in _df.columns:
        lta_map[c] = f"{c}_LTA"

st.markdown("---")
view_mode = st.radio("View / Analysis", ["Global view","By Season", "Summary (all regions)"])

def apply_mark(chart: alt.Chart, chart_type: str, point: bool=False):
    """Return chart with the requested mark."""
    return chart.mark_line(point=point) if chart_type == "Line" else chart.mark_bar()

#----------------------------
# Global View (Across all years) — Polished
#----------------------------
if view_mode == "Global view":
    sel_regions = st.multiselect("Regions to plot", options=base_regions, default=base_regions)
    chart_type = st.radio("Chart Type", ["Line", "Bar"], horizontal=True, key="chart_global")

    # Rolling/LTA controls
    add_ma = st.checkbox("Add rolling average (long-term average)", value=True)
    window_unit = st.radio("Window unit", ["Seasons", "Days"], horizontal=True, key="ma_unit")
    if window_unit == "Seasons":
        ma_win_seasons = st.slider("Rolling window (in seasons)", 2, 10, 5, key="ma_win_seasons")
    else:
        ma_win_days = st.slider("Rolling window (in days)", 30, 365, 90, step=30, key="ma_win_days")

    if not sel_regions:
        st.warning("Choose at least one region to plot.")
    else:
        # Base seasonal (observed)
        m = seasonal[["season_year", "season_label"] + sel_regions].melt(
            id_vars=["season_year", "season_label"], var_name="region", value_name="value"
        ).sort_values(["region", "season_year"])

        # Rolling data (time-based if Days, k-season if Seasons)
        ma_title = ""
        if add_ma:
            if window_unit == "Seasons":
                ma_title = f"{ma_win_seasons}-season rolling avg"
                roll_series = (
                    m.groupby("region", group_keys=False)["value"]
                     .apply(lambda s: s.rolling(window=ma_win_seasons, min_periods=1).mean())
                )
                m["roll"] = roll_series
                ma_for_plot = m[["season_label", "region", "roll"]].dropna().copy()
            else:
                ma_title = f"{ma_win_days}-day rolling avg"
                pieces = []
                for r in sel_regions:
                    tmp = filtered[[date_col, "season_label", r]].dropna().sort_values(date_col).copy()
                    tmp = tmp.set_index(pd.to_datetime(tmp[date_col]))
                    tmp["roll"] = tmp[r].rolling(f"{ma_win_days}D", min_periods=1).mean()
                    collapsed = (
                        tmp.reset_index(drop=True)
                           .groupby("season_label", as_index=False)["roll"].agg(agg_func)
                           .assign(region=r)
                    )
                    pieces.append(collapsed)
                ma_for_plot = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

            # merge so tooltips on the observed line can also show rolling value
            if not ma_for_plot.empty:
                m = m.merge(ma_for_plot, on=["season_label", "region"], how="left", suffixes=("", "_ma"))

        # Interactive legend selection (click legend to focus)
        sel = alt.selection_point(fields=["region"], bind="legend")

        # Common encodings
        x_enc = alt.X(
            "season_label:N",
            title="Season",
            axis=alt.Axis(labelAngle=-35, labelLimit=140, ticks=False)
        )
        color_enc = alt.Color("region:N", title="Region", scale=alt.Scale(scheme="tableau10"))

        # Title & subtitle
        chart_title = {
            "text": f"{agg_method} by season",
            "subtitle": ["Solid = Observed", "Dashed diamond = Rolling average"] if add_ma else []
        }
        if add_ma and ma_title:
            chart_title["subtitle"] = [*chart_title["subtitle"], f"Overlay: {ma_title}"]

        # Observed series (line or bar) with larger markers
        base_obs = alt.Chart(m).encode(
            x=x_enc,
            y=alt.Y("value:Q", title=f"{agg_method} over regions"),
            color=color_enc,
            tooltip=[
                alt.Tooltip("season_label:N", title="Season"),
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("value:Q", title="Observed", format=",.3f"),
                alt.Tooltip("roll:Q", title="Rolling Avg", format=",.3f") if add_ma else alt.value(None),
            ],
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.25))
        ).properties(height=460, title=chart_title)

        if chart_type == "Line":
            ch_obs = base_obs.mark_line(point=alt.OverlayMarkDef(size=85, filled=True, strokeWidth=2))
        else:
            ch_obs = base_obs.mark_bar()

        layers = [ch_obs]

        # Rolling overlay: dashed line + diamond markers
        if add_ma and not ma_for_plot.empty:
            ma_line = (
                alt.Chart(ma_for_plot)
                  .mark_line(strokeDash=[6, 4], strokeWidth=2, opacity=0.95)
                  .encode(x=x_enc, y="roll:Q", color=color_enc, opacity=alt.condition(sel, alt.value(1.0), alt.value(0.25)))
            )
            ma_pts = (
                alt.Chart(ma_for_plot)
                  .mark_point(shape="diamond", size=70, filled=True, opacity=0.9)
                  .encode(x=x_enc, y="roll:Q", color=color_enc, opacity=alt.condition(sel, alt.value(1.0), alt.value(0.25)))
            )
            layers += [ma_line, ma_pts]

        # Optional: overall mean reference line (across selected regions & seasons)
        if not m.empty:
            overall_mean = float(m["value"].mean())
            rule_df = pd.DataFrame({"y": [overall_mean]})
            rule = (
                alt.Chart(rule_df)
                  .mark_rule(strokeDash=[2, 2], opacity=0.6)
                  .encode(y="y:Q")
                  .properties()
            )
            layers.append(rule)

        chart = alt.layer(*layers).add_params(sel)
        st.altair_chart(chart, use_container_width=True)


# ---------------------------
# VIEW: BY SEASON (compare regions within a chosen season)
# ---------------------------
if view_mode == "By Season":
    sel_season = st.selectbox("Season", options=season_labels,
                              index=len(season_labels)-1 if season_labels else 0)
    sel_regions = st.multiselect("Regions to plot", options=base_regions, default=base_regions)

    chart_type = st.radio("Chart Type", ["Line", "Bar"], horizontal=True, key="chart_by_season")

    if not sel_regions:
        st.warning("Choose at least one region to plot.")
    else:
        y = label_to_year[sel_season]
        within = filtered[filtered["season_year"] == y].copy()
        m = within[[date_col] + sel_regions].melt(id_vars=[date_col], var_name="region", value_name="value")

        base = alt.Chart(m).encode(
            x=alt.X(field=date_col, type="temporal", title="Date"),
            y=alt.Y(field="value", type="quantitative", title="Value"),
            color=alt.Color(field="region", type="nominal", title="Region"),
            tooltip=[
                alt.Tooltip(field="region", type="nominal"),
                alt.Tooltip(field=date_col, type="temporal"),
                alt.Tooltip(field="value", type="quantitative", format=",.3f")
            ]
        ).properties(height=420, title=f"{sel_season}")

        chart = apply_mark(base, chart_type, point=True)
        st.altair_chart(chart, use_container_width=True)
# ---------------------------
# VIEW: SUMMARY (pivot by season x region + per-region stats)
# ---------------------------
else:
    st.markdown("### Season × Region summary")
    pivot = seasonal[["season_label"] + base_regions].set_index("season_label").sort_index()
    st.dataframe(pivot)
    st.download_button(
        "Download Season × Region (CSV)",
        pivot.reset_index().to_csv(index=False).encode("utf-8"),
        file_name="season_region_summary.csv",
        mime="text/csv"
    )

    st.markdown("### Overall stats per region (across seasons)")
    stats = seasonal[base_regions].agg(["count", "mean", "std", "min", "max"]).T
    stats = stats.rename(columns={"count": "n_seasons", "mean": f"mean_{agg_method.lower()}",
                                  "std": "std", "min": "min", "max": "max"})
    st.dataframe(stats)
    st.download_button(
        "Download Region Stats (CSV)",
        stats.reset_index().rename(columns={"index": "region"}).to_csv(index=False).encode("utf-8"),
        file_name="region_stats.csv",
        mime="text/csv"
    )

st.markdown("---")

# Existing downloads for filtered + seasonal
st.subheader("⬇️ Downloads")
col1, col2 = st.columns(2)
with col1:
    st.download_button("Download seasonal summary (CSV)", seasonal.to_csv(index=False).encode("utf-8"),
                       file_name="seasonal_summary.csv", mime="text/csv")
with col2:
    st.download_button("Download filtered rows (CSV)", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_rows.csv", mime="text/csv")

st.caption("Tip: If the file is huge, pre-aggregate (daily→monthly) before upload to speed things up.")
