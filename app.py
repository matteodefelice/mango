"""Interactive Streamlit app for exploring mango climate indicators.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import itertools

import pandas as pd
import plotly.express as px
import streamlit as st
import xarray as xr

from mango import indicators
from mango.data.edh import build_url
from mango.output.plots import C_ERA5, C_FUT, C_HIST
from mango.workflow import Workflow

MODELS = [
    ("CMCC-CM2-SR5-historical-r1i1p1f1-day-gn-v0.zarr",
     "CMCC-CM2-SR5-ScenarioMIP-r1i1p1f1-day-gn-v0.zarr"),
    ("DKRZ-MPI-ESM1-2-HR-historical-r1i1p1f1-day-gn-v0.zarr",
     "DKRZ-MPI-ESM1-2-HR-ScenarioMIP-r1i1p1f1-day-gn-v0.zarr"),
    ("EC-Earth3-CC-historical-r1i1p1f1-day-gr-v0.zarr",
     "EC-Earth3-CC-ScenarioMIP-r1i1p1f1-day-gr-v0.zarr"),
    ("IPSL-CM6A-LR-historical-r1i1p1f1-day-gr-v0.zarr",
     "IPSL-CM6A-LR-ScenarioMIP-r1i1p1f1-day-gr-v0.zarr"),
    ("NCAR-CESM2-historical-r1i1p1f1-day-gn-v0.zarr",
     "NCAR-CESM2-ScenarioMIP-r10i1p1f1-day-gn-v0.zarr"),
    ("NCC-NorESM2-MM-historical-r1i1p1f1-day-gn-v0.zarr",
     "NCC-NorESM2-MM-ScenarioMIP-r1i1p1f1-day-gn-v0.zarr"),
]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

SCENARIOS = ["ssp370", "ssp245", "ssp585"]

st.set_page_config(page_title="mango — climate indicators", layout="wide")
st.title("🥭 mango — interactive climate indicators")


@st.cache_resource(show_spinner=False)
def get_workflow(lat: float, lon: float, scenario: str) -> Workflow:
    """Return a cached Workflow shell (no data loaded yet).

    Label is set by the caller after retrieval so renaming doesn't
    invalidate the cached data.
    """
    urls = [[build_url(h), build_url(f)] for h, f in MODELS]
    return Workflow(lat=lat, lon=lon, label="", urls=urls, scenario=scenario)


def resolve_workflow(lat: float, lon: float, scenario: str, label: str) -> Workflow:
    wf = get_workflow(lat, lon, scenario)
    wf.label = label
    return wf


with st.sidebar:
    st.header("Location")
    label = st.text_input("Label", value="Porto")
    lat = st.number_input("Latitude (°N)", value=44.1427, format="%.4f")
    lon = st.number_input("Longitude (0–360°)", value=355.0, format="%.4f",
                          min_value=0.0, max_value=360.0,
                          help="Use 0–360 convention (e.g. 355.0 for 5°W)")
    scenario = st.selectbox("Scenario", SCENARIOS, index=0)

    if st.button("Load / debias data", type="primary"):
        wf = resolve_workflow(lat, lon, scenario, label)
        try:
            if wf.is_loaded:
                st.info("Already loaded for this location — skipping.")
            else:
                progress = st.progress(0.0, text="Starting…")
                wf.load(on_step=lambda msg, p: progress.progress(
                    p, text=f"Step · {msg}"
                ))
            st.session_state["wf_key"] = (lat, lon, scenario, label)
            st.success("Ready.")
        except Exception as e:
            st.session_state.pop("wf_key", None)
            st.error(f"Load failed: {e}")

    st.caption("Data and debiased outputs are cached on disk — repeat runs are fast.")


if "wf_key" not in st.session_state:
    st.info("👈 Pick a location and click **Load / debias data** to start.")
    st.stop()

wf = resolve_workflow(*st.session_state["wf_key"])
if not wf.is_loaded:
    st.warning("Data not fully loaded for this location. Click **Load / debias data** again.")
    st.stop()

st.subheader(f"📍 {wf.location_label}")

col1, col2 = st.columns([1, 2])
with col1:
    indicator_name = st.selectbox("Indicator", indicators.available())
    reqs = indicators.get_required_vars(indicator_name)
    st.caption(f"Requires: `{', '.join(reqs)}`")

with col2:
    start, end = st.select_slider(
        "Month range",
        options=MONTH_NAMES,
        value=(MONTH_NAMES[0], MONTH_NAMES[-1]),
    )
    i, j = MONTH_NAMES.index(start) + 1, MONTH_NAMES.index(end) + 1
    months_range = list(range(i, j + 1)) if i <= j else list(range(i, 13)) + list(range(1, j + 1))

# Treat "full year" as no filter so full_year_required indicators stay available.
months = None if months_range == list(range(1, 13)) else months_range

full_year_only = indicator_name not in indicators.available_for_months_filter()
if months is not None and full_year_only:
    st.warning(
        f"`{indicator_name}` requires a full annual series and cannot be "
        "computed on a month-filtered dataset. Clear the month filter to run it."
    )
    st.stop()


@st.cache_data(show_spinner="Computing indicator…", max_entries=64)
def compute_indicator(
    wf_key: tuple, indicator_name: str, months: tuple[int, ...] | None
) -> pd.DataFrame:
    # wf_key is passed only so Streamlit's cache keys on location changes.
    wf = resolve_workflow(*wf_key)

    def _filter(ds: xr.Dataset) -> xr.Dataset:
        if months is None:
            return ds
        return ds.sel(time=ds["time.month"].isin(months))

    dfs: list[pd.DataFrame] = []
    for ds in itertools.chain(wf.list_hist, wf.list_fut):
        dfs.extend(indicators.compute_all(
            _filter(ds), suffix="_bc", names=[indicator_name],
        ))
    dfs.extend(indicators.compute_all(
        _filter(wf.obs), suffix="", experiment_id="ERA5", names=[indicator_name],
    ))
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs).reset_index()
    try:
        out["year"] = pd.to_datetime(out["time"]).dt.year
    except (TypeError, ValueError):
        out["year"] = [t.year for t in out["time"]]
    return out


df = compute_indicator(
    st.session_state["wf_key"],
    indicator_name,
    tuple(months) if months else None,
)

if df.empty:
    st.warning("No model has the variables required by this indicator.")
    st.stop()

experiments = list(df["experiment_id"].unique())
color_map = {"ERA5": C_ERA5, "historical": C_HIST}
for exp in experiments:
    color_map.setdefault(exp, C_FUT)

order = ["ERA5", "historical"] + [e for e in experiments if e not in ("ERA5", "historical")]

fig = px.box(
    df,
    x="experiment_id",
    y="value",
    color="experiment_id",
    points="all",
    hover_data={"model": True, "year": True, "experiment_id": False},
    color_discrete_map=color_map,
    category_orders={"experiment_id": order},
    title=f"{indicator_name} — {wf.location_label}"
    + (f" (months: {', '.join(MONTH_NAMES[m-1] for m in months)})" if months else ""),
)
fig.update_traces(boxmean=True, jitter=0.4, pointpos=0, marker=dict(size=7, opacity=0.85))
fig.update_layout(
    xaxis_title="",
    yaxis_title=indicator_name,
    legend_title_text="experiment",
    height=550,
    plot_bgcolor="#f5f5f5",
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Raw values"):
    st.dataframe(
        df[["model", "experiment_id", "year", "value"]].sort_values(["experiment_id", "model", "year"]),
        use_container_width=True,
        hide_index=True,
    )
