"""Example: reproduce the original starting_point.py workflow using the mango library.

Configuration is read from mango.yaml (see mango.yaml.example).
Set edh_token there, or via the MANGO_CONFIG env var.
"""

import pandas as pd

from mango import config
from mango.data import load_cmip6_datasets_from_edh, load_era5
from mango.data.edh import build_url
from mango.debias import debias_temperature, debias_precipitation
from mango import indicators

# ── Configuration (loaded automatically from mango.yaml) ─────────────────
# You can also override programmatically:
#   config.load("path/to/custom.yaml")
#   config.override("edh_token", "...")

# ── Model URLs ────────────────────────────────────────────────────────────
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

urls = [[build_url(hist), build_url(fut)] for hist, fut in MODELS]

# Porto
sel_lat = 44.1427
sel_lon = 360 - 5
vars_to_select = ["pr", "tas", "tasmin", "tasmax"]

# ── 1. Load data ──────────────────────────────────────────────────────────
list_hist, list_fut = load_cmip6_datasets_from_edh(
    urls, lat=sel_lat, lon=sel_lon, variables=vars_to_select,
)

e5_t = load_era5(lat=sel_lat, lon=sel_lon)

# ── 2. Bias correction ───────────────────────────────────────────────────
debias_temperature(obs=e5_t, list_hist=list_hist, list_fut=list_fut)

debias_precipitation(obs=e5_t, list_hist=list_hist, list_fut=list_fut)

# ── 3. Prepare ERA5 for indicators ───────────────────────────────────────
e5_t = e5_t.rename({
    "valid_time": "time",
    "longitude": "lon",
    "latitude": "lat",
})

# ── 4. Compute indicators ────────────────────────────────────────────────
all_dfs = []

# CMIP6 models: use bias-corrected variables (suffix="_bc", the default)
for ds in list_hist + list_fut:
    all_dfs.extend(indicators.compute_all(ds, suffix="_bc"))

# ERA5: use raw variables (suffix="", no bias correction needed)
all_dfs.extend(indicators.compute_all(e5_t, suffix="", experiment_id="ERA5"))

all_df = pd.concat(all_dfs)

# ── 5. Analysis: relative change vs ERA5 baseline ────────────────────────
sel = all_df.drop(
    columns=[c for c in ["areacella", "sftlf", "orog", "percentiles"] if c in all_df.columns],
)
avg = sel.groupby(["indicator", "model", "experiment_id"])["value"].median().reset_index()
avg = avg.set_index(["indicator", "experiment_id"])

baseline = avg.xs("ERA5", level="experiment_id").drop(columns="model")

diff = (
    avg.set_index(["model"], append=True)
    .sub(baseline, level="indicator")
    .div(baseline, level="indicator")
    .reset_index()
    .query('model != ""')
)
# ── 6. Summary table with great_tables ────────────────────────────────────
from great_tables import GT, html, style, loc

# Drop extra coordinate columns that may be present
extra_cols = [c for c in ["areacella", "sftlf", "orog", "percentiles"] if c in all_df.columns]
clean = all_df.drop(columns=extra_cols)

# Per-model median across years
model_medians = (
    clean
    .groupby(["indicator", "model", "experiment_id"])["value"]
    .median()
    .reset_index()
)

# ERA5 baseline: one value per indicator
era5 = (
    model_medians
    .query('experiment_id == "ERA5"')
    .set_index("indicator")["value"]
    .rename("era5")
)

# Historical ensemble stats (across models)
hist_stats = (
    model_medians
    .query('experiment_id == "historical"')
    .groupby("indicator")["value"]
    .agg(hist_median="median", hist_min="min", hist_max="max")
)

# SSP3-7.0 ensemble stats (across models)
fut_stats = (
    model_medians
    .query('experiment_id == "ssp370"')
    .groupby("indicator")["value"]
    .agg(ssp370_median="median", ssp370_min="min", ssp370_max="max")
)

# Combine into a single summary DataFrame
summary = hist_stats.join(fut_stats).join(era5).reset_index()
summary["bias_vs_era5"] = summary["hist_median"] - summary["era5"]
summary["delta_pct"] = (
    (summary["ssp370_median"] - summary["hist_median"]) / summary["hist_median"]
)

# Pretty indicator names
summary["indicator"] = summary["indicator"].str.replace("_", " ").str.title()

# Build the table
gt_table = (
    GT(summary, rowname_col="indicator")
    .tab_header(
        title="Climate Indicators Summary",
        subtitle=html(
            f"Porto ({sel_lat}°N, {abs(sel_lon - 360):.1f}°W) &mdash; "
            "CMIP6 multi-model ensemble (bias-corrected)"
        ),
    )
    .tab_spanner(label="Historical (1985-2014)", columns=["hist_median", "hist_min", "hist_max"])
    .tab_spanner(label="SSP3-7.0 (2035-2064)", columns=["ssp370_median", "ssp370_min", "ssp370_max"])
    .cols_label(
        era5="ERA5",
        hist_median="Median",
        hist_min="Min",
        hist_max="Max",
        ssp370_median="Median",
        ssp370_min="Min",
        ssp370_max="Max",
        bias_vs_era5=html("Bias vs<br>ERA5"),
        delta_pct=html("&Delta; SSP3-7.0<br>vs Historical"),
    )
    .fmt_number(
        columns=["era5", "hist_median", "hist_min", "hist_max",
                 "ssp370_median", "ssp370_min", "ssp370_max", "bias_vs_era5"],
        decimals=1,
        use_seps=True,
    )
    .fmt_percent(columns="delta_pct", decimals=1)
    .data_color(
        columns="delta_pct",
        palette=["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
        domain=[-0.5, 0.5],
    )
    .data_color(
        columns="bias_vs_era5",
        palette=["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
        domain=[-50, 50],
    )
    .cols_align(align="center")
    .cols_align(align="left", columns="indicator")
    .tab_source_note("Spread shown as min/max across ensemble members.")
)

gt_table.save("summary_table.png", scale=2)
print("Table saved to summary_table.png")
