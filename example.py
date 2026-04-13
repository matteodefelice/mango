"""Example: run the mango workflow for Porto and produce visualisations.

Configuration is read from mango.yaml (see mango.yaml.example).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from great_tables import GT, html

from mango.data.edh import build_url
from mango.workflow import Workflow

# ── Model datasets ────────────────────────────────────────────────────────
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
urls = [[build_url(h), build_url(f)] for h, f in MODELS]

# ── Run workflow ──────────────────────────────────────────────────────────
wf = Workflow(lat=44.1427, lon=355.0, label="Porto", urls=urls)
results = wf.run()

# ── Distribution comparison plots ────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#f5f5f5"})

C_ERA5 = "#1a1a1a"
C_HIST = "#2166ac"
C_FUT  = "#b2182b"


def _tas_celsius(ds, var):
    return ds[var].values.flatten() - 273.15


def _pr_mmday(ds, var):
    pr = ds[var].values.flatten() * 86400
    return pr[pr > 0.1]


def _exceedance(values):
    sv = np.sort(values)
    n = len(sv)
    return sv, 1.0 - np.arange(1, n + 1) / (n + 1)


fig = plt.figure(figsize=(15, 11))
fig.suptitle(
    f"Temperature and Precipitation Distributions\n"
    f"{wf.label} ({wf.lat}°N, {abs(wf.lon - 360):.1f}°W) — "
    "CMIP6 multi-model ensemble vs ERA5",
    fontsize=14, fontweight="bold",
)

gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.28)

# ── Row 0: Temperature KDE (before / after BC) ───────────────────────────
for col_i, (version, col_title) in enumerate([
    ("raw", "Before bias correction"),
    ("bc",  "After bias correction"),
]):
    ax = fig.add_subplot(gs[0, col_i])
    var = "tas" if version == "raw" else "tas_bc"

    hist_tas_arrays, fut_tas_arrays = [], []
    for j, ds in enumerate(wf.list_hist):
        arr = _tas_celsius(ds, var)
        hist_tas_arrays.append(arr)
        sns.kdeplot(arr, ax=ax, color=C_HIST, linewidth=0.8, alpha=0.25,
                    label="Historical models" if j == 0 else None)
    for j, ds in enumerate(wf.list_fut):
        arr = _tas_celsius(ds, var)
        fut_tas_arrays.append(arr)
        sns.kdeplot(arr, ax=ax, color=C_FUT, linewidth=0.8, alpha=0.25,
                    label="SSP3-7.0 models" if j == 0 else None)

    all_hist_tas = np.concatenate(hist_tas_arrays)
    all_fut_tas  = np.concatenate(fut_tas_arrays)
    sns.kdeplot(all_hist_tas, ax=ax, color=C_HIST, linewidth=2.2,
                linestyle="--", label="Historical (pooled)")
    sns.kdeplot(all_fut_tas,  ax=ax, color=C_FUT,  linewidth=2.2,
                linestyle="--", label="SSP3-7.0 (pooled)")
    sns.kdeplot(_tas_celsius(wf.obs, "tas"), ax=ax, color=C_ERA5,
                linewidth=2.5, label="ERA5")

    ax.set_title(col_title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Daily Mean Temperature (°C)", fontsize=10)
    ax.set_ylabel("Density" if col_i == 0 else "", fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.85)
    ax.tick_params(labelsize=9)
    if col_i == 0:
        ax.annotate("Temperature", xy=(-0.2, 0.5), xycoords="axes fraction",
                    fontsize=12, fontweight="bold", rotation=90,
                    ha="center", va="center")

# ── Row 1, Col 0: Precipitation Q-Q plot (before vs after BC) ────────────
ax_qq = fig.add_subplot(gs[1, 0])

quantiles = np.linspace(0.01, 0.99, 200)
era5_pr   = _pr_mmday(wf.obs, "pr")
era5_q    = np.quantile(era5_pr, quantiles)

hist_pr_arrays = []
for j, ds in enumerate(wf.list_hist):
    pr_raw = _pr_mmday(ds, "pr")
    pr_bc  = _pr_mmday(ds, "pr_bc")
    hist_pr_arrays.append(pr_bc)
    ax_qq.plot(era5_q, np.quantile(pr_raw, quantiles),
               color=C_HIST, linewidth=1.0, alpha=0.35, linestyle="--",
               label="Historical (before BC)" if j == 0 else None)
    ax_qq.plot(era5_q, np.quantile(pr_bc, quantiles),
               color=C_HIST, linewidth=1.2, alpha=0.55,
               label="Historical (after BC)" if j == 0 else None)

fut_pr_arrays = []
for j, ds in enumerate(wf.list_fut):
    pr_bc = _pr_mmday(ds, "pr_bc")
    fut_pr_arrays.append(pr_bc)
    ax_qq.plot(era5_q, np.quantile(pr_bc, quantiles),
               color=C_FUT, linewidth=1.2, alpha=0.55,
               label="SSP3-7.0 (after BC)" if j == 0 else None)

ref_max = era5_q.max() * 1.1
ax_qq.plot([0, ref_max], [0, ref_max], color="gray", linewidth=1.5,
           linestyle=":", label="1 : 1 reference")

ax_qq.set_xlabel("ERA5 quantiles (mm day⁻¹)", fontsize=10)
ax_qq.set_ylabel("Model quantiles (mm day⁻¹)", fontsize=10)
ax_qq.set_title("Precipitation Q-Q vs ERA5", fontsize=11, fontweight="bold", pad=10)
ax_qq.legend(fontsize=8.5, framealpha=0.85)
ax_qq.tick_params(labelsize=9)
ax_qq.annotate("Precipitation", xy=(-0.2, 0.5), xycoords="axes fraction",
               fontsize=12, fontweight="bold", rotation=90,
               ha="center", va="center")

# ── Row 1, Col 1: Exceedance probability (log-log, after BC) ─────────────
ax_exc = fig.add_subplot(gs[1, 1])

for j, arr in enumerate(hist_pr_arrays):
    x, y = _exceedance(arr)
    ax_exc.plot(x, y, color=C_HIST, linewidth=0.7, alpha=0.25,
                label="Historical models" if j == 0 else None)
for j, arr in enumerate(fut_pr_arrays):
    x, y = _exceedance(arr)
    ax_exc.plot(x, y, color=C_FUT, linewidth=0.7, alpha=0.25,
                label="SSP3-7.0 models" if j == 0 else None)

x, y = _exceedance(np.concatenate(hist_pr_arrays))
ax_exc.plot(x, y, color=C_HIST, linewidth=2.2, linestyle="--", label="Historical (pooled)")
x, y = _exceedance(np.concatenate(fut_pr_arrays))
ax_exc.plot(x, y, color=C_FUT, linewidth=2.2, linestyle="--", label="SSP3-7.0 (pooled)")
x, y = _exceedance(era5_pr)
ax_exc.plot(x, y, color=C_ERA5, linewidth=2.5, label="ERA5")

ax_exc.set_xscale("log")
ax_exc.set_yscale("log")
ax_exc.set_xlabel("Daily Precipitation (mm day⁻¹)", fontsize=10)
ax_exc.set_ylabel("Exceedance probability", fontsize=10)
ax_exc.set_title("Precipitation Exceedance Probability (after BC)",
                 fontsize=11, fontweight="bold", pad=10)
ax_exc.legend(fontsize=8.5, framealpha=0.85)
ax_exc.tick_params(labelsize=9)

plt.savefig("distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Distribution plots saved to distributions.png")

# ── Summary table ─────────────────────────────────────────────────────────
extra_cols = [c for c in ["areacella", "sftlf", "orog", "percentiles", "location"]
              if c in results.columns]
clean = results.drop(columns=extra_cols)

model_medians = (
    clean
    .groupby(["indicator", "model", "experiment_id"])["value"]
    .median()
    .reset_index()
)
era5_base = (
    model_medians.query('experiment_id == "ERA5"')
    .set_index("indicator")["value"].rename("era5")
)
hist_stats = (
    model_medians.query('experiment_id == "historical"')
    .groupby("indicator")["value"]
    .agg(hist_median="median", hist_min="min", hist_max="max")
)
fut_stats = (
    model_medians.query('experiment_id == "ssp370"')
    .groupby("indicator")["value"]
    .agg(ssp370_median="median", ssp370_min="min", ssp370_max="max")
)

summary = hist_stats.join(fut_stats).join(era5_base).reset_index()
summary["bias_vs_era5"] = summary["hist_median"] - summary["era5"]
summary["delta_pct"] = (summary["ssp370_median"] - summary["hist_median"]) / summary["hist_median"]
summary["indicator"] = summary["indicator"].str.replace("_", " ").str.title()

gt_table = (
    GT(summary, rowname_col="indicator")
    .tab_header(
        title="Climate Indicators Summary",
        subtitle=html(
            f"{wf.label} ({wf.lat}°N, {abs(wf.lon - 360):.1f}°W) &mdash; "
            "CMIP6 multi-model ensemble (bias-corrected)"
        ),
    )
    .tab_spanner(label="Historical (1985-2014)", columns=["hist_median", "hist_min", "hist_max"])
    .tab_spanner(label="SSP3-7.0 (2035-2064)", columns=["ssp370_median", "ssp370_min", "ssp370_max"])
    .cols_label(
        era5="ERA5",
        hist_median="Median", hist_min="Min", hist_max="Max",
        ssp370_median="Median", ssp370_min="Min", ssp370_max="Max",
        bias_vs_era5=html("Bias vs<br>ERA5"),
        delta_pct=html("&Delta; SSP3-7.0<br>vs Historical"),
    )
    .fmt_number(
        columns=["era5", "hist_median", "hist_min", "hist_max",
                 "ssp370_median", "ssp370_min", "ssp370_max", "bias_vs_era5"],
        decimals=1, use_seps=True,
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
