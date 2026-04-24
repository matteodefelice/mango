"""Distribution plots for climate model ensembles."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.gridspec as gridspec
import seaborn as sns

from mango.workflow import Workflow

C_ERA5 = "#1a1a1a"
C_HIST = "#2166ac"
C_FUT  = "#b2182b"

# From Kelvin to Celsius
def _tas_celsius(ds, var):
    return ds[var].values.flatten() - 273.15


# From kg/m2/s to mm/day (returning only wet days > 0.1 mm/day)
def _pr_mmday(ds, var):
    pr = ds[var].values.flatten() * 86400
    return pr[pr > 0.1]


def _exceedance(values):
    sv = np.sort(values)
    n = len(sv)
    return sv, 1.0 - np.arange(1, n + 1) / (n + 1)


class DistributionPlot:
    """2×2 distribution figure from a completed :class:`Workflow`.

    Layout:
      - Row 0: Temperature KDE before / after bias correction
      - Row 1: Precipitation Q-Q vs ERA5 / Exceedance probability (log-log)

    Args:
        workflow: A :class:`Workflow` instance after :meth:`~Workflow.run`
            has been called.
    """

    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        sns.set_theme(style="whitegrid", font_scale=1.1)
        plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#f5f5f5"})

    def build(self) -> matplotlib.figure.Figure:
        """Return the fully rendered :class:`matplotlib.figure.Figure`."""
        wf = self.workflow

        fig = plt.figure(figsize=(15, 11))
        fig.suptitle(
            f"Temperature and Precipitation Distributions\n"
            f"{wf.location_label} — CMIP6 multi-model ensemble vs ERA5",
            fontsize=14, fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.28)

        self._temperature_kde(fig, gs, wf)
        hist_pr, fut_pr, era5_pr = self._precipitation_qq(fig, gs, wf)
        self._precipitation_exceedance(fig, gs, hist_pr, fut_pr, era5_pr)

        return fig

    def save(self, path: str = "distributions.png", dpi: int = 150) -> None:
        """Build the figure and save it to *path*."""
        fig = self.build()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Distribution plots saved to {path}")

    def _temperature_kde(self, fig, gs, wf):
        era5_tas = _tas_celsius(wf.obs, "tas")

        for col_i, (var, col_title) in enumerate([
            ("tas",    "Before bias correction"),
            ("tas_bc", "After bias correction"),
        ]):
            ax = fig.add_subplot(gs[0, col_i])

            hist_arrays, fut_arrays = [], []
            for j, ds in enumerate(wf.list_hist):
                arr = _tas_celsius(ds, var)
                hist_arrays.append(arr)
                sns.kdeplot(arr, ax=ax, color=C_HIST, linewidth=0.8, alpha=0.25,
                            label="Historical models" if j == 0 else None)
            for j, ds in enumerate(wf.list_fut):
                arr = _tas_celsius(ds, var)
                fut_arrays.append(arr)
                sns.kdeplot(arr, ax=ax, color=C_FUT, linewidth=0.8, alpha=0.25,
                            label="SSP3-7.0 models" if j == 0 else None)

            sns.kdeplot(np.concatenate(hist_arrays), ax=ax, color=C_HIST,
                        linewidth=2.2, linestyle="--", label="Historical (pooled)")
            sns.kdeplot(np.concatenate(fut_arrays), ax=ax, color=C_FUT,
                        linewidth=2.2, linestyle="--", label="SSP3-7.0 (pooled)")
            sns.kdeplot(era5_tas, ax=ax, color=C_ERA5, linewidth=2.5, label="ERA5")

            ax.set_title(col_title, fontsize=11, fontweight="bold", pad=10)
            ax.set_xlabel("Daily Mean Temperature (°C)", fontsize=10)
            ax.set_ylabel("Density" if col_i == 0 else "", fontsize=10)
            ax.legend(fontsize=8.5, framealpha=0.85)
            ax.tick_params(labelsize=9)
            if col_i == 0:
                ax.annotate("Temperature", xy=(-0.2, 0.5), xycoords="axes fraction",
                            fontsize=12, fontweight="bold", rotation=90,
                            ha="center", va="center")

    def _precipitation_qq(self, fig, gs, wf):
        """Draw the Q-Q panel; return (hist_pr_arrays, fut_pr_arrays, era5_pr) for reuse."""
        ax = fig.add_subplot(gs[1, 0])

        quantiles = np.linspace(0.01, 0.99, 200)
        era5_pr   = _pr_mmday(wf.obs, "pr")
        era5_q    = np.quantile(era5_pr, quantiles)

        hist_pr_arrays = []
        for j, ds in enumerate(wf.list_hist):
            pr_raw = _pr_mmday(ds, "pr")
            pr_bc  = _pr_mmday(ds, "pr_bc")
            hist_pr_arrays.append(pr_bc)
            ax.plot(era5_q, np.quantile(pr_raw, quantiles),
                    color=C_HIST, linewidth=1.0, alpha=0.35, linestyle="--",
                    label="Historical (before BC)" if j == 0 else None)
            ax.plot(era5_q, np.quantile(pr_bc, quantiles),
                    color=C_HIST, linewidth=1.2, alpha=0.55,
                    label="Historical (after BC)" if j == 0 else None)

        fut_pr_arrays = []
        for j, ds in enumerate(wf.list_fut):
            pr_bc = _pr_mmday(ds, "pr_bc")
            fut_pr_arrays.append(pr_bc)
            ax.plot(era5_q, np.quantile(pr_bc, quantiles),
                    color=C_FUT, linewidth=1.2, alpha=0.55,
                    label="SSP3-7.0 (after BC)" if j == 0 else None)

        ref_max = era5_q.max() * 1.1
        ax.plot([0, ref_max], [0, ref_max], color="gray", linewidth=1.5,
                linestyle=":", label="1 : 1 reference")

        ax.set_xlabel("ERA5 quantiles (mm day⁻¹)", fontsize=10)
        ax.set_ylabel("Model quantiles (mm day⁻¹)", fontsize=10)
        ax.set_title("Precipitation Q-Q vs ERA5", fontsize=11, fontweight="bold", pad=10)
        ax.legend(fontsize=8.5, framealpha=0.85)
        ax.tick_params(labelsize=9)
        ax.annotate("Precipitation", xy=(-0.2, 0.5), xycoords="axes fraction",
                    fontsize=12, fontweight="bold", rotation=90,
                    ha="center", va="center")

        return hist_pr_arrays, fut_pr_arrays, era5_pr

    def _precipitation_exceedance(self, fig, gs, hist_pr_arrays, fut_pr_arrays, era5_pr):
        ax = fig.add_subplot(gs[1, 1])

        for j, arr in enumerate(hist_pr_arrays):
            x, y = _exceedance(arr)
            ax.plot(x, y, color=C_HIST, linewidth=0.7, alpha=0.25,
                    label="Historical models" if j == 0 else None)
        for j, arr in enumerate(fut_pr_arrays):
            x, y = _exceedance(arr)
            ax.plot(x, y, color=C_FUT, linewidth=0.7, alpha=0.25,
                    label="SSP3-7.0 models" if j == 0 else None)

        x, y = _exceedance(np.concatenate(hist_pr_arrays))
        ax.plot(x, y, color=C_HIST, linewidth=2.2, linestyle="--", label="Historical (pooled)")
        x, y = _exceedance(np.concatenate(fut_pr_arrays))
        ax.plot(x, y, color=C_FUT,  linewidth=2.2, linestyle="--", label="SSP3-7.0 (pooled)")
        x, y = _exceedance(era5_pr)
        ax.plot(x, y, color=C_ERA5, linewidth=2.5, label="ERA5")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Daily Precipitation (mm day⁻¹)", fontsize=10)
        ax.set_ylabel("Exceedance probability", fontsize=10)
        ax.set_title("Precipitation Exceedance Probability (after BC)",
                     fontsize=11, fontweight="bold", pad=10)
        ax.legend(fontsize=8.5, framealpha=0.85)
        ax.tick_params(labelsize=9)
