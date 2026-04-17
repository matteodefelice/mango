"""Summary table of climate indicators using great_tables."""

import pandas as pd
from great_tables import GT, html

from mango.workflow import Workflow

_EXTRA_COLS = ["areacella", "sftlf", "orog", "percentiles", "location"]
_DIVERGING  = ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"]


class IndicatorTable:
    """Build and save a summary table from a completed :class:`Workflow`.

    Args:
        workflow: A :class:`Workflow` instance after :meth:`~Workflow.run`
            has been called.
        scenario: Scenario experiment_id used in the workflow (default
            ``"ssp370"``).
    """

    def __init__(self, workflow: Workflow, scenario: str = "ssp370"):
        self.workflow = workflow
        self.scenario = scenario

    def _summarise(self) -> pd.DataFrame:
        """Aggregate results into one row per indicator."""
        scen = self.scenario
        results = self.workflow.results
        clean = results.drop(
            columns=[c for c in _EXTRA_COLS if c in results.columns]
        )
        model_medians = (
            clean
            .groupby(["indicator", "model", "experiment_id"])["value"]
            .median()
            .reset_index()
        )
        era5 = (
            model_medians.query('experiment_id == "ERA5"')
            .set_index("indicator")["value"].rename("era5")
        )
        hist = (
            model_medians.query('experiment_id == "historical"')
            .groupby("indicator")["value"]
            .agg(hist_median="median", hist_min="min", hist_max="max")
        )
        fut = (
            model_medians.query(f'experiment_id == "{scen}"')
            .groupby("indicator")["value"]
            .agg(**{f"{scen}_median": "median", f"{scen}_min": "min", f"{scen}_max": "max"})
        )
        summary = hist.join(fut).join(era5).reset_index()
        summary["bias_vs_era5"] = summary["hist_median"] - summary["era5"]
        summary["delta_pct"] = (
            (summary[f"{scen}_median"] - summary["hist_median"])
            / summary["hist_median"]
        )
        summary["indicator"] = summary["indicator"].str.replace("_", " ").str.title()
        return summary

    def build(self) -> GT:
        """Return the :class:`GT` table object."""
        wf = self.workflow
        scen = self.scenario
        summary = self._summarise()

        hist_label = f"Historical ({wf.hist_period[0]}\u2013{wf.hist_period[1]})"
        fut_label  = f"{scen.upper()} ({wf.fut_period[0]}\u2013{wf.fut_period[1]})"
        fut_cols   = [f"{scen}_median", f"{scen}_min", f"{scen}_max"]

        col_labels = dict(
            era5="ERA5",
            hist_median="Median", hist_min="Min", hist_max="Max",
            bias_vs_era5=html("Bias vs<br>ERA5"),
            delta_pct=html(f"&Delta; {scen.upper()}<br>vs Historical"),
        )
        col_labels[f"{scen}_median"] = "Median"
        col_labels[f"{scen}_min"]    = "Min"
        col_labels[f"{scen}_max"]    = "Max"

        return (
            GT(summary, rowname_col="indicator")
            .tab_header(
                title="Climate Indicators Summary",
                subtitle=html(
                    f"{wf.location_label}"
                    " &mdash; CMIP6 multi-model ensemble (bias-corrected)"
                ),
            )
            .tab_spanner(label=hist_label,
                         columns=["hist_median", "hist_min", "hist_max"])
            .tab_spanner(label=fut_label, columns=fut_cols)
            .cols_label(**col_labels)
            .fmt_number(
                columns=["era5", "hist_median", "hist_min", "hist_max",
                         *fut_cols, "bias_vs_era5"],
                decimals=1, use_seps=True,
            )
            .fmt_percent(columns="delta_pct", decimals=1)
            .data_color(columns="delta_pct", palette=_DIVERGING, domain=[-0.5, 0.5])
            .data_color(columns="bias_vs_era5", palette=_DIVERGING, domain=[-50, 50])
            .cols_align(align="center")
            .cols_align(align="left", columns="indicator")
            .tab_source_note("Spread shown as min/max across ensemble members.")
        )

    def save(self, path: str = "summary_table.png", scale: int = 2) -> None:
        """Build the table and save it to *path*."""
        self.build().save(path, scale=scale)
        print(f"Table saved to {path}")
