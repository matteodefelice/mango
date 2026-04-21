"""Summary table of climate indicators using great_tables."""

from __future__ import annotations

import pandas as pd
from PIL import Image, ImageChops
from great_tables import GT, html, loc, style

from mango.workflow import Workflow

def _crop_whitespace(path: str) -> None:
    """Trim pure-white margins from a PNG saved by great_tables."""
    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bbox = ImageChops.difference(img, bg).getbbox()
    if bbox:
        img.crop(bbox).save(path)


_EXTRA_COLS = ["areacella", "sftlf", "orog", "percentiles", "location", "label"]
_DIVERGING  = ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"]

# Indicators whose raw values are in kg m-2 s-1 and must be converted to mm/day
_PR_INDICATORS = {"max_1day_precipitation_amount"}
_KG_M2_S1_TO_MM = 86400


class IndicatorTable:
    """Build and save a summary table from a completed :class:`Workflow` or a
    DataFrame previously exported via :meth:`~Workflow.to_parquet`.

    Args:
        source: Either a :class:`Workflow` instance (after :meth:`~Workflow.run`)
            or a :class:`~pandas.DataFrame` loaded from a parquet file.
        scenario: Scenario experiment_id (default ``"ssp370"``).
        hist_period: ``(start, end)`` years for the historical label.  Required
            when *source* is a DataFrame; ignored when *source* is a Workflow.
        fut_period: ``(start, end)`` years for the future label.  Required when
            *source* is a DataFrame; ignored when *source* is a Workflow.
    """

    def __init__(
        self,
        source: Workflow | pd.DataFrame,
        scenario: str = "ssp370",
        hist_period: tuple[str, str] | None = None,
        fut_period: tuple[str, str] | None = None,
    ):
        if isinstance(source, Workflow):
            self._results = source.results
            self._hist_period = source.hist_period
            self._fut_period = source.fut_period
            self._location_label = source.location_label
        else:
            if hist_period is None or fut_period is None:
                raise ValueError(
                    "hist_period and fut_period are required when source is a DataFrame"
                )
            self._results = source
            self._hist_period = hist_period
            self._fut_period = fut_period
            self._location_label = (
                source["label"].iloc[0] if "label" in source.columns else ""
            )
        self.scenario = scenario

    def _summarise(self) -> pd.DataFrame:
        """Aggregate results into one row per indicator."""
        scen = self.scenario
        results = self._results.copy()
        results.loc[results["indicator"].isin(_PR_INDICATORS), "value"] *= _KG_M2_S1_TO_MM
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
        scen = self.scenario
        summary = self._summarise()

        hist_label   = f"Historical ({self._hist_period[0]}\u2013{self._hist_period[1]})"
        fut_label    = f"{scen.upper()} ({self._fut_period[0]}\u2013{self._fut_period[1]})"
        hist_cols    = ["hist_min", "hist_median", "hist_max"]
        fut_cols     = [f"{scen}_min", f"{scen}_median", f"{scen}_max"]
        median_cols  = ["hist_median", f"{scen}_median"]
        minmax_cols  = ["hist_min", "hist_max", f"{scen}_min", f"{scen}_max"]

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
                    f"{self._location_label}"
                    " &mdash; CMIP6 multi-model ensemble (bias-corrected)"
                ),
            )
            .tab_spanner(label=hist_label, columns=hist_cols)
            .tab_spanner(label=fut_label, columns=fut_cols)
            .cols_label(**col_labels)
            .fmt_number(
                columns=["era5", *hist_cols, *fut_cols, "bias_vs_era5"],
                decimals=0, use_seps=True,
            )
            .fmt_percent(columns="delta_pct", decimals=1)
            .data_color(columns="delta_pct", palette=_DIVERGING, domain=[-0.95, 0.95])
            .tab_style(
                style=style.text(weight="bold"),
                locations=loc.body(columns=median_cols),
            )
            .tab_style(
                style=style.text(color="#aaaaaa"),
                locations=loc.body(columns=minmax_cols),
            )
            .cols_align(align="center")
            .cols_align(align="left", columns="indicator")
            .tab_source_note("Spread shown as min/max across ensemble members.")
        )

    def save(self, path: str = "summary_table.png", scale: int = 2) -> None:
        """Build the table and save it to *path*.

        The PNG is auto-cropped to remove the blank viewport space that
        great_tables leaves below the rendered table.
        """
        self.build().save(path, scale=scale)
        _crop_whitespace(path)
        print(f"Table saved to {path}")
