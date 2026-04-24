"""End-to-end climate indicator workflow."""

import itertools
from typing import Callable

import pandas as pd
import xarray as xr

from mango.data import load_cmip6_datasets_from_edh, load_era5
from mango.debias import debias_with_cache
from mango import indicators


class Workflow:
    """Run the full pipeline for a single location.
    A caching mechanism is used whenever possible to speed up consecutive runs on the same location. 

    Steps executed:
      1. Load CMIP6 historical + future data from EarthDataHub (EDH)
      2. Load ERA5 reanalysis from Copernicus Data Store (CDS)
      3. Apply ISIMIP bias correction (with disk cache) using ibicus
      4. Compute all applicable climate indicators
      5. Return results as a tidy DataFrame

    Typical usage::

        wf = Workflow(...)
        wf.load()           # fetch data + bias correction
        wf.run()            # compute indicators → wf.results

    After `load` + `run` the following attributes are available for
    downstream use (e.g. distribution plots):

    - ``results``   — tidy DataFrame with columns: value, experiment_id,
                      indicator, model, location
    - ``list_hist`` — bias-corrected historical model datasets
    - ``list_fut``  — bias-corrected future model datasets
    - ``obs``       — ERA5 dataset (ready for indicators)

    Args:
        lat: Latitude of the location.
        lon: Longitude (0-360 convention for EDH).
        label: Human-readable location name, added as a ``location`` column.
        urls: List of ``[historical_url, scenario_url]`` pairs.
        variables: CMIP6 variables to load.
        hist_period: (start, end) years for the historical slice.
        fut_period: (start, end) years for the future slice.
        scenario: Scenario experiment_id (default ``"ssp370"``).
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        label: str,
        urls: list[list[str]],
        variables: list[str] | None = None,
        hist_period: tuple[str, str] = ("1985", "2014"),
        fut_period: tuple[str, str] = ("2035", "2064"),
        scenario: str = "ssp370",
    ):
        self.lat = lat
        self.lon = lon
        self.label = label
        self.urls = urls
        self.variables = variables or ["pr", "tas", "tasmin", "tasmax"]
        self.hist_period = hist_period
        self.fut_period = fut_period
        self.scenario = scenario

        self.results: pd.DataFrame | None = None
        self.list_hist: list[xr.Dataset] = []
        self.list_fut: list[xr.Dataset] = []
        self.obs: xr.Dataset | None = None

    @property
    def location_label(self) -> str:
        """Human-readable location string, e.g. ``'Porto (44.1°N, 5.0°W)'``."""
        return f"{self.label} ({self.lat}°N, {abs(self.lon - 360):.1f}°W)"

    @property
    def is_loaded(self) -> bool:
        """True once :meth:`load` has populated all three data attributes."""
        return bool(self.list_hist) and bool(self.list_fut) and self.obs is not None

    def load(self, on_step: Callable[[str, float], None] | None = None) -> None:
        """Load CMIP6 and ERA5 data and apply bias correction.

        Populates ``list_hist``, ``list_fut``, and ``obs``.
        Must be called before `run`.

        Args:
            on_step: Optional callback invoked before each of the three phases
                as ``on_step(label, progress)`` where ``progress`` ∈ [0, 1].
                Useful for progress bars in UIs.
        """
        def _step(label: str, progress: float) -> None:
            print(f"[{self.label}] {label}")
            if on_step is not None:
                on_step(label, progress)

        try:
            _step("Loading CMIP6 data...", 0.05)
            self.list_hist, self.list_fut = load_cmip6_datasets_from_edh(
                self.urls,
                lat=self.lat,
                lon=self.lon,
                variables=self.variables,
                hist_period=self.hist_period,
                fut_period=self.fut_period,
                scenario=self.scenario,
            )

            _step("Loading ERA5...", 0.45)
            self.obs = load_era5(lat=self.lat, lon=self.lon)

            _step("Bias correction...", 0.75)
            debias_with_cache(obs=self.obs, list_hist=self.list_hist, list_fut=self.list_fut)

            _step("Done.", 1.0)
        except Exception:
            # Leave no partial state behind on failure.
            self.list_hist, self.list_fut, self.obs = [], [], None
            raise

    def run(
        self,
        bias_correction: bool = True,
        months: list[int] | None = None,
    ) -> pd.DataFrame:
        """Compute climate indicators and return the results DataFrame.

        Args:
            bias_correction: Use bias-corrected variables (suffix ``"_bc"``)
                when ``True``; use raw model output (no suffix) when ``False``.
            months: If given, restrict computation to these calendar months
                (1 = Jan … 12 = Dec) before passing data to each indicator.

        Returns:
            Tidy DataFrame with columns: value, experiment_id, indicator,
            model, location.
        """
        if not self.list_hist:
            raise RuntimeError("No data loaded. Call `load()` first.")

        suffix = "_bc" if bias_correction else ""

        def _filter(ds: xr.Dataset) -> xr.Dataset:
            if months is None:
                return ds
            return ds.sel(time=ds["time.month"].isin(months))

        names = indicators.available_for_months_filter() if months is not None else None

        print(f"[{self.label}] Computing indicators...")
        dfs = []
        # Compute indicators for each dataset, applying month filter and suffix as needed
        for ds in itertools.chain(self.list_hist, self.list_fut):
            dfs.extend(indicators.compute_all(_filter(ds), suffix=suffix, names=names))
        dfs.extend(indicators.compute_all(_filter(self.obs), suffix="", experiment_id="ERA5", names=names))

        self.results = pd.concat(dfs).assign(
            location=self.label,
            bias_correction=bias_correction,
            months=",".join(map(str, months)) if months is not None else None,
        )
        print(f"[{self.label}] Done.")
        return self.results

    def to_parquet(self, path: str) -> None:
        """Export results to a Parquet file with an added ``label`` column."""
        if self.results is None:
            raise RuntimeError("No results available. Run `run()` first.")
        self.results.assign(label=self.label).to_parquet(path, index=False)
