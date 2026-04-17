"""End-to-end climate indicator workflow."""

import itertools

import pandas as pd
import xarray as xr

from mango.data import load_cmip6_datasets_from_edh, load_era5
from mango.debias import debias_with_cache
from mango import indicators


class Workflow:
    """Run the full pipeline for a single location.

    Steps executed by :meth:`run`:
      1. Load CMIP6 historical + future data from EDH
      2. Load ERA5 reanalysis from CDS
      3. Apply ISIMIP bias correction (with disk cache)
      4. Compute all applicable climate indicators
      5. Return results as a tidy DataFrame

    After :meth:`run` the following attributes are available for
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

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and return the results DataFrame."""
        print(f"[{self.label}] Loading CMIP6 data...")
        self.list_hist, self.list_fut = load_cmip6_datasets_from_edh(
            self.urls,
            lat=self.lat,
            lon=self.lon,
            variables=self.variables,
            hist_period=self.hist_period,
            fut_period=self.fut_period,
            scenario=self.scenario,
        )

        print(f"[{self.label}] Loading ERA5...")
        self.obs = load_era5(lat=self.lat, lon=self.lon)

        print(f"[{self.label}] Bias correction...")
        debias_with_cache(obs=self.obs, list_hist=self.list_hist, list_fut=self.list_fut)

        print(f"[{self.label}] Computing indicators...")
        dfs = []
        for ds in itertools.chain(self.list_hist, self.list_fut):
            dfs.extend(indicators.compute_all(ds, suffix="_bc"))
        dfs.extend(indicators.compute_all(self.obs, suffix="", experiment_id="ERA5"))

        self.results = pd.concat(dfs).assign(location=self.label)
        print(f"[{self.label}] Done.")
        return self.results
