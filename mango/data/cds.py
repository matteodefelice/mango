"""Access ERA5 reanalysis data from the Copernicus Climate Data Store (CDS)."""

import os
import tempfile
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

from mango import config


def load_era5(
    lat: float,
    lon: float,
    period: tuple[str, str] = ("1980", "2014"),
    date_range: str = "1980-01-01/2026-04-06",
    cache_dir: Path | None = None,
    point_selection: bool = True,
) -> xr.Dataset:
    """Load ERA5 daily time-series data, downloading from CDS if not cached.
    Using the dataset [ERA5 hourly time-series data on single levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=download)
    The class download ERA5 hourly data for precipitation and temperature, converting it to daily data and saving to a local cache for future runs. 
    The returned dataset has the same format as the bias-corrected CMIP6 datasets, ready for indicator calculation.
    The conversion to daily data generates: max, min, and mean temperature, and total daily precipitation (converted to CMIP6-compatible units).
    Args:
        lat: Latitude for selection.
        lon: Longitude (0-360 convention; converted internally for CDS).
        period: (start, end) years to slice.
        date_range: Date range string for the CDS request.
        cache_dir: Local cache directory. Falls back to config if None.
        point_selection: If True, this is a point request (current behavior).
            If False in the future, could request a bounding box.

    Returns:
        xr.Dataset with daily variables: pr, tasmax, tasmin, tas.
            Precipitation is in CMIP6-compatible units (kg m-2 s-1).
    """
    resolved_dir = config.resolve_cache_dir(cache_dir)
    derived_dir = resolved_dir / "derived"
    derived_dir.mkdir(exist_ok=True)

    cached_filename = f"era5-{period[0]}_{period[1]}-{lat}-{lon}.daily_4_vars.nc"
    cached_full_path = derived_dir / cached_filename

    if cached_full_path.exists():
        return xr.open_dataset(cached_full_path)  # saved with canonical coord names
    # Checking longitude range
    cds_lon = lon if lon < 180 else lon - 360

    request = {
        "variable": ["2m_temperature", "total_precipitation"],
        "location": {"longitude": cds_lon, "latitude": lat},
        "date": [date_range],
        "data_format": "netcdf",
    }

    client = cdsapi.Client()
    tmp_filename = tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name
    try:
        client.retrieve("reanalysis-era5-single-levels-timeseries", request, tmp_filename)

        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(tmp_filename, "r") as z:
                nc_filename = [f for f in z.namelist() if f.endswith(".nc")][0]
                extracted_path = z.extract(nc_filename, path=tmpdir)

            with xr.open_dataset(extracted_path, engine="netcdf4") as e5:
                e5.load()
    finally:
        os.unlink(tmp_filename)

    e5_t = e5.sel(valid_time=slice(*period))
    tp_daily = e5_t["tp"].resample(valid_time="D")
    t2m_daily = e5_t["t2m"].resample(valid_time="D")
    e5_t = xr.merge([
        (tp_daily.sum() / 86.4).rename("pr"), # converting `m` to kg m-2 s-1
        t2m_daily.max().rename("tasmax"),
        t2m_daily.min().rename("tasmin"),
        t2m_daily.mean().rename("tas"),
    ])
    e5_t = e5_t.rename({"valid_time": "time", "longitude": "lon", "latitude": "lat"})
    e5_t["pr"].attrs["units"] = "kg m-2 s-1" # needed for xclim
    e5_t.to_netcdf(cached_full_path)
    return e5_t
