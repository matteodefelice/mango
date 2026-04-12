"""Bias correction for climate model data using ibicus."""

from pathlib import Path

import numpy as np
import xarray as xr
from ibicus.debias import ISIMIP

from mango import config


def _apply_debias(
    debiaser: ISIMIP,
    obs_vals: np.ndarray,
    list_hist: list[xr.Dataset],
    list_fut: list[xr.Dataset],
    var: str,
) -> None:
    """Apply a debiaser to a single variable across all model pairs in-place."""
    for hist_ds, fut_ds in zip(list_hist, list_fut, strict=True):
        if var not in hist_ds.data_vars:
            continue

        hist_vals = hist_ds[var].values[:, np.newaxis, np.newaxis]

        hist_db = debiaser.apply(obs_vals, hist_vals, hist_vals)
        hist_ds[f"{var}_bc"] = xr.DataArray(
            hist_db.squeeze(),
            dims=hist_ds.dims,
            coords=hist_ds.coords,
            attrs=hist_ds[var].attrs,
        )

        fut_vals = fut_ds[var].values[:, np.newaxis, np.newaxis]
        fut_db = debiaser.apply(obs_vals, hist_vals, fut_vals)
        fut_ds[f"{var}_bc"] = xr.DataArray(
            fut_db.squeeze(),
            dims=fut_ds.dims,
            coords=fut_ds.coords,
            attrs=fut_ds[var].attrs,
        )


def debias_temperature(
    obs: xr.Dataset,
    list_hist: list[xr.Dataset],
    list_fut: list[xr.Dataset],
    obs_vars: dict[str, str] | None = None,
) -> None:
    """Apply ISIMIP bias correction to temperature variables in-place.

    Args:
        obs: Observational reference dataset (e.g. ERA5).
        list_hist: Historical model datasets.
        list_fut: Future model datasets.
        obs_vars: Mapping from model var name to obs var name.
            Defaults to identity (same name in obs and model).
    """
    if obs_vars is None:
        obs_vars = {"tas": "tas", "tasmin": "tasmin", "tasmax": "tasmax"}

    debiaser = ISIMIP.from_variable(variable="tas")

    for var, obs_var in obs_vars.items():
        obs_vals = obs[obs_var].values[:, np.newaxis, np.newaxis]
        _apply_debias(debiaser, obs_vals, list_hist, list_fut, var)


def debias_precipitation(
    obs: xr.Dataset,
    list_hist: list[xr.Dataset],
    list_fut: list[xr.Dataset],
    obs_var: str = "pr",
    model_var: str = "pr",
) -> None:
    """Apply ISIMIP bias correction to precipitation in-place.

    Args:
        obs: Observational reference dataset (e.g. ERA5).
        list_hist: Historical model datasets.
        list_fut: Future model datasets.
        obs_var: Name of the precipitation variable in obs.
        model_var: Name of the precipitation variable in models.
    """
    debiaser = ISIMIP.from_variable(variable="pr")
    obs_vals = obs[obs_var].values[:, np.newaxis, np.newaxis]
    _apply_debias(debiaser, obs_vals, list_hist, list_fut, model_var)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _cache_key(ds: xr.Dataset) -> str:
    """Derive a unique filename stem from a dataset's metadata."""
    source = ds.attrs.get("source_id", "unknown")
    exp = str(ds["experiment_id"].values) if "experiment_id" in ds.coords else "unknown"
    t0 = str(ds["time"].values[0])[:10]
    t1 = str(ds["time"].values[-1])[:10]

    # lat/lon may be scalar (point) or arrays (cube)
    lat = ds["lat"].values
    lon = ds["lon"].values
    lat_str = f"{float(lat):.2f}" if lat.ndim == 0 else "spatial"
    lon_str = f"{float(lon):.2f}" if lon.ndim == 0 else "spatial"

    return f"{source}_{exp}_{lat_str}_{lon_str}_{t0}_{t1}"


def _debiased_cache_path(ds: xr.Dataset, cache_dir: Path) -> Path:
    """Return the full cache path for a debiased dataset."""
    debiased_dir = cache_dir / "debiased"
    debiased_dir.mkdir(exist_ok=True)
    return debiased_dir / f"{_cache_key(ds)}.nc"


def _load_cached(ds: xr.Dataset, cache_dir: Path) -> xr.Dataset | None:
    """Load a cached debiased dataset, or return None if not found."""
    path = _debiased_cache_path(ds, cache_dir)
    if path.exists():
        return xr.open_dataset(path).compute()
    return None


def _save_cached(ds: xr.Dataset, cache_dir: Path) -> None:
    """Save a debiased dataset to cache."""
    path = _debiased_cache_path(ds, cache_dir)
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Convenience function with caching
# ---------------------------------------------------------------------------

def debias_with_cache(
    obs: xr.Dataset,
    list_hist: list[xr.Dataset],
    list_fut: list[xr.Dataset],
    cache_dir: Path | None = None,
) -> None:
    """Run full bias correction (temperature + precipitation) with disk caching.

    For each (historical, future) model pair:
      - If both are found in the cache, load them directly.
      - Otherwise, run ISIMIP debiasing and save the results to cache.

    The lists are modified in-place (entries may be replaced with cached
    versions).

    Args:
        obs: Observational reference dataset (e.g. ERA5).
        list_hist: Historical model datasets.
        list_fut: Future model datasets.
        cache_dir: Cache directory. Falls back to config if None.
    """
    resolved_dir = config.resolve_cache_dir(cache_dir)

    for i, (hist_ds, fut_ds) in enumerate(zip(list_hist, list_fut, strict=True)):
        cached_hist = _load_cached(hist_ds, resolved_dir)
        cached_fut = _load_cached(fut_ds, resolved_dir)

        if cached_hist is not None and cached_fut is not None:
            list_hist[i] = cached_hist
            list_fut[i] = cached_fut
            source = hist_ds.attrs.get("source_id", "?")
            print(f"  {source}: loaded from cache")
            continue

        # Debias this single pair using the existing functions
        pair_hist = [hist_ds]
        pair_fut = [fut_ds]
        debias_temperature(obs, pair_hist, pair_fut)
        debias_precipitation(obs, pair_hist, pair_fut)

        _save_cached(pair_hist[0], resolved_dir)
        _save_cached(pair_fut[0], resolved_dir)
        list_hist[i] = pair_hist[0]
        list_fut[i] = pair_fut[0]

        source = hist_ds.attrs.get("source_id", "?")
        print(f"  {source}: debiased and cached")
