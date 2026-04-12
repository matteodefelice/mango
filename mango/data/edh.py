"""Access CMIP6 data from Earth Data Hub (EDH) via cached zarr stores."""

from pathlib import Path

import xarray as xr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import FsspecStore, LocalStore

from mango import config


# Template for EDH URLs — use with .format(token=..., dataset=...)
URL_TEMPLATE = "https://edh:{token}@data.earthdatahub.destine.eu/cmip6/{dataset}"


def build_url(dataset: str, token: str | None = None) -> str:
    """Build an EDH URL, reading the token from config if not given."""
    token = token or config.edh_token()
    if not token:
        raise ValueError(
            "EDH token not configured. Set 'edh_token' in mango.yaml "
            "or pass it explicitly."
        )
    return URL_TEMPLATE.format(token=token, dataset=dataset)


def open_cached_zarr(url: str, cache_dir: Path) -> xr.Dataset:
    """Open a remote zarr store with local disk caching."""
    http_store = FsspecStore.from_url(
        url,
        storage_options={"client_kwargs": {"trust_env": True}},
        read_only=True,
    )
    local_store = LocalStore(cache_dir / "edh" / Path(url).name)
    cache_store = CacheStore(
        store=http_store,
        cache_store=local_store,
        max_size=None,
    )
    return xr.open_dataset(cache_store, engine="zarr")


def load_cmip6_datasets_from_edh(
    urls: list[list[str]],
    lat: float,
    lon: float,
    variables: list[str],
    hist_period: tuple[str, str] = ("1985", "2014"),
    fut_period: tuple[str, str] = ("2035", "2064"),
    scenario: str = "ssp370",
    cache_dir: Path | None = None,
    point_selection: bool = True,
) -> tuple[list[xr.Dataset], list[xr.Dataset]]:
    """Load CMIP6 historical and future datasets from EDH.

    Args:
        urls: List of [historical_url, scenario_url] pairs.
        lat: Latitude for selection.
        lon: Longitude (0-360 for EDH).
        variables: Variable names to select (e.g. ['pr', 'tas', 'tasmin', 'tasmax']).
        hist_period: (start, end) years for historical slice.
        fut_period: (start, end) years for future slice.
        scenario: Scenario experiment_id to filter on.
        cache_dir: Local cache directory. Falls back to config if None.
        point_selection: If True, select nearest grid point. If False, return
            the full spatial extent (for future data-cube workflows).

    Returns:
        (list_hist, list_fut) tuple of loaded datasets.
    """
    resolved_dir = config.resolve_cache_dir(cache_dir)

    list_hist = []
    list_fut = []

    for model_urls in urls:
        cfut = open_cached_zarr(model_urls[1], cache_dir=resolved_dir)
        if scenario not in cfut["experiment_id"]:
            continue

        chist = open_cached_zarr(model_urls[0], cache_dir=resolved_dir)

        selected = [v for v in variables if v in chist.data_vars]

        hist_ds = chist.sel(time=slice(*hist_period))
        fut_ds = cfut.sel(time=slice(*fut_period), experiment_id=scenario)

        if point_selection:
            hist_ds = hist_ds.sel(lat=lat, lon=lon, method="nearest")
            fut_ds = fut_ds.sel(lat=lat, lon=lon, method="nearest")

        list_hist.append(hist_ds[selected].compute())
        list_fut.append(fut_ds[selected].compute())

    return list_hist, list_fut
