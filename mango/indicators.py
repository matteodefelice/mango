"""Climate indicator registry and built-in indicators using xclim.

Indicators are written against canonical variable names (``tasmin``,
``tasmax``, ``tas``, ``pr``).  At compute time a *suffix* (default
``"_bc"``) selects which version of the variable to use: the framework
renames e.g. ``tasmin_bc`` → ``tasmin`` before calling the indicator, so
indicator functions stay suffix-agnostic.

To add a new indicator, define a function that takes an xr.Dataset and
returns an xr.DataArray, then register it:

    @register("my_indicator", requires=["tas"])
    def my_indicator(ds: xr.Dataset) -> xr.DataArray:
        return xclim.indices.some_index(tas=ds["tas"])
"""

from typing import Callable

import pandas as pd
import xarray as xr
import xclim


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict] = {}


def register(name: str, requires: list[str]):
    """Decorator to register an indicator function.

    Args:
        name: Indicator name (used as key).
        requires: List of *canonical* (base) variable names the dataset must
            contain (e.g. ``["tasmin", "tasmax"]``, **without** suffix).
    """
    def decorator(fn: Callable[[xr.Dataset], xr.DataArray]):
        _REGISTRY[name] = {"fn": fn, "requires": requires}
        return fn
    return decorator


def available() -> list[str]:
    """Return names of all registered indicators."""
    return list(_REGISTRY.keys())


def get_required_vars(name: str) -> list[str]:
    """Return the required canonical variables for an indicator."""
    return _REGISTRY[name]["requires"]


def _prepare_dataset(
    ds: xr.Dataset,
    required: list[str],
    suffix: str,
) -> xr.Dataset | None:
    """Resolve suffixed variable names and rename to canonical names.

    Returns a lightweight renamed dataset, or *None* if the required
    variables are not present.
    """
    if not suffix:
        # No suffix — check that canonical names exist directly
        if not all(base in ds.data_vars for base in required):
            return None
        return ds

    suffixed = {f"{base}{suffix}": base for base in required}

    # Check that every required variable (with suffix) exists
    if not all(s in ds.data_vars for s in suffixed):
        return None

    # Drop conflicting canonical names before renaming, then rename
    conflicts = [base for base in required if base in ds.data_vars]
    subset = ds.drop_vars(conflicts) if conflicts else ds
    return subset.rename(suffixed)


def compute(
    ds: xr.Dataset,
    name: str,
    suffix: str = "_bc",
    model: str = "",
    experiment_id: str = "historical",
) -> pd.DataFrame:
    """Compute a single indicator and return it as a tidy DataFrame.

    Args:
        ds: Input dataset.
        name: Registered indicator name.
        suffix: Variable suffix to use (e.g. ``"_bc"`` for bias-corrected,
            ``""`` for raw model output).
        model: Model identifier for the output DataFrame.
        experiment_id: Experiment identifier for the output DataFrame.

    Returns:
        DataFrame with columns: value, experiment_id, indicator, model.
    """
    entry = _REGISTRY[name]
    prepared = _prepare_dataset(ds, entry["requires"], suffix)
    if prepared is None:
        raise KeyError(
            f"Indicator '{name}' requires {entry['requires']} with suffix "
            f"'{suffix}', but dataset contains: {list(ds.data_vars)}"
        )

    indicator = entry["fn"](prepared)
    df = indicator.to_dataframe(name="value")
    df["experiment_id"] = (
        ds["experiment_id"].values if "experiment_id" in ds.coords else experiment_id
    )
    df["indicator"] = name
    df["model"] = ds.attrs.get("source_id", model)
    return df


def compute_all(
    ds: xr.Dataset,
    suffix: str = "_bc",
    model: str = "",
    experiment_id: str = "historical",
    names: list[str] | None = None,
) -> list[pd.DataFrame]:
    """Compute all applicable indicators for a dataset.

    Args:
        ds: Input dataset.
        suffix: Variable suffix to use (``"_bc"`` or ``""``).
        model: Model identifier.
        experiment_id: Experiment identifier.
        names: If given, only compute these indicators.  Otherwise compute
            all whose required variables are present.

    Returns:
        List of DataFrames, one per computed indicator.
    """
    results = []
    targets = names if names is not None else available()
    for name in targets:
        reqs = _REGISTRY[name]["requires"]
        suffixed_reqs = [f"{base}{suffix}" for base in reqs]
        if all(v in ds.data_vars for v in suffixed_reqs):
            results.append(
                compute(ds, name, suffix=suffix, model=model, experiment_id=experiment_id)
            )
    return results


# ---------------------------------------------------------------------------
# Built-in indicators  (all use canonical names: tasmin, tasmax, tas, pr)
# ---------------------------------------------------------------------------

@register("biologically_effective_degree_days", requires=["tasmin", "tasmax"])
def _bedd(ds: xr.Dataset) -> xr.DataArray:
    return xclim.indices.biologically_effective_degree_days(
        tasmin=ds["tasmin"], tasmax=ds["tasmax"],
    )


@register("corn_heat_units", requires=["tasmin", "tasmax"])
def _corn_heat(ds: xr.Dataset) -> xr.DataArray:
    return xclim.indices.corn_heat_units(
        tasmin=ds["tasmin"], tasmax=ds["tasmax"],
    ).resample(time="YS").sum()


@register("growing_season_length", requires=["tas"])
def _gsl(ds: xr.Dataset) -> xr.DataArray:
    return xclim.indices.growing_season_length(tas=ds["tas"])


@register("frost_days", requires=["tasmin"])
def _frost(ds: xr.Dataset) -> xr.DataArray:
    return xclim.indices.frost_days(tasmin=ds["tasmin"])


@register("cold_spell_duration_index", requires=["tasmin"])
def _csdi(ds: xr.Dataset) -> xr.DataArray:
    tn10 = xclim.core.calendar.percentile_doy(
        ds["tasmin"], per=10,
    ).sel(percentiles=10)
    return xclim.indices.cold_spell_duration_index(
        tasmin=ds["tasmin"], tasmin_per=tn10,
    )


@register("maximum_consecutive_dry_days", requires=["pr"])
def _cdd(ds: xr.Dataset) -> xr.DataArray:
    return xclim.indices.maximum_consecutive_dry_days(pr=ds["pr"])


@register("cool_night_index", requires=["tasmin"])
def _cni(ds: xr.Dataset) -> xr.DataArray:
    return xclim.indices.cool_night_index(tasmin=ds["tasmin"])


@register("cooling_degree_days", requires=["tas"])
def _cdd_temp(ds: xr.Dataset) -> xr.DataArray:
    return xclim.indices.cooling_degree_days(tas=ds["tas"])


@register("cold_and_dry_days", requires=["tas", "pr"])
def _cold_dry(ds: xr.Dataset) -> xr.DataArray:
    tas_per = xclim.core.calendar.percentile_doy(
        ds["tas"], window=5, per=25,
    ).sel(percentiles=25)
    pr_ref_wet = ds["pr"].where(ds["pr"] >= 0.0001)
    pr_per = xclim.core.calendar.percentile_doy(
        pr_ref_wet, window=5, per=25,
    ).sel(percentiles=25)
    return xclim.indices._multivariate.cold_and_dry_days(
        tas=ds["tas"], pr=ds["pr"], tas_per=tas_per, pr_per=pr_per,
    )


@register("max_1day_precipitation_amount", requires=["pr"])
def _rx1day(ds: xr.Dataset) -> xr.DataArray:
    return xclim.indices.max_1day_precipitation_amount(pr=ds["pr"])


@register("dry_days", requires=["pr"])
def _dry_days(ds: xr.Dataset) -> xr.DataArray:
    return xclim.indices.dry_days(pr=ds["pr"])
