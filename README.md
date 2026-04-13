# mango

A Python library for computing climate indicators from CMIP6 model ensembles,
with ERA5 reanalysis as observational reference and ISIMIP bias correction.

## Purpose

Given a location and a set of CMIP6 models, mango:

1. Fetches model data (historical + future scenario) from the Earth Data Hub (EDH)
2. Fetches ERA5 reanalysis data from the Copernicus Climate Data Store (CDS)
3. Applies ISIMIP bias correction to align model distributions to ERA5
4. Computes a set of agro-climatic indicators (growing season length, frost days, etc.)
5. Summarises results as a comparison table: ERA5 baseline, historical ensemble, future scenario

All remote data and intermediate results (ERA5 download, debiased datasets) are cached locally so repeated runs are fast.

## Project structure

```
mango/
├── config.py          # YAML-based configuration (token, cache dir)
├── data/
│   ├── edh.py         # CMIP6 data access from Earth Data Hub (zarr, cached)
│   └── cds.py         # ERA5 data access from CDS API (netCDF, cached)
├── debias.py          # ISIMIP bias correction + disk cache for debiased data
└── indicators.py      # Registry of climate indicators built on xclim
example.py             # End-to-end workflow script
mango.yaml.example     # Configuration template
```

## Configuration

Copy `mango.yaml.example` to `mango.yaml` and fill in your credentials:

```yaml
edh_token: "your_edh_personal_access_token"
cache_dir: ~/.climate_cache   # optional, this is the default
```

The library searches for the config file in this order:
1. Path passed to `config.load(path)`
2. `MANGO_CONFIG` environment variable
3. `./mango.yaml` (working directory)
4. `~/.config/mango/config.yaml`

## Usage

```python
from mango.data import load_cmip6_datasets_from_edh, load_era5
from mango.data.edh import build_url
from mango.debias import debias_with_cache
from mango import indicators

# Build URLs from dataset names (token read from mango.yaml)
urls = [[build_url("CMCC-CM2-SR5-historical-...zarr"),
         build_url("CMCC-CM2-SR5-ScenarioMIP-...zarr")]]

# Load data
list_hist, list_fut = load_cmip6_datasets_from_edh(
    urls, lat=44.14, lon=355.0, variables=["pr", "tas", "tasmin", "tasmax"]
)
obs = load_era5(lat=44.14, lon=355.0)

# Bias correction (reads/writes ~/.climate_cache/debiased/)
debias_with_cache(obs=obs, list_hist=list_hist, list_fut=list_fut)

# Compute all applicable indicators
import pandas as pd
all_dfs = []
for ds in list_hist + list_fut:
    all_dfs.extend(indicators.compute_all(ds, suffix="_bc"))
all_dfs.extend(indicators.compute_all(obs, suffix="", experiment_id="ERA5"))

results = pd.concat(all_dfs)
```

See `example.py` for the full workflow including the summary table.

## Extending indicators

Register a new indicator by decorating a function with `@register`:

```python
from mango.indicators import register
import xclim

@register("heat_wave_index", requires=["tasmax"])
def heat_wave_index(ds):
    return xclim.indices.heat_wave_index(tasmax=ds["tasmax"])
```

The `requires` list uses canonical variable names (without suffix). The
framework resolves the correct variables at compute time based on the `suffix`
parameter (`"_bc"` for bias-corrected, `""` for raw).

## Caching

| What | Where | Format |
|---|---|---|
| ERA5 daily data | `<cache_dir>/derived/` | netCDF |
| EDH zarr chunks | `<cache_dir>/edh/` | zarr (local) |
| Debiased model data | `<cache_dir>/debiased/` | netCDF |

Delete the relevant subdirectory to force a re-download or re-computation.

## Dependencies

- `xarray`, `numpy` — data handling
- `xclim` — climate index computation
- `ibicus` — ISIMIP bias correction
- `cdsapi` — ERA5 download
- `zarr`, `fsspec` — remote zarr access
- `great_tables`, `pandas` — results table
- `pyyaml` — configuration
