#!/usr/bin/env python
# coding: utf-8

# In[12]:


# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xclim
import pandas as pd
from ibicus.debias import LinearScaling, ISIMIP
from pathlib import Path
import os, cdsapi, tempfile, zipfile
from zarr.experimental.cache_store import CacheStore
from zarr.storage import FsspecStore, LocalStore


# In[3]:


CACHE_DIR = Path.home() / ".climate_cache"
CACHE_DIR.mkdir(exist_ok=True)


# In[4]:


token = "edh_pat_28142c8f3482fac46fabc58594bca5ace512a01ddceeca0799aaf19c264f80601547ea019347ac9b0b4c2f10dcb6da77"


# In[5]:


# List of CMIP6 models from https://earthdatahub.destine.eu/collections/cmip6
urls = [
    [
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/CMCC-CM2-SR5-historical-r1i1p1f1-day-gn-v0.zarr",
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/CMCC-CM2-SR5-ScenarioMIP-r1i1p1f1-day-gn-v0.zarr"
    ],
    [
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/DKRZ-MPI-ESM1-2-HR-historical-r1i1p1f1-day-gn-v0.zarr",
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/DKRZ-MPI-ESM1-2-HR-ScenarioMIP-r1i1p1f1-day-gn-v0.zarr"
    ],
    [
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/EC-Earth3-CC-historical-r1i1p1f1-day-gr-v0.zarr",
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/EC-Earth3-CC-ScenarioMIP-r1i1p1f1-day-gr-v0.zarr",
    ],
    [
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/IPSL-CM6A-LR-historical-r1i1p1f1-day-gr-v0.zarr",
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/IPSL-CM6A-LR-ScenarioMIP-r1i1p1f1-day-gr-v0.zarr"
    ],
    [
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/NCAR-CESM2-historical-r1i1p1f1-day-gn-v0.zarr",
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/NCAR-CESM2-ScenarioMIP-r10i1p1f1-day-gn-v0.zarr"
    ],
    [
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/NCC-NorESM2-MM-historical-r1i1p1f1-day-gn-v0.zarr",
        f"https://edh:{token}@data.earthdatahub.destine.eu/cmip6/NCC-NorESM2-MM-ScenarioMIP-r1i1p1f1-day-gn-v0.zarr"
    ]
]


# In[6]:


# List of locations
# Longitude 0-360 (needed by EDH)
# SIDNEY
sel_lat = 33.865143
sel_lon = 151.209900
# MINNESOTA
sel_lat = 45.36
sel_lon = 360-96.59
# PORTO
sel_lat = 44.1427
sel_lon = 360-5 

vars_to_select = ['pr', 'tas', 'tasmin', 'tasmax']


# In[7]:


# Access cached zarr data
def get_cached_access(url:str)->xr.Dataset:
    http_store = FsspecStore.from_url(
        url,
        storage_options={"client_kwargs": {"trust_env": True}},
        read_only=True,
    )    
    local_store = LocalStore(os.path.join(CACHE_DIR, 'edh', os.path.basename(url)))
    cache_store = CacheStore(
        store=http_store,
        cache_store=local_store,
        max_size=None,
    )
    return xr.open_dataset(cache_store, engine="zarr")


# In[8]:


# Read CMIP6 data
list_hist = []
list_fut  = []
for model_urls in urls:
    chist = get_cached_access(model_urls[0])
    cfut  = get_cached_access(model_urls[1])


    selected = [v for v in vars_to_select if v in chist.data_vars]
    if 'ssp370' in cfut['experiment_id']:
        ch_t = chist.sel(
            time = slice('1985', '2014')
        ).sel(
            lat = sel_lat,
            lon = sel_lon,
            method = 'nearest'
        )[selected].compute()
        list_hist.append(ch_t)   
        cf_t = cfut.sel(
            time = slice('2035', '2064'),
            experiment_id = 'ssp370'
        ).sel(
            lat = sel_lat,
            lon = sel_lon,
            method = 'nearest'
        )[selected].compute()
        list_fut.append(cf_t)


# In[93]:


# Read ERA5 data from CDS 
era5_cached_filename = f"era5-1980_2014-{sel_lat}-{sel_lon}.daily_4_vars.nc"
cached_full_path = os.path.join(CACHE_DIR, 'derived', era5_cached_filename)
if os.path.exists(cached_full_path):
    e5_t = xr.open_dataset(cached_full_path)
else:
    dataset = "reanalysis-era5-single-levels-timeseries"
    request = {
        "variable": [
            "2m_temperature",
            "total_precipitation"
        ],
        "location": {"longitude": sel_lon if sel_lon < 180 else sel_lon - 360, 
                     "latitude": sel_lat},
        "date": ["1980-01-01/2026-04-06"],
        "data_format": "netcdf"
    }

    client = cdsapi.Client()
    tmp_filename = tempfile.NamedTemporaryFile(suffix=".zip", delete=False).name
    client.retrieve(dataset, request, tmp_filename)
    # 1. Open the temporary directory context
    with tempfile.TemporaryDirectory() as tmpdir:

        # 2. Extract the file
        with zipfile.ZipFile(tmp_filename, 'r') as z:
            nc_filename = [f for f in z.namelist() if f.endswith('.nc')][0]
            extracted_path = z.extract(nc_filename, path=tmpdir)

        # 3. Open the dataset using a context manager to FORCE it to close
        with xr.open_dataset(extracted_path, engine="netcdf4") as e5:
            # Load data into RAM so it survives after the file closes
            e5.load()

    e5_t = e5.sel(
        valid_time = slice('1980', '2014')
    )

    e5_t = xr.merge([
        e5_t["tp"].resample(valid_time = 'D').sum().rename("tp_sum"),
        e5_t["t2m"].resample(valid_time = 'D').max().rename("tasmax"),
        e5_t["t2m"].resample(valid_time = 'D').min().rename("tasmin"),
        e5_t["t2m"].resample(valid_time = 'D').mean().rename("tas")
    ])
    e5_t.to_netcdf(cached_full_path)


# ## Debiasing temperature

# In[16]:


tas_debiaser_ISIMIP = ISIMIP.from_variable(variable = 'tas')


# In[17]:


for i, _ in enumerate(list_hist):
    for var in ['tas', 'tasmin', 'tasmax']:
        if var in list_hist[i].data_vars:
            hist_db = tas_debiaser_ISIMIP.apply(
                e5_t[var].values[:, np.newaxis, np.newaxis], 
                list_hist[i][var].values[:, np.newaxis, np.newaxis], 
                list_hist[i][var].values[:, np.newaxis, np.newaxis])

            list_hist[i][f'{var}_bc'] = xr.DataArray(
                        hist_db.squeeze(),
                dims=list_hist[i].dims,
                coords=list_hist[i].coords,
                attrs = list_hist[i][var].attrs
            )

            fut_db = tas_debiaser_ISIMIP.apply(
                e5_t[var].values[:, np.newaxis, np.newaxis], 
                list_hist[i][var].values[:, np.newaxis, np.newaxis], 
                list_fut[i][var].values[:, np.newaxis, np.newaxis])

            list_fut[i][f'{var}_bc'] = xr.DataArray(
                fut_db.squeeze(),
                dims=list_fut[i].dims,
                coords=list_fut[i].coords,
                attrs = list_fut[i][var].attrs
            )  


# ## Debiasing precipitation

# In[97]:


pr_debiaser_ISIMIP = ISIMIP.from_variable(variable = 'pr')


# In[100]:


# Converting ERA5 to the same precipitation units of CMIP6
e5_t['tp_sum'] /= 86.4
e5_t['tp_sum'].attrs['units'] =  'kg m-2 s-1' # 'm/day'


# In[103]:


for i, _ in enumerate(list_hist):
    var = 'pr'
    hist_db = pr_debiaser_ISIMIP.apply(
        e5_t['tp_sum'].values[:, np.newaxis, np.newaxis], 
        list_hist[i][var].values[:, np.newaxis, np.newaxis] , 
        list_hist[i][var].values[:, np.newaxis, np.newaxis] )

    list_hist[i][f'{var}_bc'] = xr.DataArray(
                hist_db.squeeze(),
        dims=list_hist[i].dims,
        coords=list_hist[i].coords,
        attrs = list_hist[i][var].attrs
    )

    fut_db = pr_debiaser_ISIMIP.apply(
        e5_t['tp_sum'].values[:, np.newaxis, np.newaxis], 
        list_hist[i][var].values[:, np.newaxis, np.newaxis], 
        list_fut[i][var].values[:, np.newaxis, np.newaxis])

    list_fut[i][f'{var}_bc'] = xr.DataArray(
        fut_db.squeeze(),
        dims=list_fut[i].dims,
        coords=list_fut[i].coords,
        attrs = list_fut[i][var].attrs
    )  


# # Calculating indicators

# In[104]:


e5_t = e5_t.rename({'valid_time': 'time'})
e5_t= e5_t.rename({'longitude': 'lon', 'latitude': 'lat'})
e5_t= e5_t.rename({'tas': 'tas_bc', 'tasmin': 'tasmin_bc', 'tasmax': 'tasmax_bc'})
e5_t= e5_t.rename({'tp_sum': 'pr_bc'})


# In[106]:


def get_indicator(
    x:xr.Dataset, 
    name:str, 
    model:str = "",
    experiment_id:str = "historical"
    )->pd.DataFrame:
    if name == 'biologically_effective_degree_days':
        # xclim.indices.biologically_effective_degree_days(tasmin, tasmax, lat=None, thresh_tasmin='10 degC', method='gladstones', cap_value=1.0, low_dtr='10 degC', high_dtr='13 degC', max_daily_degree_days='9 degC', start_date='04-01', end_date='11-01', freq='YS')[source]
        indicator = xclim.indices.biologically_effective_degree_days(
            tasmin = x['tasmin_bc'],
            tasmax = x['tasmax_bc']
        )
    elif name == 'corn_heat_units':
        # xclim.indices.corn_heat_units(tasmin, tasmax, thresh_tasmin='4.44 degC', thresh_tasmax='10 degC')[source]
        indicator = xclim.indices.corn_heat_units(
            tasmin = x['tasmin_bc'],
            tasmax = x['tasmax_bc']
        ).resample(time = 'YS').sum()
    elif name == 'growing_season_length':
        # xclim.indices.growing_season_length(tas, thresh='5.0 degC', window=6, mid_date='07-01', freq='YS', op='>=')
        indicator = xclim.indices.growing_season_length(
            tas = x['tas_bc']
        )
    elif name == 'frost_days':
        # xclim.indices.frost_days(tasmin, thresh='0 degC', freq='YS')[source]

        indicator = xclim.indices.frost_days(
            tasmin = x['tasmin_bc']
        )
    elif name == 'cold_spell_duration_index':
        # xclim.indices.cold_spell_duration_index(tasmin, tasmin_per, window=6, freq='YS', resample_before_rl=True, bootstrap=False, op='<')
        tn10 = xclim.core.calendar.percentile_doy(x['tasmin_bc'], per=10).sel(percentiles=10)

        indicator = xclim.indices.cold_spell_duration_index(
                    tasmin = x['tasmin_bc'],
                    tasmin_per = tn10
                )
    elif name == 'maximum_consecutive_dry_days':
        # xclim.indices.maximum_consecutive_dry_days(pr, thresh='1 mm/day', freq='YS', resample_before_rl=True)[source]
        indicator = xclim.indices.maximum_consecutive_dry_days(
                    pr = x['pr_bc']
                )
    elif name == 'cool_night_index':
        # xclim.indices.cool_night_index(tasmin, lat=None, freq='YS')
        indicator = xclim.indices.cool_night_index(tasmin = x['tasmin_bc'])
    elif name == 'cooling_degree_days':
        # xclim.indices.cooling_degree_days(tas, thresh='18 degC', freq='YS')
        indicator = xclim.indices.cooling_degree_days(
                    tas = x['tas_bc']
        )
    elif name == 'cold_and_dry_days':
        # xclim.indices._multivariate.cold_and_dry_days(tas, pr, tas_per, pr_per, freq='YS')
        tas_per =  xclim.core.calendar.percentile_doy(x['tas_bc'], window=5, per=25).sel(percentiles=25)
        pr_ref_wet = x['pr_bc'].where(x['pr_bc'] >= 0.0001)
        pr_per =  xclim.core.calendar.percentile_doy(pr_ref_wet, window=5, per=25).sel(percentiles=25)

        indicator = xclim.indices._multivariate.cold_and_dry_days(
            tas = x['tas_bc'],
            pr  = x['pr_bc'],
            tas_per=tas_per,
            pr_per=pr_per)
    elif name == 'max_1day_precipitation_amount':
        # xclim.indices.max_1day_precipitation_amount(pr, freq='YS')[source]
        indicator = xclim.indices.max_1day_precipitation_amount(pr = x['pr_bc'])
    elif name == 'dry_days':
        # xclim.indices.dry_days(pr, thresh='0.2 mm/d', freq='YS', op='<')[source]
        indicator = xclim.indices.dry_days(pr = x['pr_bc'])

    # COMMON PART
    to_save = indicator.to_dataframe(name = 'value')
    to_save['experiment_id'] = x['experiment_id'].values if 'experiment_id' in x.coords else experiment_id
    to_save['indicator'] = name
    to_save['model'] = x.attrs['source_id'] if 'source_id' in x.attrs else model
    return to_save


# In[108]:


list_indicators = []
for x in list_hist+list_fut:
    if 'tas_bc' in x.data_vars:
        list_indicators.append(get_indicator(x,'growing_season_length' ))
        list_indicators.append(get_indicator(x,'cooling_degree_days' ))
    if 'pr_bc' in x.data_vars:
        list_indicators.append(get_indicator(x,'maximum_consecutive_dry_days' ))
        list_indicators.append(get_indicator(x,'max_1day_precipitation_amount' ))
        list_indicators.append(get_indicator(x,'dry_days' ))
    if 'tasmin_bc' in x.data_vars:
        list_indicators.append(get_indicator(x,'biologically_effective_degree_days' ))
        list_indicators.append(get_indicator(x,'corn_heat_units' ))
        list_indicators.append(get_indicator(x,'frost_days' ))
        list_indicators.append(get_indicator(x,'cold_spell_duration_index' ))
        list_indicators.append(get_indicator(x,'cool_night_index' ))
        list_indicators.append(get_indicator(x,'cold_and_dry_days' ))


# In[109]:


list_indicators.append(get_indicator(e5_t,'growing_season_length' , experiment_id = 'ERA5'))
list_indicators.append(get_indicator(e5_t,'biologically_effective_degree_days' , experiment_id = 'ERA5'))
list_indicators.append(get_indicator(e5_t,'corn_heat_units', experiment_id = 'ERA5' ))
list_indicators.append(get_indicator(e5_t,'frost_days', experiment_id = 'ERA5' ))
list_indicators.append(get_indicator(e5_t,'cold_spell_duration_index', experiment_id = 'ERA5' ))
list_indicators.append(get_indicator(e5_t,'maximum_consecutive_dry_days', experiment_id = 'ERA5' ))
list_indicators.append(get_indicator(e5_t,'cool_night_index', experiment_id = 'ERA5' ))
list_indicators.append(get_indicator(e5_t,'cold_and_dry_days', experiment_id = 'ERA5'  ))
list_indicators.append(get_indicator(e5_t,'cooling_degree_days', experiment_id = 'ERA5'  ))
list_indicators.append(get_indicator(e5_t,'max_1day_precipitation_amount', experiment_id = 'ERA5'  ))
list_indicators.append(get_indicator(e5_t,'dry_days', experiment_id = 'ERA5'  ))


# In[110]:


all_df = pd.concat(list_indicators)


# In[40]:


all_df.columns
sel = all_df.drop(columns = ['areacella', 'sftlf', 'orog',  'percentiles'])
# sel['label'] = sel['model'] + '_' + sel['experiment_id']
# sel = sel.drop(columns = ['model', 'experiment_id'])
avg = sel.groupby(['indicator', 'model', 'experiment_id'])['value'].median().reset_index().set_index(['indicator', 'experiment_id'])
# avg
baseline = avg.xs('ERA5', level = 'experiment_id').drop(columns = 'model')
# baseline
diff = (
    avg
        .set_index(['model'], append = True)
        .sub(baseline, level ='indicator')
        .div(baseline, level = 'indicator')
    .reset_index()
    .query('model != ""')
)

