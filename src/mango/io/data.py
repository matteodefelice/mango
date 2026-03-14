# %%
import xarray as xr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import FsspecStore, LocalStore
import hashlib
import pickle
from pathlib import Path

CACHE_DIR = Path.home() / ".climate_cache"
CACHE_DIR.mkdir(exist_ok=True)
class data_cube_access:
    """
    A class to access and process climate data from different sources (EDH or Earthmover). 

    Attributes:
        method (str): The method to access data ('edh' or 'earthmover').
        token (str): The authentication token for EDH access.
        data (xr.Dataset): The raw dataset loaded from the source.
        data_daily (xr.Dataset): The processed daily statistics dataset.
    """

    def __init__(self, method:str = 'edh', token=None):
        self.method = method
        self.token = token
        self.status = "unselected"

        print("Initializing data cube access...")
        if self.method == 'edh':
            if self.token is None:
                raise ValueError("Token must be provided for EDH access.")
            else:
                self.edh_access()
        elif self.method == 'earthmover':
            self.earthmover_access()
        else:
            raise ValueError("Invalid method. Choose 'edh' or 'earthmover'.")

    def edh_access(self)-> None:
        # Load specific
        url = f"https://edh:{self.token}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"
        http_store = FsspecStore.from_url(
            url,
            storage_options={"client_kwargs": {"trust_env": True}},
            read_only=True,
        )    
        local_store = LocalStore(CACHE_DIR)
        cache_store = CacheStore(
            store=http_store,
            cache_store=local_store,
            max_size=None,
        )
        self.data = xr.open_dataset(cache_store, engine="zarr")

        # Rename
        self.data = self.data.rename({
            "valid_time": "time",
            "latitude": "lat",
            "longitude": "lon"
            })

    def earthmover_access(self)-> None:
        from arraylake import Client
        client = Client()
        client.login()
        repo = client.get_repo("exploratory-matteodefelice/era5")
        session = repo.readonly_session(branch="main")
        self.data = xr.open_zarr(session.store, zarr_format=3, group="temporal")

        # Rename
        self.data = self.data.rename({
            "latitude": "lat",
            "longitude": "lon",
            "t2": "t2m",
            "lsp": "tp"
            })

    def select_data(self, lat_range=(39, 35), lon_range=(11, 16), time_range=('2015', '2025')):
        self.sel = (
            self.data
            .sel(
                lat = slice(*lat_range),
                lon = slice(*lon_range),
                time = slice(*time_range)
            )
        )[["t2m", "tp"]]
        self.status = "newly selected"

    def calculate_daily_stats(self):
        self.tmean  = self.sel["t2m"].resample(time="1D").mean()
        self.tmin  = self.sel["t2m"].resample(time="1D").min()
        self.tmax  = self.sel["t2m"].resample(time="1D").max()
        self.precip_daily = self.sel["tp"].resample(time="1D").sum()

        self.data_daily = xr.Dataset({
            "t2m": self.tmean, 
            "t2min": self.tmin,
            "t2max": self.tmax,
            "tp": self.precip_daily
            })
        # Set units for precipitation to ensure compatibility with xclim
        self.data_daily["tp"].attrs["units"] = "m/day"

    def get_daily(self):
        if self.status == "unselected":
            raise ValueError("Data not selected. Please call select_data() first.")
        elif self.status == "newly selected":
            print("Computing...")

            self.sel = self.sel.compute()  # Compute the selected data to speed up subsequent calculations
            print("Daily statistics...")
            
            self.calculate_daily_stats()
            
        return self.data_daily        

