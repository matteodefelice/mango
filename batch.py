"""Batch run: compute the mango workflow for 20 locations and export parquet files.

Each location is processed sequentially.  Results are written to
``output/<label>.parquet`` so they can be loaded later without re-running
the full pipeline.

Configuration is read from mango.yaml (see mango.yaml.example).
"""

import pathlib

from mango.data.edh import build_url
from mango.workflow import Workflow
from mango.output import IndicatorTable

# ── Model datasets (same ensemble as example.py) ──────────────────────────
MODELS = [
    ("CMCC-CM2-SR5-historical-r1i1p1f1-day-gn-v0.zarr",
     "CMCC-CM2-SR5-ScenarioMIP-r1i1p1f1-day-gn-v0.zarr"),
    ("DKRZ-MPI-ESM1-2-HR-historical-r1i1p1f1-day-gn-v0.zarr",
     "DKRZ-MPI-ESM1-2-HR-ScenarioMIP-r1i1p1f1-day-gn-v0.zarr"),
    ("EC-Earth3-CC-historical-r1i1p1f1-day-gr-v0.zarr",
     "EC-Earth3-CC-ScenarioMIP-r1i1p1f1-day-gr-v0.zarr"),
    ("IPSL-CM6A-LR-historical-r1i1p1f1-day-gr-v0.zarr",
     "IPSL-CM6A-LR-ScenarioMIP-r1i1p1f1-day-gr-v0.zarr"),
    ("NCAR-CESM2-historical-r1i1p1f1-day-gn-v0.zarr",
     "NCAR-CESM2-ScenarioMIP-r10i1p1f1-day-gn-v0.zarr"),
    ("NCC-NorESM2-MM-historical-r1i1p1f1-day-gn-v0.zarr",
     "NCC-NorESM2-MM-ScenarioMIP-r1i1p1f1-day-gn-v0.zarr"),
]

# ── Locations (lat, lon in 0-360 convention, label) ───────────────────────
# Covering a spread of climates and continents
LOCATIONS = [
    # Europe
    (48.85,  2.35 + 360,  "Paris"),
    (51.51, 359.88,       "London"),        # 360 - 0.12
    (41.90, 12.49,        "Rome"),
    (52.52, 13.41,        "Berlin"),
    (40.42, 356.53,       "Madrid"),        # 360 - 3.47
    # Americas
    (40.71, 286.00,       "New York"),      # 360 - 74
    (19.43, 260.91,       "Mexico City"),   # 360 - 99.09
    (-23.55, 313.45,      "Sao Paulo"),     # 360 - 46.55
    (-34.61, 301.35,      "Buenos Aires"),  # 360 - 58.65
    (45.42, 284.65,       "Toronto"),       # 360 - 75.35
    # Africa
    (30.06, 31.25,        "Cairo"),
    (-1.29, 36.82,        "Nairobi"),
    (6.37,  3.38,         "Lagos"),
    (-26.20, 28.04,       "Johannesburg"),
    # Asia & Oceania
    (35.69, 139.69,       "Tokyo"),
    (28.61, 77.21,        "New Delhi"),
    (31.23, 121.47,       "Shanghai"),
    (1.35,  103.82,       "Singapore"),
    (-33.87, 151.21,      "Sydney"),
    # Middle East
    (24.47, 54.37,        "Abu Dhabi"),
]

# ── Output directory ──────────────────────────────────────────────────────
OUT_DIR = pathlib.Path("output")
OUT_DIR.mkdir(exist_ok=True)

# ── Run ───────────────────────────────────────────────────────────────────
urls = [[build_url(h), build_url(f)] for h, f in MODELS]

for lat, lon, label in LOCATIONS:
    out_path = OUT_DIR / f"{label.lower().replace(' ', '_')}.parquet"
    if out_path.exists():
        print(f"[{label}] Skipping — {out_path} already exists.")
        continue

    wf = Workflow(lat=lat, lon=lon, label=label, urls=urls)
    wf.run()
    wf.to_parquet(str(out_path))
    table_path = OUT_DIR / f"{label.lower().replace(' ', '_')}_table.png"
    IndicatorTable(wf).save(str(table_path))
    print(f"[{label}] Saved to {out_path}")
