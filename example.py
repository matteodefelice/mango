"""Example: run the mango workflow for Porto and produce visualisations.

Configuration is read from mango.yaml (see mango.yaml.example).
"""
# %%
from mango.data.edh import build_url
from mango.workflow import Workflow
from mango.output import DistributionPlot, IndicatorTable

# ── Model datasets ────────────────────────────────────────────────────────
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
urls = [[build_url(h), build_url(f)] for h, f in MODELS]

# ── Run workflow ──────────────────────────────────────────────────────────
wf = Workflow(lat=44.1427, lon=355.0, label="Porto", urls=urls)
wf.load()
wf.run()
# %%
# ── Distribution comparison plots ────────────────────────────────────────
DistributionPlot(wf).save("distributions.png")

# ── Summary table ─────────────────────────────────────────────────────────
IndicatorTable(wf).save("summary_table.png")
