import cdsapi
import pandas as pd
import numpy as np
import os
from collections import defaultdict

# ------------------------------
# Step 1: Generate random days from selected time period
# ------------------------------

# Create full date range from 2015-01-01 to 2024-12-31
full_date_range = pd.date_range(start="2015-01-01", end="2024-12-31", freq="D")

# Randomly sample 200 unique dates
sampled_dates = np.random.default_rng(seed=42).choice(full_date_range, size=200, replace=False)
sampled_dates = pd.to_datetime(sampled_dates)
sampled_dates = pd.Series(sampled_dates).sort_values()

# Group dates by year → month → list of days
dates_by_year_month = defaultdict(lambda: defaultdict(list))
for date in sampled_dates:
    year = date.year
    month = f"{date.month:02d}"
    day = f"{date.day:02d}"
    dates_by_year_month[year][month].append(day)

# ------------------------------
# Step 2: ERA5 Data Download Loop (by year/month)
# ------------------------------

client = cdsapi.Client()
dataset = "reanalysis-era5-pressure-levels"
output_dir = "cds_downloads_test_pl"
os.makedirs(output_dir, exist_ok=True)

for year in sorted(dates_by_year_month.keys()):
    for month in sorted(dates_by_year_month[year].keys()):
        days = sorted(dates_by_year_month[year][month])

        request = {
            "product_type": "reanalysis",
            "variable": [
                "relative_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "vorticity"
            ],
            "year": [str(year)],
            "month": [month],
            "day": days,
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "pressure_level": ["1000", "800", "600", "400"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [60, 100, 0, 180]
        }

        filename = os.path.join(output_dir, f"events_{year}_{month}.nc")
        client.retrieve(dataset, request).download(filename)

print(f"Downloaded data for {len(sampled_dates)} random days from 2015.")

# ------------------------------
# Step 2: ERA5 Data Download Loop (by year/month)
# ------------------------------

client = cdsapi.Client()
dataset = "reanalysis-era5-single-levels"
output_dir = "cds_downloads_test_sl"
os.makedirs(output_dir, exist_ok=True)

for year in sorted(dates_by_year_month.keys()):
    for month in sorted(dates_by_year_month[year].keys()):
        days = sorted(dates_by_year_month[year][month])

        request = {
            "product_type": "reanalysis",
            "variable": [
                "sea_surface_temperature",
                "2m_temperature"
            ],
            "year": [str(year)],
            "month": [month],
            "day": days,
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "pressure_level": ["1000", "800", "600", "400"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [60, 100, 0, 180]
        }

        filename = os.path.join(output_dir, f"negative_events_sl_{year}_{month}.nc")
        client.retrieve(dataset, request).download(filename)

print(f"Downloaded data for {len(sampled_dates)} random days from 2015. SL")

import xarray as xr
from pathlib import Path

import xarray as xr
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIG
# ─────────────────────────────────────────────────────────────────────────────
yearly_dir = Path("cds_downloads_test_pl")  # folder with e.g. 1990_combined.nc, 1991_combined.nc, ...
final_path = Path("2015_2024_test_pl.nc")

# ─────────────────────────────────────────────────────────────────────────────
# Gather all yearly .nc files and combine
# ─────────────────────────────────────────────────────────────────────────────
yearly_files = sorted(yearly_dir.glob("*.nc"))

print(f" Found {len(yearly_files)} yearly files to combine.")

ds_all = xr.open_mfdataset(
    [str(p) for p in yearly_files],
    engine="h5netcdf",       # same as used for yearly combining
    combine="by_coords",
    parallel=True,
    coords="minimal",
    data_vars="all",
    chunks={}
)

# Save final dataset
ds_all.to_netcdf(final_path)
ds_all.close()
print(f"✅ Wrote combined dataset to: {final_path}")

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIG
# ─────────────────────────────────────────────────────────────────────────────
yearly_dir = Path("cds_downloads_test_sl")  # folder with e.g. 1990_combined.nc, 1991_combined.nc, ...
final_path = Path("2015_2024_test_sl.nc")

# ─────────────────────────────────────────────────────────────────────────────
# Gather all yearly .nc files and combine
# ─────────────────────────────────────────────────────────────────────────────
yearly_files = sorted(yearly_dir.glob("*.nc"))

print(f" Found {len(yearly_files)} yearly files to combine.")

ds_all = xr.open_mfdataset(
    [str(p) for p in yearly_files],
    engine="h5netcdf",       # same as used for yearly combining
    combine="by_coords",
    parallel=True,
    coords="minimal",
    data_vars="all",
    chunks={}
)

# Save final dataset
ds_all.to_netcdf(final_path)
ds_all.close()
print(f"✅ Wrote combined dataset to: {final_path}")


