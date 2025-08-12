import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import dask.array as darray
# from imblearn.over_sampling import SMOTE  # not used in full-domain setup

# -----------------------------------------------------------------------------
# STEP 1: Open ERA5 pressure‐level file with sensible Dask chunks
# -----------------------------------------------------------------------------
ds = xr.open_dataset(
    "1980_2000_combined_pl.nc",
    engine="h5netcdf",
    chunks="auto",
)

# -----------------------------------------------------------------------------
# STEP 3: Stack five vars + all pressure levels into one “channel” axis
# -----------------------------------------------------------------------------
vars5 = ds[["vo", "r", "u", "v", "t"]]
da = vars5.to_array(dim="var")  # (var, time, lat, lon, level)
da = da.transpose("valid_time", "latitude", "longitude", "var", "pressure_level")
da = da.stack(channel=("var", "pressure_level"))  # (time, lat, lon, channel)

# -----------------------------------------------------------------------------
# STEP 4: Load & process single-level fields to append as channels
# -----------------------------------------------------------------------------
mask_ds = xr.open_dataset(
    "1980_2000_combined_sl.nc",
    engine="h5netcdf",
    chunks="auto",
)

# 4a) Sea‐land mask (1 over ocean, 0 over land)
sea_mask = (~xr.ufuncs.isnan(mask_ds.sst.isel(valid_time=0))).astype("int8")
sea_mask = (
    sea_mask
    .expand_dims(valid_time=da.valid_time, axis=0)
    .expand_dims(channel=[-2], axis=-1)
    .transpose("valid_time", "latitude", "longitude", "channel")
)

# 4b) SST with NaNs filled by T2M
sst = mask_ds.sst.fillna(mask_ds.t2m)
sst = (
    sst
    .expand_dims(channel=[-1], axis=-1)
    .transpose("valid_time", "latitude", "longitude", "channel")
)

# -----------------------------------------------------------------------------
# STEP 5: Re‐chunk everything to uniform blocks (including channel!)
# -----------------------------------------------------------------------------
t_chunks, y_chunks, x_chunks, _ = da.chunks
n_channel = da.sizes["channel"]

chunk_dict = {
    "valid_time": t_chunks,
    "latitude":   y_chunks,
    "longitude":  x_chunks,
    "channel":    n_channel,  # put entire channel axis in one chunk
}

da       = da.astype("float32").chunk(chunk_dict)
sea_mask = sea_mask.astype("float32").chunk(chunk_dict)
sst      = sst.astype("float32").chunk(chunk_dict)

# fix the channel coordinates so concat works cleanly
da       = da.assign_coords(channel=        da.channel.values)
sea_mask = sea_mask.assign_coords(channel=np.array([-2], dtype=np.int64))
sst      = sst.assign_coords(channel=     np.array([-1], dtype=np.int64))

# now concatenate along channel
da = xr.concat([da, sea_mask, sst], dim="channel", coords="minimal", join="exact", compat="override")

# -----------------------------------------------------------------------------
# STEP 6: Rename dims and carry over time coordinate
# -----------------------------------------------------------------------------
da = da.assign_coords(channel=np.arange(da.sizes["channel"]))
da = da.rename({
    "valid_time": "time",
    "latitude":   "lat",
    "longitude":  "lon",
})
times = da.time.values      # lazy until slice/compute
lats  = da.lat.values
lons  = da.lon.values

n_time    = da.sizes["time"]
lat_size  = da.sizes["lat"]
lon_size  = da.sizes["lon"]
n_channel = da.sizes["channel"]

# Also rename lat/lon dims to height/width for U-Net convention (keep order)
inp_da = da.rename({"lat": "height", "lon": "width"})  # dims: (time, height, width, channel)

# -----------------------------------------------------------------------------
# STEP 7: Load & filter IBTrACS data (find cyclogenesis candidates)
# -----------------------------------------------------------------------------
df = pd.read_csv(
    "ibtracs.WP.list.v04r01.csv",
    usecols=["SID", "ISO_TIME", "LAT", "LON", "TOKYO_GRADE"],
)
df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
df = df.dropna(subset=["ISO_TIME"]).sort_values(["SID", "ISO_TIME"])

def filter_cyclogenesis(df):
    events = []
    for sid, grp in df.groupby("SID"):
        grp = (
            grp.assign(TOKYO_GRADE=grp.TOKYO_GRADE.astype(str).str.strip())
               .query("TOKYO_GRADE.str.isnumeric()", engine="python")
               .assign(TOKYO_GRADE=lambda x: x.TOKYO_GRADE.astype(int))
        )
        if 5 not in grp.TOKYO_GRADE.values:
            continue
        if not grp.TOKYO_GRADE.isin([2, 3, 4]).any():
            continue

        # only consider up to first grade 5
        first5 = grp[grp.TOKYO_GRADE == 5].index[0]
        dev = grp.loc[:first5].copy()

        # compute diffs, but fill the first NaN with 0
        dev["Grade_Diff"] = dev["TOKYO_GRADE"].diff().fillna(0)

        # now NaN never appears, so initial row with grade in [2,3,4] will pass
        valid = dev.query("(TOKYO_GRADE in [2,3,4]) & (Grade_Diff>=0)", engine="python")

        if not valid.empty:
            events.append(valid)

    return pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=df.columns)

df_cyclogen = filter_cyclogenesis(df)
print("Cyclogenesis events:", len(df_cyclogen))

# -----------------------------------------------------------------------------
# STEP 8: Build per-pixel labels (time, height, width) for the whole domain
# -----------------------------------------------------------------------------
labels_np = np.zeros((n_time, lat_size, lon_size), dtype=np.uint8)
tol = pd.Timedelta("3h")

# Optional: label a small blob around the mapped genesis pixel (radius=1 → 3x3)
radius = 1

for _, row in df_cyclogen.iterrows():
    t = np.datetime64(row.ISO_TIME)
    # nearest time index
    td = np.abs(times - t)
    ti = int(td.argmin())
    if td[ti] > tol:
        continue
    # nearest spatial indices on full grid (no subgrids)
    yi = int(np.abs(lats - row.LAT).argmin())
    xi = int(np.abs(lons - row.LON).argmin())

    y0 = max(0, yi - radius); y1 = min(lat_size, yi + radius + 1)
    x0 = max(0, xi - radius); x1 = min(lon_size, xi + radius + 1)
    labels_np[ti, y0:y1, x0:x1] = 1

print("Per-pixel cyclogenesis labels built:", labels_np.shape)

# -----------------------------------------------------------------------------
# STEP 9: Assemble whole-domain Dataset and split train/val by time
# -----------------------------------------------------------------------------
# Dask-ify labels with chunks to match inputs
t_chunks, y_chunks, x_chunks, _ = inp_da.chunks
labels_da = darray.from_array(labels_np, chunks=(t_chunks, y_chunks, x_chunks))

# Match dims: inputs -> (time, height, width, channel), labels -> (time, height, width)
full_ds = xr.Dataset(
    {
        "inputs": (("time", "height", "width", "channel"), inp_da.data),
        "labels": (("time", "height", "width"), labels_da),
    },
    coords={"time": inp_da.time}
)

# Train/Val split (90/10) along time
N = full_ds.dims["time"]
rng = np.random.default_rng(42)
perm = rng.permutation(N)
n_tr = int(0.9 * N)
train_idx = perm[:n_tr]
val_idx   = perm[n_tr:]

train_ds = full_ds.isel(time=train_idx)
val_ds   = full_ds.isel(time=val_idx)

print("Train/Val time sizes:", train_ds.dims["time"], val_ds.dims["time"])

# -----------------------------------------------------------------------------
# STEP 10: Normalize inputs per-channel (fit on train), save to Zarr
# -----------------------------------------------------------------------------
def _normalize_block(block, channel_mins, channel_maxs):
    scale = channel_maxs - channel_mins
    scale[scale == 0] = 1.0
    return ((block - channel_mins) / scale).astype(np.float32)

print("Preparing normalization (fit stats from TRAIN only)...")
train_inputs_da = train_ds["inputs"].data
channel_mins = train_inputs_da.min(axis=(0, 1, 2)).compute()
channel_maxs = train_inputs_da.max(axis=(0, 1, 2)).compute()

print("Normalizing train/val inputs lazily with Dask...")
train_inputs_norm = train_inputs_da.map_blocks(
    _normalize_block, channel_mins, channel_maxs, dtype=np.float32
)
val_inputs_norm = val_ds["inputs"].data.map_blocks(
    _normalize_block, channel_mins, channel_maxs, dtype=np.float32
)

# Ensure each sample (time slice) is one chunk for convenient loading
train_inputs_norm = train_inputs_norm.rechunk({0: 1})
val_inputs_norm   = val_inputs_norm.rechunk({0: 1})
train_labels_da   = train_ds["labels"].data.rechunk({0: 1})
val_labels_da     = val_ds["labels"].data.rechunk({0: 1})

# Build final datasets
train_out = xr.Dataset(
    {
        "inputs": (("time", "height", "width", "channel"), train_inputs_norm),
        "labels": (("time", "height", "width"), train_labels_da),
    },
    coords={"time": train_ds["time"]}
)

val_out = xr.Dataset(
    {
        "inputs": (("time", "height", "width", "channel"), val_inputs_norm),
        "labels": (("time", "height", "width"), val_labels_da),
    },
    coords={"time": val_ds["time"]}
)

print("Saving normalized datasets to Zarr...")
train_out.to_zarr("train_data_normalised.zarr", mode="w", consolidated=True)
val_out.to_zarr("val_data_normalised.zarr",   mode="w", consolidated=True)
print("Done. Files written: train_data_normalised.zarr, val_data_normalised.zarr")