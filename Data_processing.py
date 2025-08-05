import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import dask.array as darray
from imblearn.over_sampling import SMOTE

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
da = vars5.to_array(dim="var")  
da = da.transpose("valid_time", "latitude", "longitude", "var", "pressure_level")
da = da.stack(channel=("var", "pressure_level"))

# -----------------------------------------------------------------------------
# STEP 4: Load & downsample your single‐level file the same way
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
    .expand_dims(channel=[-2],         axis=-1)
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
# grab the existing chunk‐structure for the first three dims
t_chunks, y_chunks, x_chunks, _ = da.chunks
n_channel = da.sizes["channel"]

chunk_dict = {
    "valid_time": t_chunks,
    "latitude":   y_chunks,
    "longitude":  x_chunks,
    "channel":    n_channel,        # put entire channel axis in one chunk
}

# cast & chunk
da       = da.astype("float32").chunk(chunk_dict)
sea_mask = sea_mask.astype("float32").chunk(chunk_dict)
sst      = sst.astype("float32").chunk(chunk_dict)

# fix the channel coordinates so concat works cleanly
da       = da.assign_coords(channel=        da.channel.values)
sea_mask = sea_mask.assign_coords(channel=np.array([-2],dtype=np.int64))
sst      = sst.assign_coords(channel=     np.array([-1],dtype=np.int64))

# now concatenate along channel
da = xr.concat(
    [da, sea_mask, sst],
    dim="channel",
    coords="minimal",
    join="exact",
    compat="override",
)

# -----------------------------------------------------------------------------
# STEP 6: Rename dims and carry over time coordinate
# -----------------------------------------------------------------------------
da = da.assign_coords(channel=np.arange(da.sizes["channel"]))


# 1) rename dims to match what expect downstream
da = da.rename({
    "valid_time": "time",
    "latitude":   "lat",
    "longitude":  "lon",
})
times = da.time.values      # still lazy until slice/compute
lats  = da.lat.values
lons  = da.lon.values

n_time    = da.sizes["time"]
lat_size  = da.sizes["lat"]
lon_size  = da.sizes["lon"]
n_channel = da.sizes["channel"]

# -----------------------------------------------------------------------------
# 2) grid geometry (3×3 with 50% overlap)
# -----------------------------------------------------------------------------
grid_rows, grid_cols, overlap = 3, 3, 0.5
cell_lat = lat_size // grid_rows
cell_lon = lon_size // grid_cols
stride_lat = int(cell_lat * (1 - overlap))
stride_lon = int(cell_lon * (1 - overlap))

lat_starts = [min(i*stride_lat, lat_size - cell_lat) for i in range(grid_rows + 2)]
lon_starts = [min(j*stride_lon, lon_size - cell_lon) for j in range(grid_cols + 2)]

# -----------------------------------------------------------------------------
# 3) load & filter IBTrACS data 
# -----------------------------------------------------------------------------
df = pd.read_csv("ibtracs.WP.list.v04r01.csv",
                 usecols=['SID','ISO_TIME','LAT','LON','TOKYO_GRADE'])
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
df = df.dropna(subset=['ISO_TIME']).sort_values(['SID','ISO_TIME'])


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
        if not grp.TOKYO_GRADE.isin([2,3,4]).any():
            continue

        # only consider up to first grade 5
        first5 = grp[grp.TOKYO_GRADE==5].index[0]
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
# 4) build a labels array (NumPy) of shape (time, n_ycells, n_xcells)
# -----------------------------------------------------------------------------
labels_np = np.zeros((n_time, len(lat_starts), len(lon_starts)), dtype=np.uint8)
tol = pd.Timedelta("3h")

for _, row in df_cyclogen.iterrows():
    t = np.datetime64(row.ISO_TIME)
    # nearest time index
    td = np.abs(times - t)
    ti = int(td.argmin())
    if td[ti] > tol:
        continue
    # spatial index
    yi = int(np.abs(lats - row.LAT).argmin())
    xi = int(np.abs(lons - row.LON).argmin())
    # which cell
    for i, y0 in enumerate(lat_starts):
        y1 = y0 + cell_lat
        if not (y0 <= yi < y1): 
            continue
        for j, x0 in enumerate(lon_starts):
            x1 = x0 + cell_lon
            if x0 <= xi < x1:
                labels_np[ti, i, j] = 1

# -----------------------------------------------------------------------------
# 5) slice `da` + `labels_np` into per‐cell Datasets (lazy!)
# -----------------------------------------------------------------------------
cell_datasets = {}
for i, y0 in enumerate(lat_starts):
    y1 = y0 + cell_lat
    latc = float(lats[y0:y1].mean())
    for j, x0 in enumerate(lon_starts):
        x1 = x0 + cell_lon
        lonc = float(lons[x0:x1].mean())

        cell_inp = da.isel(lat=slice(y0,y1), lon=slice(x0,x1))

        cell_inp = cell_inp.chunk({
            "time":    50,
            "lat":     80,
            "lon":     107,
            "channel": 22
        })

        cell_lab = xr.DataArray(
            labels_np[:, i, j],
            dims=["time"],
            coords={"time": da.time}
        )

        ds_cell = xr.Dataset(
            {
                "inputs": cell_inp,
                "labels": cell_lab
            },
            coords={
                "lat_center": latc,
                "lon_center": lonc,
            }
        )

        cell_datasets[(i,j)] = ds_cell

# now `cell_datasets[(i,j)]` is an xarray.Dataset for grid cell (i,j)
# with lazy `inputs` and a small in‐memory `labels`.

print("Prepared", len(cell_datasets), "grid‐cell datasets.")
# e.g. to peek at cell (1,2):
print(cell_datasets[(1,2)])


# =============================================================================
# STEP 1: Use the in-memory `cell_datasets` from previous blocks
# =============================================================================
print("Found grid cells:", list(cell_datasets.keys()))

# =============================================================================
# STEP 2: Stack each cell's inputs/labels lazily 
# =============================================================================
input_chunks = []
label_chunks = []
center_chunks = []

for ds in cell_datasets.values():
    # 1) grab lazy Dask arrays (no .values → no eager copy)
    inp = ds["inputs"].data       # shape: (T, H, W, C)
    lab = ds["labels"].data       # shape: (T,)

    # 2) get the scalar centre coords
    latc = float(ds.coords["lat_center"])
    lonc = float(ds.coords["lon_center"])

    # 3) build a small in-RAM (time,2) array of repeated centres
    #    then wrap it as a Dask array with the same time-axis chunks
    ctr = np.tile([latc, lonc], (inp.shape[0], 1))  
    center_chunks.append(
        darray.from_array(ctr, chunks=(inp.chunks[0], 2))
    )

    input_chunks.append(inp)
    label_chunks.append(lab)

# 4) lazily concatenate **all** time-slices from every cell
all_inputs  = darray.concatenate(input_chunks, axis=0)  # (N, H, W, C)
all_labels  = darray.concatenate(label_chunks, axis=0)  # (N,)
all_centers = darray.concatenate(center_chunks, axis=0) # (N, 2)

print("Total samples:", all_labels.shape[0])

# Determine the maximum chunk size
max_chunk_size = 50

# Rechunk the dataset
all_inputs = all_inputs.rechunk({0: max_chunk_size})
all_centers = all_centers.rechunk({0: max_chunk_size})
all_labels = all_labels.rechunk({0: max_chunk_size})

print("Rechunking complete!")

# Save the datasets to Xarray
print("Saving datasets to Xarray...")
train_bs = xr.Dataset(
    {
        "inputs": (("time", "height", "width", "channel"), all_inputs),
        "centers": (("time", "coord"), all_centers),
        "labels": (("time",), all_labels),
    },
    coords={"time": np.arange(all_inputs.shape[0])}
)

# Save to disk with chunking for efficient lazy loading
train_bs.to_zarr('train_data.zarr', mode='w', consolidated=True)

print("Datasets saved successfully.")

# Load the datasets lazily
train_bs = xr.open_zarr('train_data.zarr', consolidated=True)

all_inputs = train_bs['inputs'].data
all_centers = train_bs['centers'].data
all_labels = train_bs['labels'].data


# =============================================================================
# STEP 3: Downsample negative class (10× more negatives than positives)
# =============================================================================
# pull just the label vector in RAM (small) to find indices
label_vec = all_labels.compute()
pos_idx = np.where(label_vec == 1)[0]
neg_idx = np.where(label_vec == 0)[0]
n_pos = len(pos_idx) 

np.random.seed(42)
neg_sel = np.random.choice(neg_idx, size=n_pos, replace=False)
bal_idx = np.concatenate([pos_idx, neg_sel])

# slice lazily
all_inputs  = all_inputs[bal_idx]
all_labels  = all_labels[bal_idx]
all_centers = all_centers[bal_idx]

print("Samples after class balancing:", all_labels.shape[0],",", len(pos_idx), "positives,", len(neg_sel), "negatives")

# =============================================================================
# STEP 4: Split into train/val (90/10)
# =============================================================================
N = all_labels.shape[0]
perm = np.random.permutation(N)
n_tr = int(0.9 * N)
n_va = int(0.1 * N)

train_idx = perm[:n_tr]
val_idx   = perm[n_tr:]

train_inputs, train_labels, train_centers = \
    all_inputs[train_idx], all_labels[train_idx], all_centers[train_idx]
val_inputs,   val_labels,   val_centers   = \
    all_inputs[val_idx],   all_labels[val_idx],   all_centers[val_idx]

print("Train/Val sizes:", train_labels.shape[0],
      val_labels.shape[0])

# # =============================================================================
# # STEP 4b: UPSAMPLE TRAINING SET WITH SMOTE (with dynamic k_neighbors)
# # =============================================================================
# # pull the training labels into memory to count the minority class
# y_tr = train_labels.compute()
# n_minority = int((y_tr == 1).sum())

# # choose k_neighbors = min(default=5, n_minority-1), but at least 1
# k = max(1, min(5, n_minority - 1))
# print(f"Using k_neighbors={k} because there are {n_minority} positive samples in train")

# # flatten images and append coords (this will pull train_inputs & train_centers into RAM)
# n_t, H, W, C = train_inputs.shape
# X_flat   = train_inputs.reshape((n_t, -1)).compute()
# centers  = train_centers.compute()
# X_feat   = np.hstack([X_flat, centers])
# y_feat   = y_tr

# # apply SMOTE with our dynamic k_neighbors
# sm = SMOTE(random_state=42, k_neighbors=k)
# X_res, y_res = sm.fit_resample(X_feat, y_feat)

# # rebuild Dask arrays from the SMOTE output
# train_inputs  = da.from_array(
#     X_res[:, : H * W * C].reshape(-1, H, W, C),
#     chunks=(None, H, W, C),
# )
# train_centers = da.from_array(
#     X_res[:, H * W * C :],
#     chunks=(None, 2),
# )
# train_labels  = da.from_array(y_res, chunks=(None,))

# print("After SMOTE, training samples:", train_labels.shape[0],
#       "(positives:", int((y_res == 1).sum()), 
#       "negatives:", int((y_res == 0).sum()), ")")

# # =============================================================================
# # STEP 5: Normalize Each Dataset Lazily Per Channel
# # =============================================================================

# # -----------------------------------------------------------------------------
# # 0) Top‐level block‐normalization function
# # -----------------------------------------------------------------------------
# def _normalize_block(block, channel_mins, channel_maxs):
#     """
#     Normalize one chunk of shape (..., C) by the precomputed per-channel mins/maxs.
#     """
#     # broadcasting: block[... , c] -> (block - channel_mins) / (channel_maxs - channel_mins)
#     # clamp division by zero if any channel_max == channel_min:
#     scale = channel_maxs - channel_mins
#     scale[scale == 0] = 1.0
#     return ((block - channel_mins) / scale).astype(np.float32)

# # -----------------------------------------------------------------------------
# # 1) normalize_dask_minmax no longer defines a nested function
# # -----------------------------------------------------------------------------
# def normalize_dask_minmax(x):
#     """
#     Perform a min–max normalization on a Dask array of shape (T, H, W, C),
#     returning values scaled to [0, 1] per channel.

#     This only pulls down a small C-length array of mins and maxs into memory,
#     then applies them lazily to each chunk.
#     """
#     # 1a) compute global per‐channel mins/maxs (C‐element vectors)
#     channel_mins = x.min(axis=(0, 1, 2)).compute()
#     channel_maxs = x.max(axis=(0, 1, 2)).compute()

#     # 1b) broadcast them to the right shape
#     #      (map_blocks will take care of aligning dims)
#     return x.map_blocks(
#         _normalize_block,
#         channel_mins,
#         channel_maxs,
#         dtype=np.float32
#     )

# print("no crash 1")
# train_inputs = normalize_dask_minmax(train_inputs).persist()
# val_inputs   = normalize_dask_minmax(val_inputs).persist()
# # test_inputs  = normalize_dask_minmax(test_inputs).persist()
# print("no crash 2")


# # 1) Rechunk so that each sample along the **time** axis is its own chunk:
# #    this ensures .compute() on inputs[i] only pulls one slice at a time.
train_inputs  = train_inputs.rechunk({0: 1})
train_centers = train_centers.rechunk({0: 1})
train_labels  = train_labels.rechunk({0: 1})

val_inputs    = val_inputs.rechunk({0: 1})
val_centers   = val_centers.rechunk({0: 1})
val_labels    = val_labels.rechunk({0: 1})





######### -----------------------------------------------------------------------------
import xarray as xr
import dask.array as da
import numpy as np

# =============================================================================
# STEP 5: Normalize Each Dataset Lazily Per Channel and Save to Xarray
# =============================================================================

def _normalize_block(block, channel_mins, channel_maxs):
    """
    Normalize one chunk of shape (..., C) by the precomputed per-channel mins/maxs.
    """
    scale = channel_maxs - channel_mins
    scale[scale == 0] = 1.0  # Avoid division by zero
    return ((block - channel_mins) / scale).astype(np.float32)

def normalize_dask_minmax(x):
    """
    Perform a min–max normalization on a Dask array of shape (T, H, W, C),
    returning values scaled to [0, 1] per channel.
    """
    channel_mins = x.min(axis=(0, 1, 2)).compute()
    channel_maxs = x.max(axis=(0, 1, 2)).compute()
    return x.map_blocks(
        _normalize_block,
        channel_mins,
        channel_maxs,
        dtype=np.float32
    )

# Normalize datasets
print("Normalizing datasets...")
train_inputs = normalize_dask_minmax(train_inputs)
val_inputs = normalize_dask_minmax(val_inputs)

# Save the datasets to Xarray
print("Saving datasets to Xarray...")
train_ds = xr.Dataset(
    {
        "inputs": (("time", "height", "width", "channel"), train_inputs),
        "centers": (("time", "coord"), train_centers),
        "labels": (("time",), train_labels),
    },
    coords={"time": np.arange(train_inputs.shape[0])}
)

val_ds = xr.Dataset(
    {
        "inputs": (("time", "height", "width", "channel"), val_inputs),
        "centers": (("time", "coord"), val_centers),
        "labels": (("time",), val_labels),
    },
    coords={"time": np.arange(val_inputs.shape[0])}
)

# Save to disk with chunking for efficient lazy loading
train_ds.to_zarr('train_data_normalised.zarr', mode='w', consolidated=True)
val_ds.to_zarr('val_data_normalised.zarr', mode='w', consolidated=True)

print("Datasets saved successfully.")

