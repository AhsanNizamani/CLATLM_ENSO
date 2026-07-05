from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader

LAT_NAMES = ["lat", "latitude", "yt_ocean", "y", "nav_lat"]
LON_NAMES = ["lon", "longitude", "xt_ocean", "x", "nav_lon"]
TIME_NAMES = ["time", "month", "date"]


def cftime_to_timestamp(t) -> pd.Timestamp:
    try:
        return pd.Timestamp(t)
    except Exception:
        return pd.Timestamp(
            year=t.year,
            month=t.month,
            day=getattr(t, "day", 1),
            hour=getattr(t, "hour", 0),
            minute=getattr(t, "minute", 0),
            second=getattr(t, "second", 0),
        )


def to_datetime_index(times) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([cftime_to_timestamp(t) for t in times])


def find_coord_name(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name
    return None


def infer_data_var(ds: xr.Dataset, preferred: str | None = None) -> str:
    if preferred is not None and preferred in ds.data_vars:
        return preferred

    for v in ds.data_vars:
        da = ds[v]
        dims_lower = [d.lower() for d in da.dims]
        has_time = any("time" in d or d in ["month", "date"] for d in dims_lower)
        has_lat = any("lat" in d or d in ["y", "yt_ocean"] for d in dims_lower)
        has_lon = any("lon" in d or d in ["x", "xt_ocean"] for d in dims_lower)
        if has_time and has_lat and has_lon and np.issubdtype(da.dtype, np.number):
            return v

    for v in ds.data_vars:
        if np.issubdtype(ds[v].dtype, np.number):
            return v

    raise ValueError("Could not infer numeric SST/tos variable from dataset.")


def normalize_lon_0_360(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    ds = ds.assign_coords({lon_name: ds[lon_name] % 360})
    return ds.sortby(lon_name)


def standardize_dim_names(da: xr.DataArray, time_name: str, lat_name: str, lon_name: str) -> xr.DataArray:
    rename = {}
    if time_name != "time":
        rename[time_name] = "time"
    if lat_name != "lat":
        rename[lat_name] = "lat"
    if lon_name != "lon":
        rename[lon_name] = "lon"
    return da.rename(rename) if rename else da


def remove_extra_dims(da: xr.DataArray) -> xr.DataArray:
    for d in list(da.dims):
        if d not in ["time", "lat", "lon"]:
            da = da.isel({d: 0})
    return da


def subset_lon_lat(da: xr.DataArray, lat_bounds: Tuple[float, float], lon_bounds: Tuple[float, float]) -> xr.DataArray:
    lat_min, lat_max = sorted(lat_bounds)
    da = da.sortby("lat").sel(lat=slice(lat_min, lat_max))

    lon_min, lon_max = lon_bounds
    lon_min %= 360
    lon_max %= 360

    if lon_min <= lon_max:
        da = da.sel(lon=slice(lon_min, lon_max))
    else:
        da1 = da.sel(lon=slice(lon_min, 360))
        da2 = da.sel(lon=slice(0, lon_max))
        da = xr.concat([da1, da2], dim="lon")

    return da


def make_target_grid(lat_bounds: Tuple[float, float], lon_bounds: Tuple[float, float], res: float):
    lat_min, lat_max = sorted(lat_bounds)
    lats = np.arange(lat_min, lat_max + 0.001, res, dtype=np.float32)

    lon_min, lon_max = lon_bounds
    lon_min %= 360
    lon_max %= 360

    if lon_min <= lon_max:
        lons = np.arange(lon_min, lon_max + 0.001, res, dtype=np.float32)
    else:
        lons = np.concatenate([
            np.arange(lon_min, 360, res, dtype=np.float32),
            np.arange(0, lon_max + 0.001, res, dtype=np.float32),
        ])

    return lats, lons


def open_prepare_sst_dataset(path, var_name, lat_bounds, lon_bounds, target_res_deg, label="dataset") -> xr.DataArray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    ds = xr.open_dataset(path, decode_times=True)
    var = infer_data_var(ds, var_name)

    time_name = find_coord_name(ds, TIME_NAMES)
    lat_name = find_coord_name(ds, LAT_NAMES)
    lon_name = find_coord_name(ds, LON_NAMES)

    if time_name is None or lat_name is None or lon_name is None:
        raise ValueError(
            f"{label} must be gridded data with time, lat and lon. "
            f"Found dims={list(ds.dims)}, coords={list(ds.coords)}."
        )

    ds = normalize_lon_0_360(ds, lon_name)
    da = ds[var]
    da = standardize_dim_names(da, time_name, lat_name, lon_name)
    da = remove_extra_dims(da)
    da = subset_lon_lat(da, lat_bounds, lon_bounds)

    target_lats, target_lons = make_target_grid(lat_bounds, lon_bounds, target_res_deg)
    da = da.interp(lat=target_lats, lon=target_lons, method="linear")
    da = da.where(np.isfinite(da)).astype("float32")
    return da


def compute_monthly_anomaly(da: xr.DataArray, base_period: tuple[str, str] | None = None):
    clim_source = da if base_period is None else da.sel(time=slice(base_period[0], base_period[1]))
    climatology = clim_source.groupby("time.month").mean("time", skipna=True)
    anomaly = da.groupby("time.month") - climatology
    return anomaly.astype("float32"), climatology.astype("float32")


def fill_missing_space_time(da: xr.DataArray) -> xr.DataArray:
    da = da.interpolate_na(dim="time", method="linear", fill_value="extrapolate")
    da = da.interpolate_na(dim="lat", method="nearest", fill_value="extrapolate")
    da = da.interpolate_na(dim="lon", method="nearest", fill_value="extrapolate")
    return da.fillna(0.0).astype("float32")


def make_windows(data, times, input_months, output_months, stride=1, require_output_start_after=None):
    X, Y, TT = [], [], []
    T = data.shape[0]
    max_start = T - input_months - output_months

    for i in range(0, max_start + 1, stride):
        output_start = i + input_months
        output_end = output_start + output_months

        if require_output_start_after is not None and output_start < require_output_start_after:
            continue

        x = data[i:output_start]
        y = data[output_start:output_end]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            continue

        X.append(x.astype("float32"))
        Y.append(y.astype("float32"))
        TT.append(times[output_start:output_end])

    if len(X) == 0:
        raise ValueError("No valid windows were created.")
    return np.stack(X), np.stack(Y), np.stack(TT)


def target_start_months(TT) -> np.ndarray:
    return np.array([cftime_to_timestamp(t[0]).month for t in TT], dtype=np.int64)


def stratified_month_split(months, val_ratio=0.30, seed=42):
    rng = np.random.default_rng(seed)
    months = np.asarray(months)
    train_idx, val_idx = [], []

    for m in np.unique(months):
        idx = np.where(months == m)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) > 1 else 0
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    train_idx = np.array(sorted(train_idx), dtype=np.int64)
    val_idx = np.array(sorted(val_idx), dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Stratified split produced an empty train or validation set.")
    return train_idx, val_idx


class SSTForecastDataset(Dataset):
    def __init__(self, X, Y, months):
        self.X = torch.as_tensor(X).float()
        self.Y = torch.as_tensor(Y).float()
        self.months = np.asarray(months)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], int(self.months[idx])


def make_loader(X, Y, months, indices, batch_size=64, shuffle=True, num_workers=0, device="cpu"):
    dataset = SSTForecastDataset(X, Y, months)
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(str(device).startswith("cuda")),
        drop_last=False,
    )


def spatial_weighted_mean_array(arr, lats):
    arr = np.asarray(arr)
    weights = np.cos(np.deg2rad(lats)).astype("float32")
    weights = weights / weights.mean()

    if arr.ndim == 4:
        w = weights.reshape(1, 1, -1, 1)
        return np.nanmean(arr * w, axis=(-2, -1))
    if arr.ndim == 3:
        w = weights.reshape(1, -1, 1)
        return np.nanmean(arr * w, axis=(-2, -1))
    raise ValueError(f"Unsupported array shape: {arr.shape}")
