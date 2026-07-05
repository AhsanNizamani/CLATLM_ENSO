#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data import (
    open_prepare_sst_dataset,
    compute_monthly_anomaly,
    fill_missing_space_time,
    make_windows,
    target_start_months,
    stratified_month_split,
    make_loader,
    SSTForecastDataset,
    spatial_weighted_mean_array,
)
from src.inference import predict
from src.metrics import all_metrics, rmse, mae, pcc, r2_score_np
from src.models import build_model
from src.train import fit_model
from src.utils import ensure_dir, get_device, save_json, seed_everything


def preprocess(cfg, out_dir):
    cache_path = out_dir / "preprocessed_cmip6_noaa_oisst_v21.npz"
    if cfg.data.cache and cache_path.exists() and not cfg.data.force_rebuild_cache:
        return np.load(cache_path, allow_pickle=True)

    cmip6_da = open_prepare_sst_dataset(
        cfg.data.cmip6_path,
        cfg.data.cmip6_var_name,
        tuple(cfg.data.lat_bounds),
        tuple(cfg.data.lon_bounds),
        cfg.data.target_res_deg,
        label="CMIP6",
    )
    oisst_da = open_prepare_sst_dataset(
        cfg.data.oisst_path,
        cfg.data.oisst_var_name,
        tuple(cfg.data.lat_bounds),
        tuple(cfg.data.lon_bounds),
        cfg.data.target_res_deg,
        label="NOAA OISST v2.1",
    )

    cmip6_anom, _ = compute_monthly_anomaly(cmip6_da)
    oisst_anom, _ = compute_monthly_anomaly(oisst_da)
    cmip6_anom = fill_missing_space_time(cmip6_anom)
    oisst_anom = fill_missing_space_time(oisst_anom)

    cmip6_np = cmip6_anom.values.astype("float32")
    oisst_np = oisst_anom.values.astype("float32")
    cmip6_times = cmip6_anom.time.values
    oisst_times = oisst_anom.time.values
    lats = cmip6_anom.lat.values.astype("float32")
    lons = cmip6_anom.lon.values.astype("float32")

    split = int(len(oisst_times) * cfg.data.oisst_finetune_fraction)
    scaler_data = np.concatenate([cmip6_np.reshape(-1), oisst_np[:split].reshape(-1)])
    scaler_data = scaler_data[np.isfinite(scaler_data)]
    scaler_mean = float(np.mean(scaler_data))
    scaler_std = float(np.std(scaler_data) + 1e-6)

    np.savez_compressed(
        cache_path,
        cmip6_anom=cmip6_np,
        oisst_anom=oisst_np,
        cmip6_times=cmip6_times,
        oisst_times=oisst_times,
        lats=lats,
        lons=lons,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )
    return np.load(cache_path, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)
    device = get_device()
    out_dir = ensure_dir(cfg.project.out_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    save_json(cfg, out_dir / "resolved_config.json")

    pack = preprocess(cfg, out_dir)
    cmip6 = (pack["cmip6_anom"] - float(pack["scaler_mean"])) / float(pack["scaler_std"])
    oisst = (pack["oisst_anom"] - float(pack["scaler_mean"])) / float(pack["scaler_std"])
    cmip6_times = pack["cmip6_times"]
    oisst_times = pack["oisst_times"]
    lats, lons = pack["lats"], pack["lons"]

    X_c, Y_c, TT_c = make_windows(cmip6, cmip6_times, cfg.data.input_months, cfg.data.output_months, cfg.data.stride)
    months_c = target_start_months(TT_c)
    tr_idx, va_idx = stratified_month_split(months_c, cfg.data.val_ratio, cfg.project.seed)

    split = int(len(oisst_times) * cfg.data.oisst_finetune_fraction)
    X_f, Y_f, TT_f = make_windows(oisst[:split], oisst_times[:split], cfg.data.input_months, cfg.data.output_months, cfg.data.stride)
    months_f = target_start_months(TT_f)
    ft_tr_idx, ft_va_idx = stratified_month_split(months_f, cfg.data.val_ratio, cfg.project.seed)

    X_t, Y_t, TT_t = make_windows(
        oisst,
        oisst_times,
        cfg.data.input_months,
        cfg.data.output_months,
        cfg.data.stride,
        require_output_start_after=split,
    )
    months_t = target_start_months(TT_t)

    train_loader = make_loader(X_c, Y_c, months_c, tr_idx, cfg.training.batch_size, True, cfg.training.num_workers, device)
    val_loader = make_loader(X_c, Y_c, months_c, va_idx, cfg.training.batch_size, False, cfg.training.num_workers, device)
    ft_train_loader = make_loader(X_f, Y_f, months_f, ft_tr_idx, cfg.training.batch_size, True, cfg.training.num_workers, device)
    ft_val_loader = make_loader(X_f, Y_f, months_f, ft_va_idx, cfg.training.batch_size, False, cfg.training.num_workers, device)

    test_loader = torch.utils.data.DataLoader(
        SSTForecastDataset(X_t, Y_t, months_t),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=str(device).startswith("cuda"),
    )

    H, W = X_c.shape[-2], X_c.shape[-1]
    model = build_model(cfg.model, H=H, W=W, T_out=cfg.data.output_months, device=device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model, hist_pre = fit_model(model, train_loader, val_loader, cfg, "cmip6_pretrain", ckpt_dir, device, cfg.training.pretrain_epochs, cfg.training.learning_rate)
    model, hist_ft = fit_model(model, ft_train_loader, ft_val_loader, cfg, "noaa_oisst_v21_finetune", ckpt_dir, device, cfg.training.finetune_epochs, cfg.training.finetune_learning_rate)
    pd.concat([hist_pre, hist_ft], ignore_index=True).to_csv(out_dir / "training_history.csv", index=False)

    preds_norm, obs_norm = predict(model, test_loader, cfg.training, device)
    preds_c = preds_norm * float(pack["scaler_std"]) + float(pack["scaler_mean"])
    obs_c = obs_norm * float(pack["scaler_std"]) + float(pack["scaler_mean"])

    np.savez_compressed(
        out_dir / "test_predictions_noaa_oisst_v21_celsius_anomaly.npz",
        preds_c=preds_c,
        obs_c=obs_c,
        target_times=TT_t,
        lats=lats,
        lons=lons,
    )

    rows = []
    for li in range(preds_c.shape[1]):
        m = all_metrics(obs_c[:, li:li + 1], preds_c[:, li:li + 1])
        m["lead"] = li + 1
        rows.append(m)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "lead_metrics.csv", index=False)
    print(metrics)

    pred_idx = spatial_weighted_mean_array(preds_c, lats)
    obs_idx = spatial_weighted_mean_array(obs_c, lats)
    nino_rows = []
    for li in range(pred_idx.shape[1]):
        nino_rows.append({
            "lead": li + 1,
            "RMSE": rmse(obs_idx[:, li], pred_idx[:, li]),
            "MAE": mae(obs_idx[:, li], pred_idx[:, li]),
            "PCC": pcc(obs_idx[:, li], pred_idx[:, li]),
            "R2": r2_score_np(obs_idx[:, li], pred_idx[:, li]),
        })
    pd.DataFrame(nino_rows).to_csv(out_dir / "nino34_index_metrics.csv", index=False)


if __name__ == "__main__":
    main()
