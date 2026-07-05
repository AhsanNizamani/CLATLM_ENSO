from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .losses import SPCLoss
from .metrics import all_metrics
from .utils import amp_dtype, ensure_dir


def train_one_epoch(model, loader, optimizer, loss_fn, cfg_training, device):
    model.train()
    use_amp = cfg_training.use_amp and str(device).startswith("cuda")
    dtype = amp_dtype(cfg_training.amp_dtype)
    use_scaler = use_amp and dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    running = {"loss": 0.0, "mse": 0.0, "spc_pattern": 0.0}
    n = 0

    for x, y, _ in tqdm(loader, leave=False, desc="train"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            pred = model(x)
            loss, parts = loss_fn(pred, y)

        if use_scaler:
            scaler.scale(loss).backward()
            if cfg_training.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg_training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_training.grad_clip)
            optimizer.step()

        running["loss"] += float(loss.detach().cpu())
        running["mse"] += float(parts["mse"].cpu())
        running["spc_pattern"] += float(parts["spc_pattern"].cpu())
        n += 1

    return {k: v / max(1, n) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, cfg_training, device) -> Dict[str, float]:
    model.eval()
    use_amp = cfg_training.use_amp and str(device).startswith("cuda")
    dtype = amp_dtype(cfg_training.amp_dtype)

    running = {"loss": 0.0, "mse": 0.0, "spc_pattern": 0.0}
    n = 0
    preds, trues = [], []

    for x, y, _ in tqdm(loader, leave=False, desc="val"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            pred = model(x)
            loss, parts = loss_fn(pred, y)

        running["loss"] += float(loss.detach().cpu())
        running["mse"] += float(parts["mse"].cpu())
        running["spc_pattern"] += float(parts["spc_pattern"].cpu())
        n += 1
        preds.append(pred.detach().cpu().float().numpy())
        trues.append(y.detach().cpu().float().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    result = {f"val_{k}": v / max(1, n) for k, v in running.items()}
    result.update({f"val_{k}": v for k, v in all_metrics(y_true, y_pred).items()})
    return result


def fit_model(model, train_loader, val_loader, cfg, stage_name: str, save_dir: str | Path, device, epochs: int, lr: float):
    """Train for all epochs. Early stopping is intentionally not used."""
    save_dir = ensure_dir(save_dir)
    loss_fn = SPCLoss(
        mse_weight=cfg.loss.mse_weight,
        spatial_weight=cfg.loss.spatial_pattern_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(3, int(epochs * 0.15)),
    )

    best_val = float("inf")
    history = []
    best_path = save_dir / f"{stage_name}_best.pt"
    last_path = save_dir / f"{stage_name}_last.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg.training, device)
        val_stats = validate(model, val_loader, loss_fn, cfg.training, device)
        scheduler.step(val_stats["val_loss"])
        elapsed = time.time() - t0

        row = {
            "stage": stage_name,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed,
            **{f"train_{k}": v for k, v in train_stats.items()},
            **val_stats,
        }
        history.append(row)

        print(
            f"[{stage_name}] {epoch:03d}/{epochs} | "
            f"train={row['train_loss']:.4f} val={row['val_loss']:.4f} "
            f"RMSE={row['val_RMSE']:.4f} MAE={row['val_MAE']:.4f} "
            f"PCC={row['val_PCC']:.3f} R2={row['val_R2']:.3f} "
            f"SSIM={row['val_SSIM']:.3f} ACC={row['val_ACC']:.3f} "
            f"{elapsed:.1f}s"
        )

        ckpt = {
            "model": model.state_dict(),
            "stage": stage_name,
            "epoch": epoch,
            "config": cfg,
        }
        torch.save(ckpt, last_path)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save({**ckpt, "best_val_loss": best_val}, best_path)

    hist = pd.DataFrame(history)
    hist.to_csv(save_dir / f"{stage_name}_history.csv", index=False)
    model.load_state_dict(torch.load(last_path, map_location=device, weights_only=False)["model"])
    return model, hist
