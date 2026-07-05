from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import cftime_to_timestamp, spatial_weighted_mean_array
from .metrics import rmse, pcc


def _time_index(times):
    return pd.DatetimeIndex([cftime_to_timestamp(t) for t in times])


def plot_training_history(history_csv, out_dir):
    hist = pd.read_csv(history_csv)
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist["epoch"], hist["train_loss"], marker="o", label="Train SPC")
    ax.plot(hist["epoch"], hist["val_loss"], marker="s", label="Val SPC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SPC loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_spc_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lead_metrics(metrics_df, out_dir):
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics_df["lead"], metrics_df["RMSE"], marker="o", label="RMSE")
    ax.plot(metrics_df["lead"], metrics_df["MAE"], marker="s", label="MAE")
    ax.set_xlabel("Lead month")
    ax.set_ylabel("Error (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lead_error_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for col in ["PCC", "ACC", "SSIM", "R2"]:
        ax.plot(metrics_df["lead"], metrics_df[col], marker="o", label=col)
    ax.set_xlabel("Lead month")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "lead_skill_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_nino34_selected_leads(preds_c, obs_c, target_times, lats, out_dir, leads=(3, 6, 12, 18, 24)):
    out_dir = Path(out_dir)
    pred_idx = spatial_weighted_mean_array(preds_c, lats)
    obs_idx = spatial_weighted_mean_array(obs_c, lats)

    fig, ax = plt.subplots(figsize=(12, 4))
    dates0 = _time_index(target_times[:, 0])
    ax.plot(dates0, obs_idx[:, 0], linewidth=2.0, label="Observation")
    for lead in leads:
        li = lead - 1
        dates = _time_index(target_times[:, li])
        ax.plot(dates, pred_idx[:, li], linestyle="--", marker="o", markersize=3, label=f"Lead-{lead}")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_ylabel("Niño3.4 index (°C)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "nino34_selected_leads.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_map_sequence(preds_c, obs_c, target_times, lats, lons, out_dir, seq_idx=0, lead_start=1):
    out_dir = Path(out_dir)
    lead_numbers = list(range(lead_start, lead_start + 12))
    values = []
    for lead in lead_numbers:
        li = lead - 1
        values.append(preds_c[seq_idx, li].ravel())
        values.append(obs_c[seq_idx, li].ravel())
    values = np.concatenate(values)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    levels = np.linspace(vmin, vmax, 17)

    fig, axes = plt.subplots(6, 4, figsize=(22, 12))
    for row in range(6):
        left_lead = lead_start + row
        right_lead = lead_start + 6 + row
        panels = [
            (left_lead, preds_c, 0, "Forecast"),
            (left_lead, obs_c, 1, "Observation"),
            (right_lead, preds_c, 2, "Forecast"),
            (right_lead, obs_c, 3, "Observation"),
        ]
        for lead, arr, col, name in panels:
            li = lead - 1
            ax = axes[row, col]
            cf = ax.contourf(lons, lats, arr[seq_idx, li], levels=levels, cmap="turbo", extend="both")
            dt = cftime_to_timestamp(target_times[seq_idx, li]).strftime("%Y-%m")
            ax.set_title(f"{name} • {dt}", fontsize=9)
            ax.set_xlabel("Longitude", fontsize=8)
            ax.set_ylabel("Latitude", fontsize=8)
            fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / f"map_sequence_seq{seq_idx}_leads{lead_start:02d}_{lead_start+11:02d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
