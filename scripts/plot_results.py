#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.plotting import plot_lead_metrics, plot_nino34_selected_leads, plot_monthly_map_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/CLATLM_ENSO")
    args = parser.parse_args()
    out_dir = Path(args.results)

    metrics = pd.read_csv(out_dir / "lead_metrics.csv")
    plot_lead_metrics(metrics, out_dir)

    pack = np.load(out_dir / "test_predictions_noaa_oisst_v21_celsius_anomaly.npz", allow_pickle=True)
    plot_nino34_selected_leads(pack["preds_c"], pack["obs_c"], pack["target_times"], pack["lats"], out_dir)
    plot_monthly_map_sequence(pack["preds_c"], pack["obs_c"], pack["target_times"], pack["lats"], pack["lons"], out_dir, seq_idx=0, lead_start=1)


if __name__ == "__main__":
    main()
