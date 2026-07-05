# CLATLM: Center Aware Spatiotemporal Framework for Long-Lead ENSO Forecasting

Official-style PyTorch implementation for the paper:

**Center Aware Spatiotemporal Framework with Linear-Time Complexity for Efficient Long Lead ENSO Forecasting**

The main model is **CLATLM**, a deterministic ENSO forecasting framework that combines:

- **Center Aware Linear Attention (CLA)** for adaptive spatial center-of-action learning with linear-time attention.
- **Temporal Long-Short Mamba (TLM)** for efficient long-range temporal dependency modeling without autoregressive rollouts.
- **Spatial Pattern and Content (SPC) Loss**, which combines MSE content fidelity with a spatial correlation term.

![CLATLM architecture](assets/Figure_1.png)

## Abstract

El Niño Southern Oscillation (ENSO) is one of the critical factors causing global climate extremes and ecosystem turbulence. It frequently causes floods, droughts, and excessive rainfall all across the world. Accurate long lead ENSO forecasting is essential, however it is difficult due to seasonal barriers, nonstationary temporal and time varying spatial correlations. We propose a deep fusion architecture, **Center aware Linear Attention with Temporal Long-short Mamba (CLATLM)**, for accurate long lead ENSO forecasting. CLATLM combines a **Center aware Linear Attention (CLA)** module, which learns adaptive spatial center of action with linear time attention, and a **Temporal Long-short Mamba (TLM)** module that captures long range temporal dependencies without autoregressive rollouts. To better align optimization, we introduce a **Spatial Pattern and Content (SPC) loss** that enhances standard loss with spatial correlation term. We construct a fused spatiotemporal dataset using CMIP6 simulations for pretraining and NOAA OISST v2.1 for fine tuning and evaluation. The proposed model achieves strong predictive performance while maintaining efficient inference.

## Highlights

- Deterministic long-lead ENSO forecasting up to 24 months.
- CMIP6 pretraining and NOAA OISST v2.1 fine-tuning/evaluation.
- CLA spatial module with center-aware linear attention.
- TLM temporal module based on long-short gated Mamba dynamics.
- SPC loss:

\[
\mathcal{L}_{SPC}=1.0\cdot\mathcal{L}_{MSE}+0.2\cdot\mathcal{L}_{pattern}
\]

where \(\mathcal{L}_{pattern}=1-r_s\), and \(r_s\) is the spatial correlation averaged over batch samples and lead months.

## Paper-reported efficiency

The manuscript reports that CLATLM processes approximately **60 samples per second** at batch size 2 and requires **0.48 GMACs** per prediction, reducing computational cost by **39%** compared with CNN and by more than **68%** relative to Transformer-based models. These values are paper-reported results; actual throughput depends on hardware, software versions, precision, and input resolution.

## Datasets

This repository is configured for:

1. **CMIP6 MIROC-ES2L historical `tos`** for pretraining.
2. **NOAA OISST v2.1 monthly SST** for fine-tuning and testing.

Data files are not included because NetCDF climate datasets are large. Place files as follows, or edit `configs/default.yaml`:

```text
data/
├── CMIP6_tos_Omon_MIROC-ES2L_historical_r1i1p1f2_gr1_185001-201412.nc
└── NOAA_OISST_v2_1/
    └── sst.mnmean.nc
```

Expected variables:

- CMIP6: `tos`
- NOAA OISST v2.1: `sst`

The data pipeline converts longitude to 0-360 format, extracts the target ENSO region, interpolates both datasets to a common grid, calculates monthly SST anomalies, normalizes without using the OISST test period, and builds sliding windows.

## Repository structure

```text
CLATLM_ENSO/
├── assets/
│   └── Figure_1.png
├── configs/
│   ├── default.yaml
│   └── paper_reported.yaml
├── docs/
│   ├── benchmark_summary.md
│   ├── github_upload_guide.md
│   ├── model_parameters.md
│   └── paper_alignment.md
├── scripts/
│   ├── run_full_pipeline.py
│   ├── train_pretrain_cmip6.py
│   ├── finetune_noaa_oisst_v21.py
│   ├── evaluate_noaa_oisst_v21.py
│   ├── plot_results.py
│   └── benchmark_efficiency.py
├── src/
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── inference.py
│   ├── plotting.py
│   └── models/
│       ├── cla.py
│       ├── tlm.py
│       ├── mamba_core.py
│       └── clatlm.py
├── tests/
│   └── test_shapes.py
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── README.md
```

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate clatlm-enso
```

### Pip

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the full experiment

```bash
python scripts/run_full_pipeline.py --config configs/default.yaml
```

This runs:

1. CMIP6 pretraining.
2. NOAA OISST v2.1 fine-tuning.
3. Held-out NOAA OISST v2.1 evaluation.
4. Lead-wise map metrics.
5. Niño3.4 index metrics.
6. Paper-ready result files.

Outputs are saved to:

```text
results/CLATLM_ENSO/
```

## Plot results

```bash
python scripts/plot_results.py --results results/CLATLM_ENSO
```

Generated plots include:

- Lead-wise RMSE/MAE.
- Lead-wise PCC/ACC/SSIM/R².
- Niño3.4 truth-vs-forecast selected lead plot.
- Forecast-vs-observation monthly SST anomaly map sequence.

## Evaluation metrics

The test pipeline reports:

- **RMSE**
- **MAE**
- **PCC**, Pearson correlation coefficient over flattened values
- **R²**
- **SSIM**
- **ACC**, mean spatial anomaly correlation over forecast maps

## Key experimental parameters

| Item | Value |
|---|---:|
| Main model | CLATLM |
| Spatial module | Center Aware Linear Attention (CLA) |
| Temporal module | Temporal Long-Short Mamba (TLM) |
| Loss | Spatial Pattern and Content (SPC) Loss |
| MSE weight | 1.0 |
| Spatial pattern weight | 0.2 |
| Pretraining dataset | CMIP6 MIROC-ES2L historical `tos` |
| Fine-tuning/testing dataset | NOAA OISST v2.1 monthly SST |
| Input months | 24 |
| Forecast months | 24 |
| Region | Niño3.4, 5°S-5°N, 190°E-240°E |
| Regridding | 2.5° |
| Optimizer | AdamW |
| Early stopping | Not used |

## GitHub upload

```bash
git init
git add .
git commit -m "Initial release: CLATLM ENSO forecasting"
git branch -M main
git remote add origin https://github.com/<username>/CLATLM-ENSO.git
git push -u origin main
```

## Citation

Add the final BibTeX entry after publication.

```bibtex
@article{clatlm_enso,
  title={Center Aware Spatiotemporal Framework with Linear-Time Complexity for Efficient Long Lead ENSO Forecasting},
  author={To be updated},
  journal={To be updated},
  year={2026},
  note={CLATLM}
}
```

## License

This repository is released under the MIT License unless otherwise changed by the authors.
