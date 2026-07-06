from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ProjectConfig:
    name: str = "CLATLM-ENSO"
    out_dir: str = "results/CLATLM_ENSO"
    seed: int = 42


@dataclass
class DataConfig:
    cmip6_path: str = "data/CMIP6_Nino34_SSTA_Multimodel_185001-201412.nc"
    cmip6_var_name: str = "tos"
    oisst_path: str = "data/NOAA_OISST_v2_1/sst.mnmean.nc"
    oisst_var_name: str = "sst"
    lat_bounds: tuple[float, float] = (-5.0, 5.0)
    lon_bounds: tuple[float, float] = (190.0, 240.0)
    target_res_deg: float = 2.5
    oisst_finetune_fraction: float = 0.70
    input_months: int = 12
    output_months: int = 24
    stride: int = 1
    val_ratio: float = 0.30
    cache: bool = True
    force_rebuild_cache: bool = False


@dataclass
class ModelConfig:
    in_channels: int = 1
    embed_dim: int = 16
    encoder_kernel: int = 3
    downsample: bool = True
    H_ds: int = 16
    W_ds: int = 16
    heads: int = 4
    cla_num_features: int = 256
    cla_max_features: int = 256
    cla_tau: float = 6.0
    cla_beta_init: float = 5.0
    cla_lam_init: float = 0.8
    cla_p_drop: float = 0.05
    tlm_blocks: int = 1
    tlm_n_layers: int = 2
    tlm_d_state: int = 64
    tlm_d_conv: int = 1
    tlm_use_gates: bool = True
    tlm_gate_activation: str = "sigmoid"
    ctx_tokens: int = 4
    dual_head_decoder: bool = True


@dataclass
class LossConfig:
    name: str = "Spatial Pattern and Content Loss"
    abbreviation: str = "SPC"
    mse_weight: float = 1.0
    spatial_pattern_weight: float = 0.2


@dataclass
class TrainingConfig:
    pretrain_epochs: int = 100
    finetune_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    finetune_learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    num_workers: int = 0
    grad_clip: float | None = None
    early_stopping: bool = False


@dataclass
class Config:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    metrics: list[str]


def _section(cls, data: Dict[str, Any]) -> Any:
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{k: v for k, v in data.items() if k in allowed})


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        project=_section(ProjectConfig, raw.get("project", {})),
        data=_section(DataConfig, raw.get("data", {})),
        model=_section(ModelConfig, raw.get("model", {})),
        loss=_section(LossConfig, raw.get("loss", {})),
        training=_section(TrainingConfig, raw.get("training", {})),
        metrics=list(raw.get("metrics", [])),
    )
