from __future__ import annotations

import torch
import torch.nn as nn

from .cla import CenterAwareLinearAttention
from .mamba_core import Mamba, MambaConfig
from .tlm import TemporalLongShortMamba


class DualHeadDecoder(nn.Module):
    """Dual decoder for map forecasts and Niño3.4 index forecasts."""

    def __init__(self, feature_dim: int, T_out: int, H: int, W: int):
        super().__init__()
        self.T_out = T_out
        self.H = H
        self.W = W
        self.spatial_head = nn.Linear(feature_dim, T_out * H * W)
        self.temporal_head = nn.Linear(feature_dim, T_out)

    def forward(self, z: torch.Tensor):
        maps = self.spatial_head(z).view(z.size(0), self.T_out, self.H, self.W)
        index = self.temporal_head(z).view(z.size(0), self.T_out)
        return maps, index


class CLATLM(nn.Module):
    """
    CLATLM model for ENSO forecasting.

    Components:
      - Conv3D Encoder
      - Center Aware Linear Attention (CLA)
      - Temporal Long-Short Mamba (TLM)
      - Dual Head Decoder

    The primary output is forecast maps [B, T_out, H, W]. If return_index=True,
    the model also returns a direct Niño3.4 index forecast [B, T_out].
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 16,
        H_ds: int = 16,
        W_ds: int = 16,
        heads: int = 4,
        cla_num_features: int = 256,
        cla_max_features: int = 256,
        cla_tau: float = 6.0,
        cla_beta_init: float = 5.0,
        cla_lam_init: float = 0.8,
        cla_p_drop: float = 0.05,
        tlm_blocks: int = 1,
        tlm_n_layers: int = 2,
        tlm_d_state: int = 64,
        tlm_d_conv: int = 1,
        tlm_use_gates: bool = True,
        tlm_gate_activation: str = "sigmoid",
        downsample: bool = True,
        T_out: int = 24,
        H: int = 40,
        W: int = 200,
        ctx_tokens: int = 4,
        dual_head_decoder: bool = True,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.T_out = T_out
        self.ctx_tokens = ctx_tokens
        self.dual_head_decoder = dual_head_decoder

        self.encoder = nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((None, H_ds, W_ds)) if downsample else nn.Identity()

        self.cla = CenterAwareLinearAttention(
            H=H_ds,
            W=W_ds,
            D=embed_dim,
            heads=heads,
            num_features=cla_num_features,
            rff_max_features=cla_max_features,
            tau=cla_tau,
            p_drop=cla_p_drop,
            beta_init=cla_beta_init,
            lam_init=cla_lam_init,
        )

        self.temporal_in = embed_dim * H_ds * W_ds
        self.tlm = nn.Sequential(*[
            TemporalLongShortMamba(
                in_dim=self.temporal_in,
                state_dim=tlm_d_state,
                n_layers=tlm_n_layers,
                d_conv=tlm_d_conv,
                p_drop=cla_p_drop,
                use_gates=tlm_use_gates,
                gate_activation=tlm_gate_activation,
            )
            for _ in range(tlm_blocks)
        ])

        dec_cfg = MambaConfig(
            d_model=self.temporal_in,
            n_layers=2,
            d_state=64,
            d_conv=1,
            expand_factor=2,
            conv_bias=False,
            use_gates=True,
            tie_forget=True,
        )
        self.decoder_mamba = Mamba(dec_cfg)
        self.decoder = DualHeadDecoder(self.temporal_in, T_out, H, W)

    def forward(self, x: torch.Tensor, return_index: bool = False):
        B, T, H, W = x.shape
        x = self.encoder(x.unsqueeze(1))
        x = self.pool(x).permute(0, 2, 3, 4, 1)

        x = self.cla(x)
        x = x.reshape(B, T, -1)
        x = self.tlm(x)

        k = min(self.ctx_tokens, x.size(1))
        z = x[:, -k:].mean(dim=1)
        z = self.decoder_mamba(z.unsqueeze(1)).squeeze(1)

        maps, index = self.decoder(z)
        maps = torch.nan_to_num(maps, nan=0.0, posinf=1.0, neginf=-1.0)
        index = torch.nan_to_num(index, nan=0.0, posinf=1.0, neginf=-1.0)

        if return_index:
            return maps, index
        return maps


def build_model(cfg_model, H: int, W: int, T_out: int, device):
    H_ds = min(cfg_model.H_ds, max(1, H))
    W_ds = min(cfg_model.W_ds, max(1, W))
    model = CLATLM(
        in_channels=cfg_model.in_channels,
        embed_dim=cfg_model.embed_dim,
        H_ds=H_ds,
        W_ds=W_ds,
        heads=cfg_model.heads,
        cla_num_features=cfg_model.cla_num_features,
        cla_max_features=cfg_model.cla_max_features,
        cla_tau=cfg_model.cla_tau,
        cla_beta_init=cfg_model.cla_beta_init,
        cla_lam_init=cfg_model.cla_lam_init,
        cla_p_drop=cfg_model.cla_p_drop,
        tlm_blocks=cfg_model.tlm_blocks,
        tlm_n_layers=cfg_model.tlm_n_layers,
        tlm_d_state=cfg_model.tlm_d_state,
        tlm_d_conv=cfg_model.tlm_d_conv,
        tlm_use_gates=cfg_model.tlm_use_gates,
        tlm_gate_activation=cfg_model.tlm_gate_activation,
        downsample=cfg_model.downsample,
        T_out=T_out,
        H=H,
        W=W,
        ctx_tokens=cfg_model.ctx_tokens,
        dual_head_decoder=cfg_model.dual_head_decoder,
    ).to(device)
    return model
