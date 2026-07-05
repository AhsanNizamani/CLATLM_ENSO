from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_core import Mamba, MambaConfig


class TemporalLongShortMamba(nn.Module):
    """
    Temporal Long-Short Mamba (TLM).

    TLM applies LayerNorm, gated Mamba temporal mixing, dropout and residual
    fusion to model long-short temporal dependencies in SST anomaly sequences.
    Input shape: [B, T, F]
    Output shape: [B, T, F]
    """

    def __init__(
        self,
        in_dim: int,
        state_dim: int = 64,
        n_layers: int = 2,
        d_conv: int = 1,
        p_drop: float = 0.05,
        use_gates: bool = True,
        gate_activation: str = "sigmoid",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        cfg = MambaConfig(
            d_model=in_dim,
            n_layers=n_layers,
            d_state=state_dim,
            d_conv=d_conv,
            expand_factor=2,
            conv_bias=False,
            use_gates=use_gates,
            gate_activation=gate_activation,
            tie_forget=True,
        )
        self.mamba = Mamba(cfg)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mamba(self.norm(x)))
