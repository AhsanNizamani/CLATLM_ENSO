from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MambaConfig:
    d_model: int
    n_layers: int = 2
    dt_rank: int | str = "auto"
    d_state: int = 64
    expand_factor: int = 2
    d_conv: int = 1
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    bias: bool = False
    conv_bias: bool = False
    use_gates: bool = True
    gate_activation: str = "sigmoid"
    tie_forget: bool = False

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        Di, Dm = config.d_inner, config.d_model

        self.in_proj = nn.Linear(Dm, 2 * Di, bias=config.bias)

        if config.use_gates:
            self.i_proj = nn.Linear(Dm, Di, bias=True)
            self.f_proj = None if config.tie_forget else nn.Linear(Dm, Di, bias=True)
        else:
            self.i_proj = None
            self.f_proj = None

        self.conv1d = nn.Conv1d(
            Di,
            Di,
            config.d_conv,
            groups=Di,
            bias=config.conv_bias,
            padding=config.d_conv - 1,
        )

        self.x_proj = nn.Linear(Di, config.dt_rank + 2 * config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, Di, bias=True)

        dt = torch.exp(
            torch.rand(Di) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(Di, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(Di))
        self.out_proj = nn.Linear(Di, Dm, bias=config.bias)

    def _gate_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.gate_activation == "silu":
            return F.silu(x)
        return torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        x_inner, z_inner = self.in_proj(x).chunk(2, dim=-1)

        if self.config.use_gates:
            i = self._gate_act(self.i_proj(x))
            if self.config.tie_forget:
                f = 1.0 - i
            else:
                f = self._gate_act(self.f_proj(x))
            x_mix = f * x_inner + i * torch.tanh(x_inner)
        else:
            x_mix = x_inner

        x_conv = x_mix.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        y = self.ssm(x_conv)
        y = y * self._gate_act(z_inner)
        return self.out_proj(y)

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        Di, Ds = self.config.d_inner, self.config.d_state
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, Ds, Ds], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)
        BX = deltaB * x.unsqueeze(-1)

        h = torch.zeros(x.size(0), Di, Ds, device=x.device, dtype=x.dtype)
        y = torch.zeros_like(x)
        for t in range(x.size(1)):
            h = deltaA[:, t] * h + BX[:, t]
            y[:, t] = (h @ C[:, t].unsqueeze(-1)).squeeze(-1) + D * x[:, t]
        return y


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x)) + x


class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)
