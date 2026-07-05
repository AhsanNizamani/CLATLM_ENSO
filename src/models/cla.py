from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterAwareLinearAttention(nn.Module):
    """
    Center Aware Linear Attention (CLA).

    CLA uses Q/K/V projections, center-aware kernel modulation and cosine
    random Fourier features to capture spatial relationships efficiently.
    Input shape: [B, T, H, W, D]
    Output shape: [B, T, H, W, D]
    """

    def __init__(
        self,
        H: int,
        W: int,
        D: int,
        heads: int,
        num_features: int = 256,
        rff_max_features: int = 256,
        tau: float = 6.0,
        p_drop: float = 0.05,
        beta_init: float = 5.0,
        lam_init: float = 0.8,
    ):
        super().__init__()
        if D % heads != 0:
            raise ValueError(f"embed_dim D={D} must be divisible by heads={heads}")

        self.norm = nn.LayerNorm(D)
        self.to_qkv = nn.Linear(D, 3 * D)
        self.to_out = nn.Linear(D, D)
        self.drop = nn.Dropout(p_drop)

        self.heads = heads
        self.D = D
        self.d = D // heads
        self.num_features = num_features
        self.rff_max_features = rff_max_features
        self.tau = tau

        self.beta = nn.Parameter(torch.full((heads,), beta_init))
        self.lam = nn.Parameter(torch.full((heads,), lam_init))

        self.register_buffer("rff_weights", torch.randn(heads, self.d, num_features))
        self.register_buffer("rff_max_weights", torch.randn(heads, self.d, rff_max_features))

    def _phi_softmax_pos(self, X_scaled: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        X_scaled = X_scaled.float()
        W = W.float()
        XW = torch.einsum("bhnd,bhdm->bhnm", X_scaled, W)
        norm2 = (X_scaled ** 2).sum(dim=-1, keepdim=True)
        logits = XW - 0.5 * norm2
        logits = logits.clamp(min=-20.0, max=20.0)
        return torch.exp(logits)

    def _softmax_max_center(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q = Q.float()
        K = K.float()
        d_quarter = self.d ** 0.25
        sqrt_tau = math.sqrt(self.tau)
        Q_tau = (Q / d_quarter) * sqrt_tau
        K_tau = (K / d_quarter) * sqrt_tau

        W = self.rff_max_weights.float().unsqueeze(0)
        phi_q = self._phi_softmax_pos(Q_tau, W)
        phi_k = self._phi_softmax_pos(K_tau, W)

        K_sum = phi_k.sum(dim=-2)
        S = torch.einsum("bhnm,bhm->bhn", phi_q, K_sum).clamp_min(1e-6)
        max_center = (torch.log(S) / self.tau).unsqueeze(-1)
        return torch.nan_to_num(max_center, nan=0.0, posinf=5.0, neginf=-5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, T, H, W, D = x.shape
        N = H * W

        x_flat = x.contiguous().view(B * T, N, D)
        qkv = self.to_qkv(self.norm(x_flat)).chunk(3, dim=-1)
        Q, K, V = [
            t.view(B * T, N, self.heads, self.d).permute(0, 2, 1, 3).contiguous().float()
            for t in qkv
        ]

        beta = F.softplus(self.beta).view(1, self.heads, 1, 1).float().clamp(1e-3, 20.0)
        lam = torch.sigmoid(self.lam).view(1, self.heads, 1, 1).float()

        Q_mean = Q.mean(dim=-2, keepdim=True)
        K_mean = K.mean(dim=-2, keepdim=True)
        mean_center = (Q_mean @ K_mean.transpose(-1, -2)) / math.sqrt(self.d)
        max_center = self._softmax_max_center(Q, K)
        mu_lam = lam * max_center + (1.0 - lam) * mean_center
        mu_lam = torch.nan_to_num(mu_lam, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        W_r = self.rff_weights.float().unsqueeze(0)
        Q_proj = torch.einsum("bhnd,bhdm->bhnm", Q, W_r)
        K_proj = torch.einsum("bhnd,bhdm->bhnm", K, W_r)

        Q_prime = F.elu(torch.cos(Q_proj)) + 1.001
        K_prime = F.elu(torch.cos(K_proj)) + 1.001

        shift = torch.exp((-beta * (mu_lam ** 2)).clamp(min=-20.0, max=0.0))
        raw_scale = (2.0 * beta * mu_lam) / math.sqrt(self.d)
        scale = F.softplus(raw_scale).clamp_min(1e-4).clamp_max(20.0)

        m_attn = K_prime.size(-1)
        shift_exp = shift.expand(-1, -1, -1, m_attn)
        scale_exp = scale.expand(-1, -1, -1, m_attn)

        KV = torch.einsum("bhnm,bhnd->bhmd", K_prime * scale_exp, V)
        ones = torch.ones_like(K_prime[..., 0])
        Z = torch.einsum("bhnm,bhn->bhm", K_prime * scale_exp, ones)

        attn_out = torch.einsum("bhnm,bhmd->bhnd", Q_prime * shift_exp, KV)
        norm = torch.einsum("bhnm,bhm->bhn", Q_prime * shift_exp, Z).clamp_min(1e-4)
        out = attn_out / norm.unsqueeze(-1)
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)

        out = out.permute(0, 2, 1, 3).reshape(B * T, N, D)
        out = self.drop(self.to_out(out)).view(B, T, H, W, D)
        return residual + out.to(residual.dtype)
