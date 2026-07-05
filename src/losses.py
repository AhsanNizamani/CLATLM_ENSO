from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPCLoss(nn.Module):
    """
    Spatial Pattern and Content (SPC) Loss.

    SPC combines content fidelity through MSE with spatial pattern similarity
    through a spatial correlation loss.

        L_SPC = mse_weight * MSE + spatial_weight * CorrLoss"""

    def __init__(self, mse_weight: float = 1.0, spatial_weight: float = 0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.spatial_weight = spatial_weight

    @staticmethod
    def spatial_corr_loss(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        B, T, H, W = y_true.shape
        yt = y_true.reshape(B * T, -1)
        yp = y_pred.reshape(B * T, -1)
        yt = yt - yt.mean(dim=1, keepdim=True)
        yp = yp - yp.mean(dim=1, keepdim=True)
        num = (yt * yp).sum(dim=1)
        den = torch.sqrt((yt.pow(2).sum(dim=1) + eps) * (yp.pow(2).sum(dim=1) + eps))
        return (1.0 - (num / den)).mean()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mse = F.mse_loss(y_pred, y_true)
        spatial = self.spatial_corr_loss(y_true, y_pred)
        total = self.mse_weight * mse + self.spatial_weight * spatial
        return total, {"mse": mse.detach(), "spc_pattern": spatial.detach()}
