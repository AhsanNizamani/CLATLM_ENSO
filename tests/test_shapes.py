import torch

from src.models.clatlm import CLATLM
from src.losses import SPCLoss


def test_model_forward_shape():
    model = CLATLM(embed_dim=16, H_ds=16, W_ds=16, T_out=24, H=40, W=200)
    x = torch.randn(2, 12, 40, 200)
    y = model(x)
    assert y.shape == (2, 24, 40, 200)


def test_spc_loss_runs():
    loss_fn = SPCLoss(mse_weight=1.0, spatial_weight=0.2)
    pred = torch.randn(2, 24, 40, 200)
    true = torch.randn(2, 24, 40, 200)
    loss, parts = loss_fn(pred, true)
    assert torch.isfinite(loss)
    assert "mse" in parts and "spc_pattern" in parts
