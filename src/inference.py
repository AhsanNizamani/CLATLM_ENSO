from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from .utils import amp_dtype


@torch.no_grad()
def predict(model, loader, cfg_training, device):
    model.eval()
    use_amp = cfg_training.use_amp and str(device).startswith("cuda")
    dtype = amp_dtype(cfg_training.amp_dtype)
    preds, obs = [], []

    for x, y, _ in tqdm(loader, leave=False, desc="predict"):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            pred = model(x)
        preds.append(pred.detach().cpu().float().numpy())
        obs.append(y.numpy())

    return np.concatenate(preds, axis=0), np.concatenate(obs, axis=0)
