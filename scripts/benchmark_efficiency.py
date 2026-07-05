#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.config import load_config
from src.models import build_model
from src.utils import get_device, seed_everything


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def benchmark(model, x, warmup=20, iters=100):
    device = x.device
    model.eval()
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    samples = x.size(0) * iters
    return samples / elapsed, elapsed / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--T-in", type=int, default=None)
    parser.add_argument("--H", type=int, default=40)
    parser.add_argument("--W", type=int, default=200)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)
    device = get_device()

    T_in = args.T_in or cfg.data.input_months
    model = build_model(cfg.model, H=args.H, W=args.W, T_out=cfg.data.output_months, device=device)
    x = torch.randn(args.batch_size, T_in, args.H, args.W, device=device)

    params = count_parameters(model)
    throughput, latency = benchmark(model, x, iters=args.iters)

    print(f"Model: CLATLM")
    print(f"Device: {device}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Trainable parameters: {params:,} ({params/1e6:.3f}M)")
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"Latency: {latency*1000:.2f} ms/batch")
    print("Note: GMAC estimation is hardware/tool dependent. The paper reports 0.48 GMACs per prediction.")


if __name__ == "__main__":
    main()
