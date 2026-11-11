"""Measure training throughput / peak memory of each (arch, mode) at a few input sizes
on synthetic data. Used once to pick the image size and batch size of the protocol.

    python -m scripts.benchmark --sizes 160 224 --batch-size 64
"""
import argparse
import time

import torch
import torch.nn as nn

from src.models import build_model, is_cnn, set_train_mode
from src.train import make_optimizer


def bench(arch, mode, size, bs, steps=20, warmup=5):
    model, frozen = build_model(arch, mode, num_classes=100, img_size=size)
    model = model.cuda()
    fmt = torch.channels_last if is_cnn(arch) else torch.contiguous_format
    model = model.to(memory_format=fmt)
    opt = make_optimizer(model, 1e-4, 1e-3, 0.05)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(bs, 3, size, size, device="cuda").contiguous(memory_format=fmt)
    y = torch.randint(0, 100, (bs,), device="cuda")
    set_train_mode(model, frozen)
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + warmup):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    ips = steps * bs / (time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    del model, opt
    torch.cuda.empty_cache()
    return ips, mem


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=["resnet50", "deit_s"])
    p.add_argument("--modes", nargs="+", default=["full", "partial", "linear"])
    p.add_argument("--sizes", nargs="+", type=int, default=[160, 224])
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args()
    torch.backends.cudnn.benchmark = True
    for size in a.sizes:
        for arch in a.archs:
            for mode in a.modes:
                try:
                    ips, mem = bench(arch, mode, size, a.batch_size)
                    print(f"{arch:9s} {mode:8s} {size:4d}px bs{a.batch_size}: {ips:7.1f} img/s  peak {mem:.2f} GB",
                          flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"{arch:9s} {mode:8s} {size:4d}px bs{a.batch_size}: OOM", flush=True)
                    torch.cuda.empty_cache()
