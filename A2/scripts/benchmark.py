"""Measure training throughput / peak memory of each model on synthetic data. Used once to pick
the crop size, batch size and number of iterations that fit a 6 GB GPU.

    python -m scripts.benchmark --crops 384 512 --batch-size 8
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from src.data import NUM_CLASSES, upsample_logits
from src.models import MODELS, build_model, count_params
from src.train import forward_gflops, make_optimizer


def bench(arch, crop, bs, literal_decoder=False, steps=12, warmup=4):
    model = build_model(arch, NUM_CLASSES, pretrained=False).cuda()
    if literal_decoder and hasattr(model.decode_head, "literal"):
        model.decode_head.literal = True
    params = count_params(model)
    gmacs = forward_gflops(model)
    opt = make_optimizer(model, 6e-5, 6e-4, 0.01)
    x = torch.randn(bs, 3, crop, crop, device="cuda")
    y = torch.randint(0, NUM_CLASSES, (bs, crop, crop), device="cuda")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + warmup):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(upsample_logits(logits, y.shape[-2:]), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    ips = steps * bs / (time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    del model, opt
    torch.cuda.empty_cache()
    return ips, mem, params, gmacs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archs", nargs="+", default=list(MODELS))
    p.add_argument("--crops", nargs="+", type=int, default=[384, 512])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--literal-decoder", action="store_true",
                   help="use the paper's concat-then-fuse MLP decoder instead of the merged one")
    p.add_argument("--json", help="measure both decoder variants and write {merged: [...], literal: [...]} here")
    a = p.parse_args()
    torch.backends.cudnn.benchmark = True
    result = {}
    modes = [("merged", False), ("literal", True)] if a.json else [("", a.literal_decoder)]
    for mode, literal in modes:
        rows = result.setdefault(mode, [])
        for crop in a.crops:
            for arch in a.archs:
                row = {"arch": arch, "crop": crop, "batch_size": a.batch_size}
                tag = f"{arch:18s} {mode:8s} {crop:4d}px bs{a.batch_size}"
                try:
                    ips, mem, (total, enc, head), gmacs = bench(arch, crop, a.batch_size, literal)
                    row.update(img_per_s=ips, peak_mem_gb=mem, params=total, gmacs_512=gmacs)
                    print(f"{tag}: {ips:6.1f} img/s  peak {mem:.2f} GB  params {total/1e6:.2f}M "
                          f"(enc {enc/1e6:.2f} + dec {head/1e6:.2f})  GMACs@512 {gmacs:.1f}", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"{tag}: OOM", flush=True)
                    torch.cuda.empty_cache()
                rows.append(row)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(result, f, indent=2)
