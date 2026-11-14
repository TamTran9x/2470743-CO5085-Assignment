"""Evaluation-only experiments that test two claims of the SegFormer paper on VOC 2012 val:

  1. robustness to image corruptions (the paper's Cityscapes-C study, Table 5): 9 corruption
     types of ImageNet-C (Hendrycks & Dietterich, 2019) re-implemented in PyTorch, 3 severities;
  2. robustness to the test resolution (Table 1c): the input is rescaled by 0.5x ... 1.5x.

Every model sees exactly the same corrupted images (noise is seeded per batch).
Writes results/robustness.json.

    python -m src.robustness
"""
import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import EVAL_SIZE, NUM_CLASSES, PAD_RGB, EvalSet, Normalize, upsample_logits
from .metrics import batch_confusion, scores
from .models import load_trained

SEVERITIES = (1, 3, 5)
SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
CATEGORY = {"gaussian_noise": "noise", "shot_noise": "noise", "impulse_noise": "noise",
            "gaussian_blur": "blur", "defocus_blur": "blur",
            "brightness": "digital", "contrast": "digital", "jpeg": "digital", "pixelate": "digital"}


def _conv_per_channel(x, kernel):
    k = kernel.to(x)[None, None].expand(3, 1, -1, -1)
    pad = kernel.shape[-1] // 2
    return F.conv2d(F.pad(x, (pad,) * 4, mode="reflect"), k, groups=3)


def gaussian_noise(x, sev, g):
    c = (0.08, 0.12, 0.18, 0.26, 0.38)[sev - 1]
    return x + c * torch.randn(x.shape, generator=g, device=x.device)


def shot_noise(x, sev, g):
    c = (60, 25, 12, 5, 3)[sev - 1]
    return torch.poisson(x * c, generator=g) / c


def impulse_noise(x, sev, g):
    c = (0.03, 0.06, 0.09, 0.17, 0.27)[sev - 1]
    u = torch.rand(x.shape, generator=g, device=x.device)
    return torch.where(u < c / 2, torch.zeros_like(x), torch.where(u > 1 - c / 2, torch.ones_like(x), x))


def gaussian_blur(x, sev, g):
    sigma = (1, 2, 3, 4, 6)[sev - 1]
    r = int(4 * sigma)
    t = torch.arange(-r, r + 1, dtype=torch.float32)
    k1 = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    return _conv_per_channel(x, k1[:, None] * k1[None, :])


def defocus_blur(x, sev, g):
    radius = (3, 4, 6, 8, 10)[sev - 1]
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    disk = ((t[:, None] ** 2 + t[None, :] ** 2) <= radius ** 2).float()
    return _conv_per_channel(x, disk / disk.sum())


def brightness(x, sev, g):
    return x + (0.1, 0.2, 0.3, 0.4, 0.5)[sev - 1]


def contrast(x, sev, g):
    c = (0.4, 0.3, 0.2, 0.1, 0.05)[sev - 1]
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - mean) * c + mean


def jpeg(x, sev, g):
    quality = (25, 18, 15, 10, 7)[sev - 1]
    arr = (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()

    def one(a):
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=quality)
        return np.asarray(Image.open(buf))

    with ThreadPoolExecutor(8) as ex:
        out = np.stack(list(ex.map(one, arr)))
    return torch.from_numpy(out).to(x.device).permute(0, 3, 1, 2).float() / 255


def pixelate(x, sev, g):
    c = (0.6, 0.5, 0.4, 0.3, 0.25)[sev - 1]
    size = x.shape[-2:]
    small = F.interpolate(x, scale_factor=c, mode="area")
    return F.interpolate(small, size=size, mode="nearest")


CORRUPTIONS = {"gaussian_noise": gaussian_noise, "shot_noise": shot_noise, "impulse_noise": impulse_noise,
               "gaussian_blur": gaussian_blur, "defocus_blur": defocus_blur, "brightness": brightness,
               "contrast": contrast, "jpeg": jpeg, "pixelate": pixelate}


def make_corrupt(name, sev, sizes, seed=0):
    """Returns fn(uint8 batch, first index) -> uint8 batch. Only the real image area is corrupted;
    the padding of the 512x512 canvas keeps the mean colour."""
    fn = CORRUPTIONS[name]
    pad = torch.tensor(PAD_RGB, dtype=torch.uint8, device="cuda").view(1, 3, 1, 1)
    ar = torch.arange(EVAL_SIZE, device="cuda")
    index = list(CORRUPTIONS).index(name)

    def corrupt(x, first):
        g = torch.Generator(device="cuda").manual_seed(seed + 1000003 * (index * 10 + sev) + first)
        hw = torch.as_tensor(sizes[first:first + len(x)], device="cuda")
        area = (ar[None, :, None] < hw[:, 0, None, None]) & (ar[None, None, :] < hw[:, 1, None, None])
        out = fn(x.float() / 255, sev, g).clamp(0, 1)
        out = (out * 255).round().to(torch.uint8)
        return torch.where(area[:, None], out, pad.expand_as(out))
    return corrupt


@torch.inference_mode()
def eval_all(models, evalset, normalize, batch_size, corrupt=None, scale=1.0):
    """mIoU of every model on the same (optionally corrupted / rescaled) inputs."""
    conf = {name: torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long, device="cuda") for name in models}
    seen = 0
    for x, y, _ in evalset.batches(batch_size):
        if corrupt is not None:
            x = corrupt(x, seen)
        x = normalize(x)
        if scale != 1.0:
            x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False, antialias=scale < 1)
        for name, model in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            pred = upsample_logits(logits, y.shape[-2:]).argmax(1)
            conf[name] += batch_confusion(pred, y).sum(0)
        seen += len(y)
    return {name: scores(c.cpu().numpy())["miou"] for name, c in conf.items()}


def save_examples(evalset, path, index=7):
    """A strip with one val image under every corruption at severity 3 (for the report)."""
    from . import plotstyle as ps
    ps.apply()
    x = evalset.images[index:index + 1].cuda()
    h, w = evalset.sizes[index]
    names = list(CORRUPTIONS)
    fig, axes = ps.plt.subplots(2, 5, figsize=(10.5, 3.6))
    panels = [("ảnh gốc", x)] + [(n.replace("_", " "), make_corrupt(n, 3, evalset.sizes)(x, index)) for n in names]
    for ax, (title, im) in zip(axes.ravel(), panels):
        ax.imshow(im[0, :, :h, :w].permute(1, 2, 0).cpu().numpy())
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=ps.PHOTO_DPI)
    ps.plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["deeplabv3plus_r50", "resnet50_mlp", "segformer_b0", "segformer_b1",
                                                 "segformer_b2"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-subset", type=int, default=0)
    p.add_argument("--out", default="results/robustness.json")
    a = p.parse_args()
    torch.backends.cudnn.benchmark = True
    runs = [r for r in a.runs if os.path.isfile(os.path.join("checkpoints", r, "final.pt"))]
    models = {r: load_trained(r) for r in runs}
    val = EvalSet("val", subset=a.eval_subset)
    normalize = Normalize()
    os.makedirs("figures", exist_ok=True)
    save_examples(val, "figures/fig_corruptions.png")

    result = None
    if os.path.isfile(a.out):  # resume an interrupted run (same models and images only)
        with open(a.out) as f:
            old = json.load(f)
        if old.get("runs") == runs and old.get("n_val") == len(val):
            result = old
            print("resuming from", a.out, flush=True)
    if result is None:
        result = {"runs": runs, "severities": list(SEVERITIES), "scales": list(SCALES), "category": CATEGORY,
                  "n_val": len(val), "clean": eval_all(models, val, normalize, a.batch_size),
                  "scale": {r: {} for r in runs}, "corruption": {r: {c: {} for c in CORRUPTIONS} for r in runs}}

    def save():
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)

    print("clean", result["clean"], flush=True)
    for s in SCALES:
        if str(s) not in result["scale"][runs[0]]:
            m = eval_all(models, val, normalize, a.batch_size, scale=s)
            for r in runs:
                result["scale"][r][str(s)] = m[r]
            save()
        print(f"scale {s}: " + "  ".join(f"{r} {result['scale'][r][str(s)]:.2f}" for r in runs), flush=True)
    for name in CORRUPTIONS:
        for sev in SEVERITIES:
            if str(sev) not in result["corruption"][runs[0]][name]:
                m = eval_all(models, val, normalize, a.batch_size, corrupt=make_corrupt(name, sev, val.sizes))
                for r in runs:
                    result["corruption"][r][name][str(sev)] = m[r]
                save()
            print(f"{name} s{sev}: " + "  ".join(f"{r} {result['corruption'][r][name][str(sev)]:.2f}" for r in runs),
                  flush=True)

    for r in runs:  # summary: mean over all corrupted conditions, and per category
        cor = result["corruption"][r]
        allv = [v for c in cor.values() for v in c.values()]
        result.setdefault("summary", {})[r] = {
            "clean": result["clean"][r], "mean_corrupted": float(np.mean(allv)),
            "retention": float(np.mean(allv) / result["clean"][r]),
            **{f"mean_{cat}": float(np.mean([v for c, d in cor.items() if CATEGORY[c] == cat for v in d.values()]))
               for cat in ("noise", "blur", "digital")},
        }
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["summary"], indent=2))
