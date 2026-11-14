"""Effective receptive field (ERF) of the encoder stages and of the decoder output, as in Fig. 3 of
the SegFormer paper (method of Luo et al., NeurIPS 2016).

For a feature map F, the ERF is the gradient of the centre activation sum_c F[c, H/2, W/2] with
respect to the input image; its magnitude (summed over RGB) is averaged over val images.
Writes results/erf.json, results/erf.npz and figures/fig_erf.png.

    python -m src.erf
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import plotstyle as ps
from .data import EvalSet, Normalize
from .models import load_trained

SIZE = 512
TARGETS = ("stage 1", "stage 2", "stage 3", "stage 4", "decoder")


def erf_maps(model, images, normalize, batch_size=2):
    """images: uint8 [N,3,SIZE,SIZE]. Returns [5,SIZE,SIZE] mean |d centre / d input| per target."""
    acc = torch.zeros((len(TARGETS), SIZE, SIZE), device="cuda")
    for i in range(0, len(images), batch_size):
        x = normalize(images[i:i + batch_size].cuda()).requires_grad_(True)
        feats = model.encoder(x)
        outs = list(feats) + [model.decode_head(feats)]
        for k, f in enumerate(outs):
            h, w = f.shape[-2:]
            # BatchNorm is in eval mode, so each image's centre activation depends on that image only
            grad, = torch.autograd.grad(f[:, :, h // 2, w // 2].sum(), x, retain_graph=k < len(outs) - 1)
            acc[k] += grad.abs().sum(1).sum(0)
    return (acc / len(images)).cpu().numpy()


def erf_stats(m):
    """Size of the ERF: side of the centred square holding 50% / 90% of the gradient mass, and the
    share of the image whose gradient exceeds 1% / 10% of the peak."""
    n = m.shape[0]
    yy, xx = np.mgrid[:n, :n]
    d = np.maximum(np.abs(yy - n // 2), np.abs(xx - n // 2)).ravel()
    order = np.argsort(d, kind="stable")
    cum = np.cumsum(m.ravel()[order]) / m.sum()
    side = lambda q: int(2 * d[order][np.searchsorted(cum, q)] + 1)  # noqa: E731
    return {"side_50": side(0.5), "side_90": side(0.9),
            "area_gt_1pct": float((m > 0.01 * m.max()).mean()), "area_gt_10pct": float((m > 0.1 * m.max()).mean())}


def figure(maps, stats, runs, path):
    fig, axes = ps.plt.subplots(len(runs), len(TARGETS), figsize=(2.05 * len(TARGETS), 2.15 * len(runs)))
    axes = np.atleast_2d(axes)
    for r, run in enumerate(runs):
        for k, name in enumerate(TARGETS):
            m = np.asarray(maps[run][k], dtype=np.float32)
            ax = axes[r, k]
            # log scale over 3 decades: the ERF decays quickly, a linear scale would show a single dot
            ax.imshow(np.log10(np.clip(m / m.max(), 1e-3, 1)), cmap=ps.HEAT, vmin=-3, vmax=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(name)
            if k == 0:
                ax.set_ylabel(ps.RUN_LABEL[run], fontsize=9)
            s = stats[run][name]
            ax.text(0.03, 0.03, f"50%: {s['side_50']} px · 90%: {s['side_90']} px", transform=ax.transAxes,
                    color="white", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=ps.PHOTO_DPI)
    ps.plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["deeplabv3plus_r50", "resnet50_mlp", "segformer_b2"])
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--replot", action="store_true", help="redraw the figure from results/erf.npz and erf.json")
    a = p.parse_args()
    ps.apply()
    if a.replot:
        with open("results/erf.json") as f:
            saved = json.load(f)
        with np.load("results/erf.npz") as z:
            figure({r: z[r] for r in saved["runs"]}, saved["runs"], list(saved["runs"]), "figures/fig_erf.png")
        raise SystemExit
    torch.backends.cudnn.benchmark = True
    val = EvalSet("val", subset=a.n)
    # stretch each image to the full 512x512 canvas so that the centre always lies inside real content
    images = torch.stack([F.interpolate(val.images[k:k + 1, :, :h, :w].float(), size=(SIZE, SIZE), mode="bilinear",
                                        align_corners=False)[0] for k, (h, w) in enumerate(val.sizes)])
    images = images.round().clamp(0, 255).to(torch.uint8)
    normalize = Normalize()
    runs = [r for r in a.runs if os.path.isfile(os.path.join("checkpoints", r, "final.pt"))]
    maps, stats = {}, {}
    for run in runs:
        model = load_trained(run)
        maps[run] = erf_maps(model, images, normalize)
        stats[run] = {name: erf_stats(maps[run][k]) for k, name in enumerate(TARGETS)}
        print(run, {name: (s["side_50"], s["side_90"]) for name, s in stats[run].items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs("figures", exist_ok=True)
    figure(maps, stats, runs, "figures/fig_erf.png")
    np.savez_compressed("results/erf.npz", **{r: (m / m.max(axis=(1, 2), keepdims=True)).astype(np.float16)
                                              for r, m in maps.items()})
    with open("results/erf.json", "w") as f:
        json.dump({"n_images": len(images), "input_size": SIZE, "targets": list(TARGETS), "runs": stats}, f, indent=2)
