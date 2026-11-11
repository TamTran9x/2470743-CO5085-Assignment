"""Exploratory data analysis of CIFAR-100: split sizes, class balance, pixel statistics,
sample grid and a visualisation of the augmentation pipeline.

    python -m src.eda
"""
import json
import os

import numpy as np
import torch

from . import plotstyle as ps
from .data import IMAGENET_MEAN, IMAGENET_STD, GPULoader, load_cifar100, random_crop_flip, stratified_split

plt = ps.plt


def sample_grid(ds, path):
    rng = np.random.RandomState(1)
    fig, axes = plt.subplots(10, 10, figsize=(9, 10))
    for c, ax in enumerate(axes.flat):
        idx = rng.choice(np.where(ds["y_train"] == c)[0])
        ax.imshow(ds["x_train"][idx], interpolation="nearest")
        ax.set_title(ds["classes"][c].replace("_", " "), fontsize=6.5, pad=2, color=ps.INK_2,
                     fontweight="normal")
        ax.axis("off")
    fig.subplots_adjust(wspace=0.08, hspace=0.35)
    fig.savefig(path)
    plt.close(fig)


def pixel_histogram(ds, path):
    x = ds["x_train"].reshape(-1, 3)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    colors = ["#e34948", "#008300", "#2a78d6"]
    for ch, (name, col) in enumerate(zip(["R", "G", "B"], colors)):
        hist = np.bincount(x[:, ch], minlength=256) / len(x)
        ax.plot(np.arange(256), hist, color=col, lw=1.6, label=f"Kênh {name}")
    ax.set_xlim(0, 255)
    ax.set_xlabel("Giá trị pixel")
    ax.set_ylabel("Tỉ lệ")
    ax.legend(ncol=3, loc="upper right")
    ax.set_title("Phân bố cường độ pixel theo kênh (tập train)", loc="left")
    fig.savefig(path)
    plt.close(fig)


def augmentation_figure(ds, path, img_size=224):
    torch.manual_seed(3)
    idx = [10, 2024, 3333, 4100]
    x = torch.from_numpy(ds["x_train"][idx]).permute(0, 3, 1, 2).cuda()
    loader = GPULoader(ds["x_train"][:1], ds["y_train"][:1], 1, img_size, train=False)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    cols = ["Ảnh gốc 32×32", "Crop+flip #1", "Crop+flip #2", "Crop+flip #3", f"Đầu vào {img_size}×{img_size}"]
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(8, 6.6))
    for r in range(len(idx)):
        augs = [random_crop_flip(x[r:r + 1])[0] for _ in range(3)]
        model_in = loader.preprocess(augs[0][None])[0].cpu() * std + mean
        panels = [x[r]] + augs
        for c, ax in enumerate(axes[r]):
            img = panels[c].permute(1, 2, 0).cpu().numpy() if c < 4 else model_in.permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(img, interpolation="nearest")
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=8, color=ps.INK_2, fontweight="normal")
        axes[r][0].text(-4, 16, ds["classes"][ds["y_train"][idx[r]]], rotation=90, ha="right", va="center",
                        fontsize=8, color=ps.INK_2)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.savefig(path)
    plt.close(fig)


def main(out_fig="figures", out_json="results/eda.json"):
    ps.apply()
    os.makedirs(out_fig, exist_ok=True)
    ds = load_cifar100("data")
    tr_idx, va_idx = stratified_split(ds["y_train"], val_per_class=50, seed=0)
    x = ds["x_train"].astype(np.float32) / 255.0
    stats = {
        "n_train_official": int(len(ds["y_train"])),
        "n_test": int(len(ds["y_test"])),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_classes": 100,
        "n_superclasses": 20,
        "image_shape": list(ds["x_train"].shape[1:]),
        "train_per_class": np.bincount(ds["y_train"][tr_idx]).tolist(),
        "val_per_class": np.bincount(ds["y_train"][va_idx]).tolist(),
        "test_per_class": np.bincount(ds["y_test"]).tolist(),
        "channel_mean": x.mean(axis=(0, 1, 2), dtype=np.float64).round(4).tolist(),
        "channel_std": x.std(axis=(0, 1, 2), dtype=np.float64).round(4).tolist(),
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "superclasses": {ds["superclasses"][s]: [ds["classes"][c] for c in range(100) if ds["fine_to_coarse"][c] == s]
                         for s in range(20)},
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print({k: v for k, v in stats.items() if not isinstance(v, (list, dict)) or k.startswith("channel")})
    sample_grid(ds, os.path.join(out_fig, "eda_samples.png"))
    pixel_histogram(ds, os.path.join(out_fig, "eda_pixel_hist.png"))
    augmentation_figure(ds, os.path.join(out_fig, "aug_examples.png"))


if __name__ == "__main__":
    main()
