"""Exploratory analysis of the VOC 2012 + SBD segmentation set: statistics -> results/eda.json,
figures -> figures/eda_*.png and figures/aug_examples.png.

    python -m src.eda
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from . import plotstyle as ps
from .data import CLASSES, IGNORE, NUM_CLASSES, ROOT, TrainTransform, load_pair, read_list, voc_palette

plt = ps.plt
PALETTE = voc_palette()


def colorize(mask):
    return PALETTE[mask]


def overlay(img, mask, alpha=0.6):
    """Blend the class colours over the image (background stays untouched, void is drawn white)."""
    img = np.asarray(img, dtype=np.float32)
    col = colorize(mask).astype(np.float32)
    a = np.where((mask > 0)[..., None], alpha, 0.0)
    return (img * (1 - a) + col * a).astype(np.uint8)


def split_stats(split):
    """One pass over the labels of a split."""
    items = read_list(split)

    def one(item):
        mask = np.asarray(Image.open(os.path.join(ROOT, item[1], item[0] + ".png")))
        counts = np.bincount(mask.ravel(), minlength=256)
        return mask.shape, counts[:NUM_CLASSES], counts[IGNORE], item[1] == "SegmentationClass"

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, items))
    shapes = np.array([r[0] for r in rows])                      # [N,2] (h, w)
    counts = np.stack([r[1] for r in rows]).astype(np.int64)     # [N,21]
    void = np.array([r[2] for r in rows], dtype=np.int64)
    is_voc = np.array([r[3] for r in rows])
    area = shapes.prod(1)
    present = counts > 0
    frac = counts / area[:, None]                                # share of the image covered by each class
    pix = counts.sum(0)
    return {
        "n": len(items), "shapes": shapes, "present": present, "frac": frac, "void": void, "area": area,
        "is_voc": is_voc,
        "summary": {
            "n_images": len(items),
            "n_voc_labels": int(is_voc.sum()), "n_sbd_labels": int((~is_voc).sum()),
            "height_mean": float(shapes[:, 0].mean()), "width_mean": float(shapes[:, 1].mean()),
            "height_min": int(shapes[:, 0].min()), "height_max": int(shapes[:, 0].max()),
            "width_min": int(shapes[:, 1].min()), "width_max": int(shapes[:, 1].max()),
            "landscape_share": float((shapes[:, 1] > shapes[:, 0]).mean()),
            "pixel_share": (pix / pix.sum()).tolist(),
            "images_with_class": present.sum(0).tolist(),
            "median_area_when_present": [float(np.median(frac[present[:, c], c])) if present[:, c].any() else 0.0
                                         for c in range(NUM_CLASSES)],
            "classes_per_image_mean": float(present[:, 1:].sum(1).mean()),
            "classes_per_image_hist": np.bincount(present[:, 1:].sum(1), minlength=7).tolist(),
            "foreground_share_mean": float(1 - frac[:, 0].mean() - (void / area).mean()),
            "void_share_voc_labels": float((void[is_voc] / area[is_voc]).mean()) if is_voc.any() else 0.0,
            "void_share_sbd_labels": float((void[~is_voc] / area[~is_voc]).mean()) if (~is_voc).any() else 0.0,
        },
    }


def fig_class_dist(stats):
    tr, va = stats["train"]["summary"], stats["val"]["summary"]
    order = np.argsort(tr["pixel_share"])[::-1]
    names = [CLASSES[i] for i in order]
    y = np.arange(NUM_CLASSES)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6), sharey=True)
    axes[0].barh(y, 100 * np.array(tr["pixel_share"])[order], color=ps.BLUE)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("% pixel của tập train (thang log)")
    axes[0].set_title("Tỉ lệ pixel theo lớp")
    for yi, v in zip(y, 100 * np.array(tr["pixel_share"])[order]):
        axes[0].text(v * 1.08, yi, f"{v:.2f}".replace(".", ","), va="center", fontsize=7, color=ps.INK_2)
    axes[0].set_xlim(right=200)
    h = 0.38
    axes[1].barh(y - h / 2, 100 * np.array(tr["images_with_class"])[order] / tr["n_images"], h, color=ps.BLUE,
                 label=f"train ({tr['n_images']:,})".replace(",", "."))
    axes[1].barh(y + h / 2, 100 * np.array(va["images_with_class"])[order] / va["n_images"], h, color=ps.AQUA,
                 label=f"val ({va['n_images']:,})".replace(",", "."))
    axes[1].set_xlabel("% ảnh có chứa lớp")
    axes[1].set_title("Tần suất xuất hiện")
    axes[1].set_xlim(0, 50)   # background (100%) is clipped so the 20 object classes stay readable
    axes[1].text(49.5, 0, "100%", ha="right", va="center", fontsize=7, color=ps.SURFACE)
    axes[1].legend(loc="lower right")
    axes[2].barh(y, 100 * np.array(tr["median_area_when_present"])[order], color=ps.YELLOW)
    axes[2].set_xlabel("% diện tích ảnh (trung vị, khi lớp có mặt)")
    axes[2].set_title("Kích thước vùng của lớp")
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig("figures/eda_class_dist.png")
    plt.close(fig)


def fig_stats(stats):
    tr = stats["train"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    hb = axes[0].hexbin(tr["shapes"][:, 1], tr["shapes"][:, 0], gridsize=24, cmap=ps.SEQ_BLUE, bins="log", mincnt=1)
    axes[0].set_xlabel("chiều rộng (px)")
    axes[0].set_ylabel("chiều cao (px)")
    axes[0].set_title("Kích thước ảnh (train)")
    axes[0].set_xlim(100, 512)
    axes[0].set_ylim(100, 512)
    axes[0].set_aspect("equal")
    fig.colorbar(hb, ax=axes[0], label="số ảnh (log)")
    w = 0.38
    for k, (name, color) in enumerate((("train", ps.BLUE), ("val", ps.AQUA))):
        hist = np.array(stats[name]["summary"]["classes_per_image_hist"][1:6], dtype=float)
        axes[1].bar(np.arange(1, 6) + (k - 0.5) * w, 100 * hist / stats[name]["n"], w, color=color, label=name)
    axes[1].set_xlabel("số lớp đối tượng trong một ảnh")
    axes[1].set_ylabel("% ảnh")
    axes[1].set_title("Số lớp trên mỗi ảnh")
    axes[1].legend()
    fg = 1 - tr["frac"][:, 0] - tr["void"] / tr["area"]
    axes[2].hist(100 * fg, bins=40, color=ps.BLUE)
    axes[2].axvline(100 * fg.mean(), color=ps.INK, lw=1, ls="--")
    axes[2].text(100 * fg.mean() + 2, axes[2].get_ylim()[1] * 0.9, f"trung bình {100 * fg.mean():.0f}%", fontsize=8)
    axes[2].set_xlabel("% pixel thuộc đối tượng (không phải nền)")
    axes[2].set_ylabel("số ảnh")
    axes[2].set_title("Tỉ lệ foreground (train)")
    fig.tight_layout()
    fig.savefig("figures/eda_stats.png")
    plt.close(fig)


def fig_samples(seed=3, n=6):
    """Images with their ground truth; chosen to cover many classes."""
    rng = np.random.RandomState(seed)
    items = read_list("val")
    picks = [items[i] for i in rng.choice(len(items), n, replace=False)]
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 3.3))
    seen = set()
    for k, item in enumerate(picks):
        img, mask = load_pair(ROOT, *item)
        mask = np.asarray(mask)
        seen |= set(np.unique(mask).tolist())
        axes[0, k].imshow(img)
        axes[1, k].imshow(colorize(mask))
        for ax in axes[:, k]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    axes[0, 0].set_ylabel("ảnh")
    axes[1, 0].set_ylabel("nhãn")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[c] / 255) for c in sorted(seen - {0, IGNORE})]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=ps.AXIS))
    labels = [CLASSES[c] for c in sorted(seen - {0, IGNORE})] + ["void (255)"]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("figures/eda_samples.png", dpi=ps.PHOTO_DPI)
    plt.close(fig)


def fig_label_sources(n=3):
    """The same image annotated by VOC 2012 (with void borders) and by SBD (no void, coarser contours)."""
    both = [i for i, d in read_list("train") if d == "SegmentationClass"
            and os.path.isfile(os.path.join(ROOT, "SegmentationClassAug", i + ".png"))]
    rng = np.random.RandomState(1)
    picks = rng.choice(both, n, replace=False)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 1.85 * n))
    agree = []
    for r, image_id in enumerate(picks):
        img = Image.open(os.path.join(ROOT, "JPEGImages", image_id + ".jpg"))
        voc = np.asarray(Image.open(os.path.join(ROOT, "SegmentationClass", image_id + ".png")))
        sbd = np.asarray(Image.open(os.path.join(ROOT, "SegmentationClassAug", image_id + ".png")))
        for ax, im, title in zip(axes[r], (img, colorize(voc), colorize(sbd)),
                                 ("ảnh", "nhãn VOC 2012 (có void)", "nhãn SBD (không void)")):
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(title)
    for image_id in both:  # agreement between the two annotations on non-void pixels
        voc = np.asarray(Image.open(os.path.join(ROOT, "SegmentationClass", image_id + ".png")))
        sbd = np.asarray(Image.open(os.path.join(ROOT, "SegmentationClassAug", image_id + ".png")))
        valid = voc != IGNORE
        agree.append(((voc == sbd) & valid).sum() / valid.sum())
    fig.tight_layout()
    fig.savefig("figures/eda_label_sources.png", dpi=ps.PHOTO_DPI)
    plt.close(fig)
    return {"n_images_with_both_labels": len(both), "pixel_agreement_mean": float(np.mean(agree))}


def fig_augmentation(seed=0, n_aug=4):
    torch.manual_seed(seed)
    items = read_list("train")
    tf = TrainTransform(512)
    picks = [items[i] for i in (25, 4000)]
    fig, axes = plt.subplots(len(picks), n_aug + 1, figsize=(2.05 * (n_aug + 1), 2.1 * len(picks)))
    for r, item in enumerate(picks):
        img, mask = load_pair(ROOT, *item)
        axes[r, 0].imshow(overlay(img, np.asarray(mask)))
        axes[r, 0].set_title(f"gốc {img.size[0]}×{img.size[1]}" if r == 0 else f"{img.size[0]}×{img.size[1]}")
        for k in range(n_aug):
            x, m = tf(img, mask)
            x, m = x.permute(1, 2, 0).numpy(), m.numpy()
            vis = overlay(x, m)
            vis[m == IGNORE] = (vis[m == IGNORE] * 0.35 + 255 * 0.65).astype(np.uint8)  # padding / void: washed out
            axes[r, k + 1].imshow(vis)
            if r == 0:
                axes[r, k + 1].set_title(f"augment {k + 1} (512×512)")
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.tight_layout()
    fig.savefig("figures/aug_examples.png", dpi=ps.PHOTO_DPI)
    plt.close(fig)


def pixel_stats(n=1500, seed=0):
    rng = np.random.RandomState(seed)
    items = read_list("train")
    s, s2, cnt = np.zeros(3), np.zeros(3), 0
    for i in rng.choice(len(items), n, replace=False):
        x = np.asarray(Image.open(os.path.join(ROOT, "JPEGImages", items[i][0] + ".jpg")).convert("RGB"),
                       dtype=np.float64).reshape(-1, 3) / 255
        s += x.sum(0)
        s2 += (x ** 2).sum(0)
        cnt += len(x)
    mean = s / cnt
    return mean.tolist(), np.sqrt(s2 / cnt - mean ** 2).tolist()


if __name__ == "__main__":
    ps.apply()
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    stats = {name: split_stats(name) for name in ("train", "minival", "val")}
    with open(os.path.join(ROOT, "meta.json")) as f:
        meta = json.load(f)
    mean, std = pixel_stats()
    eda = {"classes": list(CLASSES), "meta": meta, "channel_mean": mean, "channel_std": std,
           "splits": {k: v["summary"] for k, v in stats.items()}}
    eda["label_sources"] = fig_label_sources()
    fig_class_dist(stats)
    fig_stats(stats)
    fig_samples()
    fig_augmentation()
    with open("results/eda.json", "w") as f:
        json.dump(eda, f, indent=2)
    tr = eda["splits"]["train"]
    print(json.dumps({k: v for k, v in tr.items() if not isinstance(v, list)}, indent=2))
    print("pixel share:", {c: round(100 * s, 2) for c, s in zip(CLASSES, tr["pixel_share"])})
    print("label sources:", eda["label_sources"])
