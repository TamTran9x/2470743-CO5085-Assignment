"""Demo: segment any image with the trained models and save a side-by-side figure.

    python -m scripts.demo --images my_photo.jpg another.png
    python -m scripts.demo --images data/voc/JPEGImages/2007_000033.jpg --runs segformer_b2 deeplabv3plus_r50

For every image it writes demo_out/<name>.png: the photo, then one overlay per model with the
detected classes and the inference time. Images larger than 512 px are resized (long side 512).
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from src import plotstyle as ps
from src.data import CLASSES, Normalize, upsample_logits, voc_palette
from src.models import load_trained

PALETTE = voc_palette()


_WARM = set()  # (model, input shape) pairs already run once: cudnn tunes its algorithms per shape


@torch.inference_mode()
def segment(model, image, normalize):
    x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].cuda()
    if (id(model), x.shape) not in _WARM:   # untimed first pass, so the reported latency is steady-state
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(normalize(x))
        _WARM.add((id(model), x.shape))
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(normalize(x))
    pred = upsample_logits(logits, x.shape[-2:]).argmax(1)[0]
    torch.cuda.synchronize()
    return pred.cpu().numpy(), 1000 * (time.time() - t0)


def overlay(image, mask, alpha=0.62):
    img = np.asarray(image, dtype=np.float32)
    a = np.where((mask > 0)[..., None], alpha, 0.0)
    return (img * (1 - a) + PALETTE[mask] * a).astype(np.uint8)


def pick_val_images(class_names, root="data/voc"):
    """First VOC 2012 val image (by id) in which each class covers 10-60% of the picture. The choice
    depends only on the labels, never on how well a model does."""
    from src.data import read_list
    wanted = {CLASSES.index(c): None for c in class_names}
    for image_id, label_dir in read_list("val", root):
        mask = np.asarray(Image.open(os.path.join(root, label_dir, image_id + ".png")))
        for c in wanted:
            if wanted[c] is None and 0.10 < (mask == c).mean() < 0.60 and image_id not in wanted.values():
                wanted[c] = image_id
                break
        if all(wanted.values()):
            break
    return [os.path.join(root, "JPEGImages", i + ".jpg") for i in wanted.values() if i]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", nargs="*", default=[])
    p.add_argument("--val-classes", nargs="*", default=[],
                   help="also demo the first val image of each of these classes (e.g. bus cat person)")
    p.add_argument("--runs", nargs="+", default=["deeplabv3plus_r50", "segformer_b2"])
    p.add_argument("--out", default="demo_out")
    p.add_argument("--gif", help="also write an animated GIF cycling through the results")
    a = p.parse_args()
    a.images = list(a.images) + pick_val_images(a.val_classes)
    if not a.images:
        p.error("give --images and/or --val-classes")
    ps.apply()
    os.makedirs(a.out, exist_ok=True)
    outputs = []
    normalize = Normalize()
    models = {r: load_trained(r) for r in a.runs}

    for path in a.images:
        image = Image.open(path).convert("RGB")
        if max(image.size) > 512:
            s = 512 / max(image.size)
            image = image.resize((round(image.width * s), round(image.height * s)), Image.BILINEAR)
        fig, axes = ps.plt.subplots(1, len(models) + 1, figsize=(3.6 * (len(models) + 1), 3.4))
        axes[0].imshow(image)
        axes[0].set_title(os.path.basename(path), fontsize=9)
        seen = set()
        for ax, (run, model) in zip(axes[1:], models.items()):
            mask, ms = segment(model, image, normalize)
            seen |= set(np.unique(mask).tolist()) - {0}
            ax.imshow(overlay(image, mask))
            ax.set_title(f"{ps.RUN_LABEL[run]} · {ms:.0f} ms", fontsize=9)
            found = ", ".join(CLASSES[c] for c in np.unique(mask) if c) or "chỉ có nền"
            print(f"{os.path.basename(path)} | {ps.RUN_LABEL[run]}: {found} ({ms:.0f} ms)")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        handles = [ps.plt.Rectangle((0, 0), 1, 1, color=PALETTE[c] / 255) for c in sorted(seen)]
        if handles:
            fig.legend(handles, [CLASSES[c] for c in sorted(seen)], loc="lower center", ncol=len(handles), fontsize=8)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        out = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0] + ".png")
        fig.savefig(out, dpi=110)
        ps.plt.close(fig)
        outputs.append(out)
        print("  ->", out)

    if a.gif:
        frames = [Image.open(o).convert("RGB") for o in outputs]
        w, h = max(f.width for f in frames), max(f.height for f in frames)
        canvas = []
        for f in frames:   # same size for every frame, white background
            c = Image.new("RGB", (w, h), "white")
            c.paste(f, ((w - f.width) // 2, (h - f.height) // 2))
            canvas.append(c.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        canvas[0].save(a.gif, save_all=True, append_images=canvas[1:], duration=1800, loop=0)
        print("  ->", a.gif)
