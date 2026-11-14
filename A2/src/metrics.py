"""Segmentation metrics from confusion matrices, and the shared evaluation loop.

A confusion matrix C[g, p] counts pixels of ground-truth class g predicted as p (void pixels are
skipped). All scores are derived from it:
    IoU_c  = TP / (TP + FP + FN)        Dice_c = 2TP / (2TP + FP + FN)
    mIoU / mDice = mean over the 21 classes of the dataset-level IoU / Dice (the VOC convention).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import IGNORE, NUM_CLASSES, upsample_logits


def confusion(pred, target, num_classes=NUM_CLASSES):
    """One matrix for a set of pixels. pred/target: 1-D tensors, void already removed."""
    return torch.bincount(target.long() * num_classes + pred.long(), minlength=num_classes ** 2).view(
        num_classes, num_classes)


def batch_confusion(pred, target, num_classes=NUM_CLASSES):
    """Per-image matrices [B,C,C] for pred/target of shape [B,H,W]."""
    b = pred.shape[0]
    offset = torch.arange(b, device=pred.device).view(b, 1, 1) * num_classes ** 2
    idx = target.long() * num_classes + pred.long() + offset
    return torch.bincount(idx[target != IGNORE], minlength=b * num_classes ** 2).view(b, num_classes, num_classes)


def scores(conf):
    """conf: [C,C] array. Returns dataset-level scores in percent."""
    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diag(conf)
    gt, pr = conf.sum(1), conf.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (gt + pr - tp)
        dice = 2 * tp / (gt + pr)
        acc = tp / gt
    return {
        "miou": 100 * float(np.nanmean(iou)),
        "mdice": 100 * float(np.nanmean(dice)),
        "pixel_acc": 100 * float(tp.sum() / conf.sum()),
        "mean_acc": 100 * float(np.nanmean(acc)),
        "iou": (100 * iou).tolist(),
        "dice": (100 * dice).tolist(),
    }


def miou(conf):
    return scores(conf)["miou"]


@torch.inference_mode()
def evaluate(model, evalset, normalize, batch_size=16, scale=1.0, corrupt=None, keep_preds=False, amp=True):
    """Single-scale inference on a padded 512x512 canvas.

    scale   : resize the network input by this factor (logits are resized back to 512x512)
    corrupt : optional fn(uint8 batch [B,3,H,W], first image index) -> uint8 batch (robustness tests)
    Returns dict(conf [N,C,C], band_conf [C,C], preds uint8 [N,H,W] | None, seconds).
    """
    model.eval()
    confs, preds = [], []
    band_conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    t0, seen = time.time(), 0
    for x, y, band in evalset.batches(batch_size):
        if corrupt is not None:
            x = corrupt(x, seen)
        x = normalize(x)
        if scale != 1.0:
            x = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False, antialias=scale < 1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        pred = upsample_logits(logits, y.shape[-2:]).argmax(1)
        confs.append(batch_confusion(pred, y))
        band_conf += confusion(pred[band], y[band])
        if keep_preds:
            preds.append(pred.to(torch.uint8).cpu())
        seen += len(y)
    torch.cuda.synchronize()
    return {"conf": torch.cat(confs).cpu().numpy(), "band_conf": band_conf.cpu().numpy(),
            "preds": torch.cat(preds).numpy() if keep_preds else None, "seconds": time.time() - t0}


def bootstrap_miou(conf_a, conf_b=None, n_boot=2000, seed=0):
    """Image-level bootstrap of the dataset mIoU. With two models the same resampled images are
    used for both (paired), and the CI of the difference a - b and a two-sided p-value are returned."""
    rng = np.random.RandomState(seed)
    n = len(conf_a)
    a = np.asarray(conf_a, dtype=np.float64).reshape(n, -1)
    b = None if conf_b is None else np.asarray(conf_b, dtype=np.float64).reshape(n, -1)
    c = int(round(a.shape[1] ** 0.5))

    def fast_miou(flat):
        m = flat.reshape(-1, c, c)
        tp = np.einsum("bii->bi", m)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 100 * np.nanmean(tp / (m.sum(1) + m.sum(2) - tp), axis=1)

    stats_a, stats_b = [], []
    for start in range(0, n_boot, 100):
        k = min(100, n_boot - start)
        w = np.stack([np.bincount(rng.randint(0, n, n), minlength=n) for _ in range(k)]).astype(np.float64)
        stats_a.append(fast_miou(w @ a))
        if b is not None:
            stats_b.append(fast_miou(w @ b))
    sa = np.concatenate(stats_a)
    out = {"ci_low": float(np.percentile(sa, 2.5)), "ci_high": float(np.percentile(sa, 97.5))}
    if b is not None:
        d = sa - np.concatenate(stats_b)
        out = {"diff": float(miou(a.sum(0).reshape(c, c)) - miou(b.sum(0).reshape(c, c))),
               "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
               "p_value": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), "n_boot": n_boot}
    return out
