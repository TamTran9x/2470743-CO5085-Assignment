"""Aggregate the runs: summary table, statistical tests, error analysis and every result figure.

Reads  results/<run>/{metrics.json,history.csv,preds.npz}, checkpoints/<run>/val_pred.npz,
       results/{eda,robustness,erf}.json (the last two are optional)
Writes results/summary.{csv,md}, results/analysis.json, figures/fig_*.png

    python -m src.analyze
"""
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

from . import plotstyle as ps
from .data import CLASSES, IGNORE, NUM_CLASSES, ROOT, voc_palette
from .metrics import bootstrap_miou, scores

plt = ps.plt
PALETTE = voc_palette()
PAIRS = [("segformer_b2", "deeplabv3plus_r50"), ("segformer_b2", "resnet50_mlp"),
         ("deeplabv3plus_r50", "resnet50_mlp"), ("segformer_b1", "segformer_b0"),
         ("segformer_b2", "segformer_b1"), ("segformer_b0", "segformer_b0_scratch"),
         ("deeplabv3plus_r50", "segformer_b1"), ("segformer_b0", "resnet50_mlp")]


def vn(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def load_runs(out="results"):
    runs = [r for r in ps.RUN_ORDER if os.path.isfile(os.path.join(out, r, "metrics.json"))]
    M, H, P = {}, {}, {}
    for r in runs:
        with open(os.path.join(out, r, "metrics.json")) as f:
            M[r] = json.load(f)
        H[r] = pd.read_csv(os.path.join(out, r, "history.csv"))
        P[r] = np.load(os.path.join(out, r, "preds.npz"))
    return runs, M, H, P


def per_image_miou(conf):
    """mIoU of each image over the classes present in its ground truth. A class that is only
    predicted still costs IoU through the pixels it takes, but a few stray pixels of a wrong
    class do not add a zero to the average."""
    conf = conf.astype(np.float64)
    tp = np.einsum("nii->ni", conf)
    gt = conf.sum(2)
    union = gt + conf.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * np.nanmean(np.where(gt > 0, tp / union, np.nan), axis=1)


# ----------------------------------------------------------------------------- tables
def build_summary(runs, M, ci):
    rows = []
    for r in runs:
        m = M[r]
        rows.append({
            "run": r, "label": ps.RUN_LABEL[r], "family": m["family"], "encoder": m["encoder"],
            "decoder": m["decoder"], "pretrained": m["pretrained"],
            "miou": m["miou"], "ci_low": ci[r]["ci_low"], "ci_high": ci[r]["ci_high"], "mdice": m["mdice"],
            "pixel_acc": m["pixel_acc"], "band_miou": m["band_miou"], "minival_miou": m["minival_miou"],
            "params_M": m["total_params"] / 1e6, "encoder_M": m["encoder_params"] / 1e6,
            "decoder_M": m["decoder_params"] / 1e6, "gmacs": m["gmacs_512"],
            "train_min": m["train_time_s"] / 60, "train_img_s": m["train_throughput"],
            "peak_mem_gb": m["peak_mem_gb"], "fps": m["fps_bf16"], "fps_fp32": m["fps_fp32"],
            "infer_img_s": m["infer_throughput"], "iters": m["iters"], "epochs": m["epochs"],
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/summary.csv", index=False, float_format="%.4f")
    with open("results/summary.md", "w", encoding="utf-8") as f:
        f.write("| Mô hình | Pretrained | mIoU (%) | mDice (%) | Pixel acc (%) | Boundary mIoU (%) | Tham số (M) | "
                "GMACs | FPS | Train (phút) | Mem (GB) |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            f.write(f"| {r.label} | {'IN-1k' if r.pretrained else '–'} | {vn(r.miou)} | {vn(r.mdice)} | "
                    f"{vn(r.pixel_acc)} | {vn(r.band_miou)} | {vn(r.params_M)} | {vn(r.gmacs, 1)} | {vn(r.fps, 1)} | "
                    f"{vn(r.train_min, 1)} | {vn(r.peak_mem_gb)} |\n")
    return df


# ----------------------------------------------------------------------------- figures
def fig_miou(df):
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = np.arange(len(df))
    ax.bar(x, df.miou, color=[ps.RUN_COLOR[r] for r in df.run], width=0.62,
           yerr=[df.miou - df.ci_low, df.ci_high - df.miou], error_kw={"ecolor": ps.INK_2, "lw": 1, "capsize": 3})
    for xi, v, hi in zip(x, df.miou, df.ci_high):
        ax.text(xi, hi + 1.0, vn(v), ha="center", fontsize=8.5, color=ps.INK)
    ax.set_xticks(x, [l.replace(" (", "\n(").replace(" + ", "\n+ ") for l in df.label], fontsize=8)
    ax.set_ylabel("mIoU trên VOC 2012 val (%)")
    ax.set_ylim(0, max(df.miou) + 9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig("figures/fig_miou.png")
    plt.close(fig)


def fig_curves(runs, H):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3))
    for r in runs:
        kw = {"color": ps.RUN_COLOR[r], "ls": ps.RUN_STYLE.get(r, "-"), "label": ps.RUN_LABEL[r]}
        axes[0].plot(H[r].iter, H[r].minival_miou, marker="o", ms=3, **kw)
        axes[1].plot(H[r].iter, H[r].train_loss, marker="o", ms=3, **kw)
    axes[0].set_ylabel("mIoU trên minival (%)")
    axes[0].set_title("Chất lượng theo số vòng lặp")
    axes[1].set_ylabel("cross-entropy trung bình (train, thang log)")
    axes[1].set_yscale("log")
    axes[1].set_yticks([0.15, 0.2, 0.3, 0.5, 1.0, 2.0], ["0,15", "0,2", "0,3", "0,5", "1", "2"])
    axes[1].set_yticks([], minor=True)
    axes[1].set_title("Loss huấn luyện")
    for ax in axes:
        ax.set_xlabel("vòng lặp (batch 8)")
    axes[0].legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig("figures/fig_curves.png")
    plt.close(fig)


def fig_per_class(M, a, b):
    ia, ib = np.array(M[a]["iou"]), np.array(M[b]["iou"])
    order = np.argsort(ia - ib)
    y = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for yi, c in zip(y, order):
        ax.plot([ib[c], ia[c]], [yi, yi], color=ps.AXIS, lw=1.5, zorder=1)
        ax.text(max(ia[c], ib[c]) + 1.2, yi, ("+" if ia[c] >= ib[c] else "−") + vn(abs(ia[c] - ib[c]), 1),
                va="center", fontsize=7.5, color=ps.INK_2)
    ax.scatter(ib[order], y, color=ps.RUN_COLOR[b], s=34, zorder=2, label=ps.RUN_LABEL[b])
    ax.scatter(ia[order], y, color=ps.RUN_COLOR[a], s=34, zorder=2, label=ps.RUN_LABEL[a])
    ax.set_yticks(y, [CLASSES[c] for c in order])
    ax.set_xlabel("IoU theo lớp trên VOC 2012 val (%)")
    ax.set_xlim(min(ia.min(), ib.min()) - 4, 103)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig("figures/fig_per_class.png")
    plt.close(fig)


def fig_cost(df):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    d = df[df.pretrained]
    for ax, (col, xlabel, log) in zip(axes, (("params_M", "số tham số (triệu)", True),
                                             ("gmacs", "GMACs cho ảnh 512×512", True),
                                             ("fps", "FPS (batch 1, 512×512, bfloat16)", False))):
        for _, r in d.iterrows():
            ax.scatter(r[col], r.miou, s=70, color=ps.RUN_COLOR[r.run], edgecolor=ps.SURFACE, zorder=3)
            # the two runs with the same mIoU get their labels above / below the point so they never collide
            offset, ha = {"segformer_b1": ((0, 7), "center"), "deeplabv3plus_r50": ((0, -13), "center")}.get(
                r.run, ((6, -3), "left"))
            ax.annotate(r.label.replace("SegFormer-", "SF-").replace("DeepLabV3+ R50", "DLv3+ R50")
                        .replace("ResNet-50 + MLP", "R50+MLP"), (r[col], r.miou), textcoords="offset points",
                        xytext=offset, ha=ha, fontsize=7.5, color=ps.INK_2)
        if log:
            ax.set_xscale("log")
            ticks = [t for t in (2, 3, 5, 10, 20, 30, 50) if d[col].min() * 0.7 <= t <= d[col].max() * 1.6]
            ax.set_xticks(ticks, [str(t) for t in ticks])
            ax.set_xticks([], minor=True)
            ax.set_xlim(d[col].min() * 0.7, d[col].max() * 1.9)
        else:
            ax.margins(x=0.22)
        ax.set_xlabel(xlabel)
        ax.margins(y=0.12)
    axes[0].set_ylabel("mIoU (%)")
    fig.tight_layout()
    fig.savefig("figures/fig_cost.png")
    plt.close(fig)


def confusion_rates(conf):
    """Row-normalised class confusion (share of the ground-truth pixels of a class), diagonal removed."""
    c = conf.astype(np.float64)
    c = c / c.sum(1, keepdims=True)
    np.fill_diagonal(c, 0)
    return 100 * c


def fig_confusion(P, a, b):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1))
    mats = [confusion_rates(P[r]["conf"].sum(0)) for r in (b, a)]
    vmax = max(m.max() for m in mats)
    for ax, r, m in zip(axes, (b, a), mats):
        im = ax.imshow(m, cmap=ps.SEQ_BLUE, vmin=0, vmax=vmax)
        ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=90, fontsize=7)
        ax.set_yticks(range(NUM_CLASSES), CLASSES, fontsize=7)
        ax.set_title(ps.RUN_LABEL[r])
        ax.set_xlabel("dự đoán")
        ax.grid(False)
    axes[0].set_ylabel("nhãn thật")
    fig.colorbar(im, ax=axes, shrink=0.75, label="% pixel của lớp thật bị gán nhầm")
    fig.savefig("figures/fig_confusion.png")
    plt.close(fig)


def fig_size_vs_iou(M, eda, a, b):
    area = 100 * np.array(eda["splits"]["train"]["median_area_when_present"])[1:]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    out = {}
    for r in (b, a):
        iou = np.array(M[r]["iou"])[1:]
        ax.scatter(area, iou, s=30, color=ps.RUN_COLOR[r], label=ps.RUN_LABEL[r], zorder=3)
        rho, p = spearmanr(area, iou)
        out[r] = {"spearman": float(rho), "p_value": float(p)}
    iou_a = np.array(M[a]["iou"])[1:]
    for c in np.argsort(iou_a)[:6].tolist() + np.argsort(area)[-3:].tolist():
        ax.annotate(CLASSES[c + 1], (area[c], iou_a[c]), textcoords="offset points", xytext=(5, 4), fontsize=7.5,
                    color=ps.INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("diện tích trung vị của lớp trong ảnh (% ảnh, thang log)")
    ax.set_ylabel("IoU theo lớp (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/fig_size_vs_iou.png")
    plt.close(fig)
    return out


def colorize(mask):
    return PALETTE[mask]


def fig_qualitative(P, a, b, extra=None, ckpt_dir="checkpoints"):
    """Images where the two main models differ most (both directions) and where both fail."""
    preds = {r: np.load(os.path.join(ckpt_dir, r, "val_pred.npz")) for r in (b, a) + ((extra,) if extra else ())}
    ids = [str(i) for i in preds[a]["ids"]]
    ma, mb = per_image_miou(P[a]["conf"]), per_image_miou(P[b]["conf"])
    conf = P[a]["conf"]
    fg = conf[:, 1:, :].sum((1, 2)) / conf.sum((1, 2))            # share of object pixels in the ground truth
    ok = (fg > 0.05) & (fg < 0.8)
    diff = np.where(ok, ma - mb, 0)
    groups = [(f"{ps.RUN_LABEL[a]} tốt hơn", np.argsort(-diff)[:2]),
              (f"{ps.RUN_LABEL[b]} tốt hơn", np.argsort(diff)[:2]),
              ("cả hai cùng sai", np.argsort(np.where(ok, np.maximum(ma, mb), 999))[:2])]
    cols = ["ảnh", "nhãn thật"] + [ps.RUN_LABEL[r] for r in preds]
    n_rows = sum(len(g[1]) for g in groups)
    fig, axes = plt.subplots(n_rows, len(cols), figsize=(1.95 * len(cols), 1.5 * n_rows))
    row, chosen, seen = 0, [], set()
    for title, idxs in groups:
        for k in idxs:
            image_id = ids[k]
            img = np.asarray(Image.open(os.path.join(ROOT, "JPEGImages", image_id + ".jpg")).convert("RGB"))
            gt = np.asarray(Image.open(os.path.join(ROOT, "SegmentationClass", image_id + ".png")))
            h, w = gt.shape
            panels = [img, colorize(gt)]
            seen |= set(np.unique(gt).tolist())
            for r in preds:
                p = preds[r]["preds"][k][:h, :w]
                seen |= set(np.unique(p).tolist())
                panels.append(colorize(p))
            per_run = {b: mb[k], a: ma[k]}
            for c, (ax, im) in enumerate(zip(axes[row], panels)):
                ax.imshow(im)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if row == 0:
                    ax.set_title(cols[c], fontsize=9)
                if c >= 2:
                    r = list(preds)[c - 2]
                    if r in per_run:
                        ax.text(0.03, 0.04, f"mIoU {vn(per_run[r], 1)}", transform=ax.transAxes, color="white",
                                fontsize=7.5, bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.5, "lw": 0})
            axes[row, 0].set_ylabel(title, fontsize=7.5)
            chosen.append({"group": title, "id": image_id, a: float(ma[k]), b: float(mb[k])})
            row += 1
    cls = sorted(seen - {0, IGNORE})
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[c] / 255) for c in cls]
    fig.legend(handles, [CLASSES[c] for c in cls], loc="lower center", ncol=min(len(cls), 9), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("figures/fig_qualitative.png", dpi=ps.PHOTO_DPI)
    plt.close(fig)
    return chosen


def fig_robustness(rob):
    runs = [r for r in ps.RUN_ORDER if r in rob["runs"]]
    names = list(rob["corruption"][runs[0]])
    sev = [0] + rob["severities"]
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.4), sharey=True)
    for ax, name in zip(axes.ravel(), names):
        for r in runs:
            y = [rob["clean"][r]] + [rob["corruption"][r][name][str(s)] for s in rob["severities"]]
            ax.plot(sev, y, marker="o", ms=3, color=ps.RUN_COLOR[r], label=ps.RUN_LABEL[r])
        ax.set_title(f"{name.replace('_', ' ')} ({rob['category'][name]})", fontsize=8.5)
        ax.set_xticks(sev, ["gốc"] + [str(s) for s in rob["severities"]])
    ax = axes.ravel()[-1]
    x = np.arange(len(runs))
    ret = [100 * rob["summary"][r]["retention"] for r in runs]
    ax.bar(x, ret, color=[ps.RUN_COLOR[r] for r in runs], width=0.65)
    for xi, v in zip(x, ret):
        ax.text(xi, v + 2, vn(v, 0), ha="center", fontsize=7.5)
    ax.set_xticks(x, [ps.RUN_LABEL[r].replace("SegFormer-", "SF-").replace("DeepLabV3+ R50", "DLv3+")
                      .replace("ResNet-50 + MLP", "R50+MLP") for r in runs], rotation=40, ha="right", fontsize=7)
    ax.set_title("mIoU giữ lại (% so với ảnh gốc)", fontsize=8.5)
    ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("mIoU (%)")
    for ax in axes[1, :4]:
        ax.set_xlabel("mức nhiễu (severity)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(runs), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("figures/fig_robustness.png")
    plt.close(fig)


def fig_scale(rob):
    runs = [r for r in ps.RUN_ORDER if r in rob["runs"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    for r in runs:
        y = np.array([rob["scale"][r][str(s)] for s in rob["scales"]])
        axes[0].plot(rob["scales"], y, marker="o", ms=4, color=ps.RUN_COLOR[r], label=ps.RUN_LABEL[r])
        axes[1].plot(rob["scales"], y - rob["scale"][r]["1.0"], marker="o", ms=4, color=ps.RUN_COLOR[r])
    axes[0].set_ylabel("mIoU (%)")
    axes[1].set_ylabel("thay đổi mIoU so với tỉ lệ 1,0 (điểm)")
    for ax in axes:
        ax.set_xlabel("tỉ lệ phóng ảnh khi suy luận")
        ax.set_xticks(rob["scales"], [vn(s) + "×" for s in rob["scales"]])
    axes[0].legend(loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig("figures/fig_scale.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    ps.apply()
    os.makedirs("figures", exist_ok=True)
    runs, M, H, P = load_runs()
    with open("results/eda.json") as f:
        eda = json.load(f)
    a, b = ps.MAIN_VIT, ps.MAIN_CNN

    ci = {r: bootstrap_miou(P[r]["conf"]) for r in runs}
    df = build_summary(runs, M, ci)
    print(df[["label", "miou", "ci_low", "ci_high", "mdice", "band_miou", "params_M", "gmacs", "fps",
              "train_min"]].to_string(index=False))

    analysis = {"runs": {}, "pairs": {}, "classes": list(CLASSES)}
    for r in runs:
        total, band = P[r]["conf"].sum(0), P[r]["band_conf"]
        pim = per_image_miou(P[r]["conf"])
        rates = confusion_rates(total)
        top = np.dstack(np.unravel_index(np.argsort(-rates, axis=None)[:8], rates.shape))[0]
        analysis["runs"][r] = {
            "ci": ci[r], "interior_miou": scores(total - band)["miou"], "band_miou": scores(band)["miou"],
            "band_pixel_share": float(band.sum() / total.sum()),
            "per_image_miou_mean": float(np.nanmean(pim)), "per_image_miou_median": float(np.nanmedian(pim)),
            "images_below_50": float(np.mean(pim < 50)),
            "worst_classes": [[CLASSES[c], M[r]["iou"][c]] for c in np.argsort(M[r]["iou"])[:5]],
            "best_classes": [[CLASSES[c], M[r]["iou"][c]] for c in np.argsort(M[r]["iou"])[::-1][:5]],
            "top_confusions": [[CLASSES[g], CLASSES[p], float(rates[g, p])] for g, p in top],
            "missed_as_background": float(100 * total[1:, 0].sum() / total[1:].sum()),
            "object_as_other_object": float(100 * (total[1:, 1:].sum() - np.trace(total[1:, 1:])) / total[1:].sum()),
            "confusion_rate": {f"{g}->{p}": float(100 * total[CLASSES.index(g), CLASSES.index(p)]
                                                  / total[CLASSES.index(g)].sum())
                               for g, p in (("bus", "train"), ("cow", "horse"), ("sofa", "chair"), ("chair", "sofa"))},
            "background_as_object": float(100 * total[0, 1:].sum() / total[0].sum()),
        }
    for x, y in PAIRS:
        if x in runs and y in runs:
            res = bootstrap_miou(P[x]["conf"], P[y]["conf"])
            px, py = per_image_miou(P[x]["conf"]), per_image_miou(P[y]["conf"])
            res.update(a=x, b=y, a_better_share=float(np.mean(px > py + 1)), b_better_share=float(np.mean(py > px + 1)),
                       per_class_diff={CLASSES[c]: M[x]["iou"][c] - M[y]["iou"][c] for c in range(NUM_CLASSES)},
                       classes_a_wins=int(sum(M[x]["iou"][c] > M[y]["iou"][c] for c in range(NUM_CLASSES))),
                       band_diff=M[x]["band_miou"] - M[y]["band_miou"])
            analysis["pairs"][f"{x}__vs__{y}"] = res
            print(f"{x} - {y}: {res['diff']:+.2f} [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}] p={res['p_value']:.4f}")

    fig_miou(df)
    fig_curves(runs, H)
    fig_cost(df)
    if a in runs and b in runs:
        fig_per_class(M, a, b)
        fig_confusion(P, a, b)
        analysis["size_vs_iou"] = fig_size_vs_iou(M, eda, a, b)
        analysis["qualitative"] = fig_qualitative(P, a, b, extra="segformer_b0" if "segformer_b0" in runs else None)
    if os.path.isfile("results/robustness.json"):
        with open("results/robustness.json") as f:
            rob = json.load(f)
        if "summary" in rob:
            fig_robustness(rob)
            fig_scale(rob)
            analysis["robustness"] = rob["summary"]
            analysis["scale_drop"] = {r: {s: rob["scale"][r][s] - rob["scale"][r]["1.0"] for s in rob["scale"][r]}
                                      for r in rob["runs"]}
    with open("results/analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print("wrote results/summary.csv, results/analysis.json and figures/")
