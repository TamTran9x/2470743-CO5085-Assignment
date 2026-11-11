"""Aggregate all runs into tables, figures and an error analysis.

Reads results/<arch>_<mode>/{metrics.json,history.csv,preds.npz} and writes
  results/summary.csv, results/summary.md, results/analysis.json
  figures/fig_*.png

    python -m src.analyze
"""
import json
import os

import numpy as np
import pandas as pd

from . import plotstyle as ps
from .data import load_cifar100

plt = ps.plt
ARCH_ORDER = ["resnet50", "deit_s"]
MARKERS = {"scratch": "o", "linear": "s", "partial": "^", "full": "D"}
# shorter tick labels so the 20x20 superclass matrices stay legible at report width
SHORT_SUPERCLASS = {
    "fruit_and_vegetables": "fruit & veg", "household_electrical_devices": "electrical devices",
    "household_furniture": "furniture", "large_man-made_outdoor_things": "man-made outdoor",
    "large_natural_outdoor_scenes": "natural outdoor", "large_omnivores_and_herbivores": "omni/herbivores",
    "non-insect_invertebrates": "invertebrates",
}


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_runs(results_dir):
    runs = {}
    for mode in ps.MODE_ORDER:
        for arch in ARCH_ORDER:
            d = os.path.join(results_dir, f"{arch}_{mode}")
            if not os.path.isfile(os.path.join(d, "metrics.json")):
                continue
            with open(os.path.join(d, "metrics.json")) as f:
                m = json.load(f)
            p = np.load(os.path.join(d, "preds.npz"))
            m["history"] = pd.read_csv(os.path.join(d, "history.csv"))
            m["logits"] = p["logits"].astype(np.float32)
            m["labels"] = p["labels"]
            m["confusion"] = p["confusion"]
            runs[(arch, mode)] = m
    return runs


def summary_table(runs):
    rows = []
    for (arch, mode), m in runs.items():
        rows.append({
            "run": m["name"], "arch": arch, "family": m["family"], "mode": mode,
            "test_acc": 100 * m["test_acc"], "test_top5": 100 * m["test_top5"],
            "macro_f1": 100 * m["test_macro_f1"], "superclass_acc": 100 * m["test_superclass_acc"],
            "best_val_acc": 100 * m["best_val_acc"], "best_epoch": m["best_epoch"], "epochs": m["epochs"],
            "trainable_M": m["trainable_params"] / 1e6, "total_M": m["total_params"] / 1e6,
            "trainable_pct": 100 * m["trainable_params"] / m["total_params"], "gmacs": m["gmacs"],
            "train_min": m["train_time_s"] / 60, "epoch_s": m["epoch_time_s"],
            "train_img_s": m["train_throughput"], "infer_img_s": m["infer_throughput"],
            "peak_mem_gb": m["peak_mem_gb"], "final_train_acc": 100 * m["final_train_acc"],
        })
    return pd.DataFrame(rows)


def to_markdown(df):
    cols = ["arch", "mode", "test_acc", "test_top5", "macro_f1", "best_val_acc", "trainable_M", "train_min",
            "epoch_s", "peak_mem_gb"]
    header = "| Model | Chế độ | Test acc (%) | Top-5 (%) | Macro-F1 (%) | Val acc (%) | Trainable (M) | " \
             "Train (phút) | s/epoch | Peak mem (GB) |"
    lines = [header, "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        lines.append(f"| {ps.ARCH_LABEL[r.arch]} | {ps.MODE_LABEL[r['mode']]} | {r.test_acc:.2f} | {r.test_top5:.2f} | "
                     f"{r.macro_f1:.2f} | {r.best_val_acc:.2f} | {r.trainable_M:.2f} | {r.train_min:.1f} | "
                     f"{r.epoch_s:.0f} | {r.peak_mem_gb:.2f} |")
    return "\n".join(lines) + "\n"


def to_html(df):
    best = {arch: df[df.arch == arch].test_acc.max() for arch in df.arch.unique()}
    head = ("<tr><th>Model</th><th>Chế độ</th><th>Top-1 (%)</th><th>Top-5 (%)</th><th>Macro-F1 (%)</th>"
            "<th>Trainable (M)</th><th>Train (phút)</th><th>Mem (GB)</th></tr>")
    vn = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")  # Vietnamese decimal comma, as in the page text
    rows = []
    for _, r in df.iterrows():
        dot = "cnn" if r.family == "CNN" else "vit"
        cls = ' class="best"' if r.test_acc == best[r.arch] else ""
        rows.append(f'<tr><td><span class="dot {dot}"></span>{ps.ARCH_LABEL[r.arch]}</td><td>{ps.MODE_LABEL[r["mode"]]}</td>'
                    f"<td{cls}>{vn(r.test_acc)}</td><td>{vn(r.test_top5)}</td><td>{vn(r.macro_f1)}</td>"
                    f"<td>{vn(r.trainable_M)}</td><td>{vn(r.train_min, 1)}</td><td>{vn(r.peak_mem_gb)}</td></tr>")
    return ('<div class="table-wrap"><table>\n' + head + "\n" + "\n".join(rows) + "\n</table></div>\n"
            '<p style="font-size:13px;color:var(--muted)">Test 10.000 ảnh; in đậm: chế độ tốt nhất của mỗi họ. '
            "Thời gian/bộ nhớ đo trên RTX 3050 Laptop 6 GB, batch 64, 224×224.</p>")


def inject(path, marker, content):
    """Replace the text between <!-- marker --> and <!-- /marker --> in `path`."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    start, end = f"<!-- {marker} -->", f"<!-- /{marker} -->"
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(text[:i + len(start)] + "\n" + content + text[j:])


# ---------------------------------------------------------------- figures

def fig_test_acc(df, path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    modes = [m for m in ps.MODE_ORDER if m in set(df["mode"])]
    width = 0.38
    for i, arch in enumerate(ARCH_ORDER):
        sub = df[df.arch == arch].set_index("mode")
        xs = [j + (i - 0.5) * (width + 0.02) for j, m in enumerate(modes) if m in sub.index]
        vals = [sub.loc[m, "test_acc"] for m in modes if m in sub.index]
        ax.bar(xs, vals, width, color=ps.ARCH_COLOR[arch], label=ps.ARCH_LABEL[arch], zorder=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=ps.INK_2)
    ax.set_xticks(range(len(modes)), [ps.MODE_LABEL[m] for m in modes])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Top-1 accuracy trên tập test CIFAR-100 (10.000 ảnh)", loc="left")
    fig.savefig(path)
    plt.close(fig)


def fig_curves(runs, path):
    modes = [m for m in ps.MODE_ORDER if any(k[1] == m for k in runs)]
    fig, axes = plt.subplots(2, len(modes), figsize=(2.6 * len(modes), 4.6), sharey="row", squeeze=False)
    for j, mode in enumerate(modes):
        for arch in ARCH_ORDER:
            if (arch, mode) not in runs:
                continue
            h = runs[(arch, mode)]["history"]
            c = ps.ARCH_COLOR[arch]
            axes[0, j].plot(h.epoch, 100 * h.val_acc, color=c, label=ps.ARCH_LABEL[arch])
            axes[1, j].plot(h.epoch, 100 * h.train_acc, color=c, label=ps.ARCH_LABEL[arch])
        axes[0, j].set_title(ps.MODE_LABEL[mode], loc="left")
        axes[1, j].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Val accuracy (%)")
    axes[1, 0].set_ylabel("Train accuracy (%)")
    axes[0, 0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_cost(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for _, r in df.iterrows():
        for ax, x in zip(axes, [r.train_min, r.peak_mem_gb]):
            ax.scatter(x, r.test_acc, s=60, marker=MARKERS[r["mode"]], color=ps.ARCH_COLOR[r.arch],
                       edgecolor="white", linewidth=1.5, zorder=3)
    axes[0].set_xlabel("Thời gian train (phút)")
    axes[1].set_xlabel("Bộ nhớ GPU đỉnh khi train (GB)")
    axes[1].set_xlim(left=0)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("Accuracy theo thời gian train", loc="left")
    axes[1].set_title("Accuracy theo bộ nhớ GPU", loc="left")
    part = df[df["mode"] == "partial"]
    if len(part) == 2:  # the two partial-freeze runs land on almost the same spot in the time panel
        axes[0].annotate("Freeze một phần: hai mô hình gần trùng\n" + ", ".join(
            f"{ps.ARCH_LABEL[r.arch].split(' ')[0]} {r.test_acc:.1f}%" for _, r in part.iterrows()),
            (part.train_min.mean(), part.test_acc.mean()), xytext=(28, 77), fontsize=7, color=ps.INK_2,
            arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=0.8))
    # colour = model family, marker = training regime; legends sit under the plots so no point is covered
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=ps.ARCH_COLOR[a], label=ps.ARCH_LABEL[a])
               for a in ARCH_ORDER]
    handles += [plt.Line2D([], [], ls="", marker=MARKERS[m], ms=7, color=ps.MUTED, label=ps.MODE_LABEL[m])
                for m in ps.MODE_ORDER if m in set(df["mode"])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7.5, frameon=False,
               handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def superclass_confusion(m, f2c):
    t, p = f2c[m["labels"]], f2c[m["logits"].argmax(1)]
    cm = np.zeros((20, 20))
    np.add.at(cm, (t, p), 1)
    return 100 * cm / cm.sum(1, keepdims=True)


def fig_superclass_confusion(runs, keys, ds, path):
    # the diagonal (88-98 %) would swamp the colour scale, so it is printed as text and only errors are coloured
    cms = [superclass_confusion(runs[key], ds["fine_to_coarse"]) for key in keys]
    vmax = np.ceil(max(np.max(cm - np.diag(np.diag(cm))) for cm in cms))
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys) + 1.4, 4.9), sharey=True, squeeze=False,
                             layout="constrained")
    names = [SHORT_SUPERCLASS.get(s, s.replace("_", " ")) for s in ds["superclasses"]]
    for ax, key, cm in zip(axes[0], keys, cms):
        off = cm.copy()
        np.fill_diagonal(off, np.nan)
        im = ax.imshow(off, cmap=ps.SEQ_BLUE, vmin=0, vmax=vmax)
        ax.set_xticks(range(20), names, rotation=90, fontsize=7)
        ax.set_yticks(range(20), names, fontsize=7)
        ax.grid(False)
        ax.set_title(f"{ps.ARCH_LABEL[key[0]]} – {ps.MODE_LABEL[key[1]]}", loc="left")
        ax.set_xlabel("Superclass dự đoán")
        for i in range(20):
            ax.text(i, i, f"{cm[i, i]:.0f}", ha="center", va="center", fontsize=6, color=ps.INK_2)
    axes[0, 0].set_ylabel("Superclass thật")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6, label="% ảnh bị nhầm sang superclass khác")
    fig.savefig(path)
    plt.close(fig)


def fig_full_confusion(m, ds, path):
    f2c = ds["fine_to_coarse"]
    order = np.lexsort((np.arange(100), f2c))
    cm = m["confusion"].astype(float)
    cm = 100 * cm / cm.sum(1, keepdims=True)
    cm = cm[order][:, order]
    np.fill_diagonal(cm, np.nan)  # show only the errors
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(cm, cmap=ps.SEQ_BLUE, vmin=0, vmax=np.nanpercentile(cm, 99.9))
    bounds = np.where(np.diff(f2c[order]) != 0)[0] + 0.5
    for b in bounds:
        ax.axhline(b, color=ps.AXIS, lw=0.5)
        ax.axvline(b, color=ps.AXIS, lw=0.5)
    ax.grid(False)
    ax.set_xticks([])
    edges = np.concatenate([[-0.5], bounds, [99.5]])
    ax.set_yticks((edges[:-1] + edges[1:]) / 2, [s.replace("_", " ") for s in ds["superclasses"]], fontsize=6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Lớp dự đoán (100 lớp, nhóm theo superclass)")
    ax.set_ylabel("Lớp thật")
    ax.set_title(f"Confusion matrix (chỉ lỗi) – {ps.ARCH_LABEL[m['arch']]} {ps.MODE_LABEL[m['mode']]}", loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8, label="% ảnh của lớp thật")
    fig.savefig(path)
    plt.close(fig)


def fig_per_class_diff(cnn, vit, classes, path, k=12):
    diff = 100 * (np.array(vit["per_class_acc"]) - np.array(cnn["per_class_acc"]))
    order = np.argsort(diff)
    idx = np.concatenate([order[:k], order[-k:]])
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    colors = [ps.ARCH_COLOR["deit_s"] if diff[i] > 0 else ps.ARCH_COLOR["resnet50"] for i in idx]
    ax.barh(range(len(idx)), diff[idx], color=colors, height=0.72, zorder=2)
    ax.set_yticks(range(len(idx)), [classes[i].replace("_", " ") for i in idx], fontsize=7.5)
    ax.axvline(0, color=ps.AXIS, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Chênh lệch accuracy theo lớp: DeiT-S − ResNet-50 (điểm %)")
    ax.set_title(f"{k} lớp CNN tốt hơn nhiều nhất (xanh) và {k} lớp ViT tốt hơn nhiều nhất (cam)", loc="left",
                 fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def fig_misclassified(m, ds, path, n=24):
    prob = softmax(m["logits"])
    pred = prob.argmax(1)
    wrong = np.where(pred != m["labels"])[0]
    pick = wrong[np.argsort(-prob[wrong, pred[wrong]])][:n]
    cols = 8
    fig, axes = plt.subplots(n // cols, cols, figsize=(10, 4.4))
    for ax, i in zip(axes.flat, pick):
        ax.imshow(ds["x_test"][i], interpolation="nearest")
        ax.set_title(f"T: {ds['classes'][m['labels'][i]]}\nP: {ds['classes'][pred[i]]} ({prob[i, pred[i]]:.2f})",
                     fontsize=6.5, color=ps.INK_2, fontweight="normal")
        ax.axis("off")
    fig.suptitle(f"Lỗi tự tin nhất của {ps.ARCH_LABEL[m['arch']]} – {ps.MODE_LABEL[m['mode']]} "
                 "(T = nhãn thật, P = dự đoán, xác suất softmax)", fontsize=9, x=0.01, ha="left",
                 fontweight=plt.rcParams["axes.titleweight"])
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.84, wspace=0.15, hspace=0.6)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- error analysis numbers

def top_confusions(m, classes, k=10):
    cm = m["confusion"].copy()
    np.fill_diagonal(cm, 0)
    flat = np.argsort(-cm, axis=None)[:k]
    return [{"true": classes[i], "pred": classes[j], "count": int(cm[i, j])}
            for i, j in zip(*np.unravel_index(flat, cm.shape))]


def within_superclass_error_share(m, f2c):
    pred = m["logits"].argmax(1)
    wrong = pred != m["labels"]
    return float((f2c[pred[wrong]] == f2c[m["labels"][wrong]]).mean())


def top_superclass_confusions(m, ds, k=6):
    cm = superclass_confusion(m, ds["fine_to_coarse"])
    np.fill_diagonal(cm, 0)
    flat = np.argsort(-cm, axis=None)[:k]
    return [{"true": ds["superclasses"][i], "pred": ds["superclasses"][j], "pct": float(cm[i, j])}
            for i, j in zip(*np.unravel_index(flat, cm.shape))]


def mcnemar(a, b):
    """Exact McNemar test on the test images where exactly one of the two models is right."""
    from scipy.stats import binomtest
    ca = a["logits"].argmax(1) == a["labels"]
    cb = b["logits"].argmax(1) == b["labels"]
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return {"only_a": only_a, "only_b": only_b, "p_value": float(binomtest(only_a, only_a + only_b, 0.5).pvalue)}


def error_overlap(a, b):
    ca = a["logits"].argmax(1) == a["labels"]
    cb = b["logits"].argmax(1) == b["labels"]
    ens = (softmax(a["logits"]) + softmax(b["logits"])).argmax(1) == a["labels"]
    return {"both_correct": float((ca & cb).mean()), "only_a": float((ca & ~cb).mean()),
            "only_b": float((~ca & cb).mean()), "both_wrong": float((~ca & ~cb).mean()),
            "ensemble_acc": float(ens.mean())}


def main(results_dir="results", fig_dir="figures"):
    ps.apply()
    os.makedirs(fig_dir, exist_ok=True)
    ds = load_cifar100("data")
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit("no finished runs in " + results_dir)
    df = summary_table(runs)
    df.to_csv(os.path.join(results_dir, "summary.csv"), index=False, float_format="%.4f")
    with open(os.path.join(results_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(to_markdown(df))
    print(to_markdown(df))
    inject("README.md", "RESULTS_TABLE", to_markdown(df))
    inject("index.html", "RESULTS_TABLE", to_html(df))

    fig_test_acc(df, os.path.join(fig_dir, "fig_test_acc.png"))
    fig_curves(runs, os.path.join(fig_dir, "fig_curves.png"))
    fig_cost(df, os.path.join(fig_dir, "fig_cost.png"))

    f2c = ds["fine_to_coarse"]
    analysis = {"runs": {}}
    for (arch, mode), m in runs.items():
        h = m["history"]
        analysis["runs"][m["name"]] = {
            "within_superclass_error_share": within_superclass_error_share(m, f2c),
            "top_confusions": top_confusions(m, ds["classes"]),
            "hardest_classes": [{"class": ds["classes"][i], "acc": m["per_class_acc"][i]}
                                for i in np.argsort(m["per_class_acc"])[:8]],
            "easiest_classes": [{"class": ds["classes"][i], "acc": m["per_class_acc"][i]}
                                for i in np.argsort(m["per_class_acc"])[::-1][:8]],
            "top_superclass_confusions": top_superclass_confusions(m, ds),
            "val_acc_at_epoch": {int(e): float(v) for e, v in zip(h.epoch, h.val_acc)},
            "train_val_gap_final": float(h.train_acc.iloc[-1] - h.val_acc.iloc[-1]),
        }

    # ResNet-50 vs DeiT-S under the same regime (a = ResNet-50, b = DeiT-S)
    analysis["mcnemar_cnn_vs_vit"] = {mode: mcnemar(runs[("resnet50", mode)], runs[("deit_s", mode)])
                                      for mode in ps.MODE_ORDER
                                      if ("resnet50", mode) in runs and ("deit_s", mode) in runs}

    # best regime of each family, selected on validation accuracy
    best = {}
    for arch in ARCH_ORDER:
        cands = [(m["best_val_acc"], k) for k, m in runs.items() if k[0] == arch]
        if cands:
            best[arch] = max(cands)[1]
    analysis["best_by_val"] = {a: f"{k[0]}_{k[1]}" for a, k in best.items()}
    if len(best) == 2:
        cnn, vit = runs[best["resnet50"]], runs[best["deit_s"]]
        analysis["overlap_best_cnn_vs_vit"] = error_overlap(cnn, vit)
        fig_superclass_confusion(runs, [best["resnet50"], best["deit_s"]], ds,
                                 os.path.join(fig_dir, "fig_superclass_confusion.png"))
        fig_per_class_diff(cnn, vit, ds["classes"], os.path.join(fig_dir, "fig_per_class_diff.png"))
        overall = max(best.values(), key=lambda k: runs[k]["best_val_acc"])
        fig_full_confusion(runs[overall], ds, os.path.join(fig_dir, "fig_confusion_full.png"))
        fig_misclassified(runs[overall], ds, os.path.join(fig_dir, "fig_misclassified.png"))
        analysis["overall_best"] = f"{overall[0]}_{overall[1]}"
    with open(os.path.join(results_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
