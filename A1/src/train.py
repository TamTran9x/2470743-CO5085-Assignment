"""Train one (architecture, regime) configuration on CIFAR-100, then evaluate the best
checkpoint (selected on the validation split) on the official test set.

Example:
    python -m src.train --arch deit_s --mode full --epochs 10
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.flop_counter import FlopCounterMode

from .data import GPULoader, load_cifar100, stratified_split
from .models import ARCHS, MODES, build_model, count_params, head_parameters, is_cnn, set_train_mode

# Per-regime defaults: backbone lr, head lr, warmup epochs, epochs
REGIME_DEFAULTS = {
    "scratch": {"lr": 5e-4, "head_lr": 5e-4, "warmup": 3, "epochs": 30},
    "linear": {"lr": 0.0, "head_lr": 1e-3, "warmup": 1, "epochs": 10},
    "partial": {"lr": 1e-4, "head_lr": 1e-3, "warmup": 1, "epochs": 10},
    "full": {"lr": 1e-4, "head_lr": 1e-3, "warmup": 1, "epochs": 10},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=list(ARCHS), required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float, help="backbone learning rate")
    p.add_argument("--head-lr", type=float, help="classifier head learning rate")
    p.add_argument("--warmup", type=float, help="warmup epochs")
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--name", help="run name (default: <arch>_<mode>)")
    p.add_argument("--train-subset", type=int, default=0, help="debug: use only N training images")
    args = p.parse_args(argv)
    for k, v in REGIME_DEFAULTS[args.mode].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    args.name = args.name or f"{args.arch}_{args.mode}"
    return args


def make_optimizer(model, lr, head_lr, wd):
    head_ids = {id(p) for p in head_parameters(model)}
    no_wd_names = set(getattr(model, "no_weight_decay", lambda: set())())
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = id(p) in head_ids
        no_decay = p.ndim <= 1 or name in no_wd_names
        groups.setdefault((is_head, no_decay), []).append(p)
    param_groups = [
        {"params": ps, "lr": head_lr if is_head else lr, "weight_decay": 0.0 if no_decay else wd}
        for (is_head, no_decay), ps in groups.items()
    ]
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), fused=True)


def warmup_cosine(total_steps, warmup_steps):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


@torch.inference_mode()
def predict(model, loader):
    """Return (logits [N,C] float32 on CPU, labels [N] on CPU, elapsed seconds)."""
    model.eval()
    outs, ys = [], []
    torch.cuda.synchronize()
    t0 = time.time()
    for xb, yb in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(model(xb).float())
        ys.append(yb)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return torch.cat(outs).cpu(), torch.cat(ys).cpu(), elapsed


def topk_acc(logits, labels, k):
    return (logits.topk(k, dim=1).indices == labels[:, None]).any(dim=1).float().mean().item()


def forward_gmacs(model, img_size):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device="cuda")
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return fc.get_total_flops() / 2 / 1e9


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_dir = os.path.join(args.out, args.name)
    ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    last_path, best_path = os.path.join(ckpt_dir, "last.pt"), os.path.join(ckpt_dir, "best.pt")

    # ---------------- data ----------------
    ds = load_cifar100(args.data)
    tr_idx, va_idx = stratified_split(ds["y_train"], val_per_class=50, seed=args.split_seed)
    if args.train_subset:
        tr_idx = np.random.RandomState(0).choice(tr_idx, args.train_subset, replace=False)
    cl = is_cnn(args.arch)
    train_loader = GPULoader(ds["x_train"][tr_idx], ds["y_train"][tr_idx], args.batch_size, args.img_size,
                             train=True, channels_last=cl)
    val_loader = GPULoader(ds["x_train"][va_idx], ds["y_train"][va_idx], 256, args.img_size,
                           train=False, channels_last=cl)
    test_loader = GPULoader(ds["x_test"], ds["y_test"], 256, args.img_size, train=False, channels_last=cl)

    # ---------------- model ----------------
    model, frozen = build_model(args.arch, args.mode, num_classes=100, img_size=args.img_size)
    model = model.cuda()
    if cl:
        model = model.to(memory_format=torch.channels_last)
    total_params, trainable_params = count_params(model)
    gmacs = forward_gmacs(model, args.img_size)
    trainable = [p for p in model.parameters() if p.requires_grad]

    opt = make_optimizer(model, args.lr, args.head_lr, args.wd)
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(args.epochs * steps_per_epoch, int(args.warmup * steps_per_epoch)))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    start_epoch, best_acc, history = 0, -1.0, []
    if os.path.isfile(last_path):
        state = torch.load(last_path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, best_acc, history = state["epoch"] + 1, state["best_acc"], state["history"]
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state(state["rng_cuda"])
        print(f"Resumed from epoch {start_epoch}")

    print(f"[{args.name}] params total={total_params/1e6:.2f}M trainable={trainable_params/1e6:.2f}M "
          f"GMACs={gmacs:.2f} train={len(tr_idx)} val={len(va_idx)} steps/epoch={steps_per_epoch}", flush=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    peak_mem_gb = 0.0  # peak GPU memory of the training phase (validation excluded)
    for epoch in range(start_epoch, args.epochs):
        set_train_mode(model, frozen)
        torch.cuda.reset_peak_memory_stats()
        loss_sum = torch.zeros((), device="cuda")
        correct = torch.zeros((), device="cuda")
        seen = 0
        torch.cuda.synchronize()
        t0 = time.time()
        for xb, yb in train_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
                loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
            opt.step()
            sched.step()
            loss_sum += loss.detach() * len(yb)
            correct += (logits.argmax(1) == yb).sum()
            seen += len(yb)
        torch.cuda.synchronize()
        epoch_time = time.time() - t0
        peak_mem_gb = max(peak_mem_gb, torch.cuda.max_memory_allocated() / 1024 ** 3)

        val_logits, val_y, _ = predict(model, val_loader)
        val_loss = nn.functional.cross_entropy(val_logits, val_y).item()
        val_acc = topk_acc(val_logits, val_y, 1)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum.item() / seen,
            "train_acc": correct.item() / seen,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr_max": max(g["lr"] for g in opt.param_groups),
            "epoch_time_s": epoch_time,
        }
        history.append(row)
        print(f"[{args.name}] ep {epoch+1:3d}/{args.epochs} loss {row['train_loss']:.3f} "
              f"train_acc {row['train_acc']:.4f} val_loss {val_loss:.3f} val_acc {val_acc:.4f} "
              f"time {epoch_time:.0f}s", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "best_acc": best_acc, "history": history,
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last_path)

    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    # ---------------- test evaluation of the best checkpoint ----------------
    best = torch.load(best_path, map_location="cuda", weights_only=False)
    model.load_state_dict(best["model"])
    predict(model, test_loader)  # warm-up pass so the timing below excludes cudnn autotuning
    logits, y, infer_time = predict(model, test_loader)
    pred = logits.argmax(1)
    y_np, pred_np = y.numpy(), pred.numpy()
    cm = confusion_matrix(y_np, pred_np, labels=np.arange(100))
    f2c = ds["fine_to_coarse"]
    metrics = {
        "name": args.name,
        "arch": args.arch,
        "family": ARCHS[args.arch]["family"],
        "timm_name": ARCHS[args.arch]["timm_name"],
        "mode": args.mode,
        "pretrained": args.mode != "scratch",
        "epochs": args.epochs,
        "best_epoch": best["epoch"],
        "best_val_acc": best_acc,
        "test_acc": topk_acc(logits, y, 1),
        "test_top5": topk_acc(logits, y, 5),
        "test_macro_f1": f1_score(y_np, pred_np, average="macro"),
        "test_superclass_acc": float((f2c[pred_np] == f2c[y_np]).mean()),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "gmacs": gmacs,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "train_time_s": sum(h["epoch_time_s"] for h in history),
        "epoch_time_s": float(np.mean([h["epoch_time_s"] for h in history])),
        "train_throughput": len(tr_idx) / float(np.mean([h["epoch_time_s"] for h in history])),
        "infer_throughput": len(y_np) / infer_time,
        "peak_mem_gb": peak_mem_gb,
        "final_train_acc": history[-1]["train_acc"],
        "per_class_acc": (np.diag(cm) / cm.sum(1)).tolist(),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "preds.npz"), logits=logits.numpy().astype(np.float16),
                        labels=y_np, confusion=cm)
    print(f"[{args.name}] TEST acc {metrics['test_acc']:.4f} top5 {metrics['test_top5']:.4f} "
          f"macroF1 {metrics['test_macro_f1']:.4f} (best epoch {best['epoch']}, "
          f"train {metrics['train_time_s']/60:.1f} min, peak {peak_mem_gb:.2f} GB)", flush=True)
    return metrics


if __name__ == "__main__":
    main()
