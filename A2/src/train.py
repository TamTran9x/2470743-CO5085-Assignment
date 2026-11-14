"""Train one segmentation model on VOC 2012 train_aug, then evaluate the final weights on VOC 2012 val.

The protocol follows the SegFormer paper (AdamW, lr 6e-5 with a 10x multiplier on the decoder,
weight decay 0.01, linear warm-up + poly schedule, random scale 0.5-2.0 / crop / flip / colour
jitter) with a schedule shortened to fit a 6 GB laptop GPU. There is no checkpoint selection:
the last iteration is evaluated, and the held-out `minival` split is only used for learning curves.

    python -m src.train --arch segformer_b2
    python -m src.train --arch segformer_b0 --no-pretrained --lr 6e-4 --head-lr-mult 1
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from .data import IGNORE, NUM_CLASSES, EvalSet, Normalize, train_loader, upsample_logits
from .metrics import evaluate, scores
from .models import MODELS, build_model, count_params


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=list(MODELS), required=True)
    p.add_argument("--name", help="run name (default: <arch>[_scratch])")
    p.add_argument("--no-pretrained", action="store_true", help="random initialisation of the encoder")
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--lr", type=float, default=6e-5, help="encoder learning rate")
    p.add_argument("--head-lr-mult", type=float, default=10.0)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500, help="linear warm-up iterations")
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--decoder-dim", type=int)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default="results")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--train-subset", type=int, default=0, help="debug: use only N training images")
    p.add_argument("--eval-subset", type=int, default=0, help="debug: use only N images of minival/val")
    args = p.parse_args(argv)
    args.name = args.name or args.arch + ("_scratch" if args.no_pretrained else "")
    return args


def make_optimizer(model, lr, head_lr, wd):
    """AdamW with a larger lr for the randomly initialised decoder; no weight decay on norms/biases."""
    groups = {}
    for part, lr_part in (("encoder", lr), ("decode_head", head_lr)):
        for _, p in getattr(model, part).named_parameters():
            if p.requires_grad:
                groups.setdefault((lr_part, 0.0 if p.ndim <= 1 else wd), []).append(p)
    return torch.optim.AdamW([{"params": ps, "lr": g_lr, "weight_decay": g_wd} for (g_lr, g_wd), ps in groups.items()],
                             betas=(0.9, 0.999), fused=True)


def warmup_poly(total, warmup, power=1.0):
    def fn(step):
        poly = (1.0 - min(step, total) / total) ** power
        return poly * min(1.0, (step + 1) / warmup) if warmup else poly
    return fn


def forward_gflops(model, size=512):
    """FLOPs of one forward pass on a size x size image (multiply-adds counted as 2 FLOPs -> GMACs)."""
    model.eval()
    x = torch.randn(1, 3, size, size, device="cuda")
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return fc.get_total_flops() / 2 / 1e9


@torch.inference_mode()
def measure_fps(model, size=512, amp=True, steps=60, warmup=15):
    """Latency of single-image inference (batch 1), in images per second."""
    model.eval()
    x = torch.randn(1, 3, size, size, device="cuda")
    for i in range(steps + warmup):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            model(x)
    torch.cuda.synchronize()
    return steps / (time.time() - t0)


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
    last_path = os.path.join(ckpt_dir, "last.pt")

    # ---------------- model ----------------
    model = build_model(args.arch, NUM_CLASSES, pretrained=not args.no_pretrained, decoder_dim=args.decoder_dim,
                        drop_path_rate=args.drop_path).cuda()
    total_params, enc_params, head_params = count_params(model)
    gflops = forward_gflops(model)
    opt = make_optimizer(model, args.lr, args.lr * args.head_lr_mult, args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_poly(args.iters, args.warmup))

    start_iter, history, train_time, peak_mem_gb = 0, [], 0.0, 0.0
    if os.path.isfile(last_path):
        state = torch.load(last_path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_iter, history = state["iter"], state["history"]
        train_time, peak_mem_gb = state["train_time"], state["peak_mem_gb"]
        print(f"Resumed from iteration {start_iter}")

    # ---------------- data ----------------
    normalize = Normalize()
    train_ds, loader = train_loader(args.batch_size, args.crop, seed=args.seed + start_iter, workers=args.workers,
                                    subset=args.train_subset)
    minival = EvalSet("minival", subset=args.eval_subset)
    print(f"[{args.name}] params {total_params/1e6:.2f}M (encoder {enc_params/1e6:.2f}M, decoder "
          f"{head_params/1e6:.2f}M) GMACs@512={gflops:.1f} train={len(train_ds)} minival={len(minival)} "
          f"iters={args.iters} (~{args.iters * args.batch_size / len(train_ds):.1f} epochs)", flush=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---------------- training ----------------
    data = iter(loader)
    it = start_iter
    while it < args.iters:
        model.train()
        torch.cuda.reset_peak_memory_stats()
        loss_sum = torch.zeros((), device="cuda")
        correct = torch.zeros((), device="cuda")
        valid = torch.zeros((), device="cuda")
        seg_start = it
        torch.cuda.synchronize()
        t0 = time.time()
        while it < args.iters and (it == seg_start or it % args.eval_every):
            xb, yb = next(data)
            xb = normalize(xb.cuda(non_blocking=True))
            yb = yb.cuda(non_blocking=True).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)
            logits = upsample_logits(logits, yb.shape[-2:])
            loss = F.cross_entropy(logits, yb, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            loss_sum += loss.detach()
            mask = yb != IGNORE
            correct += (logits.detach().argmax(1) == yb)[mask].sum()
            valid += mask.sum()
        torch.cuda.synchronize()
        seg_time = time.time() - t0
        train_time += seg_time
        peak_mem_gb = max(peak_mem_gb, torch.cuda.max_memory_allocated() / 1024 ** 3)

        ev = evaluate(model, minival, normalize)
        s = scores(ev["conf"].sum(0))
        row = {"iter": it, "epoch": it * args.batch_size / len(train_ds),
               "train_loss": loss_sum.item() / (it - seg_start), "train_pixel_acc": 100 * (correct / valid).item(),
               "minival_miou": s["miou"], "minival_pixel_acc": s["pixel_acc"],
               "lr_encoder": opt.param_groups[0]["lr"], "train_time_s": train_time,
               "img_per_s": (it - seg_start) * args.batch_size / seg_time}
        history.append(row)
        print(f"[{args.name}] it {it:6d}/{args.iters} loss {row['train_loss']:.3f} "
              f"train_acc {row['train_pixel_acc']:.2f} minival mIoU {s['miou']:.2f} "
              f"{row['img_per_s']:.1f} img/s  elapsed {train_time/60:.1f} min", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "iter": it,
                    "history": history, "train_time": train_time, "peak_mem_gb": peak_mem_gb}, last_path)
    del data, loader

    with open(os.path.join(run_dir, "history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    torch.save({"model": model.state_dict(), "arch": args.arch, "decoder_dim": args.decoder_dim},
               os.path.join(ckpt_dir, "final.pt"))

    # ---------------- test: VOC 2012 val, final weights ----------------
    del minival
    val = EvalSet("val", subset=args.eval_subset)
    evaluate(model, val, normalize)  # warm-up pass so the timing below excludes cudnn autotuning
    ev = evaluate(model, val, normalize, keep_preds=True)
    s, band = scores(ev["conf"].sum(0)), scores(ev["band_conf"])
    cfg = MODELS[args.arch]
    metrics = {
        "name": args.name, "arch": args.arch, "label": cfg["label"], "family": cfg["family"],
        "encoder": cfg["encoder"], "decoder": cfg["decoder"], "pretrained": not args.no_pretrained,
        "pretrained_id": None if args.no_pretrained else cfg["pretrained_id"],
        "iters": args.iters, "epochs": args.iters * args.batch_size / len(train_ds),
        "batch_size": args.batch_size, "crop": args.crop, "lr": args.lr, "head_lr_mult": args.head_lr_mult,
        "miou": s["miou"], "mdice": s["mdice"], "pixel_acc": s["pixel_acc"], "mean_acc": s["mean_acc"],
        "band_miou": band["miou"], "iou": s["iou"], "dice": s["dice"], "band_iou": band["iou"],
        "minival_miou": history[-1]["minival_miou"],
        "total_params": total_params, "encoder_params": enc_params, "decoder_params": head_params,
        "gmacs_512": gflops, "train_time_s": train_time,
        "train_throughput": args.iters * args.batch_size / train_time, "peak_mem_gb": peak_mem_gb,
        "infer_throughput": len(val) / ev["seconds"],
        "fps_bf16": measure_fps(model, amp=True), "fps_fp32": measure_fps(model, amp=False),
        "final_train_loss": history[-1]["train_loss"], "n_val": len(val),
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(os.path.join(run_dir, "preds.npz"), conf=ev["conf"].astype(np.int32),
                        band_conf=ev["band_conf"], ids=np.array(val.ids))
    np.savez_compressed(os.path.join(ckpt_dir, "val_pred.npz"), preds=ev["preds"], ids=np.array(val.ids))
    print(f"[{args.name}] VOC12 val: mIoU {s['miou']:.2f} mDice {s['mdice']:.2f} pixel acc {s['pixel_acc']:.2f} "
          f"boundary mIoU {band['miou']:.2f} | train {train_time/60:.1f} min, peak {peak_mem_gb:.2f} GB, "
          f"{metrics['fps_bf16']:.1f} FPS", flush=True)
    return metrics


if __name__ == "__main__":
    main()
