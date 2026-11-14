"""Run the A2 experiment grid sequentially.

Finished runs (results/<name>/metrics.json exists) are skipped and interrupted runs resume from
their last evaluation point, so the script can simply be re-launched.

    python -m scripts.run_all                       # everything
    python -m scripts.run_all --only segformer_b2 deeplabv3plus_r50
"""
import argparse
import os
import subprocess
import sys
import time

# (run name, extra arguments of src.train). Main comparison first, ablations after.
EXPERIMENTS = [
    ("segformer_b2", ["--arch", "segformer_b2"]),
    ("deeplabv3plus_r50", ["--arch", "deeplabv3plus_r50"]),
    ("segformer_b0", ["--arch", "segformer_b0"]),
    ("resnet50_mlp", ["--arch", "resnet50_mlp"]),
    ("segformer_b1", ["--arch", "segformer_b1"]),
    # no ImageNet pre-training: one lr for the whole network, longer warm-up
    ("segformer_b0_scratch", ["--arch", "segformer_b0", "--no-pretrained", "--lr", "6e-4", "--head-lr-mult", "1",
                              "--warmup", "1500"]),
]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="run names to run (default: all)")
    p.add_argument("--out", default="results")
    p.add_argument("--logs", default="logs")
    a, extra = p.parse_known_args()  # extra args are forwarded to src.train
    os.makedirs(a.logs, exist_ok=True)
    if sys.platform == "win32":
        # Keep Windows from idle-sleeping while this process runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    failed = []
    for name, train_args in EXPERIMENTS:
        if a.only and name not in a.only:
            continue
        if os.path.isfile(os.path.join(a.out, name, "metrics.json")):
            print(f"skip {name} (done)", flush=True)
            continue
        cmd = [sys.executable, "-W", "ignore", "-m", "src.train", "--name", name, "--out", a.out] + train_args + extra
        print(f"=== {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(os.path.join(a.logs, f"{name}.log"), "a", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace")
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            proc.wait()
        print(f"=== {name} finished with code {proc.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
        if proc.returncode != 0:
            failed.append(name)
    print("ALL DONE" if not failed else f"FAILED: {failed}", flush=True)
    sys.exit(1 if failed else 0)
