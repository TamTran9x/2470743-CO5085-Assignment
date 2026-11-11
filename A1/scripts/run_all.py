"""Run the full A1 experiment grid sequentially (2 architectures x 4 regimes).

Finished runs (results/<name>/metrics.json exists) are skipped and interrupted runs
resume from their last epoch, so the script can simply be re-launched.

    python -m scripts.run_all              # everything
    python -m scripts.run_all --only deit_s_full resnet50_full
"""
import argparse
import os
import subprocess
import sys
import time

# Pretrained regimes first (short), from-scratch last (long)
EXPERIMENTS = [
    ("deit_s", "linear"), ("resnet50", "linear"),
    ("deit_s", "full"), ("resnet50", "full"),
    ("deit_s", "partial"), ("resnet50", "partial"),
    ("resnet50", "scratch"), ("deit_s", "scratch"),
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
    for arch, mode in EXPERIMENTS:
        name = f"{arch}_{mode}"
        if a.only and name not in a.only:
            continue
        if os.path.isfile(os.path.join(a.out, name, "metrics.json")):
            print(f"skip {name} (done)")
            continue
        cmd = [sys.executable, "-m", "src.train", "--arch", arch, "--mode", mode, "--out", a.out] + extra
        print(f"=== {name}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        with open(os.path.join(a.logs, f"{name}.log"), "a", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
                log.flush()
            proc.wait()
        print(f"=== {name} finished with code {proc.returncode} in {(time.time()-t0)/60:.1f} min", flush=True)
        if proc.returncode != 0:
            sys.exit(proc.returncode)
