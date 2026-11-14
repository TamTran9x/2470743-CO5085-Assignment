"""Render index.html (the A2 page) and README.md from their templates in page/.

Every number is taken from results/ so the page cannot drift from the runs. A template contains
`{{ expression }}` placeholders that are evaluated with these names:
    S[run]   row of results/summary.csv (floats)        A   results/analysis.json
    R        results/robustness.json                     E   results/erf.json
    D        results/eda.json                            B   results/benchmark.json
    V        results/verify_segformer.json
    f(x, d)  number with a decimal comma                 sg(x, d)  the same with an explicit sign
    n(x)     integer with thousands dots                 pair(a, b)  paired-bootstrap entry of A
    results_table(), results_table_md(), robustness_table()

    python -m scripts.build_page
"""
import json
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["deeplabv3plus_r50", "resnet50_mlp", "segformer_b0", "segformer_b1", "segformer_b2", "segformer_b0_scratch"]


def load(name):
    with open(os.path.join(ROOT, "results", name), encoding="utf-8") as fh:
        return json.load(fh)


def f(x, d=2):
    return f"{float(x):.{d}f}".replace(".", ",").replace("-", "−")


def sg(x, d=2):
    return ("+" if float(x) >= 0 else "") + f(x, d)


def n(x):
    return f"{int(x):,}".replace(",", ".")


df = pd.read_csv(os.path.join(ROOT, "results", "summary.csv"))
S = {r.run: r for r in df.itertuples()}
A, R, E, D, B, V = (load(x) for x in ("analysis.json", "robustness.json", "erf.json", "eda.json", "benchmark.json",
                                      "verify_segformer.json"))
RUNS = [r for r in ORDER if r in S]


def pair(a, b):
    return A["pairs"][f"{a}__vs__{b}"]


def ci(p):
    return f"[{sg(p['ci_low'])}; {sg(p['ci_high'])}]"


def dot(run):
    return '<span class="dot {}"></span>'.format("cnn" if S[run].family == "CNN" else
                                                 ("vit" if S[run].pretrained else "scratch"))


def results_table():
    best = max(S[r].miou for r in RUNS)
    rows = ["<tr><th>Mô hình</th><th>Pre-train</th><th>mIoU (%)</th><th>mDice (%)</th><th>Boundary mIoU (%)</th>"
            "<th>Tham số (M)</th><th>GMACs</th><th>FPS</th><th>Train (phút)</th><th>Mem (GB)</th></tr>"]
    for r in RUNS:
        s = S[r]
        cls = ' class="best"' if s.miou == best else ""
        rows.append(f"<tr><td>{dot(r)}{s.label}</td><td>{'IN-1k' if s.pretrained else '–'}</td>"
                    f"<td{cls}>{f(s.miou)}</td><td>{f(s.mdice)}</td><td>{f(s.band_miou)}</td><td>{f(s.params_M, 1)}</td>"
                    f"<td>{f(s.gmacs, 1)}</td><td>{f(s.fps, 0)}</td><td>{f(s.train_min, 0)}</td>"
                    f"<td>{f(s.peak_mem_gb, 1)}</td></tr>")
    return '<div class="table-wrap"><table>\n' + "\n".join(rows) + "\n</table></div>"


def results_table_md():
    rows = ["| Mô hình | Pre-train | mIoU (%) | mDice (%) | Boundary mIoU (%) | Tham số (M) | GMACs | FPS | "
            "Train (phút) | Mem (GB) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in RUNS:
        s = S[r]
        rows.append(f"| {s.label} | {'IN-1k' if s.pretrained else '–'} | {f(s.miou)} | {f(s.mdice)} | "
                    f"{f(s.band_miou)} | {f(s.params_M, 1)} | {f(s.gmacs, 1)} | {f(s.fps, 0)} | {f(s.train_min, 0)} | "
                    f"{f(s.peak_mem_gb, 1)} |")
    return "\n".join(rows)


def robustness_table():
    rows = ["<tr><th>Mô hình</th><th>Ảnh gốc</th><th>Nhiễu (noise)</th><th>Mờ (blur)</th><th>Digital</th>"
            "<th>Trung bình 27 điều kiện</th><th>Giữ lại (%)</th></tr>"]
    for r in [x for x in ORDER if x in R["summary"]]:
        s = R["summary"][r]
        rows.append(f"<tr><td>{dot(r)}{S[r].label}</td><td>{f(s['clean'])}</td><td>{f(s['mean_noise'])}</td>"
                    f"<td>{f(s['mean_blur'])}</td><td>{f(s['mean_digital'])}</td><td>{f(s['mean_corrupted'])}</td>"
                    f"<td>{f(100 * s['retention'], 1)}</td></tr>")
    return '<div class="table-wrap"><table class="plain">\n' + "\n".join(rows) + "\n</table></div>"


def render(template, out):
    with open(os.path.join(ROOT, "page", template), encoding="utf-8") as fh:
        text = fh.read()
    text = re.sub(r"\{\{(.+?)\}\}", lambda m: str(eval(m.group(1).strip(), globals())), text, flags=re.S)
    with open(os.path.join(ROOT, out), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("wrote", out)


if __name__ == "__main__":
    render("index.template.html", "index.html")
    render("README.template.md", "README.md")
