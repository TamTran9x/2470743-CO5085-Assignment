"""Render the Overleaf (XeLaTeX) version of the report from report/overleaf/main.template.tex.

The template holds the same text as report/report.typ; `<< expression >>` placeholders are
evaluated against results/ exactly like the page templates, so the numbers cannot drift from
the Typst report. Figures are copied next to main.tex and everything is zipped for upload.

    python -m scripts.build_overleaf
"""
import csv
import json
import os
import re
import shutil
import zipfile

from scripts.build_page import A, B, D, E, R, RUNS, S, V, n, pair

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "overleaf")
B2, B1, B0 = "segformer_b2", "segformer_b1", "segformer_b0"
DL, RM, SC = "deeplabv3plus_r50", "resnet50_mlp", "segformer_b0_scratch"
T = D["splits"]["train"]
META = D["meta"]
M = {}
for _r in RUNS:
    with open(os.path.join(ROOT, "results", _r, "metrics.json")) as _f:
        M[_r] = json.load(_f)
LAB = {r: S[r].label for r in RUNS}
H = {}
for _r in RUNS:
    with open(os.path.join(ROOT, "results", _r, "history.csv"), encoding="utf-8") as _f:
        H[_r] = list(csv.DictReader(_f))


def hist(run, i, d=2, key="minival_miou"):
    """Value of the learning curve at evaluation point i (0 = first, -1 = last)."""
    return f(H[run][i][key], d)


def f(x, d=2):
    """Number with a decimal comma; a negative sign is typeset in math mode."""
    s = f"{abs(float(x)):.{d}f}".replace(".", ",")
    return ("$-$" if round(float(x), d) < 0 else "") + s


def sg(x, d=2):
    return ("+" if round(float(x), d) >= 0 else "") + f(x, d)


def v(run, key, d=2):
    return f(getattr(S[run], key), d)


def miou(run):
    return v(run, "miou")


def gain(a, b, key="miou", d=2):
    return f(getattr(S[a], key) - getattr(S[b], key), d)


def ci(p):
    return f"[{sg(p['ci_low'])}; {sg(p['ci_high'])}]"


def pval(p):
    return "< 0,001" if p < 0.001 else f(p, 3)


def bench(kind, arch):
    return next(r for r in B[kind] if r["arch"] == arch)


def conf(run, key):
    return f(A["runs"][run]["confusion_rate"][key], 1)


def erf(run, target, key=None):
    s = E["runs"][run][target]
    return str(s[key]) if key else f"{s['side_50']} / {s['side_90']}"


def rows(lines):
    return " \\\\\n".join(" & ".join(str(c) for c in line) for line in lines) + " \\\\"


def encoder_name(run):
    return "ResNet-50" if S[run].encoder == "resnet50" else "MiT-" + S[run].encoder.upper()


ROLE = {DL: "baseline CNN", RM: "ablation: CNN + decoder MLP", B2: "mô hình chính, cùng cỡ với baseline",
        SC: "không pretrain", B0: "ablation: kích thước", B1: "ablation: kích thước"}


def rows_models():
    return rows([LAB[r], encoder_name(r), "ASPP + skip (V3+)" if S[r].decoder == "aspp" else "MLP",
                 v(r, "params_M"), v(r, "gmacs", 1), ROLE[r]] for r in RUNS)


def rows_decoder():
    out = []
    for a in (B0, B1, B2, RM):
        lit, mer = bench("literal", a), bench("merged", a)
        out.append([LAB[a], f(lit["gmacs_512"], 1), f(lit["peak_mem_gb"]), f(lit["img_per_s"], 1),
                    f(mer["gmacs_512"], 1), f(mer["peak_mem_gb"]), f(mer["img_per_s"], 1)])
    return rows(out)


def rows_main():
    return rows([LAB[r], "ImageNet-1k" if S[r].pretrained else "không", v(r, "miou"),
                 f"[{v(r, 'ci_low', 1)}; {v(r, 'ci_high', 1)}]", v(r, "mdice"), v(r, "pixel_acc"), v(r, "band_miou"),
                 v(r, "minival_miou")] for r in RUNS)


def rows_cost():
    return rows([LAB[r], v(r, "params_M"), v(r, "gmacs", 1), v(r, "train_min", 1), v(r, "train_img_s", 1),
                 v(r, "peak_mem_gb"), v(r, "fps", 1), v(r, "fps_fp32", 1)] for r in RUNS)


def rs(run, key):
    return R["summary"][run][key]


def sc(run, s):
    return R["scale"][run][s]


def dsc(run, s):
    return sg(sc(run, s) - sc(run, "1.0"), 1)


def rows_pairs():
    order = [(B2, DL), (B2, RM), (DL, RM), (B0, RM), (B1, B0), (B2, B1), (DL, B1), (B0, SC)]
    return rows([LAB[a], LAB[b], sg(pair(a, b)["diff"]), ci(pair(a, b)), pval(pair(a, b)["p_value"])]
                for a, b in order)


def rows_boundary():
    return rows([LAB[r], v(r, "miou"), f(A["runs"][r]["interior_miou"]), f(A["runs"][r]["band_miou"]),
                 f(A["runs"][r]["interior_miou"] - A["runs"][r]["band_miou"], 1)] for r in RUNS if S[r].pretrained)


def rows_erf():
    return rows([LAB[r]] + [erf(r, t) for t in E["targets"]] for r in (DL, RM, B0, B2))


def rows_robust():
    out = []
    for r in [x for x in RUNS if x in R["summary"]]:
        s = R["summary"][r]
        out.append([LAB[r], f(s["clean"]), f(s["mean_noise"]), f(s["mean_blur"]), f(s["mean_digital"]),
                    f(s["mean_corrupted"]), f(100 * s["retention"], 1)])
    return rows(out)


def rows_scale():
    return rows([LAB[r]] + [f(R["scale"][r][str(s)]) for s in R["scales"]] for r in RUNS if r in R["scale"])


def render():
    with open(os.path.join(OUT, "main.template.tex"), encoding="utf-8") as fh:
        text = fh.read()
    text = re.sub(r"<<(.+?)>>", lambda m: str(eval(m.group(1).strip(), globals())), text, flags=re.S)
    with open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def in_math_block(lines):
    """True while the last kept line is inside a display-math block opened with \\[ ."""
    text = "\n".join(lines)
    return text.count("\\[") > text.count("\\]")


def check(text):
    """Cheap static checks, since there is no TeX installation on the training machine."""
    problems = []
    in_verbatim, in_tabular, stack, kept = False, 0, [], []
    for no, line in enumerate(text.splitlines(), 1):
        if "\\begin{Verbatim}" in line:
            in_verbatim = True
        if in_verbatim:
            in_verbatim = "\\end{Verbatim}" not in line
            continue
        if line.lstrip().startswith("%"):
            continue
        kept.append(line)
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", line):
            if m.group(1) == "begin":
                stack.append(m.group(2))
            elif not stack or stack.pop() != m.group(2):
                problems.append(f"line {no}: \\end{{{m.group(2)}}} does not match")
            in_tabular += (m.group(2).startswith("tabular")) * (1 if m.group(1) == "begin" else -1)
        if re.search(r"(?<!\\)%", line):
            problems.append(f"line {no}: unescaped %")
        if re.search(r"(?<!\\)#", line) and "\\newcommand" not in line:
            problems.append(f"line {no}: unescaped #")
        if len(re.findall(r"(?<!\\)\$", line)) % 2:
            problems.append(f"line {no}: odd number of $")
        no_math = re.sub(r"\$[^$]*\$", "", line)
        no_math = re.sub(r"\\(label|ref|includegraphics(\[[^\]]*\])?|href)\{[^}]*\}", "", no_math)
        if re.search(r"(?<!\\)[_^]", no_math) and "\\[" not in line and not in_math_block(kept):
            problems.append(f"line {no}: unescaped _ or ^")
        if not in_tabular and re.search(r"(?<!\\)&", no_math) and "\\begin{tabular" not in line:
            problems.append(f"line {no}: & outside a table")
    body = "\n".join(kept)
    if body.count("{") - body.count("\\{") != body.count("}") - body.count("\\}"):
        problems.append("unbalanced braces")
    problems += [f"unclosed environment {e}" for e in stack]
    if re.search(r"<<|>>", body):
        problems.append("unrendered placeholder")
    figures = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body))
    missing = [x for x in figures if not os.path.isfile(os.path.join(ROOT, "figures", x))]
    problems += [f"missing figure {x}" for x in missing]
    return problems, sorted(figures)


if __name__ == "__main__":
    text = render()
    problems, figures = check(text)
    fig_dir = os.path.join(OUT, "figures")
    shutil.rmtree(fig_dir, ignore_errors=True)
    os.makedirs(fig_dir)
    for name in figures:
        if os.path.isfile(os.path.join(ROOT, "figures", name)):
            shutil.copy2(os.path.join(ROOT, "figures", name), fig_dir)
    zip_path = os.path.join(ROOT, "report", "A2_report_overleaf.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("main.tex", "latexmkrc"):
            z.write(os.path.join(OUT, name), name)
        for name in figures:
            z.write(os.path.join(fig_dir, name), "figures/" + name)
    print(f"wrote report/overleaf/main.tex ({len(text) / 1000:.0f} kB), {len(figures)} figures, {zip_path}")
    print("static checks:", "OK" if not problems else problems)
