"""Compile report/report.typ -> report/report.pdf (tables are read from results/*.csv|json).

    python -m scripts.build_report
"""
import os

import typst

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    src = os.path.join(ROOT, "report", "report.typ")
    out = os.path.join(ROOT, "report", "report.pdf")
    _, warnings = typst.compile_with_warnings(src, output=out, root=ROOT)
    for w in warnings:
        print("warning:", w)
    print("wrote", out)
