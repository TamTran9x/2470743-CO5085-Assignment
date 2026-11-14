"""Assemble the folders to upload to GitHub (git is not required on this machine).

  dist/2470743-CO5085-Assignment/   -> repository 2470743-CO5085-Assignment
      index.html, README.md         (landing page of A1-A2, updated with the A2 entry)
      A2/                           (this assignment: page, code, results, figures, logs, report)
  dist/2470743-CO5085-Exercise1/    -> repository 2470743-CO5085-Exercise1
      index.html                    (home page with the links to E1-E3, A1 and A2)

Data, checkpoints and virtual environments are not copied.

    python -m scripts.make_dist
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A2_ITEMS = ["index.html", "README.md", "requirements.txt", ".gitignore", "src", "scripts", "page", "results",
            "figures", "logs", "report", "demo"]
# per-run training logs are kept; driver logs, pid files and the large ERF arrays are not
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pid", "run_all.*.log", "pipeline.*.log", "post_train.*.log",
                              "robustness.*.log", "erf.npz")

if __name__ == "__main__":
    dist = os.path.join(ROOT, "dist")
    if os.path.isdir(dist):
        shutil.rmtree(dist)
    repo = os.path.join(dist, "2470743-CO5085-Assignment")
    a2 = os.path.join(repo, "A2")
    os.makedirs(a2)
    for item in A2_ITEMS:
        src = os.path.join(ROOT, item)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(a2, item), ignore=SKIP)
        elif os.path.isfile(src):
            shutil.copy2(src, os.path.join(a2, item))
    shutil.copy2(os.path.join(ROOT, "landing", "assignment_index.html"), os.path.join(repo, "index.html"))
    shutil.copy2(os.path.join(ROOT, "landing", "assignment_README.md"), os.path.join(repo, "README.md"))
    home = os.path.join(dist, "2470743-CO5085-Exercise1")
    os.makedirs(home)
    shutil.copy2(os.path.join(ROOT, "landing", "index.html"), os.path.join(home, "index.html"))

    total = 0
    for base, _, files in os.walk(dist):
        total += sum(os.path.getsize(os.path.join(base, f)) for f in files)
    print(f"wrote {dist} ({total / 1e6:.1f} MB)")
    for name in sorted(os.listdir(a2)):
        print("  A2/" + name)
