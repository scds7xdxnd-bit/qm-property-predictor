#!/usr/bin/env python3
"""Assemble the Hugging Face Space directory from the main project.

A Space is a flat repo with app.py at the root, so this copies the
modules it needs, precomputes the dataset and features, and writes them
alongside -- the Space must not download or featurize at startup, or
every cold boot takes minutes.

    python space/build_space.py
    cd space/build && git init && git add -A && git commit -m "init"
    git remote add origin https://huggingface.co/spaces/<user>/<name>
    git push origin main
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = Path(__file__).resolve().parent
BUILD = SPACE / "build"

MODULES = ["__init__.py", "config.py", "data.py", "features.py",
           "splits.py", "models.py", "evaluate.py", "qm.py",
           "explain.py", "external.py"]


SMOKE = r"""
import os, sys
os.environ["QMPROP_NO_LAUNCH"] = "1"
sys.path.insert(0, ".")
import numpy as np
import app

# A normal query must work, and must not silently return an error card.
_, md, _ = app.predict("aspirin")
assert "Predicted log S" in md, f"normal query failed: {md[:200]}"

# The regression this smoke test exists for: RDKit's Ipc reaches 1e54 on
# large molecules, which is finite in float64 and inf in float32. A stale
# features.py shipped that bug to production once, where it surfaced as
# an xgboost "Check failed: valid" on any large input.
big = ".".join(["c1ccc2c(c1)ccc1c2cccc1"] * 12)
_, md, _ = app.predict(big)
assert "Predicted log S" in md, f"large molecule failed: {md[:200]}"

# And the domain warning must still fire for chemistry it has not seen.
_, md, _ = app.predict("table salt")
assert "applicability domain" in md, "domain warning missing"
print("smoke: normal, oversized and out-of-domain queries all OK")
"""


def smoke_test() -> None:
    """Load the built app and run three queries before anything ships.

    Building the Space copies modules from src/; nothing guarantees those
    copies are current or that the app still imports. This has already
    caught one production bug the hard way, so it now runs every build.
    """
    print("\nrunning smoke test against the built app...")
    result = subprocess.run(
        [sys.executable, "-c", SMOKE], cwd=BUILD,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-3000:])
        raise SystemExit("smoke test FAILED -- not shipping this build")
    print("  " + result.stdout.strip().splitlines()[-1])


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))

    dataset = ROOT / "data" / "processed" / "dataset.csv"
    features = ROOT / "data" / "interim" / "features.npz"
    for path, script in [(dataset, "01_fetch_data.py"),
                         (features, "02_featurize.py")]:
        if not path.exists():
            print(f"{path.name} missing -- running scripts/{script}")
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script)],
                cwd=ROOT / "scripts", check=True,
            )

    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "qmprop").mkdir(parents=True)
    (BUILD / "data" / "processed").mkdir(parents=True)
    (BUILD / "data" / "interim").mkdir(parents=True)

    for name in MODULES:
        shutil.copy2(ROOT / "src" / "qmprop" / name, BUILD / "qmprop" / name)

    shutil.copy2(ROOT / "app" / "app.py", BUILD / "app.py")
    shutil.copy2(ROOT / "config.yaml", BUILD / "config.yaml")
    shutil.copy2(SPACE / "README.md", BUILD / "README.md")
    shutil.copy2(SPACE / "requirements.txt", BUILD / "requirements.txt")
    shutil.copy2(dataset, BUILD / "data" / "processed" / "dataset.csv")
    shutil.copy2(features, BUILD / "data" / "interim" / "features.npz")

    # app.py walks up two parents to find src/; in the flat Space layout
    # qmprop sits beside it, so drop that line.
    app = BUILD / "app.py"
    app.write_text(app.read_text().replace(
        'sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))',
        "# (Space layout: qmprop/ sits beside app.py, no path juggling needed)",
    ))

    # config.py resolves paths from parents[2], which is wrong when
    # flattened. Point it at the Space root instead.
    cfg = BUILD / "qmprop" / "config.py"
    cfg.write_text(cfg.read_text().replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[2]",
        "PROJECT_ROOT = Path(__file__).resolve().parents[1]",
    ))

    # Testing the build locally leaves __pycache__ behind; bytecode from
    # this machine's interpreter is useless (and wrong) on the Space's.
    for pc in BUILD.rglob("__pycache__"):
        shutil.rmtree(pc, ignore_errors=True)

    smoke_test()

    # The smoke run leaves fresh bytecode behind; clear it again so the
    # Space never receives .pyc files built by this machine's interpreter.
    for pc in BUILD.rglob("__pycache__"):
        shutil.rmtree(pc, ignore_errors=True)

    size = sum(f.stat().st_size for f in BUILD.rglob("*") if f.is_file())
    print(f"\nbuilt {BUILD}  ({size / 1e6:.1f} MB)")
    for f in sorted(BUILD.rglob("*")):
        if f.is_file():
            print("  ", f.relative_to(BUILD))
    print("\nNext:")
    print("  cd space/build")
    print("  git init -b main && git add -A && git commit -m 'Solubility predictor'")
    print("  git remote add origin https://huggingface.co/spaces/<user>/<space>")
    print("  git push origin main")
    print("\nOr, with the token from `hf auth login` already stored:")
    print("  from huggingface_hub import HfApi")
    print("  HfApi().upload_folder(folder_path='space/build',")
    print("      repo_id='<user>/<space>', repo_type='space',")
    print("      ignore_patterns=['__pycache__/*', '*.pyc'])")


if __name__ == "__main__":
    main()
