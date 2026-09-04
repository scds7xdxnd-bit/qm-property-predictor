#!/usr/bin/env python3
"""B3LYP/6-31G* single points on a subset -> QM descriptors.

The slow step. Roughly 5-60 s per molecule on a laptop depending on size,
so 200 molecules is a coffee-to-overnight run. Results are appended to
CSV after every molecule, so interrupting and rerunning resumes instead
of starting over.

    python scripts/03_run_qm.py            # subset_size from config.yaml
    python scripts/03_run_qm.py --limit 10 # smoke test first, always
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from qmprop import load_config
from qmprop.qm import qm_descriptors

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("run_qm")


def select_subset(df: pd.DataFrame, n: int, max_heavy: int, seed: int) -> pd.DataFrame:
    """Pick molecules small enough to be affordable, sampled reproducibly.

    Sampling rather than taking the first n matters: the raw file is
    loosely ordered by compound class, so a head() slice would be a
    biased chemical subset and the ablation would generalize to nothing.
    """
    from rdkit import Chem

    heavy = df["smiles"].map(
        lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms()
    )
    affordable = df[heavy <= max_heavy]
    log.info(
        "%d/%d molecules within %d heavy atoms",
        len(affordable), len(df), max_heavy,
    )
    if len(affordable) <= n:
        return affordable.copy()
    return affordable.sample(n=n, random_state=seed).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="override subset size (use a small value first)")
    args = parser.parse_args()

    cfg = load_config()
    qcfg = cfg["qm"]
    df = pd.read_csv(cfg["data_dir"] / "processed" / "dataset.csv")

    n = args.limit if args.limit is not None else qcfg["subset_size"]
    subset = select_subset(df, n, qcfg["max_heavy_atoms"], cfg["split"]["seed"])

    out = Path(cfg["data_dir"] / "interim" / "qm_descriptors.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out.exists():
        done = set(pd.read_csv(out)["smiles"])
        log.info("resuming: %d molecules already computed", len(done))

    todo = [s for s in subset["smiles"] if s not in done]
    log.info("computing %d molecules at %s/%s",
             len(todo), qcfg["xc"], qcfg["basis"])

    t0 = time.time()
    for i, smi in enumerate(todo, 1):
        t1 = time.time()
        res = qm_descriptors(
            smi,
            basis=qcfg["basis"],
            xc=qcfg["xc"],
            max_heavy_atoms=qcfg["max_heavy_atoms"],
            conformer_seed=qcfg["conformer_seed"],
            mmff_max_iters=qcfg["mmff_max_iters"],
        )
        row = pd.DataFrame([res.as_row()])
        row.to_csv(out, mode="a", header=not out.exists(), index=False)

        status = "ok " if res.ok else "FAIL"
        detail = (
            f"gap {res.qm_gap_ev:6.3f} eV" if res.ok else res.reason[:48]
        )
        log.info("[%3d/%3d] %s %5.1fs  %s  %s",
                 i, len(todo), status, time.time() - t1, detail, smi[:44])

    if todo:
        log.info("total %.1f min", (time.time() - t0) / 60)

    results = pd.read_csv(out)
    n_ok = int(results["ok"].sum())
    print(f"\n{out}: {n_ok}/{len(results)} succeeded")
    if n_ok < len(results):
        print("\nfailure reasons:")
        print(results.loc[~results["ok"], "reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
