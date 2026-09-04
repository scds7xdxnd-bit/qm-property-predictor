#!/usr/bin/env python3
"""B3LYP/6-31G* single points on a subset -> QM descriptors.

The slow step. Roughly 5-60 s per molecule on a laptop depending on size.
Results are appended to CSV after every molecule, so interrupting and
rerunning resumes instead of starting over.

Parallelism is across molecules, not inside one SCF. For systems this
small PySCF's own threading barely helps -- measured at ~90% of a single
core with OMP_NUM_THREADS=6 -- because the matrices are too small to
amortize the sync. Independent single-threaded workers scale nearly
linearly instead. Each worker pins itself to one thread; without that
the pool oversubscribes and runs slower than serial.

MEMORY, not cores, is the binding constraint. A 20-heavy-atom molecule
at 6-31G* was measured here at 1-2.6 GB resident, so the default is
derived from RAM rather than CPU count. Seven workers on an 8 GB machine
drove it into swap: worker CPU collapsed to ~28% each while kernel_task
burned a full core on memory compression, and throughput was worse than
with three. More workers is not more throughput once you are swapping.

    python scripts/03_run_qm.py             # subset_size from config.yaml
    python scripts/03_run_qm.py --limit 10  # smoke test first, always
    python scripts/03_run_qm.py --workers 1 # serial, for debugging
"""

import argparse
import logging
import os
import time
from multiprocessing import Pool, cpu_count
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


def _default_workers() -> int:
    """Pick a worker count from RAM, falling back to cores.

    Budget ~2 GB per worker -- the high end of what a 20-heavy-atom
    6-31G* job was measured to need here -- and reserve 2 GB for the OS
    and whatever else the machine is doing. Without that reservation an
    8 GB laptop is told it can run 4 workers, and swaps. Capped by
    cores-1 so the parent still gets scheduled.
    """
    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        usable = max(total_bytes - 2 * 1024**3, 2 * 1024**3)
        by_memory = int(usable / (2 * 1024**3))
    except (ValueError, OSError, AttributeError):
        by_memory = 4
    return max(1, min(by_memory, cpu_count() - 1))


def _init_worker() -> None:
    """One thread per worker, set before PySCF is ever imported.

    `qm.py` imports pyscf lazily inside the function, so putting these in
    the environment here still lands before the library reads them.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _compute(job: tuple[str, dict]):
    """Run one molecule. Settings travel *with* the job, deliberately.

    macOS spawns rather than forks, so a child re-imports this module
    under __mp_main__ and never executes main(). Anything main() stuffed
    into a module-level global is therefore empty in the worker, and the
    molecules come back computed at the wrong level of theory with no
    error to show for it. Passing the kwargs in the payload is immune.
    """
    smiles, kwargs = job
    return qm_descriptors(smiles, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="override subset size (use a small value first)")
    parser.add_argument("--workers", type=int, default=None,
                        help="molecules computed in parallel "
                             "(default: from available RAM, see _default_workers)")
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

    kwargs = dict(
        basis=qcfg["basis"],
        xc=qcfg["xc"],
        max_heavy_atoms=qcfg["max_heavy_atoms"],
        conformer_seed=qcfg["conformer_seed"],
        mmff_max_iters=qcfg["mmff_max_iters"],
        density_fit=qcfg.get("density_fit", True),
        max_memory_mb=qcfg.get("max_memory_mb", 400),
    )
    def record(i: int, res, elapsed: float) -> None:
        """Append one result. The parent owns the file, so the workers
        never race for it -- and a crash still leaves the rows already
        earned on disk."""
        pd.DataFrame([res.as_row()]).to_csv(
            out, mode="a", header=not out.exists(), index=False)
        status = "ok " if res.ok else "FAIL"
        detail = f"gap {res.qm_gap_ev:6.3f} eV" if res.ok else res.reason[:48]
        log.info("[%3d/%3d] %s %5.1fs  %s  %s",
                 i, len(todo), status, elapsed, detail, res.smiles[:44])

    t0 = time.time()
    requested = args.workers if args.workers is not None else _default_workers()
    workers = max(1, min(requested, len(todo))) if todo else 1

    if workers == 1:
        for i, smi in enumerate(todo, 1):
            t1 = time.time()
            record(i, qm_descriptors(smi, **kwargs), time.time() - t1)
    else:
        log.info("using %d parallel workers, 1 thread each", workers)
        _init_worker()   # keep the parent from oversubscribing too
        jobs = [(smi, kwargs) for smi in todo]
        with Pool(workers, initializer=_init_worker) as pool:
            for i, res in enumerate(pool.imap_unordered(_compute, jobs), 1):
                # Per-molecule timing is meaningless when they overlap;
                # report the running average instead of a fake duration.
                record(i, res, (time.time() - t0) / i)

    if todo:
        log.info("total %.1f min for %d molecules (%.1f s/molecule wall)",
                 (time.time() - t0) / 60, len(todo),
                 (time.time() - t0) / len(todo))

    # Interrupted runs can leave a molecule written twice (killed between
    # compute and the parent's read of `done`). Duplicates would silently
    # weight those molecules twice in the ablation join, so collapse them
    # here rather than trusting every earlier run to have exited cleanly.
    results = pd.read_csv(out)
    before = len(results)
    results = results.drop_duplicates(subset="smiles", keep="last")
    if len(results) < before:
        log.info("dropped %d duplicate row(s)", before - len(results))
        results.to_csv(out, index=False)

    n_ok = int(results["ok"].sum())
    print(f"\n{out}: {n_ok}/{len(results)} succeeded")
    if n_ok < len(results):
        print("\nfailure reasons:")
        print(results.loc[~results["ok"], "reason"].value_counts().to_string())


if __name__ == "__main__":
    main()
