#!/usr/bin/env python3
"""Does more data change the Ch 3 vs Ch 4 verdict? (Ch 2 + Ch 4)

The README claims graph networks lose on ESOL because 893 training
molecules is too few, and that they start winning "around 10^5". That is
a claim, and AqSolDB makes it cheap to test part of it: 9,982 curated
solubility measurements, ~9x ESOL, same property, same units.

This script builds the enriched dataset and reruns the comparison on a
learning curve. If the gap between XGBoost and the GNN narrows as n
grows, the capacity explanation holds. If it does not, the explanation
was wrong and the GNN is simply worse here.

Every point on the curve uses a scaffold split, so the trend is not an
artifact of easier test sets at larger n.

    python scripts/09_scale.py                 # fetch + curve
    python scripts/09_scale.py --sizes 1000 3000 9000
    python scripts/09_scale.py --no-gnn        # trees only, fast
"""

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from qmprop import load_config
from qmprop.evaluate import regression_metrics
from qmprop.external import build_enriched
from qmprop.features import build_features
from qmprop.models import build_model
from qmprop.splits import scaffold_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("scale")


def subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if n >= len(df) else df.sample(n=n, random_state=seed).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[1000, 2000, 4000, 8000, 10000])
    ap.add_argument("--no-gnn", action="store_true")
    ap.add_argument("--gnn-epochs", type=int, default=120)
    args = ap.parse_args()

    cfg = load_config()
    scfg = cfg["split"]
    seed = scfg["seed"]

    df = build_enriched(cfg)
    print(f"\nenriched dataset: {len(df)} molecules "
          f"({len(df) / 1117:.1f}x ESOL)\n")

    rows = []
    for n in args.sizes:
        sub = subsample(df, n, seed)
        smiles = sub["smiles"].tolist()
        y = sub["logS"].to_numpy(dtype=float)

        X, _ = build_features(smiles, morgan_kwargs=cfg["features"]["morgan"])
        tr, _, te = scaffold_split(smiles, scfg["frac_train"],
                                   scfg["frac_valid"], scfg["frac_test"], seed)

        for name in ("xgboost", "random_forest"):
            t = time.time()
            m = build_model(name, seed=seed)
            m.fit(X[tr], y[tr])
            met = regression_metrics(y[te], m.predict(X[te]))
            met.update(model=name, n=len(sub), n_train=len(tr),
                       seconds=round(time.time() - t, 1))
            rows.append(met)
            log.info("n=%-6d %-14s RMSE %.3f  R2 %.3f", len(sub), name,
                     met["rmse"], met["r2"])

        if not args.no_gnn:
            met = run_gnn(smiles, y, tr, te, seed, args.gnn_epochs)
            met.update(model="gnn", n=len(sub), n_train=len(tr))
            rows.append(met)
            log.info("n=%-6d %-14s RMSE %.3f  R2 %.3f", len(sub), "gnn",
                     met["rmse"], met["r2"])

    report(rows, cfg)


def run_gnn(smiles, y, tr, te, seed, epochs):
    import torch

    from qmprop.gnn import build_gnn, collate, mol_to_graph

    torch.set_num_threads(2)     # the QM run may still own the other cores
    graphs = [mol_to_graph(s) for s in smiles]
    keep = [i for i, g in enumerate(graphs) if g is not None]
    valid = set(keep)
    tr = np.array([i for i in tr if i in valid])
    te = np.array([i for i in te if i in valid])

    model = build_gnn(seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)

    t0 = time.time()
    for _ in range(epochs):
        model.train()
        order = rng.permutation(tr)
        for s in range(0, len(order), 32):
            chunk = order[s:s + 32]
            x, e, k, b = collate([graphs[i] for i in chunk])
            opt.zero_grad()
            out = model(torch.tensor(x), torch.tensor(e), torch.tensor(k),
                        torch.tensor(b), len(chunk))
            loss = loss_fn(out, torch.tensor(y[chunk], dtype=torch.float32))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    model.eval()
    preds = np.zeros(len(te))
    with torch.no_grad():
        for s in range(0, len(te), 64):
            chunk = te[s:s + 64]
            x, e, k, b = collate([graphs[i] for i in chunk])
            preds[s:s + len(chunk)] = model(
                torch.tensor(x), torch.tensor(e), torch.tensor(k),
                torch.tensor(b), len(chunk)).numpy()

    met = regression_metrics(y[te], preds)
    met["seconds"] = round(time.time() - t0, 1)
    return met


def report(rows, cfg) -> None:
    out = cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "scale.json").write_text(json.dumps(rows, indent=2))

    models = sorted({r["model"] for r in rows})
    sizes = sorted({r["n"] for r in rows})
    print("\n== scaffold-split RMSE vs training-set size ==\n")
    header = f"{'n':>7}" + "".join(f"{m:>16}" for m in models)
    print(header)
    print("-" * len(header))
    for n in sizes:
        line = f"{n:>7}"
        for m in models:
            hit = [r for r in rows if r["n"] == n and r["model"] == m]
            line += f"{hit[0]['rmse']:>16.3f}" if hit else f"{'-':>16}"
        print(line)

    if "gnn" in models and "xgboost" in models:
        print("\ngap (gnn RMSE - xgboost RMSE), negative = GNN ahead:")
        for n in sizes:
            g = [r for r in rows if r["n"] == n and r["model"] == "gnn"]
            x = [r for r in rows if r["n"] == n and r["model"] == "xgboost"]
            if g and x:
                print(f"  n={n:<7} {g[0]['rmse'] - x[0]['rmse']:+.3f}")
        print("\nA gap that shrinks as n grows supports the capacity")
        print("explanation. A flat one means the architecture, not the data,")
        print("is the limit -- and the README claim needs rewriting.")
    print(f"\nwrote {out / 'scale.json'}")


if __name__ == "__main__":
    main()
