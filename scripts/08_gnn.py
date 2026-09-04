#!/usr/bin/env python3
"""Ch 4 stretch goal: a graph network, scored on the same split.

The comparison only means something if nothing else changes, so this
uses the identical scaffold split, the identical target, and the
identical test set as `04_train.py`. The only difference is that the
model sees the molecular graph instead of a fingerprint.

    python scripts/08_gnn.py
    python scripts/08_gnn.py --epochs 300 --hidden 128

A held-out slice of the training set is used for early stopping. It is
carved out by scaffold as well -- validating on a random slice would
pick the epoch that best fits scaffolds the model has already seen, and
that choice does not transfer to the scaffold-split test set.
"""

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from qmprop import load_config
from qmprop.evaluate import parity_plot, regression_metrics
from qmprop.gnn import build_gnn, collate, mol_to_graph
from qmprop.splits import scaffold_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("gnn")


def batches(indices, graphs, y, batch_size, rng=None):
    order = np.array(indices)
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        yield chunk, collate([graphs[i] for i in chunk]), y[chunk]


def run_epoch(model, opt, loss_fn, indices, graphs, y, batch_size, torch,
              rng=None, train: bool = True):
    model.train() if train else model.eval()
    total, seen = 0.0, 0
    predictions = np.zeros(len(y))

    for chunk, (x, e, k, b), target in batches(
            indices, graphs, y, batch_size, rng):
        args = (torch.tensor(x), torch.tensor(e), torch.tensor(k),
                torch.tensor(b), len(chunk))
        if train:
            opt.zero_grad()
            out = model(*args)
            loss = loss_fn(out, torch.tensor(target, dtype=torch.float32))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        else:
            with torch.no_grad():
                out = model(*args)
                loss = loss_fn(out, torch.tensor(target, dtype=torch.float32))
        predictions[chunk] = out.detach().numpy()
        total += loss.item() * len(chunk)
        seen += len(chunk)
    return total / max(seen, 1), predictions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=40)
    args = ap.parse_args()

    import torch

    cfg = load_config()
    out_dir = cfg["output_dir"]
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["data_dir"] / "processed" / "dataset.csv")
    target = cfg["dataset"]["target_name"]
    smiles = df["smiles"].tolist()
    y = df[target].to_numpy(dtype=float)

    graphs = [mol_to_graph(s) for s in smiles]
    bad = [i for i, g in enumerate(graphs) if g is None]
    if bad:
        log.warning("%d molecules failed to graph; dropping", len(bad))
        keep = [i for i in range(len(graphs)) if i not in set(bad)]
        graphs = [graphs[i] for i in keep]
        smiles = [smiles[i] for i in keep]
        y = y[keep]

    s = cfg["split"]
    train_idx, _, test_idx = scaffold_split(
        smiles, s["frac_train"], s["frac_valid"], s["frac_test"], s["seed"])

    # Inner scaffold split for early stopping -- see module docstring.
    inner_train_pos, _, inner_val_pos = scaffold_split(
        [smiles[i] for i in train_idx], 0.85, 0.0, 0.15, s["seed"])
    fit_idx = train_idx[inner_train_pos]
    val_idx = train_idx[inner_val_pos]
    log.info("fit %d | val %d | test %d", len(fit_idx), len(val_idx),
             len(test_idx))

    model = build_gnn(hidden=args.hidden, rounds=args.rounds, seed=s["seed"])
    n_params = sum(p.numel() for p in model.parameters())
    log.info("%d parameters for %d training molecules -- %.0f params per "
             "molecule", n_params, len(fit_idx), n_params / len(fit_idx))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=0.5, patience=15)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(s["seed"])

    best_rmse, best_state, best_epoch = float("inf"), None, -1
    t0 = time.time()
    for epoch in range(args.epochs):
        run_epoch(model, opt, loss_fn, fit_idx, graphs, y, args.batch_size,
                  torch, rng=rng, train=True)
        _, val_pred = run_epoch(model, opt, loss_fn, val_idx, graphs, y,
                                args.batch_size, torch, train=False)
        val_rmse = float(np.sqrt(np.mean((val_pred[val_idx] - y[val_idx]) ** 2)))
        sched.step(val_rmse)

        if val_rmse < best_rmse - 1e-4:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if epoch - best_epoch >= args.patience:
            log.info("early stop at epoch %d (best %d)", epoch, best_epoch)
            break
        if epoch % 25 == 0:
            log.info("epoch %3d  val RMSE %.3f  (best %.3f @ %d)",
                     epoch, val_rmse, best_rmse, best_epoch)

    model.load_state_dict(best_state)
    _, test_pred = run_epoch(model, opt, loss_fn, test_idx, graphs, y,
                             args.batch_size, torch, train=False)
    metrics = regression_metrics(y[test_idx], test_pred[test_idx])
    metrics.update(model="gnn", split=s["method"], params=n_params,
                   epochs_run=best_epoch + 1,
                   seconds=round(time.time() - t0, 1))

    print("\n== GNN on the scaffold split ==\n")
    print(f"RMSE {metrics['rmse']:.3f}   MAE {metrics['mae']:.3f}   "
          f"R2 {metrics['r2']:.3f}   n={metrics['n']}")

    baseline_path = out_dir / f"baselines_{s['method']}.json"
    if baseline_path.exists():
        rows = json.loads(baseline_path.read_text())
        rows = [r for r in rows if r["model"] != "gnn"] + [metrics]
        rows.sort(key=lambda r: r["rmse"])
        baseline_path.write_text(json.dumps(rows, indent=2))

        print(f"\n{'model':<16}{'RMSE':>8}{'MAE':>8}{'R2':>8}")
        for r in rows:
            print(f"{r['model']:<16}{r['rmse']:>8.3f}{r['mae']:>8.3f}"
                  f"{r['r2']:>8.3f}")
        winner = rows[0]
        if winner["model"] != "gnn":
            print(f"\nThe GNN does not win: {winner['model']} is ahead by "
                  f"{metrics['rmse'] - winner['rmse']:.3f} log units. With "
                  f"{len(fit_idx)} training molecules that is the expected "
                  f"result -- graph networks need far more data before their "
                  f"inductive bias pays for their parameters.")

    figure = out_dir / "figures" / f"parity_{s['method']}_gnn.png"
    parity_plot(y[test_idx], test_pred[test_idx],
                f"GNN, {s['method']} split", figure)
    print(f"wrote {figure}")


if __name__ == "__main__":
    main()
