#!/usr/bin/env python3
"""Train the GNN in a process of its own. Not an optimization -- required.

PyTorch and XGBoost each bundle their own libomp.dylib. Loading both
into one process on macOS deadlocks: the second runtime's threads park
in __kmp_fork_barrier and never wake. Observed as a training loop that
sits at 0% CPU forever, with a stack ending

    at::native::addmm -> copy_ -> TensorIteratorBase::for_each
      -> __kmp_join_call -> __kmp_join_barrier -> __psynch_cvwait

and two different libomp load addresses in the same backtrace.

KMP_DUPLICATE_LIB_OK=TRUE silences the usual warning, but Intel
documents it as unsafe -- it can crash or silently corrupt results,
which is worse than hanging. So the GNN gets its own process, which
never imports xgboost, and the two runtimes never meet.

Reads a job JSON on argv[1], writes metrics JSON to argv[2].
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qmprop.evaluate import regression_metrics
from qmprop.gnn import build_gnn, collate, mol_to_graph


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    smiles = job["smiles"]
    y = np.array(job["y"], dtype=float)
    tr = np.array(job["train"], dtype=int)
    te = np.array(job["test"], dtype=int)
    seed, epochs = job["seed"], job["epochs"]

    import torch

    torch.set_num_threads(job.get("threads", 2))

    from qmprop.splits import scaffold_split

    graphs = [mol_to_graph(s) for s in smiles]
    valid = {i for i, g in enumerate(graphs) if g is not None}
    tr = np.array([i for i in tr if i in valid])
    te = np.array([i for i in te if i in valid])

    # Inner scaffold split for early stopping. This has to be scaffold
    # too: validating on a random slice picks the epoch that best fits
    # scaffolds the model has already seen, and that choice does not
    # transfer to a scaffold-split test set.
    inner_fit, _, inner_val = scaffold_split(
        [smiles[i] for i in tr], 0.85, 0.0, 0.15, seed)
    fit_idx, val_idx = tr[inner_fit], tr[inner_val]

    model = build_gnn(seed=seed, pooling=job.get("pooling", "sum"))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                       patience=15)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)
    patience = job.get("patience", 40)

    def predict(idx):
        model.eval()
        out = np.zeros(len(idx))
        with torch.no_grad():
            for s in range(0, len(idx), 64):
                chunk = idx[s:s + 64]
                x, e, k, b = collate([graphs[i] for i in chunk])
                out[s:s + len(chunk)] = model(
                    torch.tensor(x), torch.tensor(e), torch.tensor(k),
                    torch.tensor(b), len(chunk)).numpy()
        return out

    best_rmse, best_state, best_epoch = float("inf"), None, -1
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(fit_idx)
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

        val_rmse = float(np.sqrt(np.mean((predict(val_idx) - y[val_idx]) ** 2)))
        sched.step(val_rmse)
        if val_rmse < best_rmse - 1e-4:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = {k_: v.detach().clone()
                          for k_, v in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            print(f"  early stop at epoch {epoch} (best {best_epoch})",
                  file=sys.stderr, flush=True)
            break
        if epoch % 20 == 0:
            print(f"  gnn epoch {epoch}/{epochs} val {val_rmse:.3f} "
                  f"(best {best_rmse:.3f} @ {best_epoch})",
                  file=sys.stderr, flush=True)

    # Without this the reported number is whatever the last epoch
    # happened to land on, which is why a flat 100-epoch run got WORSE
    # as n grew: more data, more steps past the optimum, no way back.
    if best_state is not None:
        model.load_state_dict(best_state)
    preds = predict(te)

    met = regression_metrics(y[te], preds)
    met["seconds"] = round(time.time() - t0, 1)
    met["params"] = sum(p.numel() for p in model.parameters())
    met["best_epoch"] = best_epoch
    met["val_rmse"] = round(best_rmse, 4)
    met["pooling"] = job.get("pooling", "sum")
    Path(sys.argv[2]).write_text(json.dumps(met))


if __name__ == "__main__":
    main()
