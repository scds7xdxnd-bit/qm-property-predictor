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

    graphs = [mol_to_graph(s) for s in smiles]
    valid = {i for i, g in enumerate(graphs) if g is not None}
    tr = np.array([i for i in tr if i in valid])
    te = np.array([i for i in te if i in valid])

    model = build_gnn(seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)

    t0 = time.time()
    for epoch in range(epochs):
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
        if epoch % 20 == 0:
            print(f"  gnn epoch {epoch}/{epochs}", file=sys.stderr, flush=True)

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
    met["params"] = sum(p.numel() for p in model.parameters())
    Path(sys.argv[2]).write_text(json.dumps(met))


if __name__ == "__main__":
    main()
