#!/usr/bin/env python3
"""Baselines on the full dataset: Ch 3 models vs. the Ch 4 network.

Uses fingerprints + descriptors, scaffold-split. No QM here -- this is
the number the QM arm has to beat in 05_ablation.py.

    python scripts/04_train.py
    python scripts/04_train.py --split random   # to see the inflation
"""

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from qmprop import load_config
from qmprop.evaluate import format_table, parity_plot, regression_metrics
from qmprop.models import build_model
from qmprop.splits import get_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=None, choices=["scaffold", "random"])
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config()
    target = cfg["dataset"]["target_name"]

    df = pd.read_csv(cfg["data_dir"] / "processed" / "dataset.csv")
    cache = np.load(cfg["data_dir"] / "interim" / "features.npz", allow_pickle=True)

    if not np.array_equal(cache["smiles"], df["smiles"].values):
        raise RuntimeError(
            "features.npz is stale -- rerun scripts/02_featurize.py"
        )

    X = np.hstack([cache["morgan"], cache["descriptors"]])
    y = df[target].to_numpy(dtype=float)
    log.info("features %s, target %s", X.shape, y.shape)

    scfg = cfg["split"]
    method = args.split or scfg["method"]
    train_idx, _, test_idx = get_split(method)(
        df["smiles"].tolist(),
        scfg["frac_train"], scfg["frac_valid"], scfg["frac_test"],
        scfg["seed"],
    )

    model_names = args.models or cfg["models"]
    out_dir = cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in model_names:
        t0 = time.time()
        model = build_model(name, seed=scfg["seed"])
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        m = regression_metrics(y[test_idx], pred)
        m["model"] = name
        m["split"] = method
        m["seconds"] = round(time.time() - t0, 1)
        rows.append(m)

        fig = parity_plot(
            y[test_idx], pred,
            title=f"{name} · {method} split · {target}",
            path=out_dir / "figures" / f"parity_{method}_{name}.png",
        )
        log.info("%-14s RMSE %.3f  R2 %.3f  (%.1fs)  -> %s",
                 name, m["rmse"], m["r2"], m["seconds"], fig.name)

    rows.sort(key=lambda r: r["rmse"])
    table = format_table(
        [{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()}
         for r in rows],
        ["model", "split", "rmse", "mae", "r2", "n", "seconds"],
    )
    print("\n" + table)

    results_path = out_dir / f"baselines_{method}.json"
    results_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
