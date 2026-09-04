#!/usr/bin/env python3
"""Does quantum chemistry actually help? The ablation.

The question the project exists to answer: once a model already has
fingerprints and cheap RDKit descriptors, do B3LYP orbital energies add
anything, or is the physics already implicit in the structure?

Design note, and the thing most ablations get wrong: every arm is scored
on EXACTLY the same molecules and the same scaffold split -- the subset
for which QM succeeded. Comparing a QM arm on 180 small molecules to a
fingerprint arm on all 1117 would measure the subset, not the features.

    python scripts/05_ablation.py
"""

import json
import logging

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from qmprop import load_config
from qmprop.evaluate import format_table, parity_plot, regression_metrics
from qmprop.features import build_features
from qmprop.models import build_model
from qmprop.qm import QM_FEATURE_NAMES
from qmprop.splits import get_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ablation")

ARMS = {
    "fingerprint":            dict(use_morgan=True,  use_descriptors=False, use_qm=False),
    "descriptors":            dict(use_morgan=False, use_descriptors=True,  use_qm=False),
    "fingerprint+desc":       dict(use_morgan=True,  use_descriptors=True,  use_qm=False),
    "qm only":                dict(use_morgan=False, use_descriptors=False, use_qm=True),
    "fingerprint+desc+qm":    dict(use_morgan=True,  use_descriptors=True,  use_qm=True),
}


def main() -> None:
    cfg = load_config()
    target = cfg["dataset"]["target_name"]
    scfg = cfg["split"]

    df = pd.read_csv(cfg["data_dir"] / "processed" / "dataset.csv")

    qm_path = cfg["data_dir"] / "interim" / "qm_descriptors.csv"
    if not qm_path.exists():
        raise SystemExit(
            f"{qm_path} not found -- run scripts/03_run_qm.py first"
        )

    qm = pd.read_csv(qm_path)
    qm = qm[qm["ok"]].drop_duplicates(subset="smiles")
    log.info("%d molecules with converged QM", len(qm))

    # Inner join is what enforces "same rows in every arm".
    merged = df.merge(qm[["smiles", *QM_FEATURE_NAMES]], on="smiles", how="inner")
    if len(merged) < 50:
        log.warning(
            "only %d molecules -- results will be noisy; raise qm.subset_size",
            len(merged),
        )

    smiles = merged["smiles"].tolist()
    y = merged[target].to_numpy(dtype=float)
    qm_matrix = merged[QM_FEATURE_NAMES].to_numpy(dtype=np.float32)

    train_idx, _, test_idx = get_split(scfg["method"])(
        smiles, scfg["frac_train"], scfg["frac_valid"], scfg["frac_test"],
        scfg["seed"],
    )
    log.info("split: %d train / %d test", len(train_idx), len(test_idx))

    out_dir = cfg["output_dir"]
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    rows = []
    for arm, flags in ARMS.items():
        X, names = build_features(
            smiles,
            use_morgan=flags["use_morgan"],
            use_descriptors=flags["use_descriptors"],
            qm_matrix=qm_matrix if flags["use_qm"] else None,
            qm_names=QM_FEATURE_NAMES,
            morgan_kwargs=cfg["features"]["morgan"],
            descriptor_names=cfg["features"]["descriptors"]["names"],
        )

        for model_name in cfg["models"]:
            model = build_model(model_name, seed=scfg["seed"])
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])

            m = regression_metrics(y[test_idx], pred)
            m.update(arm=arm, model=model_name, n_features=X.shape[1])
            rows.append(m)
            log.info("%-22s %-14s RMSE %.3f  R2 %6.3f  (%d feats)",
                     arm, model_name, m["rmse"], m["r2"], X.shape[1])

            if arm == "fingerprint+desc+qm":
                parity_plot(
                    y[test_idx], pred,
                    title=f"{model_name} · +QM · {target}",
                    path=out_dir / "figures" / f"ablation_qm_{model_name}.png",
                )

    print("\n" + format_table(
        [{
            "arm": r["arm"], "model": r["model"],
            "features": r["n_features"],
            "rmse": f"{r['rmse']:.3f}", "mae": f"{r['mae']:.3f}",
            "r2": f"{r['r2']:.3f}",
        } for r in rows],
        ["arm", "model", "features", "rmse", "mae", "r2"],
    ))

    # The headline: per model, what did QM buy?
    print("\nQM delta (negative = QM helped, RMSE)")
    base = {r["model"]: r["rmse"] for r in rows if r["arm"] == "fingerprint+desc"}
    withqm = {r["model"]: r["rmse"] for r in rows if r["arm"] == "fingerprint+desc+qm"}
    for model_name in cfg["models"]:
        if model_name in base and model_name in withqm:
            delta = withqm[model_name] - base[model_name]
            pct = 100 * delta / base[model_name]
            print(f"  {model_name:<14} {delta:+.4f}  ({pct:+.1f}%)")

    print(
        f"\n  n = {len(test_idx)} test molecules. A delta smaller than roughly"
        f"\n  {1.96 * np.std(y[test_idx]) / np.sqrt(len(test_idx)):.3f} is inside the noise —"
        " report it as 'no measurable effect',\n  not as a win."
    )

    path = out_dir / "ablation.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
