#!/usr/bin/env python3
"""Does quantum chemistry actually help? The ablation.

The question the project exists to answer: once a model already has
fingerprints and cheap RDKit descriptors, do B3LYP orbital energies add
anything, or is the physics already implicit in the structure?

Design note, and the thing most ablations get wrong: every arm is scored
on EXACTLY the same molecules and the same scaffold split -- the subset
for which QM succeeded. Comparing a QM arm on 180 small molecules to a
fingerprint arm on all 1117 would measure the subset, not the features.

Because the arms share a test set, the comparison is PAIRED, and the
significance test has to be too. An earlier version quoted
1.96*sigma_y/sqrt(n) as the noise floor, which is the confidence interval
for the mean of y -- a different quantity, and far too wide here: two
models scored on the same molecules make correlated errors, and the
uncertainty on their *difference* is much smaller than the uncertainty on
either one alone. Using the wrong interval would hide a real effect. So
the delta is bootstrapped by resampling molecules and recomputing both
RMSEs on the same resample.

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

N_BOOTSTRAP = 10_000

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
    predictions: dict[tuple[str, str], np.ndarray] = {}
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
            predictions[(arm, model_name)] = pred
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
    y_test = y[test_idx]
    rng = np.random.default_rng(scfg["seed"])
    verdicts = []

    print(f"\nQM delta: does adding 8 quantum features to fingerprints+descriptors help?")
    print(f"(negative = QM helped. {N_BOOTSTRAP:,} paired bootstrap resamples, 95% CI)\n")
    print(f"{'model':<16}{'base':>8}{'+QM':>8}{'delta':>9}{'95% CI':>18}  verdict")
    print("-" * 74)

    for model_name in cfg["models"]:
        base_pred = predictions.get(("fingerprint+desc", model_name))
        qm_pred = predictions.get(("fingerprint+desc+qm", model_name))
        if base_pred is None or qm_pred is None:
            continue

        base_err = base_pred - y_test
        qm_err = qm_pred - y_test
        rmse = lambda e: float(np.sqrt(np.mean(e**2)))
        delta = rmse(qm_err) - rmse(base_err)

        # Resample MOLECULES, not residuals, and score both arms on the
        # same resample -- that is what makes the test paired.
        idx = rng.integers(0, len(y_test), size=(N_BOOTSTRAP, len(y_test)))
        deltas = (np.sqrt((qm_err[idx] ** 2).mean(axis=1))
                  - np.sqrt((base_err[idx] ** 2).mean(axis=1)))
        lo, hi = np.percentile(deltas, [2.5, 97.5])

        if hi < 0:
            verdict = "QM helps"
        elif lo > 0:
            verdict = "QM hurts"
        else:
            verdict = "no measurable effect"
        verdicts.append(verdict)

        print(f"{model_name:<16}{rmse(base_err):>8.3f}{rmse(qm_err):>8.3f}"
              f"{delta:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>18}  {verdict}")

    print(f"\n  n = {len(y_test)} test molecules, {len(train_idx)} training.")
    if verdicts and all(v == "no measurable effect" for v in verdicts):
        print("  Every interval straddles zero: on this target, at this sample")
        print("  size, QM features add nothing measurable on top of cheap ones.")
        print("  That is a real finding, not a failed experiment -- solubility is")
        print("  dominated by polarity and H-bonding, which TPSA and LogP already")
        print("  encode. Report it as a null result, with the honest caveat that")
        print(f"  {len(y_test)} test molecules cannot resolve an effect smaller than")
        print("  the intervals above -- absence of evidence at this n, not proof")
        print("  of absence.")

    path = out_dir / "ablation.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
