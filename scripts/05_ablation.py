#!/usr/bin/env python3
"""Does quantum chemistry actually help? The ablation.

The question the project exists to answer: once a model already has
fingerprints and cheap RDKit descriptors, do B3LYP orbital energies add
anything, or is the physics already implicit in the structure?

Three design decisions, each fixing a way this experiment is usually
gotten wrong:

1. **Same molecules in every arm.** Every arm is scored on exactly the
   subset for which QM converged. Comparing a QM arm on 180 small
   molecules to a fingerprint arm on all 1117 would measure the subset,
   not the features.

2. **Scaffold k-fold, not one split.** A single 80/20 split of a
   200-molecule subset leaves ~40 test molecules, which cannot resolve
   the effect being looked for. Rotating folds gives an out-of-fold
   prediction for every molecule -- 5x the comparison data for a
   training cost measured in seconds. Scaffold groups stay whole, so
   each fold is still an honest test.

3. **A paired test, because the arms share a test set.** An earlier
   version quoted 1.96*sigma_y/sqrt(n) as the noise floor, which is the
   confidence interval for the mean of y -- a different quantity, and
   far too wide: two models scored on the same molecules make correlated
   errors, so the uncertainty on their *difference* is much smaller. The
   delta is bootstrapped by resampling molecules and recomputing both
   RMSEs on each resample.

    python scripts/05_ablation.py
    python scripts/05_ablation.py --folds 10
    python scripts/05_ablation.py --single-split   # the old 80/20
"""

import argparse
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
from qmprop.splits import get_split, scaffold_kfold

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

BASE_ARM = "fingerprint+desc"
QM_ARM = "fingerprint+desc+qm"


def run_controls(smiles, y, qm_matrix, folds, cfg, models, seed):
    """Is a "QM helps" verdict information, or just more columns?

    Three controls, each scored exactly like the real arm:

      shuffled QM  the real matrix with its rows permuted -- identical
                   marginal distributions, zero link to the molecule
      gaussian     eight standard normals
      heavy atoms  eight copies of the heavy-atom count

    An MLP will often improve when handed any block of well-scaled
    continuous columns, whatever is in them, so the noise controls
    separate "the features carry chemistry" from "the model liked the
    extra inputs". The size control matters because QM descriptors are
    partly a size proxy -- the HOMO-LUMO gap correlates about -0.54 with
    heavy-atom count here -- and molecule size predicts solubility on
    its own.

    If a control reproduces the gain, the gain is not quantum chemistry.
    """
    from rdkit import Chem

    rng = np.random.default_rng(seed)
    heavy = np.array([Chem.MolFromSmiles(s).GetNumHeavyAtoms()
                      for s in smiles], dtype=float)

    print("\n== controls: does anything else reproduce the gain? ==\n")
    print("QM feature correlation with heavy-atom count:")
    for i, name in enumerate(QM_FEATURE_NAMES):
        print(f"  {name:28} r = {np.corrcoef(qm_matrix[:, i], heavy)[0, 1]:+.3f}")

    blocks = {
        "none (base)": None,
        "real QM": qm_matrix,
        "shuffled QM": qm_matrix[rng.permutation(len(qm_matrix))],
        "gaussian noise": rng.standard_normal(qm_matrix.shape).astype(np.float32),
        "heavy-atom count": np.repeat(heavy[:, None], qm_matrix.shape[1],
                                      axis=1).astype(np.float32),
    }

    header = f"\n{'extra block':<20}" + "".join(f"{m:>16}" for m in models)
    print(header)
    print("-" * len(header))

    baseline, out = {}, {}
    for label, extra in blocks.items():
        X, _ = build_features(
            smiles, morgan_kwargs=cfg["features"]["morgan"],
            qm_matrix=extra,
            qm_names=QM_FEATURE_NAMES if extra is not None else None,
            descriptor_names=cfg["features"]["descriptors"]["names"])
        line = f"{label:<20}"
        for name in models:
            pred = out_of_fold(name, X, y, folds, seed)
            rmse = regression_metrics(y, pred)["rmse"]
            baseline.setdefault(name, rmse)
            out.setdefault(label, {})[name] = rmse
            line += (f"{rmse:>16.3f}" if extra is None
                     else f"{rmse:>10.3f}{rmse - baseline[name]:>+6.2f}")
        print(line)
    return out


def out_of_fold(model_name, X, y, folds, seed):
    """Train once per fold, predict the held-out fold. Returns predictions
    for every molecule, each made by a model that never saw it."""
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in folds:
        model = build_model(model_name, seed=seed)
        model.fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])
    assert not np.isnan(pred).any(), "a molecule was never in a test fold"
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--single-split", action="store_true",
                    help="one 80/20 scaffold split instead of k-fold "
                         "(faster, far less statistical power)")
    args = ap.parse_args()

    cfg = load_config()
    target = cfg["dataset"]["target_name"]
    scfg = cfg["split"]

    df = pd.read_csv(cfg["data_dir"] / "processed" / "dataset.csv")

    qm_path = cfg["data_dir"] / "interim" / "qm_descriptors.csv"
    if not qm_path.exists():
        raise SystemExit(f"{qm_path} not found -- run scripts/03_run_qm.py first")

    qm = pd.read_csv(qm_path)
    n_attempted = len(qm)
    qm = qm[qm["ok"]].drop_duplicates(subset="smiles")
    log.info("%d/%d molecules with converged QM", len(qm), n_attempted)
    if n_attempted > len(qm):
        reasons = pd.read_csv(qm_path)
        failed = reasons[~reasons["ok"]]["reason"].value_counts()
        for reason, n in failed.items():
            log.info("  %d failed: %s", n, str(reason)[:70])

    # Inner join is what enforces "same rows in every arm".
    merged = df.merge(qm[["smiles", *QM_FEATURE_NAMES]], on="smiles", how="inner")
    if len(merged) < 50:
        log.warning("only %d molecules -- results will be noisy; "
                    "raise qm.subset_size", len(merged))

    smiles = merged["smiles"].tolist()
    y = merged[target].to_numpy(dtype=float)
    qm_matrix = merged[QM_FEATURE_NAMES].to_numpy(dtype=np.float32)

    if args.single_split:
        train_idx, _, test_idx = get_split(scfg["method"])(
            smiles, scfg["frac_train"], scfg["frac_valid"], scfg["frac_test"],
            scfg["seed"])
        folds = [(train_idx, test_idx)]
        scored = test_idx
        log.info("single split: %d train / %d test", len(train_idx), len(test_idx))
    else:
        folds = scaffold_kfold(smiles, args.folds, scfg["seed"])
        scored = np.arange(len(smiles))
        log.info("%d-fold scaffold CV over %d molecules (fold sizes %s)",
                 args.folds, len(smiles), [len(t) for _, t in folds])

    out_dir = cfg["output_dir"]
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    rows = []
    predictions: dict[tuple[str, str], np.ndarray] = {}
    for arm, flags in ARMS.items():
        X, _ = build_features(
            smiles,
            use_morgan=flags["use_morgan"],
            use_descriptors=flags["use_descriptors"],
            qm_matrix=qm_matrix if flags["use_qm"] else None,
            qm_names=QM_FEATURE_NAMES,
            morgan_kwargs=cfg["features"]["morgan"],
            descriptor_names=cfg["features"]["descriptors"]["names"],
        )

        for model_name in cfg["models"]:
            pred = out_of_fold(model_name, X, y, folds, scfg["seed"])
            m = regression_metrics(y[scored], pred[scored])
            m.update(arm=arm, model=model_name, n_features=X.shape[1])
            rows.append(m)
            predictions[(arm, model_name)] = pred
            log.info("%-22s %-14s RMSE %.3f  R2 %6.3f  (%d feats)",
                     arm, model_name, m["rmse"], m["r2"], X.shape[1])

            if arm == QM_ARM:
                parity_plot(y[scored], pred[scored],
                            title=f"{model_name} · +QM · {target}",
                            path=out_dir / "figures" / f"ablation_qm_{model_name}.png")

    print("\n" + format_table(
        [{"arm": r["arm"], "model": r["model"], "features": r["n_features"],
          "rmse": f"{r['rmse']:.3f}", "mae": f"{r['mae']:.3f}",
          "r2": f"{r['r2']:.3f}"} for r in rows],
        ["arm", "model", "features", "rmse", "mae", "r2"],
    ))

    # The headline: per model, what did QM buy?
    y_scored = y[scored]
    rng = np.random.default_rng(scfg["seed"])
    verdicts, deltas_out = [], {}

    print(f"\nQM delta: 8 quantum features on top of fingerprints+descriptors")
    print(f"(negative = QM helped; {N_BOOTSTRAP:,} paired bootstrap resamples, 95% CI)\n")
    print(f"{'model':<16}{'base':>8}{'+QM':>8}{'delta':>9}{'95% CI':>19}  verdict")
    print("-" * 76)

    for model_name in cfg["models"]:
        base_pred = predictions.get((BASE_ARM, model_name))
        qm_pred = predictions.get((QM_ARM, model_name))
        if base_pred is None or qm_pred is None:
            continue

        base_err = base_pred[scored] - y_scored
        qm_err = qm_pred[scored] - y_scored
        rmse = lambda e: float(np.sqrt(np.mean(e**2)))
        delta = rmse(qm_err) - rmse(base_err)

        # Resample MOLECULES and score both arms on the same resample --
        # that is what makes the test paired.
        idx = rng.integers(0, len(y_scored), size=(N_BOOTSTRAP, len(y_scored)))
        boot = (np.sqrt((qm_err[idx] ** 2).mean(axis=1))
                - np.sqrt((base_err[idx] ** 2).mean(axis=1)))
        lo, hi = (float(v) for v in np.percentile(boot, [2.5, 97.5]))

        verdict = "QM helps" if hi < 0 else "QM hurts" if lo > 0 else "no measurable effect"
        verdicts.append(verdict)
        deltas_out[model_name] = {"base_rmse": rmse(base_err),
                                  "qm_rmse": rmse(qm_err), "delta": delta,
                                  "ci_low": lo, "ci_high": hi, "verdict": verdict}

        print(f"{model_name:<16}{rmse(base_err):>8.3f}{rmse(qm_err):>8.3f}"
              f"{delta:>+9.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>19}  {verdict}")

    width = np.mean([d["ci_high"] - d["ci_low"] for d in deltas_out.values()]) \
        if deltas_out else float("nan")
    print(f"\n  n = {len(y_scored)} molecules scored out-of-fold.")
    print(f"  Mean 95% CI width on the delta: {width:.3f} log units.")

    if verdicts and all(v == "no measurable effect" for v in verdicts):
        print("\n  Every interval straddles zero. On this target, at this sample")
        print("  size, QM features add nothing measurable on top of cheap ones.")
        print("  That is a real finding, not a failed experiment: solubility is")
        print("  dominated by polarity and H-bonding, which TPSA and LogP already")
        print("  encode. Report it as a null result, with the honest caveat that")
        print(f"  an effect smaller than ~{width / 2:.2f} log units could not have been")
        print("  seen here -- absence of evidence at this n, not proof of absence.")
    controls = None
    if any(v == "QM helps" for v in verdicts):
        print("\n  At least one model improved beyond the interval. Running the")
        print("  controls before believing it.")
        helped = [m for m in cfg["models"]
                  if deltas_out.get(m, {}).get("verdict") == "QM helps"]
        controls = run_controls(smiles, y[scored], qm_matrix, folds, cfg,
                                helped + ["ridge"], scfg["seed"])

        real = {m: controls["real QM"][m] - controls["none (base)"][m]
                for m in helped}
        noise = {m: controls["gaussian noise"][m] - controls["none (base)"][m]
                 for m in helped}
        size = {m: controls["heavy-atom count"][m] - controls["none (base)"][m]
                for m in helped}
        for m in helped:
            print(f"\n  {m}: real QM {real[m]:+.2f}, but gaussian noise alone "
                  f"gives {noise[m]:+.2f}")
            print(f"  and heavy-atom count alone gives {size[m]:+.2f}.")
            if abs(size[m]) >= 0.8 * abs(real[m]) or abs(noise[m]) >= 0.5 * abs(real[m]):
                print(f"  The gain survives neither control -- it is the model"
                      f" liking extra")
                print(f"  scaled columns, plus molecular size that QM features"
                      f" proxy. Not")
                print(f"  quantum chemistry. Report it as a null result.")
            else:
                print("  The gain survives both controls; it may be real.")

    payload = {"metrics": rows, "qm_delta": deltas_out, "controls": controls,
               "n_scored": int(len(y_scored)),
               "evaluation": "single 80/20 scaffold split" if args.single_split
               else f"{args.folds}-fold scaffold CV"}
    path = out_dir / "ablation.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
