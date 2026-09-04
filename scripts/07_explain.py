#!/usr/bin/env python3
"""Ch 7: ask in English, get a prediction that shows its work.

    python scripts/07_explain.py caffeine
    python scripts/07_explain.py "CC(=O)Oc1ccccc1C(=O)O"
    python scripts/07_explain.py aspirin ibuprofen "table salt"
    python scripts/07_explain.py caffeine --llm     # if ANTHROPIC_API_KEY is set

The model is XGBoost rather than the random forest used elsewhere, for
one reason: XGBoost exposes exact TreeSHAP, so "which features drove
this" is arithmetic rather than a plausible story. It also happens to be
the better model on the scaffold split (RMSE 0.900 vs 0.909).
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from qmprop import load_config
from qmprop.explain import explain, render_llm, render_text
from qmprop.features import build_features
from qmprop.models import build_model
from qmprop.splits import get_split

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def load_everything(cfg: dict):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    df = pd.read_csv(cfg["data_dir"] / "processed" / "dataset.csv")
    cache = np.load(cfg["data_dir"] / "interim" / "features.npz",
                    allow_pickle=True)
    X = np.hstack([cache["morgan"], cache["descriptors"]])
    y = df[cfg["dataset"]["target_name"]].to_numpy(dtype=float)

    s = cfg["split"]
    train_idx, _, _ = get_split(s["method"])(
        df["smiles"].tolist(), s["frac_train"], s["frac_valid"],
        s["frac_test"], s["seed"])

    model = build_model("xgboost", seed=s["seed"])
    model.fit(X[train_idx], y[train_idx])

    descriptor_names = [str(n) for n in cache["descriptor_names"]]
    morgan_names = [str(n) for n in cache["morgan_names"]] \
        if "morgan_names" in cache else \
        [f"ecfp4_{i}" for i in range(cache["morgan"].shape[1])]
    feature_names = morgan_names + descriptor_names

    train_smiles = df["smiles"].iloc[train_idx].tolist()
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg["features"]["morgan"]["radius"],
        fpSize=cfg["features"]["morgan"]["n_bits"])
    train_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s_))
                 for s_ in train_smiles]

    def feature_builder(smiles: str):
        """Must reproduce the training columns exactly -- passing the cached
        descriptor names is what stops the silent feature-count drift."""
        Xq, names = build_features(
            [smiles],
            morgan_kwargs=cfg["features"]["morgan"],
            descriptor_names=descriptor_names,
        )
        if Xq.shape[1] != X.shape[1]:
            raise RuntimeError(
                f"featurizer produced {Xq.shape[1]} columns, model expects "
                f"{X.shape[1]}")
        return Xq[0], names

    measured = dict(zip(df["smiles"], y))
    return (model, feature_builder, train_smiles, train_fps,
            y[train_idx], measured, feature_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+", help="names or SMILES")
    ap.add_argument("--llm", action="store_true",
                    help="also print an LLM rephrasing (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--json", action="store_true", help="emit facts as JSON")
    args = ap.parse_args()

    cfg = load_config()
    print("training XGBoost on the scaffold-split training half...")
    (model, feature_builder, train_smiles, train_fps, train_y,
     measured, _) = load_everything(cfg)

    for query in args.queries:
        print("\n" + "=" * 72)
        exp = explain(query, model, feature_builder, train_smiles, train_fps,
                      train_y, measured_lookup=measured)
        if exp is None:
            print(f"Could not resolve {query!r} to a structure. "
                  f"Try a SMILES string or a common chemical name.")
            continue

        if args.json:
            print(json.dumps(exp.as_facts(), indent=2))
            continue

        print(render_text(exp))
        if args.llm:
            text = render_llm(exp)
            print("\n**In plain language**\n" + text if text else
                  "\n(no ANTHROPIC_API_KEY set, so no LLM rephrasing)")


if __name__ == "__main__":
    main()
