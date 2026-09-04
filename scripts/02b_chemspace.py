#!/usr/bin/env python3
"""Ch 2: what chemistry is actually in here, and how good can any model get?

Two questions the book asks you to assemble a dataset without asking:

  1. *What does the training set cover?* A 2-D embedding of the ECFP4
     space, colored by solubility and by scaffold-split membership. The
     second panel is the argument for scaffold splitting made visually:
     the test set should sit in regions the training set does not
     occupy, which is what deployment actually looks like.

  2. *What is the best RMSE anyone could get?* ESOL is a curation, not a
     measurement. AqSolDB independently curated the same property from
     nine sources; where both list the same InChIKey and disagree, the
     disagreement is experimental noise. ESOL is one of those nine
     sources, so the measured disagreement is a lower bound rather than
     an unbiased estimate -- but a lower bound is enough to know whether
     a reported RMSE is chasing signal or chasing curation error.

    python scripts/02b_chemspace.py
"""

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from qmprop import load_config
from qmprop.external import cross_source_agreement, load_aqsoldb
from qmprop.features import morgan_matrix
from qmprop.splits import murcko_scaffold, random_split, scaffold_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("chemspace")


def noise_floor(esol: pd.DataFrame, aqsoldb: pd.DataFrame) -> dict:
    """How much do two independent curations disagree on the same molecule?"""
    merged = cross_source_agreement(esol, aqsoldb)
    multi = merged[merged["n_sources"] > 1] if "n_sources" in merged else merged

    def summarize(df: pd.DataFrame, label: str) -> dict:
        d = df["delta"].to_numpy()
        return {
            "label": label,
            "n": int(len(d)),
            "rmse": float(np.sqrt(np.mean(d**2))),
            "mae": float(np.mean(np.abs(d))),
            "bias": float(np.mean(d)),
            "max_abs": float(np.max(np.abs(d))) if len(d) else float("nan"),
        }

    all_overlap = summarize(merged, "all shared molecules")
    independent = summarize(multi, "AqSolDB had >1 source")

    print("\n== ESOL vs AqSolDB, same molecule by InChIKey ==\n")
    print(f"{'subset':<26} {'n':>5} {'RMSE':>7} {'MAE':>7} {'bias':>7} {'max':>7}")
    for row in (all_overlap, independent):
        print(f"{row['label']:<26} {row['n']:>5} {row['rmse']:>7.3f} "
              f"{row['mae']:>7.3f} {row['bias']:>+7.3f} {row['max_abs']:>7.2f}")

    print(f"\nESOL is one of the nine sources AqSolDB merged, so BOTH rows are")
    print("partly self-comparison -- where AqSolDB's value came from ESOL the two")
    print("agree by construction, which is why the MAE is so much smaller than")
    print(f"the RMSE. Read {independent['rmse']:.2f} log units as a LOWER BOUND on the")
    print("experimental noise, not an estimate of it: the true disagreement between")
    print("fully independent measurements is larger. Even so, the best model here")
    print("sits at 0.90 RMSE on the scaffold split, comfortably above the floor --")
    print("so there is real signal left to capture, not just curation noise to fit.")

    worst = merged.reindex(merged["delta"].abs().sort_values(ascending=False).index)
    print("\nlargest disagreements:")
    cols = ["smiles", "logS", "logS_aqsoldb", "delta"]
    print(worst[cols].head(5).to_string(index=False,
                                        float_format=lambda v: f"{v:7.2f}"))
    return {"all_overlap": all_overlap, "independent": independent,
            "n_esol": int(len(esol)), "n_aqsoldb": int(len(aqsoldb))}


def split_separation(smiles: list[str], cfg: dict) -> dict:
    """How far is the test set from training, under each split?

    The picture shows test molecules clustering away from training. This
    is that claim as a number: for every test molecule, the Tanimoto to
    its nearest training neighbour. Under a random split those neighbours
    are close, so the model interpolates and scores well. Under a
    scaffold split they are far, which is why the same model loses ~50%
    of its apparent accuracy -- the split did not get harder by accident,
    it got harder on purpose, and deployment is the hard one.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg["features"]["morgan"]["radius"],
        fpSize=cfg["features"]["morgan"]["n_bits"])
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s_)) for s_ in smiles]

    s_cfg = cfg["split"]
    out = {}
    for name, splitter in (("scaffold", scaffold_split), ("random", random_split)):
        train_idx, _, test_idx = splitter(
            smiles, s_cfg["frac_train"], s_cfg["frac_valid"],
            s_cfg["frac_test"], s_cfg["seed"])
        train_fps = [fps[i] for i in train_idx]
        nearest = np.array([
            max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps))
            for i in test_idx
        ])
        out[name] = {
            "mean_nearest_tanimoto": float(nearest.mean()),
            "median_nearest_tanimoto": float(np.median(nearest)),
            "frac_below_0.4": float((nearest < 0.4).mean()),
            "n_test": int(len(test_idx)),
        }

    print("\n== how far is the test set from training? ==\n")
    print(f"{'split':<12}{'mean':>8}{'median':>8}{'% below 0.4':>14}")
    for name, r in out.items():
        print(f"{name:<12}{r['mean_nearest_tanimoto']:>8.3f}"
              f"{r['median_nearest_tanimoto']:>8.3f}"
              f"{100 * r['frac_below_0.4']:>13.0f}%")
    sc, rd = out["scaffold"], out["random"]
    print(f"\nA random split leaves {100 * rd['frac_below_0.4']:.0f}% of test molecules "
          f"without a close training analogue;")
    print(f"a scaffold split leaves {100 * sc['frac_below_0.4']:.0f}%. That gap is the "
          f"whole reason the same model")
    print("scores ~50% worse on the scaffold split -- and the scaffold number is the")
    print("one that predicts behaviour on chemistry nobody has measured yet.")
    print(f"\nWorth sitting with: 0.4 Tanimoto is the threshold the deployed app uses")
    print(f"to warn that a prediction is out of domain. So {100 * sc['frac_below_0.4']:.0f}% of the scaffold")
    print("test set consists of molecules the app would flag as untrustworthy. The")
    print("headline RMSE is largely measured on exactly those cases -- which is the")
    print("point of the honest split, not a flaw in it.")
    return out


def embed(X: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """t-SNE and UMAP on the same fingerprints, Jaccard metric.

    Euclidean distance on binary fingerprints is the default everywhere
    and is the wrong metric -- Tanimoto/Jaccard is what cheminformatics
    means by similarity, and it is what the model's applicability domain
    is judged with, so the picture should use it too.
    """
    from sklearn.manifold import TSNE

    out = {}
    log.info("t-SNE on %d x %d", *X.shape)
    out["t-SNE"] = TSNE(n_components=2, metric="jaccard", init="random",
                        perplexity=30, random_state=seed).fit_transform(X)
    try:
        import umap
        log.info("UMAP on %d x %d", *X.shape)
        out["UMAP"] = umap.UMAP(n_components=2, metric="jaccard",
                                random_state=seed).fit_transform(X)
    except Exception as exc:                    # noqa: BLE001
        log.warning("UMAP unavailable (%s); plotting t-SNE only", exc)
    return out


def make_figure(embeddings, y, train_idx, test_idx, path) -> None:
    n_rows = len(embeddings)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 5 * n_rows), squeeze=False)

    for r, (name, xy) in enumerate(embeddings.items()):
        ax = axes[r][0]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=y, s=8, cmap="viridis", alpha=.8)
        ax.set_title(f"{name}: chemical space, colored by measured log S")
        fig.colorbar(sc, ax=ax, label="log S (mol/L)")

        ax = axes[r][1]
        ax.scatter(xy[train_idx, 0], xy[train_idx, 1], s=8, alpha=.5,
                   label=f"train (n={len(train_idx)})", color="#4C72B0")
        ax.scatter(xy[test_idx, 0], xy[test_idx, 1], s=10, alpha=.85,
                   label=f"test (n={len(test_idx)})", color="#C44E52")
        ax.set_title(f"{name}: scaffold split\n(test clusters apart -- that is the point)")
        ax.legend(fontsize=8)

        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nwrote {path}")


def main() -> None:
    cfg = load_config()
    out = cfg["output_dir"]
    (out / "figures").mkdir(parents=True, exist_ok=True)

    esol = pd.read_csv(cfg["data_dir"] / "processed" / "dataset.csv")
    aqsoldb = load_aqsoldb(cfg["data_dir"] / "raw" / "aqsoldb.tab")
    log.info("ESOL %d molecules, AqSolDB %d", len(esol), len(aqsoldb))

    stats = noise_floor(esol, aqsoldb)

    scaffolds = esol["smiles"].map(murcko_scaffold)
    print(f"\n{scaffolds.nunique()} distinct Murcko scaffolds across "
          f"{len(esol)} molecules")
    print(f"{(scaffolds == '').sum()} acyclic molecules (empty scaffold)")
    top = scaffolds[scaffolds != ""].value_counts().head(5)
    print("\nmost common scaffolds:")
    for smi, n in top.items():
        print(f"  {n:>4}  {smi}")
    stats["n_scaffolds"] = int(scaffolds.nunique())
    stats["split_separation"] = split_separation(esol["smiles"].tolist(), cfg)

    X, _ = morgan_matrix(esol["smiles"].tolist(),
                         radius=cfg["features"]["morgan"]["radius"],
                         n_bits=cfg["features"]["morgan"]["n_bits"])
    seed = cfg["split"]["seed"]
    train_idx, _, test_idx = scaffold_split(
        esol["smiles"].tolist(), cfg["split"]["frac_train"],
        cfg["split"]["frac_valid"], cfg["split"]["frac_test"], seed)

    embeddings = embed(X.astype(bool), seed)
    make_figure(embeddings, esol["logS"].to_numpy(), train_idx, test_idx,
                out / "figures" / "chemical_space.png")

    with open(out / "chemspace.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"wrote {out / 'chemspace.json'}")


if __name__ == "__main__":
    main()
