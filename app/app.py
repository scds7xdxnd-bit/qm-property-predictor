#!/usr/bin/env python3
"""Gradio front end: SMILES in, predicted solubility out, with neighbors.

The neighbor panel is the honest part. A prediction for a molecule with
no close analog in the training set is extrapolation, and the app says
so rather than quietly returning a confident-looking number
(the applicability-domain idea from Ch 2.1).

    python app/app.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator

from qmprop import load_config
from qmprop.models import build_model
from qmprop.splits import get_split

NEIGHBOR_WARN = 0.4  # below this max-Tanimoto, call it extrapolation

# ZeroGPU requires at least one @spaces.GPU function to exist at startup,
# or the scheduler refuses to start the container:
#
#   errorMessage: "No @spaces.GPU function detected during startup"
#
# This app is CPU-only -- a random forest over fingerprints -- and a free
# account cannot move a Space off ZeroGPU. So register one decorated
# function to satisfy the check. It is never called on the request path,
# so it consumes no GPU quota; predictions run on CPU exactly as they do
# locally. `spaces` is injected by the Space image and absent elsewhere.
try:
    import spaces
except ImportError:
    spaces = None

if spaces is not None:

    @spaces.GPU(duration=5)
    def _zerogpu_registration() -> str:
        """Present so ZeroGPU will schedule the container. Not used."""
        return "ok"


cfg = load_config()
TARGET = cfg["dataset"]["target_name"]
_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=cfg["features"]["morgan"]["radius"],
    fpSize=cfg["features"]["morgan"]["n_bits"],
)


def _load():
    """Train once at startup on the scaffold-split training half."""
    df = pd.read_csv(cfg["data_dir"] / "processed" / "dataset.csv")
    cache = np.load(cfg["data_dir"] / "interim" / "features.npz", allow_pickle=True)
    X = np.hstack([cache["morgan"], cache["descriptors"]])
    y = df[TARGET].to_numpy(dtype=float)

    s = cfg["split"]
    train_idx, _, _ = get_split(s["method"])(
        df["smiles"].tolist(), s["frac_train"], s["frac_valid"],
        s["frac_test"], s["seed"],
    )

    model = build_model("random_forest", seed=s["seed"])
    model.fit(X[train_idx], y[train_idx])

    train_smiles = df["smiles"].iloc[train_idx].tolist()
    train_fps = [_gen.GetFingerprint(Chem.MolFromSmiles(s_)) for s_ in train_smiles]

    # The exact descriptor columns that survived training-time filtering.
    # Inference must reproduce this list verbatim or the feature count
    # silently drifts and the model rejects the input.
    descriptor_names = [str(n) for n in cache["descriptor_names"]]

    return model, df, train_idx, train_smiles, train_fps, descriptor_names


MODEL, DF, TRAIN_IDX, TRAIN_SMILES, TRAIN_FPS, DESCRIPTOR_NAMES = _load()


def predict(smiles: str):
    smiles = (smiles or "").strip()
    if not smiles:
        return None, "Enter a SMILES string.", None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"RDKit could not parse `{smiles}` — not a valid molecule.", None

    from qmprop.features import build_features

    canonical = Chem.MolToSmiles(mol)
    X, _ = build_features(
        [canonical],
        morgan_kwargs=cfg["features"]["morgan"],
        descriptor_names=DESCRIPTOR_NAMES,
    )
    if X.shape[1] != MODEL.n_features_in_:
        return None, (
            f"Feature mismatch: built {X.shape[1]}, model expects "
            f"{MODEL.n_features_in_}. Rerun scripts/02_featurize.py."
        ), None
    pred = float(MODEL.predict(X)[0])

    query_fp = _gen.GetFingerprint(mol)
    sims = np.array(DataStructs.BulkTanimotoSimilarity(query_fp, TRAIN_FPS))
    order = np.argsort(sims)[::-1][:5]
    best = float(sims[order[0]])

    lines = [
        f"### Predicted {TARGET}: **{pred:.2f}** log(mol/L)",
        f"`{canonical}`",
        "",
    ]
    if best < NEIGHBOR_WARN:
        lines.append(
            f"⚠️ **Outside the applicability domain.** Nearest training "
            f"molecule is only {best:.2f} Tanimoto away, so this number is "
            f"extrapolation. Treat it as a guess, not a prediction."
        )
    else:
        lines.append(f"Nearest training neighbor: {best:.2f} Tanimoto.")

    neighbors = pd.DataFrame({
        "tanimoto": [round(float(sims[i]), 3) for i in order],
        "measured": [
            round(float(DF[TARGET].iloc[TRAIN_IDX[i]]), 2) for i in order
        ],
        "smiles": [TRAIN_SMILES[i] for i in order],
    })

    return Draw.MolToImage(mol, size=(340, 260)), "\n".join(lines), neighbors


def build_ui() -> "gr.Blocks":
    """Construct the interface.

    `demo` MUST exist at module scope: a Hugging Face Space imports this
    file and looks for a module-level Blocks object. Building it inside
    a function leaves nothing for the runner to serve, and the container
    starts, reports a local URL, then exits.
    """
    with gr.Blocks(title="Solubility predictor") as demo:
        gr.Markdown(
            "# Aqueous solubility predictor\n"
            "Random forest on ECFP4 + RDKit descriptors, trained on the "
            "scaffold-split ESOL training half. Predictions for molecules "
            "unlike anything in training are flagged."
        )
        with gr.Row():
            with gr.Column(scale=2):
                smiles_in = gr.Textbox(
                    label="SMILES",
                    value="CC(=O)Oc1ccccc1C(=O)O",
                    placeholder="CC(=O)Oc1ccccc1C(=O)O",
                )
                gr.Examples(
                    examples=[
                        ["CC(=O)Oc1ccccc1C(=O)O"],          # aspirin
                        ["CCO"],                             # ethanol
                        ["c1ccc2cc3ccccc3cc2c1"],            # anthracene
                        ["CN1C=NC2=C1C(=O)N(C)C(=O)N2C"],    # caffeine
                    ],
                    inputs=smiles_in,
                )
                go = gr.Button("Predict", variant="primary")
                result = gr.Markdown()
            with gr.Column(scale=1):
                image = gr.Image(label="Structure", height=260)
        neighbors = gr.Dataframe(label="Nearest training molecules", wrap=True)

        go.click(predict, smiles_in, [image, result, neighbors])
        smiles_in.submit(predict, smiles_in, [image, result, neighbors])

    return demo


demo = build_ui()


# Launch at MODULE level, with no __main__ guard.
#
# A Hugging Face Space imports this file rather than running it as a
# script, so everything under `if __name__ == "__main__":` is dead code
# there. That is why three earlier fixes had no effect: the Space showed
# launch() output but exited, and even an explicit block_thread() inside
# the guard never ran (it cannot return silently -- it would have logged).
#
# Module-level launch is also what HF's own Space template does.
#
# ssr_mode=False because the Space's Node SSR proxy dies at startup and
# takes the app down with it. Nothing here needs server-side rendering.
#
# Set QMPROP_NO_LAUNCH=1 to import this module without starting a server
# (used by the tests and the build smoke check).
if os.environ.get("QMPROP_NO_LAUNCH") != "1":
    # Do not rely on gradio's automatic thread block. From its source:
    #
    #   is_in_interactive_mode = bool(getattr(sys, "ps1", sys.flags.interactive))
    #   if not prevent_thread_lock and not is_in_interactive_mode:
    #       self.block_thread()
    #
    # The Space invokes this file with Python's interactive flag set, so
    # gradio SKIPS the block on purpose: launch() prints its URLs,
    # returns, the process ends, and the container reports RUNTIME_ERROR
    # with no traceback. Locally the flag is unset, so it blocks and the
    # bug is invisible.
    #
    # So: return deterministically, then hold the thread ourselves.
    demo.launch(ssr_mode=False, prevent_thread_lock=True)
    demo.block_thread()
