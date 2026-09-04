#!/usr/bin/env python3
"""Gradio front end: a name or a SMILES in, an explained prediction out.

Everything after the number is the point (Ch 7):

  * a name is resolved to a structure through PubChem/ChEMBL, so you can
    type "caffeine" instead of Cn1cnc2c1c(=O)n(C)c(=O)n2C;
  * the nearest training molecules are shown with their MEASURED values,
    so the prediction can be checked against real data rather than
    trusted;
  * exact TreeSHAP says which features moved this particular prediction;
  * anything with no close training analogue is flagged as extrapolation
    (the applicability-domain idea from Ch 2.1).

XGBoost rather than the random forest used earlier, for two reasons: it
is the better model on the scaffold split (RMSE 0.900 vs 0.909), and it
exposes exact TreeSHAP, so the attribution table is arithmetic rather
than a plausible story.

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
from qmprop.explain import explain, render_text
from qmprop.models import build_model
from qmprop.splits import get_split

# ZeroGPU requires at least one @spaces.GPU function to exist at startup,
# or the scheduler refuses to start the container:
#
#   errorMessage: "No @spaces.GPU function detected during startup"
#
# This app is CPU-only -- gradient boosting over fingerprints -- and a
# free account cannot move a Space off ZeroGPU. So register one decorated
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

    model = build_model("xgboost", seed=s["seed"])
    model.fit(X[train_idx], y[train_idx])

    train_smiles = df["smiles"].iloc[train_idx].tolist()
    train_fps = [_gen.GetFingerprint(Chem.MolFromSmiles(s_)) for s_ in train_smiles]

    # The exact descriptor columns that survived training-time filtering.
    # Inference must reproduce this list verbatim or the feature count
    # silently drifts and the model rejects the input.
    descriptor_names = [str(n) for n in cache["descriptor_names"]]
    morgan_names = [str(n) for n in cache["morgan_names"]]

    return (model, df, train_idx, train_smiles, train_fps, descriptor_names,
            morgan_names, y)


(MODEL, DF, TRAIN_IDX, TRAIN_SMILES, TRAIN_FPS, DESCRIPTOR_NAMES,
 MORGAN_NAMES, Y_ALL) = _load()
TRAIN_Y = Y_ALL[TRAIN_IDX]
FEATURE_NAMES = MORGAN_NAMES + DESCRIPTOR_NAMES
MEASURED = dict(zip(DF["smiles"], Y_ALL))


def _feature_builder(smiles: str):
    """Reproduce the training columns exactly for one molecule."""
    from qmprop.features import build_features

    X, names = build_features(
        [smiles],
        morgan_kwargs=cfg["features"]["morgan"],
        descriptor_names=DESCRIPTOR_NAMES,
    )
    if X.shape[1] != MODEL.n_features_in_:
        raise RuntimeError(
            f"Feature mismatch: built {X.shape[1]}, model expects "
            f"{MODEL.n_features_in_}. Rerun scripts/02_featurize.py."
        )
    return X[0], names


def predict(query: str):
    """Resolve, predict, explain. Returns (image, markdown, neighbour table)."""
    query = (query or "").strip()
    if not query:
        return None, "Enter a molecule name or a SMILES string.", None

    try:
        result = explain(
            query, MODEL, _feature_builder, TRAIN_SMILES, TRAIN_FPS, TRAIN_Y,
            measured_lookup=MEASURED, k_neighbors=5,
        )
    except RuntimeError as exc:                  # feature drift, see above
        return None, f"❌ {exc}", None
    except Exception as exc:                     # noqa: BLE001 - show, don't crash
        return None, f"❌ Unexpected error: {exc}", None

    if result is None:
        return None, (
            f"Could not turn `{query}` into a structure. Either it is not a "
            f"valid SMILES string, or PubChem and ChEMBL have no compound "
            f"under that name. Try a SMILES string, or a more common name."
        ), None

    mol = Chem.MolFromSmiles(result.smiles)
    neighbors = pd.DataFrame({
        "tanimoto": [round(n["similarity"], 3) for n in result.neighbors],
        "measured": [round(n["measured_logS"], 2) for n in result.neighbors],
        "smiles": [n["smiles"] for n in result.neighbors],
    })
    return (Draw.MolToImage(mol, size=(340, 260)),
            render_text(result),
            neighbors)


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
            "Type a **name** (`caffeine`) or a **SMILES** string. XGBoost on "
            "ECFP4 + RDKit descriptors, trained on the scaffold-split ESOL "
            "training half.\n\n"
            "Every prediction comes with the training molecules it is "
            "reasoning from, the features that moved it (exact TreeSHAP), "
            "and a warning when the molecule is unlike anything in training. "
            "On the honest scaffold split this model scores RMSE 0.90 — and "
            "two independent curations of this same property disagree by "
            "~0.34, so that is the floor, not zero."
        )
        with gr.Row():
            with gr.Column(scale=2):
                smiles_in = gr.Textbox(
                    label="Molecule name or SMILES",
                    value="aspirin",
                    placeholder="caffeine   or   CC(=O)Oc1ccccc1C(=O)O",
                )
                gr.Examples(
                    examples=[
                        ["aspirin"],                         # name lookup
                        ["caffeine"],                        # in training set
                        ["CCO"],                             # ethanol, as SMILES
                        ["c1ccc2cc3ccccc3cc2c1"],            # anthracene
                        ["table salt"],                      # out of domain
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
