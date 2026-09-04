"""Ch 7: turn a question into a prediction that shows its work.

The chapter's framing is "add an LLM". The load-bearing part is
narrower and more useful: a number with no provenance is not an answer.
So the pipeline is

    "how soluble is caffeine?"
        -> resolve the name to a structure          (PubChem / ChEMBL)
        -> predict                                   (the Ch 3 model)
        -> say WHY: nearest training neighbours and
           the features that moved this prediction   (Tanimoto + TreeSHAP)
        -> say WHETHER TO BELIEVE IT                 (applicability domain)

An LLM is optional and deliberately last. `render_text` produces the
whole explanation deterministically from computed numbers; `render_llm`
only rephrases those same numbers when an API key happens to be set.
The language model never supplies a quantity -- that is the difference
between an explanation and a plausible-sounding one.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

log = logging.getLogger(__name__)

# Below this Tanimoto to anything in training, the model is extrapolating
# and the prediction should be read as a guess. Matches app/app.py.
NEIGHBOR_WARN = 0.4


@dataclass
class Explanation:
    query: str
    smiles: str
    prediction: float
    resolved_from: str = "smiles"
    resolution: dict | None = None
    neighbors: list[dict] = field(default_factory=list)
    contributions: list[dict] = field(default_factory=list)
    in_domain: bool = True
    max_similarity: float = 0.0
    measured: float | None = None
    shap_bias: float = 0.0
    n_contributing: int = 0

    def as_facts(self) -> dict:
        """Everything an explanation may assert, and nothing else."""
        return {
            "query": self.query,
            "smiles": self.smiles,
            "predicted_logS": round(self.prediction, 2),
            "predicted_mg_per_L": round(self.mg_per_litre, 1),
            "resolved_from": self.resolved_from,
            "max_train_similarity": round(self.max_similarity, 2),
            "in_applicability_domain": self.in_domain,
            "measured_logS": self.measured,
            "nearest_neighbors": self.neighbors,
            "top_features": self.contributions,
            "n_contributing_features": self.n_contributing,
        }

    @property
    def mg_per_litre(self) -> float:
        """log S is log10(mol/L); mg/L needs the molecular weight."""
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(self.smiles)
        return float(10**self.prediction * Descriptors.MolWt(mol) * 1000.0)


def looks_like_smiles(text: str) -> bool:
    """Is this already a structure, or a name to look up?

    Rule: no whitespace, and RDKit can parse it.

    An earlier version also demanded a bond/branch/ring character so
    that an all-letter string could not be mistaken for a structure.
    That rejected `CCO` -- ethanol, and about the most ordinary SMILES
    there is -- along with `CC`, `CCC` and every other simple alkane.
    Guarding against a rare failure by breaking a common input is a bad
    trade.

    The residual ambiguity is real but small: `CO` is methanol as SMILES
    and carbon monoxide as a name, and a mistyped name that happens to
    parse becomes a structure. Both are survivable because the caller
    echoes the interpreted SMILES back, so a wrong reading is visible in
    the answer rather than silent. Users who mean the name can say
    "carbon monoxide".
    """
    text = text.strip()
    if not text or any(c.isspace() for c in text):
        return False

    # Failing to parse is the *expected* outcome for every name typed
    # into the box, so it must not be logged as an error. Suppress only
    # around this speculative parse -- genuine failures elsewhere should
    # still be visible.
    RDLogger.DisableLog("rdApp.*")
    try:
        return Chem.MolFromSmiles(text) is not None
    finally:
        RDLogger.EnableLog("rdApp.*")


def resolve_query(text: str) -> tuple[str | None, str, dict | None]:
    """-> (smiles, how_it_was_resolved, lookup_record)."""
    text = text.strip()
    if looks_like_smiles(text):
        return Chem.CanonSmiles(text), "smiles", None

    from .external import resolve_name

    record = resolve_name(text)
    if record is None:
        return None, "unresolved", None
    smiles = Chem.CanonSmiles(record["smiles"])
    return smiles, record.get("source", "lookup"), record


def _generator(radius: int = 2, n_bits: int = 2048):
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits)


def nearest_neighbors(
    smiles: str,
    train_smiles: list[str],
    train_fps: list,
    train_y: np.ndarray,
    k: int = 3,
    generator=None,
) -> list[dict]:
    """Top-k training molecules by Tanimoto, with their measured values.

    This is the part of the explanation a chemist can check. If the
    neighbours look nothing like the query, the prediction is not
    trustworthy no matter how confident the model sounds.
    """
    gen = generator or _generator()
    fp = gen.GetFingerprint(Chem.MolFromSmiles(smiles))
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
    order = np.argsort(-sims)[:k]
    return [
        {
            "smiles": train_smiles[i],
            "similarity": float(sims[i]),
            "measured_logS": float(train_y[i]),
        }
        for i in order
    ]


def contributions(
    model,
    x_row: np.ndarray,
    feature_names: list[str],
    k: int = 6,
) -> tuple[list[dict], float, int]:
    """Which features moved THIS prediction, and by how much.

    Uses XGBoost's built-in `pred_contribs`, which is exact TreeSHAP:
    across ALL features the contributions plus the bias reproduce the
    prediction to ~1e-6, so the explanation is arithmetically the model
    rather than a story told beside it. Note the "across all features"
    part -- a typical prediction here has several hundred non-zero
    contributions, so the top-k table below is a summary and does not
    itself sum to the prediction. `n_contributing` is returned so the
    caller can say so instead of implying otherwise.

    Returns ([], 0.0, 0) for models without TreeSHAP rather than
    substituting global feature importances, which answer a different
    question ("what does the model use in general") and would mislead.
    """
    booster = getattr(model, "get_booster", None)
    if booster is None:
        log.debug("%s has no TreeSHAP; skipping attribution",
                  type(model).__name__)
        return [], 0.0, 0

    import xgboost as xgb

    dmatrix = xgb.DMatrix(x_row.reshape(1, -1))
    shap = booster().predict(dmatrix, pred_contribs=True)[0]
    values, bias = shap[:-1], float(shap[-1])

    order = np.argsort(-np.abs(values))[:k]
    top = [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"f{i}",
            "contribution": float(values[i]),
            "value": float(x_row[i]),
        }
        for i in order if values[i] != 0.0
    ]
    return top, bias, int(np.count_nonzero(values))


def _describe_feature(name: str) -> str:
    """Plain-language gloss for the descriptor names that carry weight."""
    glossary = {
        "MolLogP": "octanol/water partition (lipophilicity)",
        "MolWt": "molecular weight",
        "TPSA": "topological polar surface area",
        "NumHDonors": "hydrogen-bond donors",
        "NumHAcceptors": "hydrogen-bond acceptors",
        "NumRotatableBonds": "rotatable bonds",
        "RingCount": "number of rings",
        "NumAromaticRings": "aromatic rings",
        "FractionCSP3": "fraction of sp3 carbons",
        "HeavyAtomCount": "heavy atoms",
        "LabuteASA": "approximate surface area",
        "BalabanJ": "Balaban branching index",
        "BertzCT": "molecular complexity",
    }
    if name in glossary:
        return glossary[name]
    if name.startswith("ecfp"):
        return f"substructure bit {name.split('_')[-1]}"
    return name


def explain(
    query: str,
    model,
    feature_builder,
    train_smiles: list[str],
    train_fps: list,
    train_y: np.ndarray,
    measured_lookup: dict[str, float] | None = None,
    k_neighbors: int = 3,
) -> Explanation | None:
    """Full Ch 7 path. `feature_builder(smiles) -> (X_row, feature_names)`."""
    smiles, how, record = resolve_query(query)
    if smiles is None:
        return None

    x_row, feature_names = feature_builder(smiles)
    prediction = float(model.predict(x_row.reshape(1, -1))[0])

    neighbors = nearest_neighbors(
        smiles, train_smiles, train_fps, train_y, k=k_neighbors)
    max_sim = neighbors[0]["similarity"] if neighbors else 0.0

    top, bias, n_nonzero = contributions(model, x_row, feature_names)

    return Explanation(
        query=query,
        smiles=smiles,
        prediction=prediction,
        resolved_from=how,
        resolution=record,
        neighbors=neighbors,
        contributions=top,
        shap_bias=bias,
        n_contributing=n_nonzero,
        in_domain=max_sim >= NEIGHBOR_WARN,
        max_similarity=max_sim,
        measured=(measured_lookup or {}).get(smiles),
    )


def render_text(exp: Explanation) -> str:
    """Deterministic explanation. No model, no network, no invented numbers."""
    lines = []
    if exp.resolved_from in ("pubchem", "chembl"):
        name = (exp.resolution or {}).get("iupac_name") or exp.query
        lines.append(f"Resolved **{exp.query}** via {exp.resolved_from} "
                     f"to `{exp.smiles}` ({name}).")
    lines.append(
        f"### Predicted log S: **{exp.prediction:.2f}** log(mol/L) "
        f"(~{exp.mg_per_litre:,.0f} mg/L)"
    )

    if exp.measured is not None:
        lines.append(
            f"This molecule is in the training set; its measured value is "
            f"**{exp.measured:.2f}**, so this is a memory check, not a "
            f"prediction."
        )

    if not exp.in_domain:
        lines.append(
            f"⚠️ **Outside the applicability domain.** The most similar "
            f"training molecule scores only {exp.max_similarity:.2f} Tanimoto. "
            f"The model has not seen this kind of chemistry; treat the number "
            f"as a guess."
        )

    if exp.neighbors:
        lines.append("\n**Closest training molecules**")
        lines.append("| similarity | measured log S | SMILES |")
        lines.append("|---:|---:|:---|")
        for n in exp.neighbors:
            lines.append(f"| {n['similarity']:.2f} | {n['measured_logS']:.2f} "
                         f"| `{n['smiles']}` |")
        spread = [n["measured_logS"] for n in exp.neighbors]
        lines.append(
            f"\nThose neighbours span {min(spread):.2f} to {max(spread):.2f}. "
            f"A prediction outside that range is the model extrapolating from "
            f"structure, not interpolating between measurements."
        )

    if exp.contributions:
        shown = sum(c["contribution"] for c in exp.contributions)
        lines.append(f"\n**What moved this prediction** (exact TreeSHAP, "
                     f"from a baseline of {exp.shap_bias:.2f})")
        lines.append("| feature | value | effect on log S |")
        lines.append("|:---|---:|---:|")
        for c in exp.contributions:
            lines.append(
                f"| {_describe_feature(c['feature'])} | {c['value']:.3g} "
                f"| {c['contribution']:+.2f} |"
            )
        lines.append(
            f"\nThese are the {len(exp.contributions)} largest of "
            f"{exp.n_contributing} contributing features, worth "
            f"{shown:+.2f} of the {exp.prediction - exp.shap_bias:+.2f} total "
            f"shift from baseline. The rest is spread thinly across "
            f"substructure bits, so this table is a summary, not the whole sum."
        )
    return "\n".join(lines)


LLM_SYSTEM = (
    "You explain a solubility prediction to a chemist. You are given a JSON "
    "object of computed facts. Use ONLY numbers that appear in it -- never "
    "estimate, round differently, or introduce a value of your own. Two "
    "short paragraphs. If in_applicability_domain is false, lead with that "
    "caveat. Do not describe the model as confident."
)


def render_llm(exp: Explanation, model_name: str = "claude-opus-5") -> str | None:
    """Optional rephrasing. Returns None when no key is configured.

    Note what is *not* delegated: the prediction, the neighbours, the
    attributions and the domain check are all computed before this is
    called. The LLM sees them as data and writes prose. If it is absent,
    unreachable, or wrong, `render_text` is still the real answer.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import json

        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model_name,
            max_tokens=400,
            system=LLM_SYSTEM,
            messages=[{"role": "user",
                       "content": json.dumps(exp.as_facts(), indent=2)}],
        )
        return "".join(b.text for b in message.content if b.type == "text")
    except Exception as exc:                    # noqa: BLE001
        log.warning("LLM rephrasing unavailable: %s", exc)
        return None
