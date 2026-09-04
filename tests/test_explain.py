"""Ch 7 layer. The tests that matter are the ones that stop the
explanation from drifting away from the model it claims to explain."""

import numpy as np
import pytest

from qmprop.explain import (
    NEIGHBOR_WARN,
    Explanation,
    contributions,
    looks_like_smiles,
    nearest_neighbors,
    render_text,
)

TRAIN = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "c1ccccc1C"]
TRAIN_Y = np.array([-0.31, -0.66, -0.88, -1.64, -2.21])


@pytest.fixture(scope="module")
def train_fps():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in TRAIN]


# --- telling a structure from a name --------------------------------------

@pytest.mark.parametrize("text", [
    "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCO", "C=CC=C", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
])
def test_recognizes_smiles(text):
    assert looks_like_smiles(text)


@pytest.mark.parametrize("text", [
    "caffeine", "table salt", "acetylsalicylic acid", "", "   ",
])
def test_rejects_names(text):
    assert not looks_like_smiles(text)


def test_plain_alkane_smiles_are_recognized():
    """The regression that motivated dropping the punctuation rule."""
    for text in ("CC", "CCC", "CCO", "CCCCCC"):
        assert looks_like_smiles(text), text


def test_parseable_letter_string_is_read_as_smiles():
    """Documents the accepted ambiguity rather than pretending it is
    handled: 'CON' parses, so it is treated as a structure. The caller
    echoes the SMILES back, which is what makes this safe enough."""
    from rdkit import Chem
    assert Chem.MolFromSmiles("CON") is not None      # premise
    assert looks_like_smiles("CON")


def test_multiword_input_is_never_smiles():
    for text in ("carbon monoxide", "table salt", "CC O"):
        assert not looks_like_smiles(text), text


# --- neighbours -----------------------------------------------------------

def test_identical_molecule_scores_one(train_fps):
    n = nearest_neighbors("CCO", TRAIN, train_fps, TRAIN_Y, k=1)
    assert n[0]["similarity"] == pytest.approx(1.0)
    assert n[0]["measured_logS"] == pytest.approx(-0.31)


def test_neighbors_sorted_descending(train_fps):
    n = nearest_neighbors("CCCCCO", TRAIN, train_fps, TRAIN_Y, k=5)
    sims = [x["similarity"] for x in n]
    assert sims == sorted(sims, reverse=True)


def test_dissimilar_query_falls_below_domain_threshold(train_fps):
    """A metal complex shares no ECFP bits with small alcohols."""
    n = nearest_neighbors("[Pt](Cl)(Cl)([NH3])[NH3]", TRAIN, train_fps,
                          TRAIN_Y, k=1)
    assert n[0]["similarity"] < NEIGHBOR_WARN


# --- attribution ----------------------------------------------------------

def test_contributions_reproduce_the_prediction():
    """TreeSHAP additivity: bias + every contribution == model.predict.

    This is the claim the explanation rests on. If it ever fails, the
    'what moved this prediction' table is fiction.
    """
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 12))
    y = 2.0 * X[:, 0] - X[:, 3] + rng.normal(scale=0.1, size=200)
    model = xgb.XGBRegressor(n_estimators=40, max_depth=3, random_state=0)
    model.fit(X, y)

    row = X[7]
    shap = model.get_booster().predict(
        xgb.DMatrix(row.reshape(1, -1)), pred_contribs=True)[0]
    assert float(shap.sum()) == pytest.approx(
        float(model.predict(row.reshape(1, -1))[0]), abs=1e-4)


def test_contributions_finds_the_real_driver():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 8))
    y = 3.0 * X[:, 2] + rng.normal(scale=0.05, size=300)   # feature 2 only
    model = xgb.XGBRegressor(n_estimators=60, max_depth=3, random_state=0)
    model.fit(X, y)

    names = [f"f{i}" for i in range(8)]
    top, _bias, n_nonzero = contributions(model, X[5], names, k=3)
    assert top[0]["feature"] == "f2"
    assert n_nonzero >= 1


def test_contributions_empty_for_models_without_treeshap():
    """A random forest gets no attribution rather than a fabricated one."""
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 5))
    model = RandomForestRegressor(n_estimators=5, random_state=0).fit(
        X, X[:, 0])
    top, bias, n = contributions(model, X[0], [f"f{i}" for i in range(5)])
    assert top == [] and bias == 0.0 and n == 0


# --- rendering ------------------------------------------------------------

def test_render_warns_when_out_of_domain():
    exp = Explanation(query="x", smiles="CCO", prediction=-1.0,
                      max_similarity=0.11, in_domain=False)
    assert "Outside the applicability domain" in render_text(exp)
    assert "0.11" in render_text(exp)


def test_render_flags_training_set_membership():
    exp = Explanation(query="x", smiles="CCO", prediction=-0.30,
                      measured=-0.31, max_similarity=1.0)
    assert "memory check" in render_text(exp)


def test_render_states_partial_coverage_of_shap_table():
    """The table shows 2 of many features; it must not imply it is the sum."""
    exp = Explanation(
        query="x", smiles="CCO", prediction=-1.0, shap_bias=-2.5,
        n_contributing=400, max_similarity=0.9,
        contributions=[{"feature": "MolLogP", "contribution": 1.0, "value": 0.5},
                       {"feature": "MolWt", "contribution": 0.3, "value": 46.0}],
    )
    text = render_text(exp)
    assert "400" in text and "summary, not the whole sum" in text
