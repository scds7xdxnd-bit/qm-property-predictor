import numpy as np
import pytest

from qmprop.data import canonical_smiles, inchikey
from qmprop.features import build_features, descriptor_matrix, morgan_matrix

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]  # ethanol, benzene, aspirin


def test_canonical_smiles_normalizes_equivalent_strings():
    assert canonical_smiles("C(C)O") == canonical_smiles("CCO")


def test_canonical_smiles_rejects_nonsense():
    assert canonical_smiles("this is not a molecule") is None


def test_inchikey_is_stable_across_writings():
    assert inchikey("C(C)O") == inchikey("CCO")


def test_morgan_shape_and_binary_values():
    X, names = morgan_matrix(SMILES, radius=2, n_bits=256)
    assert X.shape == (3, 256)
    assert len(names) == 256
    assert set(np.unique(X)).issubset({0.0, 1.0})


def test_descriptors_are_finite_and_varying():
    X, names = descriptor_matrix(SMILES)
    assert X.shape[0] == 3
    assert X.shape[1] == len(names)
    assert np.isfinite(X).all()


def test_unknown_descriptor_name_raises():
    with pytest.raises(KeyError):
        descriptor_matrix(SMILES, names=["NotARealDescriptor"])


def test_build_features_concatenates_blocks():
    qm = np.zeros((3, 2), dtype=np.float32)
    X, names = build_features(
        SMILES,
        use_morgan=True, use_descriptors=True,
        qm_matrix=qm, qm_names=["qm_a", "qm_b"],
        morgan_kwargs={"n_bits": 64, "radius": 2},
    )
    assert X.shape[1] == len(names)
    assert names[-2:] == ["qm_a", "qm_b"]


def test_build_features_rejects_mismatched_qm_rows():
    with pytest.raises(ValueError, match="expected 3"):
        build_features(SMILES, qm_matrix=np.zeros((2, 2)))


def test_build_features_requires_at_least_one_family():
    with pytest.raises(ValueError, match="no feature family"):
        build_features(SMILES, use_morgan=False, use_descriptors=False)


def test_explicit_names_keep_every_column_for_one_molecule():
    """Regression: the constant-column filter is training-time logic.

    With one molecule every column has zero variance, so running the
    filter at inference drops all of them and the model is handed the
    wrong feature count. Explicit names must round-trip exactly.
    """
    _, names = descriptor_matrix(SMILES)          # training: filter runs
    X_one, names_one = descriptor_matrix(["CCO"], names=names)
    assert names_one == names
    assert X_one.shape == (1, len(names))
    assert np.isfinite(X_one).all()


def test_single_molecule_matches_training_feature_count():
    """The exact failure the Space hit: 2048 built vs 2247 expected."""
    _, desc_names = descriptor_matrix(SMILES)
    X_train, train_names = build_features(
        SMILES, morgan_kwargs={"n_bits": 64, "radius": 2},
        descriptor_names=desc_names,
    )
    X_one, _ = build_features(
        ["CCO"], morgan_kwargs={"n_bits": 64, "radius": 2},
        descriptor_names=desc_names,
    )
    assert X_one.shape[1] == X_train.shape[1] == len(train_names)


def test_drop_degenerate_can_be_forced_off():
    X, names = descriptor_matrix(["CCO"], drop_degenerate=False)
    assert X.shape[0] == 1 and X.shape[1] == len(names) > 100
