"""The split is the part most worth testing: a silent bug here inflates
every downstream number and looks like success."""

import numpy as np
import pytest

from qmprop.splits import murcko_scaffold, random_split, scaffold_split

BENZENES = [
    "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1O",
]
PYRIDINES = ["c1ccncc1C", "c1ccncc1CC", "c1ccncc1O"]
ACYCLIC = ["CCO", "CCCO", "CCCCO"]
ALL = BENZENES + PYRIDINES + ACYCLIC


def test_murcko_extracts_ring_core():
    assert murcko_scaffold("c1ccccc1CCC") == "c1ccccc1"
    assert murcko_scaffold("c1ccncc1CC") == "c1ccncc1"


def test_murcko_empty_for_acyclic():
    assert murcko_scaffold("CCCCO") == ""


def test_murcko_survives_bad_smiles():
    assert murcko_scaffold("not-a-molecule") == ""


def test_split_partitions_every_index_exactly_once():
    train, valid, test = scaffold_split(ALL, 0.6, 0.0, 0.4, seed=0)
    combined = np.concatenate([train, valid, test])
    assert sorted(combined) == list(range(len(ALL)))


def test_no_scaffold_spans_train_and_test():
    """The whole point: a chemical series lands on one side of the wall."""
    train, _, test = scaffold_split(ALL, 0.6, 0.0, 0.4, seed=0)
    train_scaffolds = {murcko_scaffold(ALL[i]) for i in train}
    test_scaffolds = {murcko_scaffold(ALL[i]) for i in test}
    assert not (train_scaffolds & test_scaffolds)


def test_split_is_deterministic():
    a = scaffold_split(ALL, 0.6, 0.0, 0.4, seed=7)
    b = scaffold_split(ALL, 0.6, 0.0, 0.4, seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        scaffold_split(ALL, 0.6, 0.0, 0.6)


def test_random_split_also_partitions_cleanly():
    train, valid, test = random_split(ALL, 0.6, 0.2, 0.2, seed=1)
    combined = np.concatenate([train, valid, test])
    assert sorted(combined) == list(range(len(ALL)))


# --- scaffold k-fold ------------------------------------------------------

def test_kfold_tests_every_molecule_exactly_once():
    """The whole reason k-fold is here: every molecule contributes one
    out-of-fold prediction, so the ablation is scored on all of them."""
    from qmprop.splits import scaffold_kfold

    folds = scaffold_kfold(ALL, n_folds=3, seed=0)
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested) == list(range(len(ALL)))


def test_kfold_never_splits_a_scaffold_across_folds():
    """If a scaffold appeared in two folds, its analogs would leak from
    training into test and the fold would stop being an honest test."""
    from qmprop.splits import murcko_scaffold, scaffold_kfold

    folds = scaffold_kfold(ALL, n_folds=3, seed=0)
    home = {}
    for fold_id, (_, test) in enumerate(folds):
        for i in test:
            scaffold = murcko_scaffold(ALL[i])
            assert home.setdefault(scaffold, fold_id) == fold_id, scaffold


def test_kfold_train_and_test_are_disjoint_and_complete():
    from qmprop.splits import scaffold_kfold

    for train, test in scaffold_kfold(ALL, n_folds=3, seed=0):
        assert set(train).isdisjoint(test)
        assert sorted(np.concatenate([train, test])) == list(range(len(ALL)))


def test_kfold_is_deterministic_for_a_seed():
    from qmprop.splits import scaffold_kfold

    a = scaffold_kfold(ALL, n_folds=3, seed=7)
    b = scaffold_kfold(ALL, n_folds=3, seed=7)
    for (t1, s1), (t2, s2) in zip(a, b):
        assert np.array_equal(t1, t2) and np.array_equal(s1, s2)


def test_kfold_rejects_impossible_requests():
    from qmprop.splits import scaffold_kfold

    with pytest.raises(ValueError):
        scaffold_kfold(ALL, n_folds=1)
    with pytest.raises(ValueError):
        scaffold_kfold(ALL[:2], n_folds=5)


def test_kfold_raises_rather_than_returning_an_empty_fold():
    """Three scaffolds cannot fill five folds. Silently returning an
    empty test fold would make out_of_fold leave NaNs behind."""
    from qmprop.splits import scaffold_kfold

    only_three = ["c1ccccc1C", "c1ccccc1CC", "c1ccncc1C", "c1ccncc1CC", "CCO", "CCCO"]
    with pytest.raises(ValueError, match="empty"):
        scaffold_kfold(only_three, n_folds=5)
