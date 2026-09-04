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
