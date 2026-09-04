"""External-source joins and name resolution.

The join logic is tested offline with fixtures -- it is the part that can
be silently wrong. The live API calls are tested too, but they skip
rather than fail when the network is unavailable, so a plane or a CI box
without egress does not produce a red build that means nothing.
"""

import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from qmprop.external import (
    chembl_resolve,
    cross_source_agreement,
    pubchem_resolve,
    resolve_name,
)


@pytest.fixture
def esol():
    return pd.DataFrame({
        "inchikey": ["AAA-N", "BBB-N", "CCC-N"],
        "smiles": ["CCO", "CCCO", "c1ccccc1"],
        "logS": [-0.31, -0.66, -1.64],
    })


@pytest.fixture
def aqsoldb():
    return pd.DataFrame({
        "inchikey": ["AAA-N", "BBB-N", "DDD-N"],
        "smiles": ["CCO", "CCCO", "CCCCO"],
        "logS_aqsoldb": [-0.31, -1.16, -0.88],
        "sd_aqsoldb": [0.0, 0.4, 0.1],
        "n_sources": [1, 3, 2],
    })


def test_join_keeps_only_shared_molecules(esol, aqsoldb):
    merged = cross_source_agreement(esol, aqsoldb)
    assert len(merged) == 2
    assert set(merged["inchikey"]) == {"AAA-N", "BBB-N"}


def test_delta_direction_is_aqsoldb_minus_esol(esol, aqsoldb):
    merged = cross_source_agreement(esol, aqsoldb)
    row = merged[merged["inchikey"] == "BBB-N"].iloc[0]
    assert row["delta"] == pytest.approx(-1.16 - (-0.66))


def test_agreeing_source_gives_zero_delta(esol, aqsoldb):
    merged = cross_source_agreement(esol, aqsoldb)
    row = merged[merged["inchikey"] == "AAA-N"].iloc[0]
    assert row["delta"] == pytest.approx(0.0)


def test_duplicate_keys_in_second_source_do_not_multiply_rows(esol, aqsoldb):
    """A duplicated InChIKey would fan the join out and double-weight
    that molecule in the noise-floor RMSE."""
    doubled = pd.concat([aqsoldb, aqsoldb.iloc[[0]]], ignore_index=True)
    merged = cross_source_agreement(esol, doubled)
    assert len(merged) == 2
    assert merged["inchikey"].is_unique


def test_missing_key_column_raises(aqsoldb):
    with pytest.raises(KeyError):
        cross_source_agreement(pd.DataFrame({"smiles": ["CCO"]}), aqsoldb)


def test_n_sources_survives_the_join_for_filtering(esol, aqsoldb):
    """The honest subset needs this column; losing it to a suffix
    collision would silently change which rows get reported."""
    merged = cross_source_agreement(esol, aqsoldb)
    assert "n_sources" in merged.columns
    assert (merged["n_sources"] > 1).sum() == 1


# --- live lookups, skipped when offline -----------------------------------

def _needs_network(fn, *args):
    try:
        return fn(*args)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        pytest.skip(f"network unavailable: {exc}")


@pytest.mark.network
def test_pubchem_resolves_a_known_name():
    from rdkit import Chem

    record = _needs_network(pubchem_resolve, "caffeine")
    if record is None:
        pytest.skip("PubChem returned no record")
    assert record["cid"] == 2519
    mol = Chem.MolFromSmiles(record["smiles"])
    assert mol is not None
    from rdkit.Chem import rdMolDescriptors
    assert rdMolDescriptors.CalcMolFormula(mol) == "C8H10N4O2"


@pytest.mark.network
def test_pubchem_returns_none_for_nonsense():
    assert _needs_network(pubchem_resolve, "zzzz-not-a-compound-xyzzy") is None


@pytest.mark.network
def test_resolve_name_falls_through_to_none():
    assert _needs_network(resolve_name, "zzzz-not-a-compound-xyzzy") is None
