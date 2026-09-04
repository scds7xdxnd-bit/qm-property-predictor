"""The contract the batch runner depends on: qm_descriptors NEVER raises.

`scripts/03_run_qm.py` walks 200 molecules and appends each result. If a
single bad molecule threw, the run would die partway and the partial CSV
would look like a completed subset. So every failure mode has to come
back as ok=False with a reason. These tests are cheap -- none of them
reaches an actual SCF.
"""

import sys

import pytest

from qmprop.qm import QM_FEATURE_NAMES, QMResult, qm_descriptors


def test_unparseable_smiles_returns_reason():
    r = qm_descriptors("not a molecule")
    assert r.ok is False and "SMILES" in r.reason


def test_oversize_molecule_is_refused_before_any_scf():
    """Cheap guard: this must not spend an hour discovering it is too big."""
    big = "C" * 60
    r = qm_descriptors(big, max_heavy_atoms=20)
    assert r.ok is False
    assert "too large" in r.reason and "60" in r.reason


def test_empty_string_returns_reason():
    r = qm_descriptors("")
    assert r.ok is False


def test_failed_result_still_has_every_column():
    """A failure row is written to the same CSV as a success. Missing
    columns would shift the header and corrupt every later row."""
    r = qm_descriptors("not a molecule")
    row = r.as_row()
    for name in QM_FEATURE_NAMES:
        assert name in row, name
    assert row["ok"] is False


def test_degrades_cleanly_when_pyscf_is_absent(monkeypatch):
    """CI installs no pyscf, and a user may not have it either. The
    import failure must surface as a reason, not a traceback."""

    class Blocker:
        # find_spec, not the removed find_module -- a find_module hook is
        # silently ignored on 3.12+, so the block would do nothing and
        # the test would pass by accident.
        def find_spec(self, name, path=None, target=None):
            if name == "pyscf" or name.startswith("pyscf."):
                raise ImportError(f"blocked for test: {name}")
            return None

    monkeypatch.setitem(sys.modules, "_dummy", None)
    for mod in [k for k in list(sys.modules) if k.startswith("pyscf")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])

    r = qm_descriptors("CCO")
    assert r.ok is False
    assert "pyscf" in r.reason


def test_qm_result_row_is_flat_and_serializable():
    import json

    row = QMResult(smiles="CCO", ok=False, reason="x").as_row()
    json.dumps(row)   # raises if a numpy scalar or nested object leaked in
