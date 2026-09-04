"""Outside data: AqSolDB, PubChem, ChEMBL (Ch 2).

Two distinct jobs, both part of "assembling a dataset" in Ch 2:

  * `load_aqsoldb` pulls a second, independently curated solubility set
    so ESOL can be checked against it. Where the two disagree on the
    same InChIKey, that disagreement is experimental noise -- and it is
    the floor no model can get under. Ch 3 reports RMSE without ever
    asking what RMSE would be achievable, which is how a project talks
    itself into chasing a tenth of a log unit that isn't there.

  * `pubchem_*` and `chembl_*` resolve human names to structures. Ch 7
    needs this: a user types "caffeine", not "Cn1cnc2c1c(=O)n(C)c(=O)n2C".

Both are network calls against public REST APIs, cached on disk. PubChem
asks for no more than 5 requests/second and will throttle you if you
ignore that, so the client sleeps between calls.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# AqSolDB (Sorkun, Khetan & Er, Sci Data 2019), doi:10.7910/DVN/OVHAW8.
# 9982 compounds merged from nine sources, with a standard-deviation
# column recording how much those sources disagreed.
AQSOLDB_URL = "https://dataverse.harvard.edu/api/access/datafile/3407241"

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

_PUBCHEM_MIN_INTERVAL = 0.25   # seconds; their published cap is 5 req/s
_last_pubchem_call = 0.0

USER_AGENT = "qmprop/0.1 (educational project; contact via repository)"


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read()


# --------------------------------------------------------------------------
# AqSolDB
# --------------------------------------------------------------------------

def load_aqsoldb(dest: Path) -> pd.DataFrame:
    """Download (once) and load AqSolDB.

    Returns the columns worth having: InChIKey, SMILES, Solubility
    (log mol/L, same units as ESOL), SD, and how many sources contributed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        log.info("downloading AqSolDB (~3.8 MB)")
        dest.write_bytes(_get(AQSOLDB_URL, timeout=120))
    else:
        log.info("using cached %s", dest.name)

    df = pd.read_csv(dest, sep="\t")
    keep = ["ID", "Name", "InChIKey", "SMILES", "Solubility", "SD", "Ocurrences"]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df.rename(columns={
        "InChIKey": "inchikey",
        "SMILES": "smiles",
        "Solubility": "logS_aqsoldb",
        "SD": "sd_aqsoldb",
        "Ocurrences": "n_sources",
    })


def cross_source_agreement(
    esol: pd.DataFrame,
    aqsoldb: pd.DataFrame,
    key: str = "inchikey",
) -> pd.DataFrame:
    """Inner-join two solubility sources on InChIKey.

    ESOL is one of the nine sources AqSolDB merged, so overlap is
    expected and the comparison is not fully independent -- where
    AqSolDB had only ESOL for a compound the two agree by construction.
    Restrict to `n_sources > 1` for the honest comparison; the returned
    frame keeps the column so the caller can.
    """
    if key not in esol.columns:
        raise KeyError(f"ESOL frame has no {key!r} column")
    merged = esol.merge(
        aqsoldb.drop_duplicates(subset=key), on=key, how="inner",
        suffixes=("", "_aq"),
    )
    merged["delta"] = merged["logS_aqsoldb"] - merged["logS"]
    return merged


# --------------------------------------------------------------------------
# PubChem
# --------------------------------------------------------------------------

def pubchem_resolve(name: str, timeout: int = 30) -> dict | None:
    """Name -> {cid, smiles, formula, iupac_name}, or None if not found.

    PubChem renamed the SMILES property: `CanonicalSMILES` became
    `ConnectivitySMILES` (and `IsomericSMILES` became `SMILES`). Older
    tutorials request the old names and get a 400, so both spellings are
    tried before giving up.
    """
    global _last_pubchem_call

    for props in ("ConnectivitySMILES,MolecularFormula,IUPACName",
                  "CanonicalSMILES,MolecularFormula,IUPACName"):
        elapsed = time.monotonic() - _last_pubchem_call
        if elapsed < _PUBCHEM_MIN_INTERVAL:
            time.sleep(_PUBCHEM_MIN_INTERVAL - elapsed)

        url = (f"{PUBCHEM_BASE}/compound/name/"
               f"{urllib.parse.quote(name)}/property/{props}/JSON")
        try:
            payload = json.loads(_get(url, timeout=timeout))
        except Exception as exc:               # noqa: BLE001 - any failure -> try next
            log.debug("PubChem lookup failed for %r (%s): %s", name, props, exc)
            _last_pubchem_call = time.monotonic()
            continue
        finally:
            _last_pubchem_call = time.monotonic()

        rows = payload.get("PropertyTable", {}).get("Properties", [])
        if not rows:
            continue
        row = rows[0]
        smiles = row.get("ConnectivitySMILES") or row.get("CanonicalSMILES")
        if not smiles:
            continue
        return {
            "cid": row.get("CID"),
            "smiles": smiles,
            "formula": row.get("MolecularFormula"),
            "iupac_name": row.get("IUPACName"),
            "source": "pubchem",
        }
    return None


# --------------------------------------------------------------------------
# ChEMBL
# --------------------------------------------------------------------------

def chembl_resolve(name: str, timeout: int = 30) -> dict | None:
    """Name -> {chembl_id, smiles, pref_name}, or None. Fallback for PubChem."""
    url = (f"{CHEMBL_BASE}/molecule.json?pref_name__iexact="
           f"{urllib.parse.quote(name)}&limit=1")
    try:
        payload = json.loads(_get(url, timeout=timeout))
    except Exception as exc:                   # noqa: BLE001
        log.debug("ChEMBL lookup failed for %r: %s", name, exc)
        return None

    molecules = payload.get("molecules") or []
    if not molecules:
        return None
    mol = molecules[0]
    structures = mol.get("molecule_structures") or {}
    smiles = structures.get("canonical_smiles")
    if not smiles:
        return None
    return {
        "chembl_id": mol.get("molecule_chembl_id"),
        "smiles": smiles,
        "pref_name": mol.get("pref_name"),
        "source": "chembl",
    }


def resolve_name(name: str) -> dict | None:
    """PubChem first, ChEMBL second. Ch 7's front door."""
    return pubchem_resolve(name) or chembl_resolve(name)
