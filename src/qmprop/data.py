"""Dataset assembly (Ch 2).

Loads ESOL/Delaney -- 1128 molecules with measured aqueous solubility --
then canonicalizes and deduplicates. The deduplication step matters more
than it looks: the same molecule written two ways survives a naive load
and lands in both train and test, which inflates every score downstream
(the point of Ch 1.6).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
from rdkit import Chem, RDLogger

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")  # RDKit is chatty about every parse failure


def download(urls: str | Sequence[str], dest: Path) -> Path:
    """Fetch the dataset once and cache it on disk.

    Takes a list of mirrors and tries them in order. Dataset URLs rot --
    the DeepChem S3 bucket that hosted ESOL for years now 404s -- and a
    pipeline that dies at step one because of someone else's bucket
    policy is a bad pipeline.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("using cached %s", dest.name)
        return dest

    import urllib.error
    import urllib.request

    if isinstance(urls, str):
        urls = [urls]

    failures = []
    for url in urls:
        try:
            log.info("downloading %s", url)
            urllib.request.urlretrieve(url, dest)
            return dest
        except (urllib.error.URLError, OSError) as exc:
            log.warning("  failed: %s", exc)
            failures.append(f"{url} -> {exc}")
            dest.unlink(missing_ok=True)

    raise RuntimeError(
        "every mirror failed:\n  " + "\n  ".join(failures)
    )


def canonical_smiles(smiles: str) -> str | None:
    """Canonical SMILES, or None if RDKit cannot parse the string.

    Also the validity check to run on anything an LLM hands you (Ch 7):
    a None here means the molecule was never real.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def inchikey(smiles: str) -> str | None:
    """Standard InChIKey -- the identity key for deduplication (Ch 1.1)."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


def load_dataset(cfg: dict) -> pd.DataFrame:
    """Return a clean frame with columns: smiles, inchikey, target.

    Rows that fail to parse are dropped and counted. Duplicates are
    collapsed to the mean of their measurements, which is the honest
    treatment when the same compound was measured twice.
    """
    ds = cfg["dataset"]
    raw_path = cfg["data_dir"] / "raw" / f"{ds['name']}.csv"
    download(ds.get("urls") or ds["url"], raw_path)

    df = pd.read_csv(raw_path)

    smi_col, tgt_col = ds["smiles_column"], ds["target_column"]
    missing = [c for c in (smi_col, tgt_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"columns {missing} not in {raw_path.name}. "
            f"Found: {list(df.columns)}"
        )

    n_raw = len(df)
    df = df[[smi_col, tgt_col]].rename(
        columns={smi_col: "smiles", tgt_col: ds["target_name"]}
    )

    df["smiles"] = df["smiles"].map(canonical_smiles)
    n_unparsed = int(df["smiles"].isna().sum())
    df = df.dropna(subset=["smiles"])

    df["inchikey"] = df["smiles"].map(inchikey)
    df = df.dropna(subset=["inchikey"])

    target = ds["target_name"]
    n_before_dedupe = len(df)
    df = (
        df.groupby("inchikey", as_index=False)
        .agg(smiles=("smiles", "first"), **{target: (target, "mean")})
    )

    log.info(
        "%s: %d rows -> %d molecules (%d unparsed, %d duplicates merged)",
        ds["name"], n_raw, len(df), n_unparsed, n_before_dedupe - len(df),
    )
    return df.reset_index(drop=True)
