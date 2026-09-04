"""Quantum-chemical descriptors via PySCF (Ch 5 and 6).

The pipeline for one molecule:

    SMILES -> add H -> ETKDG embed -> MMFF optimize   (RDKit, cheap)
           -> B3LYP/6-31G* single point               (PySCF, expensive)
           -> HOMO, LUMO, gap, dipole, Mulliken range

This is the free substitute for the Gaussian workflow in Ch 6.5. The
concepts are identical -- build a geometry, choose a method and basis,
run, parse the orbital energies -- and `pip install pyscf` costs nothing.

Two honest caveats the book's framing can obscure:

  * This is a SINGLE POINT on an MMFF geometry, not a DFT geometry
    optimization. It is roughly 50x cheaper and good enough for
    descriptors. It is not good enough to publish a barrier height.
  * Orbital energies from Kohn-Sham DFT are not ionization potentials.
    They correlate with reactivity, which is all a feature needs to do.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import asdict, dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)

HARTREE_TO_EV = 27.211386245988

QM_FEATURE_NAMES = [
    "qm_homo_ev",
    "qm_lumo_ev",
    "qm_gap_ev",
    "qm_dipole_debye",
    "qm_charge_min",
    "qm_charge_max",
    "qm_charge_span",
    "qm_energy_per_electron_ha",
]


@dataclass
class QMResult:
    smiles: str
    ok: bool
    reason: str = ""
    qm_homo_ev: float = np.nan
    qm_lumo_ev: float = np.nan
    qm_gap_ev: float = np.nan
    qm_dipole_debye: float = np.nan
    qm_charge_min: float = np.nan
    qm_charge_max: float = np.nan
    qm_charge_span: float = np.nan
    qm_energy_per_electron_ha: float = np.nan

    def as_row(self) -> dict:
        return asdict(self)


def embed_3d(smiles: str, seed: int = 0xF00D, max_iters: int = 500):
    """SMILES -> 3D molecule with hydrogens, MMFF-relaxed.

    Returns (mol, reason). `mol` is None when embedding fails, which
    happens for a small fraction of strained cages and large macrocycles.
    Never raises: the batch runner's whole design depends on a bad
    molecule costing one row rather than the run.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "unparseable SMILES"

    # An empty SMILES parses to a valid Mol with zero atoms rather than
    # to None, so it slips past the check above and then makes
    # EmbedMolecule raise "molecule has no atoms". That would take a
    # 200-molecule batch down mid-run and leave a partial CSV looking
    # like a finished subset.
    if mol.GetNumAtoms() == 0:
        return None, "empty molecule (no atoms)"

    mol = Chem.AddHs(mol)

    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            # Retry with random coordinates before giving up.
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                return None, "ETKDG embedding failed"
    except Exception as exc:      # noqa: BLE001 - never raise, see docstring
        return None, f"embedding error: {type(exc).__name__}: {exc}"[:120]

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
    except Exception as exc:  # MMFF has no parameters for some elements
        log.debug("MMFF skipped for %s: %s", smiles, exc)

    return mol, ""


def _pyscf_atoms(mol) -> list[tuple[str, tuple[float, float, float]]]:
    """RDKit conformer -> PySCF atom list, coordinates in Angstrom."""
    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append((atom.GetSymbol(), (pos.x, pos.y, pos.z)))
    return atoms


def qm_descriptors(
    smiles: str,
    basis: str = "6-31G*",
    xc: str = "b3lyp",
    max_heavy_atoms: int = 20,
    conformer_seed: int = 0xF00D,
    mmff_max_iters: int = 500,
    max_memory_mb: int = 400,
) -> QMResult:
    """Compute QM features for one molecule. Never raises -- failures
    come back as `ok=False` with a reason, so a batch run does not die
    on molecule 137 of 200.

    `max_memory_mb` caps PySCF's working budget (its default is 4000 MB).
    It matters because parallel workers multiply it: three of them
    swapping an 8 GB laptop was measured at ~8% of a core each, with
    kernel_task pinned near 95% running the memory compressor.

    Measured on 4-aminophenyl sulfone (17 heavy atoms, ~250 basis
    functions), one process, peak RSS:

        budget 4000 MB -> 1169 MB,  269 s
        budget  900 MB -> 1113 MB,  266 s
        budget  400 MB ->  766 MB,  255 s

    Two things to take from that. The cap works, but not linearly -- it
    buys almost nothing until it is tight enough to bind, and the useful
    setting here is 400, not 900. And the answers are unchanged: gap
    5.0784 eV and dipole 8.9588 D at every budget, identical to four
    decimals, so this is a memory-strategy knob and not a accuracy knob.
    Time does not get worse either, which it would if this were forcing
    a much slower algorithm.

    A caveat on the mechanism, since it is easy to assume: at the default
    budget this molecule peaked at 1169 MB, nowhere near the ~3.9 GB a
    stored 250^4/8 ERI tensor would need, so PySCF was not holding the
    full integral list here regardless. The larger jobs that pushed
    workers to 2.6-3.6 GB may well have been; that was not measured.
    Raise the budget if you have the RAM and are running few workers.
    """
    mol2d = Chem.MolFromSmiles(smiles)
    if mol2d is None:
        return QMResult(smiles=smiles, ok=False, reason="unparseable SMILES")

    n_heavy = mol2d.GetNumHeavyAtoms()
    if n_heavy > max_heavy_atoms:
        return QMResult(
            smiles=smiles, ok=False,
            reason=f"too large ({n_heavy} > {max_heavy_atoms} heavy atoms)",
        )

    mol3d, reason = embed_3d(smiles, conformer_seed, mmff_max_iters)
    if mol3d is None:
        return QMResult(smiles=smiles, ok=False, reason=reason)

    try:
        from pyscf import dft, gto
    except ImportError:
        return QMResult(smiles=smiles, ok=False, reason="pyscf not installed")

    charge = Chem.GetFormalCharge(mol3d)
    n_radical = sum(a.GetNumRadicalElectrons() for a in mol3d.GetAtoms())

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            pyscf_mol = gto.M(
                atom=_pyscf_atoms(mol3d),
                basis=basis,
                charge=charge,
                spin=n_radical,   # 2S, i.e. count of unpaired electrons
                verbose=0,
                unit="Angstrom",
                max_memory=max_memory_mb,
            )

            mf = dft.UKS(pyscf_mol) if n_radical else dft.RKS(pyscf_mol)
            mf.xc = xc
            mf.max_memory = max_memory_mb
            energy = mf.kernel()

        if not mf.converged:
            return QMResult(smiles=smiles, ok=False, reason="SCF did not converge")

        # Flatten the alpha/beta axis for open-shell (UKS) results so the
        # HOMO/LUMO extraction below is the same code either way.
        mo_energy = np.concatenate([np.ravel(e) for e in np.atleast_2d(mf.mo_energy)])
        mo_occ = np.concatenate([np.ravel(o) for o in np.atleast_2d(mf.mo_occ)])

        occupied = mo_energy[mo_occ > 0]
        virtual = mo_energy[mo_occ == 0]
        if occupied.size == 0 or virtual.size == 0:
            return QMResult(smiles=smiles, ok=False, reason="no frontier orbitals")

        homo = float(occupied.max()) * HARTREE_TO_EV
        lumo = float(virtual.min()) * HARTREE_TO_EV

        dipole = float(np.linalg.norm(mf.dip_moment(unit="Debye", verbose=0)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, charges = mf.mulliken_pop(verbose=0)
        charges = np.asarray(charges, dtype=float)

        return QMResult(
            smiles=smiles,
            ok=True,
            qm_homo_ev=homo,
            qm_lumo_ev=lumo,
            qm_gap_ev=lumo - homo,
            qm_dipole_debye=dipole,
            qm_charge_min=float(charges.min()),
            qm_charge_max=float(charges.max()),
            qm_charge_span=float(charges.max() - charges.min()),
            qm_energy_per_electron_ha=float(energy) / pyscf_mol.nelectron,
        )

    except Exception as exc:
        return QMResult(
            smiles=smiles, ok=False, reason=f"{type(exc).__name__}: {exc}"[:200]
        )
