"""A small message-passing GNN, written out rather than imported (Ch 4).

The Ch 4 stretch goal. Deliberately no torch-geometric: the whole model
is ~80 lines of torch, and the point of the chapter is to see what a
graph network actually does to a molecule, not to configure one.

The architecture is the plain one from Gilmer et al. 2017, stripped to
essentials:

    atom features -> embed
    T rounds of:  h_v <- GRU(h_v, sum over neighbours W h_u)
    sum-pool over atoms -> MLP -> one number

Two design notes that matter for honesty:

  * Sum pooling, not mean. Solubility is roughly extensive -- a bigger
    molecule has more surface to solvate -- and mean-pooling throws away
    size, which is one of the strongest signals in the data.
  * The bond type enters as a separate weight matrix per bond order, the
    cheap version of edge conditioning. Without it the network cannot
    tell a single bond from a double, which for solubility matters.

This is not expected to beat XGBoost on 893 training molecules, and it
does not. Graph networks earn their keep at 10^5 molecules and up; below
that the inductive bias does not pay for the parameters. Reporting that
plainly is the honest version of the Ch 4 comparison.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# One-hot vocabularies. Anything outside them lands in a trailing "other"
# slot rather than raising -- a single exotic atom should not kill a run.
ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si"]
DEGREES = [0, 1, 2, 3, 4, 5]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

ATOM_FEATURE_DIM = (
    len(ATOMS) + 1 + len(DEGREES) + 1 + len(HYBRIDIZATIONS) + 1 + 4
)


def _one_hot(value, vocabulary: list) -> list[float]:
    """One-hot with an explicit 'other' slot at the end."""
    vec = [0.0] * (len(vocabulary) + 1)
    vec[vocabulary.index(value) if value in vocabulary else len(vocabulary)] = 1.0
    return vec


def atom_features(atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOMS)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS)
        + [
            float(atom.GetFormalCharge()),
            float(atom.GetIsAromatic()),
            float(atom.GetTotalNumHs()),
            float(atom.IsInRing()),
        ]
    )


def mol_to_graph(smiles: str):
    """SMILES -> (atom features, edge index, edge bond-type index).

    Edges are stored in both directions: message passing is symmetric,
    and a directed edge list would quietly make the graph a DAG.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)

    src, dst, kind = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        name = str(bond.GetBondType())
        b = BOND_TYPES.index(name) if name in BOND_TYPES else len(BOND_TYPES)
        src += [i, j]
        dst += [j, i]
        kind += [b, b]

    if not src:                      # single-atom molecules have no bonds
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_kind = np.zeros((0,), dtype=np.int64)
    else:
        edge_index = np.array([src, dst], dtype=np.int64)
        edge_kind = np.array(kind, dtype=np.int64)
    return x, edge_index, edge_kind


def collate(graphs: list):
    """Batch graphs by block-diagonal concatenation.

    Standard trick: offset each graph's node indices and concatenate, so
    one sparse scatter handles the whole batch. `batch_index` records
    which graph each atom came from, for pooling at the end.
    """
    xs, edges, kinds, batch_index = [], [], [], []
    offset = 0
    for g, (x, edge_index, edge_kind) in enumerate(graphs):
        xs.append(x)
        edges.append(edge_index + offset)
        kinds.append(edge_kind)
        batch_index.append(np.full(len(x), g, dtype=np.int64))
        offset += len(x)
    return (
        np.concatenate(xs),
        np.concatenate(edges, axis=1) if edges else np.zeros((2, 0), np.int64),
        np.concatenate(kinds),
        np.concatenate(batch_index),
    )


def build_gnn(hidden: int = 96, rounds: int = 3, seed: int = 42):
    """Construct the model. Imported lazily so torch stays optional."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)

    class MPNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(ATOM_FEATURE_DIM, hidden)
            # One message matrix per bond order (+1 for 'other').
            self.message = nn.ModuleList(
                [nn.Linear(hidden, hidden) for _ in range(len(BOND_TYPES) + 1)]
            )
            self.update = nn.GRUCell(hidden, hidden)
            self.readout = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Dropout(0.1), nn.Linear(hidden, 1),
            )
            self.rounds = rounds

        def forward(self, x, edge_index, edge_kind, batch_index, n_graphs):
            h = torch.relu(self.embed(x))
            src, dst = edge_index[0], edge_index[1]

            for _ in range(self.rounds):
                if src.numel():
                    messages = torch.zeros_like(h)
                    for b, layer in enumerate(self.message):
                        mask = edge_kind == b
                        if not mask.any():
                            continue
                        contribution = layer(h[src[mask]])
                        messages.index_add_(0, dst[mask], contribution)
                else:
                    messages = torch.zeros_like(h)
                h = self.update(messages, h)

            pooled = torch.zeros(n_graphs, h.shape[1], device=h.device)
            pooled.index_add_(0, batch_index, h)      # sum pooling, see docstring
            return self.readout(pooled).squeeze(-1)

    return MPNN()
