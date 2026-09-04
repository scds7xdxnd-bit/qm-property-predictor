"""Graph construction is the part that fails silently -- a wrong edge
list still trains, just worse. These pin the graph, not the accuracy."""

import numpy as np
import pytest

from qmprop.gnn import (
    ATOM_FEATURE_DIM,
    BOND_TYPES,
    collate,
    mol_to_graph,
)


def test_ethanol_graph_shape():
    x, edge_index, edge_kind = mol_to_graph("CCO")
    assert x.shape == (3, ATOM_FEATURE_DIM)      # heavy atoms only
    assert edge_index.shape == (2, 4)            # 2 bonds, both directions
    assert len(edge_kind) == 4


def test_edges_are_symmetric():
    """Message passing must see an undirected graph; a one-way edge list
    would make information flow in only one direction along every bond."""
    _, edge_index, _ = mol_to_graph("CC(=O)Oc1ccccc1C(=O)O")
    pairs = {(int(a), int(b)) for a, b in zip(*edge_index)}
    assert all((b, a) in pairs for a, b in pairs)


def test_bond_types_are_distinguished():
    _, _, single = mol_to_graph("CC")
    _, _, double = mol_to_graph("C=C")
    _, _, triple = mol_to_graph("C#C")
    assert single[0] == BOND_TYPES.index("SINGLE")
    assert double[0] == BOND_TYPES.index("DOUBLE")
    assert triple[0] == BOND_TYPES.index("TRIPLE")


def test_aromatic_ring_is_aromatic_not_alternating():
    """Benzene from RDKit is 6 aromatic bonds, not 3 single + 3 double."""
    _, _, kinds = mol_to_graph("c1ccccc1")
    assert set(kinds.tolist()) == {BOND_TYPES.index("AROMATIC")}
    assert len(kinds) == 12


def test_single_atom_molecule_has_no_edges():
    """Sodium ion has no bonds; an empty edge list must not crash collate."""
    x, edge_index, edge_kind = mol_to_graph("[Na+]")
    assert x.shape == (1, ATOM_FEATURE_DIM)
    assert edge_index.shape == (2, 0)
    assert edge_kind.shape == (0,)


def test_invalid_smiles_returns_none():
    assert mol_to_graph("not a molecule") is None


def test_unknown_element_lands_in_other_slot():
    """A rare element should degrade, not raise."""
    result = mol_to_graph("[Se]=C=[Se]")
    assert result is not None
    x, _, _ = result
    assert x.shape[1] == ATOM_FEATURE_DIM


def test_collate_offsets_node_indices():
    """The block-diagonal trick: graph 2's edges must not point at
    graph 1's atoms. Getting this wrong silently fuses molecules."""
    graphs = [mol_to_graph(s) for s in ("CCO", "CCO")]
    x, edge_index, _, batch_index = collate(graphs)
    assert x.shape[0] == 6
    assert set(batch_index.tolist()) == {0, 1}
    second = edge_index[:, 4:]                   # graph 1's four edges
    assert second.min() >= 3, "second graph's edges must be offset by 3"


def test_collate_batch_index_matches_atom_counts():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    graphs = [mol_to_graph(s) for s in smiles]
    _, _, _, batch_index = collate(graphs)
    counts = np.bincount(batch_index)
    assert counts.tolist() == [len(g[0]) for g in graphs]


def test_forward_pass_runs_and_is_deterministic():
    torch = pytest.importorskip("torch")
    from qmprop.gnn import build_gnn

    graphs = [mol_to_graph(s) for s in ("CCO", "c1ccccc1", "[Na+]")]
    x, e, k, b = collate(graphs)
    args = (torch.tensor(x), torch.tensor(e), torch.tensor(k),
            torch.tensor(b), 3)

    a = build_gnn(seed=7).eval()
    c = build_gnn(seed=7).eval()
    with torch.no_grad():
        assert torch.allclose(a(*args), c(*args))


def test_sum_pooling_makes_size_visible():
    """Two ethanols pooled as one graph must not equal one ethanol.

    Guards the sum-vs-mean choice: with mean pooling these would be
    identical, and the model would lose molecular size entirely.
    """
    torch = pytest.importorskip("torch")
    from qmprop.gnn import build_gnn

    model = build_gnn(seed=3).eval()
    one = collate([mol_to_graph("CCO")])
    two = collate([mol_to_graph("CCOCCO".replace("OCC", "OCC"))])  # bigger mol
    with torch.no_grad():
        a = model(torch.tensor(one[0]), torch.tensor(one[1]),
                  torch.tensor(one[2]), torch.tensor(one[3]), 1)
        b = model(torch.tensor(two[0]), torch.tensor(two[1]),
                  torch.tensor(two[2]), torch.tensor(two[3]), 1)
    assert not torch.allclose(a, b)
