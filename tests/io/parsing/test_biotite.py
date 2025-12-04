
"""Tests for Biotite parsing utilities."""

import tempfile

from priox.io.parsing import biotite
from priox.io.parsing.structures import ProcessedStructure
from priox.chem import residues as rc
from priox.io.parsing.utils import (
    atom_array_dihedrals,
)


def test_atom_array_dihedrals():
    """Test the atom_array_dihedrals function."""
    with open("tests/data/1ubq.pdb") as f:
        pdb_string = f.read()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.write(pdb_string)
        filepath = tmp.name
    # Use biotite.load_structure_with_hydride to get AtomArray directly
    atom_array = biotite.load_structure_with_hydride(filepath)
    dihedrals = atom_array_dihedrals(atom_array)
    assert dihedrals is not None
