"""Tests for JAX MD bridge."""

import jax.numpy as jnp
import pytest

from priox.md import jax_md_bridge
from priox.core.containers import ProteinTuple
from priox.chem import residues as residue_constants


@pytest.fixture
def mock_force_field():
  """Create a mock force field."""
  class MockFF:
    def __init__(self):
      # Basic parameter methods
      self.get_charge = lambda res, atom: 0.0
      self.get_lj_params = lambda res, atom: (1.0, 0.1)
      
      # Required attributes accessed by parameterize_system
      self.bonds = []  # List of (class1, class2, length, k)
      self.angles = []  # List of (class1, class2, class3, theta, k)
      self.atom_class_map = {}  # Maps "RES_ATOM" -> class
      self.atom_type_map = {}  # Maps "RES_ATOM" -> type
      self.atom_key_to_id = {}  # Maps (res, atom) -> id
      self.residue_templates = {}  # Maps res -> [(atom1, atom2), ...]
      self.propers = []  # List of dihedral definitions
      self.cmap_torsions = []  # List of CMAP torsion definitions
      self.impropers = []  # List of improper definitions
      self.cmap_energy_grids = []  # CMAP energy grids

  return MockFF()


@pytest.fixture(autouse=True)
def mock_stereo_chemical_props(monkeypatch):
  """Mock stereo chemical props loading."""
  def mock_load():
    # Return (residue_bonds, residue_virtual_bonds, residue_bond_angles)
    # We need minimal data for ALA and GLY
    # Bond(atom1, atom2, length, stddev)
    # BondAngle(atom1, atom2, atom3, rad, stddev)
    
    from priox.chem.residues import Bond, BondAngle
    
    bonds = {
        "ALA": [
            Bond("N", "CA", 1.46, 0.01),
            Bond("CA", "C", 1.52, 0.01),
            Bond("C", "O", 1.23, 0.01),
            Bond("CA", "CB", 1.53, 0.01)
        ],
        "GLY": [
            Bond("N", "CA", 1.45, 0.01),
            Bond("CA", "C", 1.52, 0.01),
            Bond("C", "O", 1.23, 0.01)
        ]
    }
    angles = {
        "ALA": [],
        "GLY": []
    }
    return bonds, {}, angles

  monkeypatch.setattr(residue_constants, "load_stereo_chemical_props", mock_load)



def test_parameterize_system_simple(mock_force_field):
  """Test parameterizing a simple dipeptide (ALA-GLY)."""
  residues = ["ALA", "GLY"]
  
  # Construct atom names
  atoms_ala = residue_constants.residue_atoms["ALA"]
  atoms_gly = residue_constants.residue_atoms["GLY"]
  atom_names = atoms_ala + atoms_gly
  
  params = jax_md_bridge.parameterize_system(
      mock_force_field, residues, atom_names
  )
  
  # Check counts
  n_atoms = len(atom_names)
  assert params["charges"].shape == (n_atoms,)
  assert params["sigmas"].shape == (n_atoms,)
  
  # Check bonds
  # ALA internal bonds + GLY internal bonds + 1 peptide bond
  # ALA has 5 atoms: N, CA, C, O, CB. 
  # Bonds: N-CA, CA-C, C-O, CA-CB (4 bonds)
  # GLY has 4 atoms: N, CA, C, O
  # Bonds: N-CA, CA-C, C-O (3 bonds)
  # Peptide: ALA.C - GLY.N (1 bond)
  # Total: 8 bonds
  
  # Note: residue_constants might define more/less depending on H.
  # residue_atoms only lists heavy atoms.
  # residue_bonds in residue_constants usually covers heavy atoms.
  
  # Let's count expected from residue_constants
  def count_bonds(res):
      bonds = 0
      atoms = set(residue_constants.residue_atoms[res])
      for b in residue_constants.load_stereo_chemical_props()[0].get(res, []):
          if b.atom1_name in atoms and b.atom2_name in atoms:
              bonds += 1
      return bonds

  expected_ala = count_bonds("ALA")
  expected_gly = count_bonds("GLY")
  expected_total = expected_ala + expected_gly + 1 # +1 for peptide
  
  assert len(params["bonds"]) == expected_total
  
  # Check backbone indices
  # Should be (2, 4)
  bb_indices = params["backbone_indices"]
  assert bb_indices.shape == (2, 4)
  
  # Verify indices for ALA (first residue)
  # ALA atoms: C, CA, CB, N, O (alphabetical in residue_atoms? No, PDB order)
  # residue_atoms["ALA"] = ["C", "CA", "CB", "N", "O"] -> Wait, check residue_constants.py
  # It says: "C", "CA", "CB", "N", "O" in the dict?
  # Let's check the file content we saw earlier.
  # residue_atoms = { "ALA": ["C", "CA", "CB", "N", "O"], ... }
  # Wait, standard PDB order is N, CA, C, O, CB.
  # The residue_atoms dict in residue_constants.py seems to be alphabetical or specific order?
  # Line 345: "ALA": ["C", "CA", "CB", "N", "O"]
  # This is NOT standard PDB order (N, CA, C, O).
  # If our bridge assumes `atom_names` matches `residue_atoms` order, then:
  # ALA indices: 0:C, 1:CA, 2:CB, 3:N, 4:O
  
  # N is at index 3
  # CA is at index 1
  # C is at index 0
  # O is at index 4
  
  assert bb_indices[0, 0] == 3 # N
  assert bb_indices[0, 1] == 1 # CA
  assert bb_indices[0, 2] == 0 # C
  assert bb_indices[0, 3] == 4 # O
