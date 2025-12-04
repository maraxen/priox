
"""Extended tests for priox.io.parsing.pqr to increase coverage."""

import unittest
from unittest import mock
import numpy as np
import pytest
from io import StringIO
from priox.io.parsing import pqr

class TestPQRExtended(unittest.TestCase):
    
    def test_parse_atom_line_invalid(self):
        """Test parsing invalid lines."""
        # Too few fields
        line = "ATOM  1  N  ALA"
        self.assertIsNone(pqr._parse_atom_line(line))
        
        # Invalid float
        line = "ATOM      1  N   ALA A  50      X       20.000  30.000  -0.500   1.850"
        self.assertIsNone(pqr._parse_atom_line(line))

    def test_parse_atom_line_merged_fields(self):
        """Test parsing lines with merged fields (record name + serial)."""
        # "ATOM" + "123456" -> "ATOM123456" (len > 6)
        # Standard: ATOM  12345 ...
        # If merged: ATOM12345 ...
        
        # pqr.py logic: if len(fields[0]) > 6:
        # fields[0] is "ATOM12345"
        # fields[1] is atom_name
        # fields[2] is res_name
        # ...
        
        line = "ATOM12345 N   ALA A  50      10.000  20.000  30.000  -0.500   1.850"
        parsed = pqr._parse_atom_line(line)
        
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["atom_name"], "N")
        self.assertEqual(parsed["res_name"], "ALA")
        self.assertEqual(parsed["coord"], [10.0, 20.0, 30.0])

    def test_parse_atom_line_water(self):
        """Test water skipping."""
        line = "ATOM      1  O   HOH A  50      10.000  20.000  30.000  -0.500   1.850"
        self.assertIsNone(pqr._parse_atom_line(line))
        
        line = "ATOM      1  O   WAT A  50      10.000  20.000  30.000  -0.500   1.850"
        self.assertIsNone(pqr._parse_atom_line(line))

    def test_parse_pqr_file_object(self):
        """Test parsing from file-like object."""
        content = """\
ATOM      1  N   ALA A  50      10.000  20.000  30.000  -0.500   1.850
ATOM      2  CA  ALA A  50      11.000  21.000  31.000   0.100   1.700
"""
        f = StringIO(content)
        result = pqr.parse_pqr_to_processed_structure(f)
        
        self.assertEqual(result.atom_array.array_length(), 2)
        self.assertEqual(result.atom_array.res_name[0], "ALA")

    def test_parse_pqr_chain_filtering_set(self):
        """Test chain filtering with a list/set."""
        content = """\
ATOM      1  N   ALA A  50      10.000  20.000  30.000  -0.500   1.850
ATOM      2  N   ALA B  50      11.000  21.000  31.000   0.100   1.700
ATOM      3  N   ALA C  50      12.000  22.000  32.000   0.100   1.700
"""
        f = StringIO(content)
        # Filter for A and C
        result = pqr.parse_pqr_to_processed_structure(f, chain_id=["A", "C"])
        
        self.assertEqual(result.atom_array.array_length(), 2)
        self.assertEqual(result.atom_array.chain_id[0], "A")
        self.assertEqual(result.atom_array.chain_id[1], "C")

    def test_parse_pqr_epsilon_lookup(self):
        """Test epsilon lookup based on element."""
        # N -> should get N epsilon
        line = "ATOM      1  N   ALA A  50      10.000  20.000  30.000  -0.500   1.850"
        parsed = pqr._parse_atom_line(line)
        self.assertIsNotNone(parsed)
        # Check against residues.van_der_waals_epsilon defaults
        # N is usually present.
        self.assertTrue(parsed["epsilon"] > 0)
        
        # Unknown element -> default 0.15
        line = "ATOM      1  X   ALA A  50      10.000  20.000  30.000  -0.500   1.850"
        parsed = pqr._parse_atom_line(line)
        self.assertEqual(parsed["epsilon"], 0.15)

if __name__ == "__main__":
    unittest.main()
