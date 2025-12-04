"""Unified dispatch for parsing protein structures."""

from __future__ import annotations

import pathlib
from typing import IO, TYPE_CHECKING, Any

from priox.io.parsing import biotite, mdtraj, pqr, utils
from priox.io.parsing.structures import ProcessedStructure

if TYPE_CHECKING:
  from collections.abc import Sequence

  from priox.core.containers import ProteinStream


def load_structure(  # noqa: C901, PLR0912, PLR0915
  file_path: str | pathlib.Path | IO[str],
  file_format: str | None = None,
  chain_id: str | Sequence[str] | None = None,
  **kwargs: Any,  # noqa: ANN401
) -> ProteinStream:
  """Load a protein structure from a file.

  Args:
      file_path: Path to the file or file-like object.
      file_format: Format of the file (e.g., "pdb", "cif", "pqr", "foldcomp").
          If None, inferred from extension.
      chain_id: Chain ID(s) to load.
      **kwargs: Additional arguments passed to specific parsers.

  Returns:
      ProcessedStructure or ProteinStream (for FoldComp/Trajectories).

  """
  if isinstance(file_path, str):
    path = pathlib.Path(file_path)
  elif isinstance(file_path, pathlib.Path):
    path = file_path
  else:
    path = None

  if file_format is None:
    if path is not None:
      suffix = path.suffix.lower()
      if suffix == ".pdb":
        file_format = "pdb"
      elif suffix in (".cif", ".mmcif"):
        file_format = "cif"
      elif suffix == ".pqr":
        file_format = "pqr"
      elif suffix in (".fcz", ".foldcomp"):
         file_format = "foldcomp"
      elif suffix in (".dcd", ".xtc", ".h5", ".hdf5"):
         file_format = "mdtraj"
    else:
      # Default to pdb for file-like objects (e.g. StringIO) if format not specified
      file_format = "pdb"

  # Extract arguments for processed_structure_to_protein_tuples
  extract_dihedrals = kwargs.pop("extract_dihedrals", False)
  populate_physics = kwargs.pop("populate_physics", False)
  force_field_name = kwargs.pop("force_field_name", "ff14SB")

  if file_format == "pqr":
    processed = pqr.parse_pqr_to_processed_structure(file_path, chain_id=chain_id)
    return utils.processed_structure_to_protein_tuples(
        processed,
        source_name=str(path or "pqr"),
        extract_dihedrals=extract_dihedrals,
        populate_physics=populate_physics,
        force_field_name=force_field_name,
    )

  if file_format in ("pdb", "cif", "mmcif"):
      # Biotite parser
      try:
          atom_array = biotite.load_structure_with_hydride(file_path, chain_id=chain_id, **kwargs)
      except Exception as e:
          msg = f"Failed to parse structure from source: {file_path}. {e}"
          raise RuntimeError(msg) from e

      # Create ProcessedStructure
      # We need to derive r_indices and chain_ids
      import numpy as np  # noqa: PLC0415

      r_indices = atom_array.res_id

      # Map chain_id strings to integers
      # We need a consistent mapping.
      # If chain_id attribute exists (it should for AtomArray from PDB/CIF)
      if hasattr(atom_array, "chain_id"):
          unique_chains = sorted(set(atom_array.chain_id))
          chain_map = {cid: i for i, cid in enumerate(unique_chains)}
          chain_ids = np.array([chain_map[cid] for cid in atom_array.chain_id], dtype=np.int32)
      else:
          # If no chain_id, assume all 0
          chain_ids = np.zeros(atom_array.array_length(), dtype=np.int32)

      processed = ProcessedStructure(
          atom_array=atom_array,
          r_indices=r_indices,
          chain_ids=chain_ids,
      )
      return utils.processed_structure_to_protein_tuples(
          processed,
          source_name=str(path or file_format),
          extract_dihedrals=extract_dihedrals,
          populate_physics=populate_physics,
          force_field_name=force_field_name,
      )

  if file_format == "foldcomp":
      # FoldComp usually takes a database name or list of IDs.
      # If file_path is a directory or name, treat as database?
      # For now, let's assume file_path is the database name if format is foldcomp
      # But get_protein_structures takes protein_ids.
      # This might need a different API for FoldComp.
      # Let's leave it for now or raise NotImplementedError if not clear.
      msg = "FoldComp loading via dispatch not yet fully unified."
      raise NotImplementedError(msg)

  if file_format == "mdtraj":
      # MDTraj parser returns Iterator[ProcessedStructure]
      iterator = mdtraj.parse_mdtraj_to_processed_structure(file_path, chain_id=chain_id, **kwargs)
      # We need to yield ProteinTuples from each ProcessedStructure
      def _generator() -> ProteinStream:
          try:
              for processed in iterator:
                  yield from utils.processed_structure_to_protein_tuples(
                      processed,
                      source_name=str(path or "mdtraj"),
                      extract_dihedrals=extract_dihedrals,
                      populate_physics=populate_physics,
                      force_field_name=force_field_name,
                  )
          except RuntimeError:
              # If parsing fails (e.g. malformed file), we yield nothing as per tests
              pass
      return _generator()

  msg = (
    f"Failed to parse structure from source: {file_path}. "
      f"Unsupported file format: {file_format}"
  )
  raise RuntimeError(
      msg,
  )

parse_input = load_structure

def _determine_h5_structure(file_path: str | pathlib.Path) -> str:
    """Determine the structure of an HDF5 file (mdcath or mdtraj)."""
    import h5py  # noqa: PLC0415
    with h5py.File(file_path, "r") as f:
        if "layout" in f.attrs and f.attrs["layout"] == "mdcath":
             return "mdcath"

        # MDTraj HDF5 files typically have 'topology', 'coordinates', 'cell_lengths', 'cell_angles'
        if "topology" in f or "coordinates" in f:
            return "mdtraj"

    # Default fallback as per test_determine_h5_structure_unknown
    return "mdcath"
