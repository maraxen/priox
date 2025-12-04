"""Data loading utilities for protein structure data."""

import dataclasses
import logging
import pathlib
from collections.abc import Iterator, Sequence
from typing import IO, Any, SupportsIndex

import grain.python as grain
import numpy as np

from priox.core.containers import ProteinTuple
from priox.io.parsing.foldcomp import (
  FoldCompDatabase,
)
from priox.ops import prefetch as prefetch_autotune
from priox.ops.transforms import BatchAndCollate

from .processing import frame_iterator_from_inputs

logger = logging.getLogger(__name__)


def _is_frame_valid(frame: ProteinTuple) -> tuple[bool, str]:
  """Check if a protein frame is valid."""
  if len(frame.aatype) == 0:
    return False, "Empty structure"
  if frame.coordinates.shape[0] != len(frame.aatype):
    return False, "Shape mismatch between coordinates and aatype"
  if np.isnan(frame.coordinates).any():
    return False, "NaN values in coordinates"
  return True, ""


class ProteinDataSource(grain.RandomAccessDataSource):
  """Implements a Grain DataSource for streaming protein structure frames."""

  def __init__(
    self,
    inputs: Sequence[str | pathlib.Path | IO[str]],
    parse_kwargs: dict[str, Any] | None = None,
    foldcomp_database: FoldCompDatabase | None = None,
  ) -> None:
    """Initialize the data source by preparing the frame iterator.

    Args:
        inputs: A sequence of input sources (file paths, file-like objects, etc.).
        parse_kwargs: Optional keyword arguments to pass to the parsing function.
        foldcomp_database: An optional FoldCompDatabase for resolving FoldComp IDs.

    """
    super().__init__()
    self.inputs = inputs
    self.parse_kwargs = parse_kwargs or {}
    self.foldcomp_database = foldcomp_database
    self.frames = []
    self.skipped_frames = []
    for frame in frame_iterator_from_inputs(
      self.inputs,
      self.parse_kwargs,
      self.foldcomp_database,
    ):
      is_valid, reason = _is_frame_valid(frame)
      if is_valid:
        self.frames.append(frame)
      else:
        logger.warning("Skipping invalid frame from %s: %s", frame.source, reason)
        self.skipped_frames.append({"source": frame.source, "reason": reason})

    self._length = len(self.frames)

  def __len__(self) -> int:
    """Return the total number of frames available."""
    return self._length

  def __getitem__(self, index: SupportsIndex) -> ProteinTuple:  # type: ignore[override]
    """Return the ProteinTuple at the specified index.

    Args:
        index (SupportsIndex): The index of the item to retrieve.

    Returns:
        ProteinTuple: The protein structure frame at the specified index.

    Raises:
        IndexError: If the index is out of range.

    """
    idx = int(index)
    if not 0 <= idx < len(self):
      msg = f"Attempted to access index {idx}, but valid indices are 0 to {len(self) - 1}."
      raise IndexError(msg)
    return self.frames[idx]




def create_protein_dataset(
  inputs: Sequence[str | pathlib.Path | IO[str]],
  batch_size: int,
  *,
  parse_kwargs: dict[str, Any] | None = None,
  foldcomp_database: FoldCompDatabase | None = None,
  ram_budget_mb: int = 1024,
  max_workers: int | None = None,
  max_buffer_size: int | None = None,
  collate_kwargs: dict[str, Any] | None = None,
  shuffle: bool = False,
  seed: int = 0,
) -> grain.DataLoader:
  """Create a Grain DataLoader for protein structure data.

  Args:
      inputs: Sequence of input files/paths.
      batch_size: Batch size.
      parse_kwargs: Arguments for parsing (e.g. chain_id).
      foldcomp_database: FoldComp database.
      ram_budget_mb: RAM budget in MB for prefetching.
      max_workers: Max workers for prefetching.
      max_buffer_size: Max buffer size for prefetching.
      collate_kwargs: Arguments for pad_and_collate_proteins (e.g. use_electrostatics).
      shuffle: Whether to shuffle the data.
      seed: Random seed for shuffling.

  Returns:
      grain.DataLoader: The data loader.

  """
  ds = ProteinDataSource(inputs, parse_kwargs, foldcomp_database)

  sampler = grain.IndexSampler(
    len(ds),
    shuffle=shuffle,
    seed=seed,
    shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
  )

  operations = []
  operations.append(
    BatchAndCollate(batch_size=batch_size, collate_kwargs=collate_kwargs or {})
  )

  config = prefetch_autotune.pick_performance_config(
    ds,
    ram_budget_mb=ram_budget_mb,
    max_workers=max_workers,
    max_buffer_size=max_buffer_size,
  )

  return grain.DataLoader(
    data_source=ds,
    sampler=sampler,
    operations=operations,
    worker_count=config.multiprocessing_options.num_workers,
    worker_buffer_size=config.read_options.prefetch_buffer_size,
  )
