"""HDF5 dataset loader for plant genome annotation.

Provides GenomicHDF5Dataset, a memory-efficient PyTorch Dataset that opens a
single HDF5 file handle per DataLoader worker and supports O(1) lookup of
adjacent chunks for hidden-state caching.

Compatible with both the original 3-label smORFeus HDF5 files and the new
7-label plant annotation HDF5 files produced by convert_plant_npz_to_hdf5.py.

Classes exposed:
    GenomicHDF5Dataset -- Per-chunk genomic sequence and label loader.
"""

import h5py
import torch
import numpy as np
import re
from collections import defaultdict
from torch.utils.data import Dataset


class GenomicHDF5Dataset(Dataset):
    """Memory-efficient HDF5 chunk loader with O(1) adjacency index for hidden-state caching."""

    def __init__(self, hdf5_path: str, overlap: int = 500, length_bins=None):
        self.hdf5_path = hdf5_path
        self.overlap   = overlap
        self._h5       = None
        self._worker_id = None

        self.length_bins = length_bins or [(0, float("inf"), 1.0)]
        self.use_length_weighting = len(self.length_bins) > 1

        self.all_chunks   = []
        self.chunk_index  = {}
        self.previous_chunk_map = {}
        self.sequence_map = defaultdict(list)

        with h5py.File(hdf5_path, "r") as f:
            for seq_id in f:
                chunks = [
                    c for c in f[seq_id].keys()
                    if re.fullmatch(r".+_chunk_\d{5}", c)
                ]
                chunks.sort(key=lambda x: int(re.search(r"chunk_(\d{5})", x).group(1)))

                prev_chunk = ""
                for chunk in chunks:
                    gidx = len(self.all_chunks)
                    self.all_chunks.append((seq_id, chunk))
                    self.chunk_index[(seq_id, chunk)] = gidx
                    self.previous_chunk_map[(seq_id, chunk)] = prev_chunk
                    self.sequence_map[seq_id].append((gidx, chunk))
                    prev_chunk = chunk

        self.sequences = self.sequence_map
        self.chunk_to_seq = {
            chunk_id: (seq_id, gidx)
            for gidx, (seq_id, chunk_id) in enumerate(self.all_chunks)
        }

    def __len__(self):
        return len(self.all_chunks)

    def __getitem__(self, idx):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")

        seq_id, chunk_id   = self.all_chunks[idx]
        prev_chunk_id      = self.previous_chunk_map.get((seq_id, chunk_id), "")
        group              = self._h5[seq_id][chunk_id]

        # Decode sequence bytes, handles both S1 arrays and raw byte objects
        seq_data = group["sequence"][:]
        if seq_data.dtype.kind == "S":
            cur_seq = b"".join(seq_data).decode("utf-8")
        elif isinstance(seq_data, bytes):
            cur_seq = seq_data.decode("utf-8")
        elif hasattr(seq_data, "tobytes"):
            cur_seq = seq_data.tobytes().decode("utf-8")
        else:
            cur_seq = "".join(
                str(x, "utf-8") if isinstance(x, bytes) else str(x) for x in seq_data
            )

        if len(cur_seq) != 10_000:
            print(f"WARNING: {chunk_id} has length {len(cur_seq)}, expected 10000")

        if self.overlap > 0:
            prev_seq = self._get_adjacent_chunk(seq_id, chunk_id, -1)
            next_seq = self._get_adjacent_chunk(seq_id, chunk_id, +1)
            full_seq = (
                (prev_seq[-self.overlap:] if prev_seq else "")
                + cur_seq
                + (next_seq[: self.overlap] if next_seq else "")
            )
        else:
            full_seq = cur_seq

        if "frame_targets" in group:
            frame_targets  = group["frame_targets"][:]          # (L, N)
            length_weights = self._compute_length_weights(frame_targets)

            overlap_count = None
            if "target" in group and group["target"].shape[1] >= 2:
                overlap_count = group["target"][:, 1]

            return {
                "sequence":          full_seq,
                "target":            frame_targets,
                "length_weights":    length_weights,
                "chunk_id":          chunk_id,
                "overlap_count":     overlap_count,
                "sequence_id":       seq_id,
                "previous_chunk_id": prev_chunk_id,
            }

        # Legacy 2-channel fallback for original smORFeus HDF5 files
        target        = group["target"][:]
        orf_presence  = target[:, 0]
        return {
            "sequence":          full_seq,
            "target":            orf_presence.reshape(-1, 1),
            "chunk_id":          chunk_id,
            "sequence_id":       seq_id,
            "previous_chunk_id": prev_chunk_id,
            "orf_indices": {
                "non_orf": np.where(target[:, 1] == 0)[0],
            },
        }

    def _get_adjacent_chunk(self, seq_id: str, cur_chunk_id: str, direction: int):
        try:
            cur_num = int(cur_chunk_id.split("_chunk_")[-1])
        except (ValueError, IndexError):
            return None
        adj_num = cur_num + direction
        if adj_num < 0:
            return None
        adj_id  = f"{seq_id}_chunk_{adj_num:05d}"
        adj_idx = self.chunk_index.get((seq_id, adj_id))
        if adj_idx is None:
            return None
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        seq_bytes = self._h5[seq_id][adj_id]["sequence"][:]
        if seq_bytes.dtype.kind == "S":
            return b"".join(seq_bytes).decode("utf-8")
        return seq_bytes.tobytes().decode("utf-8")

    def _get_weight_for_length(self, length: int) -> float:
        for min_len, max_len, weight in self.length_bins:
            if min_len <= length < max_len:
                return weight
        return 1.0

    def _extract_orf_runs(self, sequence: np.ndarray) -> list:
        runs   = []
        start  = None
        for i, val in enumerate(sequence):
            if val == 1 and start is None:
                start = i
            elif val != 1 and start is not None:
                runs.append((start, i, i - start))
                start = None
        if start is not None:
            runs.append((start, len(sequence), len(sequence) - start))
        return runs

    def _compute_length_weights(self, frame_targets: np.ndarray) -> np.ndarray:
        if not self.use_length_weighting:
            return np.ones_like(frame_targets, dtype=np.float32)

        L, num_labels = frame_targets.shape
        weights = np.ones((L, num_labels), dtype=np.float32)

        for label_idx in range(num_labels):
            label_col = frame_targets[:, label_idx]
            runs = self._extract_orf_runs(label_col)
            for start, end, length in runs:
                weights[start:end, label_idx] = self._get_weight_for_length(length)

        return weights