"""Converts plant genome .npz files (output of prepare_data.py) into the
sequential HDF5 format expected by GenomicHDF5Dataset and the
hidden-state caching pipeline.

Input layout (produced by prepare_data.py):
  <dataset_root>/
    <species_name>/
      output_labels/          ← searched first; falls back to species root
        Chr1.npz              ← sequence (str) + labels (L, 7) uint8
        Chr2.npz
        ...

Output HDF5 layout (mirrors existing smORFeus convention):
  hdf5_file[seq_id][chunk_id]["sequence"]      S1 byte array  (CHUNK_SIZE,)
  hdf5_file[seq_id][chunk_id]["frame_targets"] uint8 array    (CHUNK_SIZE, 7)
  hdf5_file[seq_id][chunk_id]["target"]        uint8 array    (CHUNK_SIZE, 2)  ← back-compat stub

  seq_id   = "{species}_{chrom}"          e.g. "arabidopsis_Chr1"
  chunk_id = "{seq_id}_chunk_{N:05d}"     e.g. "arabidopsis_Chr1_chunk_00003"

Chunk naming satisfies the regex  r".+_chunk_\\d{5}"  used by new_dataset.py.

Labels (7 channels, matching prepare_data.py LABELS list):
  0  protein_coding_gene
  1  five_prime_UTR
  2  three_prime_UTR
  3  exon
  4  intron
  5  splice_donor
  6  splice_acceptor
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

CHUNK_SIZE   = 10_000           # must match smORFeus chunk_size
MIN_CHUNK_BP = 5_000            # discard trailing remnants shorter than this
NUM_LABELS   = 7
LABEL_NAMES  = [
    "protein_coding_gene",
    "five_prime_UTR",
    "three_prime_UTR",
    "exon",
    "intron",
    "splice_donor",
    "splice_acceptor",
]
SEQ_DTYPE    = "S1"             # one byte per nucleotide – matches existing pipeline
COMPRESSION  = "lzf"           # same as existing pipeline


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _sanitise_name(name: str) -> str:
    """Replace characters illegal in HDF5 group names."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", name)


def _load_npz(path: str) -> tuple[str, np.ndarray]:
    """Load a .npz file; returns (sequence_str, labels_array(L,7))."""
    data = np.load(path, allow_pickle=True)

    # sequence field
    seq_raw = data["sequence"]
    if isinstance(seq_raw, np.ndarray) and seq_raw.ndim == 0:
        sequence = str(seq_raw.item()).upper()
    else:
        sequence = str(seq_raw).upper()

    # labels field
    labels = data["labels"]
    if labels.ndim != 2 or labels.shape[1] != NUM_LABELS:
        raise ValueError(
            f"{path}: expected labels shape (L, {NUM_LABELS}), got {labels.shape}"
        )
    if len(sequence) != labels.shape[0]:
        raise ValueError(
            f"{path}: sequence length {len(sequence)} != labels rows {labels.shape[0]}"
        )

    return sequence, labels.astype(np.uint8)


def _pad_to_chunk(seq_slice: str, label_slice: np.ndarray) -> tuple[str, np.ndarray]:
    """Right-pad a partial final chunk with 'N' nucleotides and zero labels."""
    deficit = CHUNK_SIZE - len(seq_slice)
    if deficit == 0:
        return seq_slice, label_slice
    seq_padded    = seq_slice + "N" * deficit
    label_padded  = np.zeros((CHUNK_SIZE, NUM_LABELS), dtype=np.uint8)
    label_padded[: len(label_slice)] = label_slice
    return seq_padded, label_padded


def _write_chunk(
    seq_group:    h5py.Group,
    chunk_id:     str,
    sequence:     str,
    frame_targets: np.ndarray,
    prev_chunk_id: str,
) -> None:
    g = seq_group.create_group(chunk_id)

    # sequence  ── stored as array of single-byte strings (existing pipeline convention)
    g.create_dataset(
        "sequence",
        data=np.array(list(sequence), dtype=SEQ_DTYPE),
        compression=COMPRESSION,
        shuffle=True,
    )

    # frame_targets  ── 7-channel plant labels
    g.create_dataset(
        "frame_targets",
        data=frame_targets.astype(np.uint8),
        compression=COMPRESSION,
        shuffle=True,
    )

    # target  ── 2-channel back-compat stub expected by legacy code paths
    stub = np.zeros((CHUNK_SIZE, 2), dtype=np.uint8)
    ds   = g.create_dataset(
        "target",
        data=stub,
        compression=COMPRESSION,
        shuffle=True,
    )
    ds.attrs["channel_names"] = ["orf_presence", "overlap_count"]

    # chunk-level metadata
    g.attrs["previous_chunk_id"] = prev_chunk_id
    g.attrs["num_labels"]        = NUM_LABELS
    g.attrs["label_names"]       = LABEL_NAMES


# ── Per-chromosome converter ──────────────────────────────────────────────────

def convert_chromosome(
    npz_path:   str,
    hdf5_file:  h5py.File,
    species_id: str,
    chrom_id:   str,
    verbose:    bool = False,
) -> dict:
    sequence, labels = _load_npz(npz_path)
    chrom_len        = len(sequence)

    seq_id    = _sanitise_name(f"{species_id}_{chrom_id}")
    seq_group = hdf5_file.create_group(seq_id)

    # store top-level metadata on the sequence group
    seq_group.attrs["species"]    = species_id
    seq_group.attrs["chrom"]      = chrom_id
    seq_group.attrs["chrom_len"]  = chrom_len
    seq_group.attrs["chunk_size"] = CHUNK_SIZE
    seq_group.attrs["num_labels"] = NUM_LABELS
    seq_group.attrs["label_names"] = LABEL_NAMES

    n_full   = chrom_len // CHUNK_SIZE
    remainder = chrom_len %  CHUNK_SIZE
    n_chunks_written = 0
    prev_chunk_id    = ""

    for i in range(n_full):
        start   = i * CHUNK_SIZE
        end     = start + CHUNK_SIZE
        seq_sl  = sequence[start:end]
        lbl_sl  = labels[start:end]

        chunk_id = f"{seq_id}_chunk_{i:05d}"
        _write_chunk(seq_group, chunk_id, seq_sl, lbl_sl, prev_chunk_id)
        prev_chunk_id = chunk_id
        n_chunks_written += 1

    # partial last chunk
    partial_written = False
    if remainder >= MIN_CHUNK_BP:
        start   = n_full * CHUNK_SIZE
        seq_sl  = sequence[start:]
        lbl_sl  = labels[start:]
        seq_sl, lbl_sl = _pad_to_chunk(seq_sl, lbl_sl)

        chunk_id = f"{seq_id}_chunk_{n_full:05d}"
        _write_chunk(seq_group, chunk_id, seq_sl, lbl_sl, prev_chunk_id)
        n_chunks_written += 1
        partial_written = True
    elif remainder > 0 and verbose:
        print(
            f"  [WARN] {chrom_id}: discarding {remainder} bp tail "
            f"(< MIN_CHUNK_BP={MIN_CHUNK_BP})"
        )

    return {
        "seq_id":          seq_id,
        "chrom_len":       chrom_len,
        "n_full_chunks":   n_full,
        "partial_written": partial_written,
        "n_chunks":        n_chunks_written,
    }


# ── Top-level driver ──────────────────────────────────────────────────────────

def convert_dataset(
    dataset_root: str,
    output_hdf5:  str,
    chunk_size:   int  = CHUNK_SIZE,
    verbose:      bool = False,
) -> None:
    if chunk_size != CHUNK_SIZE:
        raise NotImplementedError(
            f"chunk_size={chunk_size} requested but CHUNK_SIZE constant is "
            f"{CHUNK_SIZE}.  Edit the constant at the top of this file."
        )

    species_dirs = sorted(
        d for d in glob.glob(os.path.join(dataset_root, "*"))
        if os.path.isdir(d)
    )
    if not species_dirs:
        print(f"[ERROR] No species directories found under {dataset_root}")
        sys.exit(1)

    print(f"Found {len(species_dirs)} species: "
          f"{[os.path.basename(d) for d in species_dirs]}")

    total_stats = []

    with h5py.File(output_hdf5, "w") as hdf5_file:
        # store global metadata
        hdf5_file.attrs["label_names"] = LABEL_NAMES
        hdf5_file.attrs["num_labels"]  = NUM_LABELS
        hdf5_file.attrs["chunk_size"]  = CHUNK_SIZE
        hdf5_file.attrs["source"]      = "convert_plant_npz_to_hdf5"

        for species_dir in species_dirs:
            species_id = _sanitise_name(os.path.basename(species_dir))

            # prepare_data.py writes to output_labels/ subdirectory
            labels_subdir = os.path.join(species_dir, "output_labels")
            search_dir    = labels_subdir if os.path.isdir(labels_subdir) else species_dir

            npz_files = sorted(glob.glob(os.path.join(search_dir, "*.npz")))
            if not npz_files:
                print(f"  [WARN] No .npz files found for {species_id}, skipping.")
                continue

            print(f"\n── {species_id}  ({len(npz_files)} chromosomes) ──")

            for npz_path in tqdm(npz_files, desc=f"  {species_id}", leave=False):
                chrom_id = Path(npz_path).stem          # e.g. "Chr1"
                try:
                    stats = convert_chromosome(
                        npz_path, hdf5_file, species_id, chrom_id, verbose
                    )
                    total_stats.append(stats)
                    if verbose:
                        print(
                            f"    {chrom_id}: {stats['chrom_len']:,} bp → "
                            f"{stats['n_chunks']} chunks  (seq_id={stats['seq_id']})"
                        )
                except Exception as exc:
                    print(f"  [ERROR] {npz_path}: {exc}")
                    raise

    # ── summary ──────────────────────────────────────────────────────────────
    total_chroms   = len(total_stats)
    total_chunks   = sum(s["n_chunks"]   for s in total_stats)
    total_bp       = sum(s["chrom_len"]  for s in total_stats)
    total_partials = sum(s["partial_written"] for s in total_stats)

    print(f"\n{'─'*60}")
    print(f"Conversion complete -> {output_hdf5}")
    print(f"   Chromosomes  : {total_chroms}")
    print(f"   Chunks total : {total_chunks:,}  ({total_partials} padded partials)")
    print(f"   Genome bp    : {total_bp:,}")
    hdf5_size = os.path.getsize(output_hdf5) / (1024 ** 2)
    print(f"   HDF5 size    : {hdf5_size:.1f} MB")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert plant .npz annotation files to smORFeus HDF5 format"
    )
    parser.add_argument(
        "--dataset_root", required=True,
        help="Root directory containing one sub-folder per species "
             "(each with an output_labels/ sub-folder of .npz files)"
    )
    parser.add_argument(
        "--output_hdf5", required=True,
        help="Path for the output HDF5 file"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-chromosome details"
    )
    args = parser.parse_args()

    convert_dataset(
        dataset_root=args.dataset_root,
        output_hdf5=args.output_hdf5,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()