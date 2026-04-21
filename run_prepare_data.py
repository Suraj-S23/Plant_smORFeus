"""Batch driver for prepare_data.py across multiple plant species.

Auto-discovers FASTA/GFF pairs by matching biological names across two
directories, handling Galaxy-style filenames like
'Galaxy2-[AagrOXF_genome.fasta]'. Duplicate copies ending with ' (N)'
are automatically skipped.

Output layout consumed by convert_plant_npz_to_hdf5.py:
    <dataset_root>/
        <species_stem>/
            output_labels/
                <seqid>.npz

Usage:
    python run_prepare_data.py \\
        --dataset_root plant_processed_data_np \\
        --fasta_dir    plant_data/fasta \\
        --gff_dir      plant_data/gff

    # Dry run (print commands, run nothing)
    python run_prepare_data.py --dry_run \\
        --dataset_root plant_processed_data_np \\
        --fasta_dir    plant_data/fasta \\
        --gff_dir      plant_data/gff

    # Single species
    python run_prepare_data.py --species AagrOXF_genome \\
        --dataset_root plant_processed_data_np \\
        --fasta_dir    plant_data/fasta \\
        --gff_dir      plant_data/gff
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Script location (used to resolve prepare_data.py relative path)
_HERE = os.path.dirname(os.path.abspath(__file__))

PREPARE_DATA_SCRIPT = os.path.join(_HERE, "prepare_data.py")

# Recognised file extensions (case-insensitive)
FASTA_EXTENSIONS = {".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz"}
GFF_EXTENSIONS   = {".gff3", ".gff", ".gff3.gz", ".gff.gz"}

# Biological suffixes to strip so FASTA and GFF stems can match automatically.
# e.g. "ORYSAJA_Osativa_204_v7.0.softmasked" and
#      "ORYSAJA_Osativa_204_v7.0.gene_exons" both become "ORYSAJA_Osativa_204_v7.0"
_STRIP_SUFFIXES = [
    ".softmasked", ".gene_exons", ".gene_models", "_annotation", "_fasta",
]

# Manual overrides for pairs whose stems can never match automatically due to
# version mismatches or different assembly naming conventions.
# Format: canonical_output_name -> {"fasta": raw_stem, "gff": raw_stem}
# The raw stems are what get_stem() returns after suffix stripping.
MANUAL_PAIRS = {
    "BRADI_Bdistachyon_556":  {"fasta": "BRADI_Bdistachyon_556_v3.0",
                                "gff":   "BRADI_Bdistachyon_556_v3.2"},
    "POPTR_Ptrichocarpa_533": {"fasta": "POPTR_Ptrichocarpa_533_v4.0",
                                "gff":   "POPTR_Ptrichocarpa_533_v4.1"},
    "ZEAMA_Zmays_284":        {"fasta": "ZEAMA_Zmays_284_AGPv3",
                                "gff":   "ZEAMA_Zmays_284_Ensembl-18_2010-01-MaizeSequence"},
    "GLYMA_Gmax":             {"fasta": "GLYMA_Gmax_880_v6.0",
                                "gff":   "GLYMA_Gmax_Wm82.a6.v1"},
}


def _strip_biological_suffixes(stem: str) -> str:
    """Strip known biological suffixes so FASTA and GFF stems can be matched."""
    for suffix in _STRIP_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def get_stem(path: Path):
    """Extract biological stem from a filename; returns None for duplicate copies."""
    name = path.name

    if re.search(r'\s*\(\d+\)\s*$', name):
        return None

    m = re.search(r'\[(.+)\]', name)
    if m:
        inner = m.group(1)
        for ext in sorted(FASTA_EXTENSIONS | GFF_EXTENSIONS, key=len, reverse=True):
            if inner.lower().endswith(ext.lower()):
                return _strip_biological_suffixes(inner[: -len(ext)])
        return _strip_biological_suffixes(inner)

    for ext in sorted(FASTA_EXTENSIONS | GFF_EXTENSIONS, key=len, reverse=True):
        if name.lower().endswith(ext.lower()):
            return _strip_biological_suffixes(name[: -len(ext)])
    return _strip_biological_suffixes(path.stem)


def discover_pairs(fasta_dir: str, gff_dir: str) -> dict:
    """Match FASTA and GFF files by biological stem; warns about unmatched files."""
    fasta_dir = Path(fasta_dir)
    gff_dir   = Path(gff_dir)

    fastas = {}
    for f in fasta_dir.iterdir():
        stem = get_stem(f)
        if stem is not None:
            fastas[stem] = f

    gffs = {}
    for g in gff_dir.iterdir():
        stem = get_stem(g)
        if stem is not None:
            gffs[stem] = g

    paired   = {}
    only_fa  = []
    only_gff = []

    for stem, fasta_path in sorted(fastas.items()):
        if stem in gffs:
            paired[stem] = {"fasta": str(fasta_path), "gff": str(gffs[stem])}
        else:
            only_fa.append(stem)

    for stem in sorted(gffs):
        if stem not in fastas:
            only_gff.append(stem)

    for canonical, mapping in MANUAL_PAIRS.items():
        fasta_stem = mapping["fasta"]
        gff_stem   = mapping["gff"]
        if fasta_stem in fastas and gff_stem in gffs:
            paired[canonical] = {
                "fasta": str(fastas[fasta_stem]),
                "gff":   str(gffs[gff_stem]),
            }
            if fasta_stem in only_fa:
                only_fa.remove(fasta_stem)
            if gff_stem in only_gff:
                only_gff.remove(gff_stem)
            print(f"  [manual] Paired {canonical}: {fasta_stem} <-> {gff_stem}")
        else:
            missing = []
            if fasta_stem not in fastas:
                missing.append(f"FASTA '{fasta_stem}'")
            if gff_stem not in gffs:
                missing.append(f"GFF '{gff_stem}'")
            print(f"  [manual] WARNING: Cannot pair {canonical} "
                  f"— missing {', '.join(missing)}")

    if only_fa:
        print(f"  FASTA with no matching GFF  ({len(only_fa)}): {only_fa}")
    if only_gff:
        print(f"  GFF with no matching FASTA  ({len(only_gff)}): {only_gff}")

    return paired


def run_species(species: str, fasta: str, gff: str,
                dataset_root: str, dry_run: bool) -> bool:
    """Run prepare_data.py for one species; skips if already processed."""
    output_dir = os.path.join(dataset_root, species, "output_labels")

    if os.path.isdir(output_dir) and any(Path(output_dir).glob("*.npz")):
        print(f"  [{species}] already has .npz files - skipping.")
        return True

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, PREPARE_DATA_SCRIPT,
        "--fasta",      fasta,
        "--gff",        gff,
        "--output_dir", output_dir,
    ]

    print(f"\n-- {species} --")
    print(f"   FASTA : {fasta}")
    print(f"   GFF   : {gff}")
    print(f"   OUT   : {output_dir}")
    if dry_run:
        print(f"   CMD   : {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  FAILED: {species} (exit {result.returncode})")
        return False

    npz_count = len(list(Path(output_dir).glob("*.npz")))
    print(f"  OK: {species} -> {npz_count} .npz files in {output_dir}")
    return True


def main():
    """Entry point for batch label preparation."""
    parser = argparse.ArgumentParser(
        description="Batch prepare_data.py across multiple plant species."
    )
    parser.add_argument(
        "--dataset_root",
        default=os.path.join(_HERE, "plant_processed_data_np"),
        help="Root directory for output species sub-directories "
             "(default: %(default)s)",
    )
    parser.add_argument(
        "--fasta_dir",
        default=os.path.join(_HERE, "plant_data", "fasta"),
        help="Directory containing FASTA files (default: %(default)s)",
    )
    parser.add_argument(
        "--gff_dir",
        default=os.path.join(_HERE, "plant_data", "gff"),
        help="Directory containing GFF3 files (default: %(default)s)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without running anything",
    )
    parser.add_argument(
        "--species", default=None,
        help="Process a single species stem only",
    )
    args = parser.parse_args()

    if not os.path.isfile(PREPARE_DATA_SCRIPT):
        sys.exit(f"ERROR: prepare_data.py not found at {PREPARE_DATA_SCRIPT}")
    if not os.path.isdir(args.fasta_dir):
        sys.exit(f"ERROR: fasta_dir not found: {args.fasta_dir}")
    if not os.path.isdir(args.gff_dir):
        sys.exit(f"ERROR: gff_dir not found: {args.gff_dir}")

    print(f"Scanning {args.fasta_dir} and {args.gff_dir} for pairs...")
    pairs = discover_pairs(args.fasta_dir, args.gff_dir)
    if not pairs:
        sys.exit("ERROR: No matched FASTA/GFF pairs found.")

    if args.species:
        if args.species not in pairs:
            sys.exit(f"ERROR: '{args.species}' not in matched pairs.\n"
                     f"Available: {sorted(pairs.keys())}")
        pairs = {args.species: pairs[args.species]}

    print(f"\nDataset root  : {args.dataset_root}")
    print(f"Matched pairs : {sorted(pairs.keys())}")
    if args.dry_run:
        print("(dry run - no commands will be executed)")

    failed = []
    for species, paths in sorted(pairs.items()):
        ok = run_species(species, paths["fasta"], paths["gff"],
                         args.dataset_root, args.dry_run)
        if not ok:
            failed.append(species)

    print(f"\n{'-'*55}")
    if failed:
        print(f"FAILED species: {failed}")
        sys.exit(1)
    else:
        print(f"Done: {len(pairs)} species processed.")
        print(f"Next: python convert_plant_npz_to_hdf5.py "
              f"--dataset_root {args.dataset_root}")


if __name__ == "__main__":
    main()
