"""FASTA + GFF3 → per-chromosome NPZ (sequence string + (L, 7) uint8 labels).

Label order: protein_coding_gene, five_prime_UTR, three_prime_UTR, exon,
intron, splice_donor, splice_acceptor. The last three are derived from
mRNA exon pairs (2 bp donor/acceptor at intron boundaries).

Output is consumed by convert_plant_npz_to_hdf5.py."""

import argparse
import os
from Bio import SeqIO
import numpy as np
from tqdm import tqdm
import gffutils

LABELS = ["protein_coding_gene", "five_prime_UTR", "three_prime_UTR",
          "exon", "intron", "splice_donor", "splice_acceptor"]


def build_gff_db(gff_path):
    """Load or (re)build a gffutils FeatureDB cached at gff_path + ".db"."""
    db_path = gff_path + ".db"
    if os.path.exists(db_path):
        try:
            return gffutils.FeatureDB(db_path, keep_order=True)
        except Exception:
            os.remove(db_path)

    print(f"Building GFF database: {db_path}")
    return gffutils.create_db(
        gff_path,
        dbfn=db_path,
        force=True,
        keep_order=True,
        merge_strategy="create_unique",
        disable_infer_genes=True,
        disable_infer_transcripts=True,
    )


def get_intervals_from_gff(db, feature_type):
    """{seqid: [(start, end, strand)]}: 0-based half-open."""
    feats = {}
    for f in db.features_of_type(feature_type, order_by="seqid"):
        feats.setdefault(f.seqid, []).append((f.start - 1, f.end, f.strand))
    return feats


def compute_introns_and_splice_sites(exon_list):
    """Introns, donors, acceptors from a sorted exon list."""
    introns, donors, acceptors = [], [], []
    for i in range(len(exon_list) - 1):
        s1, e1, strand = exon_list[i]
        s2, e2, _ = exon_list[i + 1]
        introns.append((e1, s2, strand))
        if strand == "+":
            donors.append((e1, e1 + 2, strand))
            acceptors.append((s2 - 2, s2, strand))
        else:
            donors.append((s2 - 2, s2, strand))
            acceptors.append((e1, e1 + 2, strand))
    return introns, donors, acceptors


def label_chromosome(record, feature_dict):
    """(L, 7) uint8 label array for a BioPython SeqRecord."""
    L = len(record.seq)
    y = np.zeros((L, len(LABELS)), dtype=np.uint8)
    for i, label in enumerate(LABELS):
        for s, e, _ in feature_dict.get(label, []):
            s2 = max(0, min(L, s))
            e2 = max(0, min(L, e))
            if e2 > s2:
                y[s2:e2, i] = 1
    return y


def process_genome(fasta_path, gff_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    db = build_gff_db(gff_path)

    features = {label: {} for label in LABELS}
    features["protein_coding_gene"] = get_intervals_from_gff(db, "gene")

    for gff_type, name in [("five_prime_UTR", "five_prime_UTR"),
                            ("three_prime_UTR", "three_prime_UTR"),
                            ("exon", "exon")]:
        features[name] = get_intervals_from_gff(db, gff_type)

    introns, donors, acceptors = {}, {}, {}
    for tr in db.features_of_type("mRNA", order_by="seqid"):
        exons = [f for f in db.children(tr, featuretype="exon", order_by="start")]
        if len(exons) < 2:
            continue
        exon_list = [(e.start - 1, e.end, e.strand) for e in exons]
        I, D, A = compute_introns_and_splice_sites(exon_list)
        for lst, store in [(I, introns), (D, donors), (A, acceptors)]:
            store.setdefault(tr.seqid, []).extend(lst)

    features["intron"] = introns
    features["splice_donor"] = donors
    features["splice_acceptor"] = acceptors

    print(f"Processing {fasta_path}...")
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Labeling chromosomes"):
        label_data = {k: v.get(record.id, []) for k, v in features.items()}
        y = label_chromosome(record, label_data)

        np.savez_compressed(
            os.path.join(output_dir, f"{record.id}.npz"),
            sequence=str(record.seq),
            labels=y,
        )
    print(f"Saved .npz files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare per-chromosome NPZ label files from FASTA + GFF3."
    )
    parser.add_argument("--fasta", required=True,
                        help="Input genome FASTA file")
    parser.add_argument("--gff", required=True,
                        help="Input GFF3 annotation file")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write per-chromosome NPZ files")
    args = parser.parse_args()

    process_genome(args.fasta, args.gff, args.output_dir)


if __name__ == "__main__":
    main()
