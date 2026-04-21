"""Inference script for the smORFeus plant annotation model.

Given a trained PlantTrainer checkpoint and a FASTA file, this module
predicts per-nucleotide genomic annotations for each sequence and writes
the results to GFF3 format. When a matching GFF3 annotation is also
provided, per-label evaluation metrics are computed and saved.

Two modes:
    1. FASTA only: produces a GFF3 file of predictions, no metrics.
    2. FASTA + GFF: additionally computes per-label metrics and saves
                     evaluation figures (ROC, PR, overlay).

Usage:
    # Inference only
    python plant_inference.py \\
        --checkpoint checkpoints_runE/best.ckpt \\
        --fasta     genome.fasta \\
        --outdir    inference_out/

    # With evaluation
    python plant_inference.py \\
        --checkpoint checkpoints_runE/best.ckpt \\
        --fasta     genome.fasta \\
        --gff       annotation.gff3 \\
        --outdir    inference_out/
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from Bio import SeqIO

# Label definitions (must match plant_config.py)
LABEL_NAMES = [
    "protein_coding_gene",
    "five_prime_UTR",
    "three_prime_UTR",
    "exon",
    "intron",
    "splice_donor",
    "splice_acceptor",
]

TOKEN_MAP  = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
PAD_TOKEN  = 5
CHUNK_SIZE = 10_000

# GFF3 feature types that map to each label
GFF_FEATURE_MAP = {
    "protein_coding_gene": ["gene"],
    "five_prime_UTR":      ["five_prime_UTR", "five_prime_utr"],
    "three_prime_UTR":     ["three_prime_UTR", "three_prime_utr"],
    "exon":                ["exon"],
    # intron, splice_donor, splice_acceptor are derived from exon pairs
}


def tokenise(seq: str) -> np.ndarray:
    """Convert a nucleotide string to int64 token array; unknowns map to PAD_TOKEN."""
    return np.array([TOKEN_MAP.get(c, PAD_TOKEN) for c in seq.upper()],
                    dtype=np.int64)


def load_model(checkpoint: str, device: torch.device):
    from plant_train import PlantTrainer

    print(f"[INFO] Loading checkpoint: {checkpoint}")
    model = PlantTrainer.load_from_checkpoint(
        checkpoint, map_location=device
    )
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded: "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def run_inference_on_sequence(
    model,
    seq: str,
    device: torch.device,
    chunk_size: int = CHUNK_SIZE,
    use_cache: bool = True,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run chunk-by-chunk inference on a full chromosome, optionally with hidden-state caching."""
    try:
        from mamba_ssm.utils.generation import InferenceParams
        _has_cache = True
    except ImportError:
        _has_cache = False
        if use_cache:
            print("[WARN] mamba_ssm not available. Running without hidden-state cache.")

    tokens = tokenise(seq)
    L      = len(tokens)
    probs  = np.zeros((L, len(LABEL_NAMES)), dtype=np.float32)

    cache_params = None

    for chunk_idx, start in enumerate(range(0, L, chunk_size)):
        end   = min(L, start + chunk_size)
        chunk = tokens[start:end]

        x = torch.from_numpy(chunk).unsqueeze(0).long().to(device)  # [1, clen]

        # Frame phase: chunk start position modulo 3, consistent with the collator
        fp = torch.tensor([start % 3], dtype=torch.long, device=device)

        if use_cache and _has_cache:
            if cache_params is None:
                cache_params = InferenceParams(
                    max_seqlen=chunk_size, max_batch_size=1
                )
                cache_params.key_value_memory_dict = {}
            cache_params.seqlen_offset = 0
        else:
            cache_params = None

        with torch.no_grad():
            out    = model.model(
                input_ids    = x,
                cache_params = cache_params,
                frame_phase  = fp,
            )
            logits = out.frame_logits          # [1, clen, 7]
            chunk_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        probs[start:end] = chunk_probs

    preds = (probs >= threshold).astype(np.uint8)
    return probs, preds


def parse_gff_labels(gff_path: str, seq_id: str, seq_len: int) -> Optional[np.ndarray]:
    """Build a (seq_len, 7) binary label array from a GFF3 file for one sequence.

    Returns None if seq_id has no annotations or gffutils is not installed.
    The gffutils DB is cached at gff_path + ".db".
    """
    try:
        import gffutils
    except ImportError:
        print("[ERROR] gffutils not installed. Run: pip install gffutils")
        return None

    db_path = gff_path + ".db"
    if Path(db_path).exists():
        try:
            db = gffutils.FeatureDB(db_path, keep_order=True)
        except Exception:
            Path(db_path).unlink()
            db = gffutils.create_db(
                gff_path, dbfn=db_path, force=True, keep_order=True,
                merge_strategy="create_unique",
                disable_infer_genes=True,
                disable_infer_transcripts=True,
            )
    else:
        print(f"[INFO] Building GFF database: {db_path}")
        db = gffutils.create_db(
            gff_path, dbfn=db_path, force=True, keep_order=True,
            merge_strategy="create_unique",
            disable_infer_genes=True,
            disable_infer_transcripts=True,
        )

    y = np.zeros((seq_len, len(LABEL_NAMES)), dtype=np.uint8)

    def _fill(label_idx, intervals):
        for s, e in intervals:
            s2, e2 = max(0, s), min(seq_len, e)
            if e2 > s2:
                y[s2:e2, label_idx] = 1

    for label, ftypes in GFF_FEATURE_MAP.items():
        idx = LABEL_NAMES.index(label)
        intervals = []
        for ftype in ftypes:
            try:
                for f in db.features_of_type(ftype):
                    if f.seqid == seq_id:
                        intervals.append((f.start - 1, f.end))
            except Exception:
                pass
        _fill(idx, intervals)

    # Derive intron, splice_donor, and splice_acceptor from exon pairs per mRNA
    introns, donors, acceptors = [], [], []
    try:
        for tr in db.features_of_type("mRNA"):
            if tr.seqid != seq_id:
                continue
            exons = sorted(
                db.children(tr, featuretype="exon", order_by="start"),
                key=lambda f: f.start,
            )
            if len(exons) < 2:
                continue
            for i in range(len(exons) - 1):
                e1_end   = exons[i].end
                e2_start = exons[i + 1].start - 1
                strand   = exons[i].strand
                introns.append((e1_end, e2_start))
                if strand == "+":
                    donors.append((e1_end, e1_end + 2))
                    acceptors.append((e2_start - 2, e2_start))
                else:
                    donors.append((e2_start - 2, e2_start))
                    acceptors.append((e1_end, e1_end + 2))
    except Exception as exc:
        print(f"[WARN] mRNA/exon parsing failed: {exc}")

    _fill(LABEL_NAMES.index("intron"),          introns)
    _fill(LABEL_NAMES.index("splice_donor"),    donors)
    _fill(LABEL_NAMES.index("splice_acceptor"), acceptors)

    if y.sum() == 0:
        print(f"[WARN] No annotations found for '{seq_id}' in {gff_path}")
        return None

    return y


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict:
    """Per-label and macro metrics dict with keys: precision, recall, f1, accuracy, auroc, ap."""
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        accuracy_score, roc_auc_score, average_precision_score,
    )

    preds = (probs >= 0.5).astype(int)
    results = {}

    for i, name in enumerate(LABEL_NAMES):
        y, p, prob = labels[:, i], preds[:, i], probs[:, i]
        pos = y.sum()
        results[name] = {
            "precision": precision_score(y, p, zero_division=0),
            "recall":    recall_score(y, p, zero_division=0),
            "f1":        f1_score(y, p, zero_division=0),
            "accuracy":  accuracy_score(y, p),
            "auroc":     roc_auc_score(y, prob) if pos > 0 and pos < len(y) else float("nan"),
            "ap":        average_precision_score(y, prob) if pos > 0 else float("nan"),
        }

    results["macro"] = {}
    for metric in ("precision", "recall", "f1", "accuracy", "auroc", "ap"):
        vals = [
            results[name][metric]
            for name in LABEL_NAMES
            if not np.isnan(results[name][metric])
        ]
        results["macro"][metric] = float(np.mean(vals)) if vals else float("nan")

    return results


def print_metrics(metrics: Dict):
    header = (f"{'Label':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} "
              f"{'Acc':>7} {'AUROC':>7} {'AP':>7}")
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name in LABEL_NAMES + ["macro"]:
        m = metrics[name]
        print(f"{name:<25} "
              f"{m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} "
              f"{m['accuracy']:>7.4f} "
              f"{m['auroc']:>7.3f} "
              f"{m['ap']:>7.3f}")
    print("=" * len(header))


def save_metrics_tsv(metrics: Dict, path: Path):
    with open(path, "w") as f:
        f.write("label\tprecision\trecall\tf1\taccuracy\tauroc\tap\n")
        for name in LABEL_NAMES + ["macro"]:
            m = metrics[name]
            f.write(f"{name}\t{m['precision']:.4f}\t{m['recall']:.4f}\t"
                    f"{m['f1']:.4f}\t{m['accuracy']:.4f}\t"
                    f"{m['auroc']:.4f}\t{m['ap']:.4f}\n")
    print(f"[INFO] Metrics saved -> {path}")


def preds_to_gff3(preds: np.ndarray, seq_id: str, path: Path,
                  threshold: float = 0.5, source: str = "smORFeus"):
    """Write binary predictions to a GFF3 file.

    Each contiguous run of 1s for a given label becomes one GFF3 feature.
    Features are sorted by start position, with label index as tiebreak.

    Args:
        preds: Uint8 array of shape (L, 7) with binary predictions.
        seq_id: Sequence identifier for the GFF3 seqname column.
        path: Destination file path.
        threshold: Not used here (preds are already binary); retained for
            interface compatibility.
        source: Source field value in the GFF3 output.
    """
    def _runs(col):
        runs, s = [], None
        for i, v in enumerate(col):
            if v == 1 and s is None:
                s = i
            elif v == 0 and s is not None:
                runs.append((s, i))
                s = None
        if s is not None:
            runs.append((s, len(col)))
        return runs

    gff_type = {
        "protein_coding_gene": "gene",
        "five_prime_UTR":      "five_prime_UTR",
        "three_prime_UTR":     "three_prime_UTR",
        "exon":                "exon",
        "intron":              "intron",
        "splice_donor":        "splice_donor_site",
        "splice_acceptor":     "splice_acceptor_site",
    }

    features = []
    for i, label in enumerate(LABEL_NAMES):
        for s, e in _runs(preds[:, i]):
            features.append((s, i, label, e))

    features.sort(key=lambda x: (x[0], x[1]))

    with open(path, "w") as f:
        f.write("##gff-version 3\n")
        f.write(f"##sequence-region {seq_id} 1 {len(preds)}\n")
        for feat_id, (s, _, label, e) in enumerate(features, start=1):
            f.write(
                f"{seq_id}\t{source}\t{gff_type[label]}\t"
                f"{s + 1}\t{e}\t.\t.\t.\t"
                f"ID=feat{feat_id};label={label}\n"
            )
    print(f"[INFO] Predictions (GFF3) saved -> {path}")


def save_eval_figures(
    probs_list:  List[np.ndarray],
    tgts_list:   List[np.ndarray],
    overlay_examples: List[dict],
    outdir: Path,
):
    """Generate ROC, PR, and overlay figures via plot_figures.py; fails gracefully if absent."""
    try:
        import plot_figures as pf
        pf.OUT_DIR = outdir
        outdir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Generating ROC curve...")
        pf.plot_roc(probs_list, tgts_list, name="inference_roc")

        print("[INFO] Generating PR curve...")
        pf.plot_pr(probs_list, tgts_list, name="inference_pr")

        if overlay_examples:
            print("[INFO] Generating annotation overlays...")
            pf.plot_overlay(overlay_examples)
        else:
            print("[INFO] No overlay examples collected, skipping overlay figures.")

    except ImportError:
        print("[WARN] plot_figures.py not importable, skipping figures.")
        print("       Place plot_figures.py alongside plant_inference.py to enable figures.")


def main():
    """Entry point for command-line inference."""
    parser = argparse.ArgumentParser(
        description="smORFeus plant genome inference and evaluation."
    )
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to PlantTrainer checkpoint (.ckpt)")
    parser.add_argument("--fasta",         required=True,
                        help="Input FASTA file (one or more sequences)")
    parser.add_argument("--gff",           default=None,
                        help="Optional GFF3 annotation for evaluation")
    parser.add_argument("--outdir",        default="inference_out",
                        help="Output directory (default: inference_out/)")
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Prediction threshold (default: 0.5)")
    parser.add_argument("--chunk_size",    type=int, default=CHUNK_SIZE,
                        help=f"Chunk size in bp (default: {CHUNK_SIZE})")
    parser.add_argument("--no_cache",      action="store_true",
                        help="Disable hidden-state caching across chunks")
    parser.add_argument("--max_overlays",  type=int, default=3,
                        help="Max annotation overlay figures to generate (default: 3)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = load_model(args.checkpoint, device)

    all_probs  = [[] for _ in LABEL_NAMES]
    all_tgts   = [[] for _ in LABEL_NAMES]
    overlays   = []
    eval_mode  = args.gff is not None

    records = list(SeqIO.parse(args.fasta, "fasta"))
    print(f"[INFO] Found {len(records)} sequence(s) in {args.fasta}")

    for rec in records:
        seq_id = rec.id
        seq    = str(rec.seq).upper()
        L      = len(seq)
        print(f"\n[INFO] Processing {seq_id} ({L:,} bp)...")

        probs, preds = run_inference_on_sequence(
            model, seq, device,
            chunk_size = args.chunk_size,
            use_cache  = not args.no_cache,
            threshold  = args.threshold,
        )

        gff_out = outdir / f"{seq_id}_predictions.gff3"
        preds_to_gff3(preds, seq_id, gff_out, threshold=args.threshold)

        if eval_mode:
            labels = parse_gff_labels(args.gff, seq_id, L)
            if labels is None:
                print(f"[WARN] Skipping metrics for {seq_id}, no GFF annotations found.")
                continue

            print(f"[INFO] Computing metrics for {seq_id}...")
            metrics = compute_metrics(probs, labels)
            print_metrics(metrics)

            tsv_path = outdir / f"{seq_id}_metrics.tsv"
            save_metrics_tsv(metrics, tsv_path)

            for i in range(len(LABEL_NAMES)):
                all_probs[i].append(probs[:, i])
                all_tgts[i].append(labels[:, i].astype(np.float32))

            if len(overlays) < args.max_overlays:
                best_start, best_score = 0, 0
                for start in range(0, L, args.chunk_size):
                    end   = min(L, start + args.chunk_size)
                    score = int(labels[start:end].sum())
                    if score > best_score:
                        best_score, best_start = score, start
                end = min(L, best_start + args.chunk_size)
                overlays.append({
                    "true":     labels[best_start:end].astype(int),
                    "pred":     preds[best_start:end].astype(int),
                    "seq_id":   seq_id,
                    "chunk_id": f"pos_{best_start}-{end}",
                })

    if eval_mode and any(len(x) > 0 for x in all_probs):
        print("\n[INFO] Generating aggregate evaluation figures...")
        probs_cat = [np.concatenate(x) for x in all_probs]
        tgts_cat  = [np.concatenate(x) for x in all_tgts]

        agg_probs   = np.stack(probs_cat, axis=1)
        agg_labels  = np.stack(tgts_cat,  axis=1).astype(np.uint8)
        agg_metrics = compute_metrics(agg_probs, agg_labels)
        print("\n[INFO] Aggregate metrics across all sequences:")
        print_metrics(agg_metrics)
        save_metrics_tsv(agg_metrics, outdir / "aggregate_metrics.tsv")

        save_eval_figures(probs_cat, tgts_cat, overlays, outdir / "figures")

    print(f"\n[INFO] All outputs saved to {outdir}/")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()