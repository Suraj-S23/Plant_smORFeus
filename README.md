# Plant_smORFeus

Per-nucleotide plant genome annotation by fine-tuning the
smORFeus bidirectional Mamba backbone
(Caduceus architecture) on 7-label genomic annotations across 8+ plant species.

## Labels predicted

| Index | Label | Description |
|-------|-------|-------------|
| 0 | `protein_coding_gene` | Entire gene body (TSS to TES) |
| 1 | `five_prime_UTR` | 5' untranslated region |
| 2 | `three_prime_UTR` | 3' untranslated region |
| 3 | `exon` | Exonic sequence |
| 4 | `intron` | Intronic sequence (derived from exon pairs) |
| 5 | `splice_donor` | 2 bp GT dinucleotide at intron start |
| 6 | `splice_acceptor` | 2 bp AG dinucleotide at intron end |

## Architecture

The backbone is a locally modified Caduceus model (`caduceus/`) with:
- Bidirectional Mamba blocks (forward + reverse-complement streams)
- Optional reverse-complement parameter sharing (RCPS)
- Frame-positional encoding (modulo-3 reading-frame embeddings)
- ConvSmoothingDecoder with per-label Gaussian smoothing kernels

The model is loaded from a pretrained smORFeus checkpoint and fine-tuned
end-to-end with binary cross-entropy loss plus an optional continuity
regularisation term.

## Installation

```bash
# 1. Clone
git clone https://github.com/<your-org>/Plant_smORFeus.git
cd Plant_smORFeus

# 2. Create conda environment
conda env create -f environment.yml
conda activate plant_smorfeues

# 3. Install mamba_ssm from source (requires CUDA GPU)
pip install causal-conv1d
pip install mamba-ssm
# See: https://github.com/state-spaces/mamba
```

## Data pipeline

### Step 1: Prepare NPZ label files

For each species, produce per-chromosome NPZ files containing the sequence
string and a binary `(L, 7)` label array:

```bash
python run_prepare_data.py \
    --dataset_root plant_processed_data_np \
    --fasta_dir    plant_data/fasta \
    --gff_dir      plant_data/gff
```

### Step 2: Convert to HDF5

Convert all species NPZ files into a single sequential HDF5 file:

```bash
python convert_plant_npz_to_hdf5.py \
    --dataset_root plant_processed_data_np \
    --output_hdf5  plant_finetune/plant_data.hdf5
```

The HDF5 layout (one group per sequence, one subgroup per 10 kbp chunk)
mirrors the existing smORFeus convention and supports hidden-state caching
across consecutive chunks.

## Training

```bash
# Quick development run (small model, 3 epochs)
python plant_train.py --config dev

# Full training with the base config (Run E values)
python plant_train.py --config configs/base_config.yaml

# Load a pretrained smORFeus checkpoint
python plant_train.py \
    --config configs/base_config.yaml \
    --pretrained_checkpoint path/to/smorfeus.ckpt
```

All hyperparameters are documented in `configs/base_config.yaml` and
`plant_config.py`.

## Evaluation

```bash
python plant_eval.py \
    --checkpoint checkpoints/best.ckpt \
    --data_path  plant_finetune/plant_data.hdf5 \
    --outdir     evaluation/
```

Produces ROC curves, precision-recall curves, confusion matrices, and
genomic overlay panels per label.

## Inference

```bash
# Inference only — writes GFF3 predictions
python plant_inference.py \
    --checkpoint checkpoints/best.ckpt \
    --fasta      new_genome.fasta \
    --outdir     inference_out/

# Inference + evaluation — also writes per-label metrics
python plant_inference.py \
    --checkpoint checkpoints/best.ckpt \
    --fasta      new_genome.fasta \
    --gff        new_genome.gff3 \
    --outdir     inference_out/
```

## Adding a new species

See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide covering
data preparation, HDF5 conversion, species registration, and split assignment.

## Project structure

```
Plant_smORFeus/
├── caduceus/                  # Self-contained Caduceus Mamba backbone
│   ├── __init__.py
│   ├── configuration_caduceus.py
│   ├── modeling_caduceus2h.py
│   └── modeling_rcps.py
├── helpers/                   # Data pipeline helpers
│   ├── __init__.py
│   ├── plant_dataset.py       # HDF5 dataset loader
│   ├── plant_datamodule.py    # Lightning DataModule with species-level splits
│   ├── plant_collator.py      # Padded batch collator
│   └── plant_sampling.py      # SequenceAwareBatchSampler
├── configs/
│   └── base_config.yaml       # Run E hyperparameters (baseline)
├── plant_config.py            # PlantConfig dataclass
├── plant_train.py             # Training loop (PlantTrainer)
├── plant_eval.py              # Evaluation callback and standalone eval
├── plant_inference.py         # Genome-wide inference
├── plot_figures.py            # Publication figure generation
├── prepare_data.py            # FASTA + GFF3 -> NPZ
├── run_prepare_data.py        # Batch prepare_data.py driver
├── convert_plant_npz_to_hdf5.py  # NPZ -> HDF5 converter
├── data_statistics.py         # Dataset statistics
├── plot_data_statistics.py    # Statistics figures
├── requirements.txt
├── environment.yml
├── CONTRIBUTING.md
└── LICENSE
```

## Citation

If you use this code, please cite the original Caduceus/smORFeus papers and
this repository.

## License

MIT — see [LICENSE](LICENSE).
