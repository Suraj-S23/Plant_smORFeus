# Contributing to Plant_smORFeus

How to add new species, run experiments, and submit pull requests.

## Development setup

```bash
# 1. Clone the repository
git clone https://github.com/Suraj-S23/Plant_smORFeus.git
cd Plant_smORFeus

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate plant_smorfeus

# 3. Install mamba_ssm from source (requires a CUDA-capable GPU)
pip install causal-conv1d
pip install mamba-ssm
# Full instructions: https://github.com/state-spaces/mamba
```

## Adding a new plant species

### 1. Obtain the data

You need two files per genome:
- A FASTA file with one sequence per chromosome or scaffold.
- A GFF3 annotation file with feature types matching the label schema
  (gene, five_prime_UTR, three_prime_UTR, exon, mRNA).

### 2. Prepare NPZ label files

```bash
python prepare_data.py \
    --fasta  path/to/genome.fasta \
    --gff    path/to/annotation.gff3 \
    --output_dir plant_processed_data_np/<SpeciesName>/output_labels/
```

Or use `run_prepare_data.py` for batch processing:

```bash
python run_prepare_data.py \
    --dataset_root plant_processed_data_np \
    --fasta_dir    plant_data/fasta \
    --gff_dir      plant_data/gff
```

If your FASTA and GFF filenames do not share a common stem after standard
suffix stripping, add a manual override entry to the `MANUAL_PAIRS`
dictionary at the top of `run_prepare_data.py`.

### 3. Convert to HDF5

```bash
python convert_plant_npz_to_hdf5.py \
    --dataset_root plant_processed_data_np \
    --output_hdf5  plant_finetune/plant_data.hdf5
```

### 4. Register the species for train/val/test assignment

Open `helpers/plant_datamodule.py` and update the `BIOLOGICAL_SPECIES`
class attribute to include a prefix-to-species mapping for the new species.
By default, unrecognised species are placed in the training split; to assign
them to val or test, add them to `FIXED_VAL` or `FIXED_TEST` inside
`setup()`.

## Running experiments

### Quick development run

```bash
python plant_train.py --preset dev
```

### Full training run

```bash
python plant_train.py --config configs/base_config.yaml
```

### Overriding specific parameters

Only flags defined in `plant_train.py`'s argument parser can be overridden
at the CLI (`--batch_size`, `--max_epochs`, `--data_path`, `--checkpoint`,
`--resume_checkpoint`, `--test_only`, `--test_checkpoint`). For all other
fields, edit the YAML config or a preset in `plant_config.py`.

```bash
python plant_train.py --config configs/base_config.yaml \
    --batch_size 2 --max_epochs 10
```

## Evaluation and inference

```bash
# Evaluate a checkpoint
python plant_eval.py \
    --checkpoint checkpoints/best.ckpt \
    --data_path  plant_finetune/plant_data.hdf5

# Run inference on new FASTA
python plant_inference.py \
    --checkpoint checkpoints/best.ckpt \
    --fasta      new_genome.fasta \
    --outdir     inference_out/
```

## Pull request guidelines

1. Fork the repository and create a feature branch from `main`.
2. Run `python -m pytest` (when tests exist) before opening a PR.
3. Keep PRs focused: one logical change per PR.
4. Document new hyperparameters in `plant_config.py` and `configs/base_config.yaml`.
5. If you add a new species, include a brief note in the PR description with
   the genome source, annotation version, and how it was assigned to a split.
6. Do not commit data files, checkpoints, or generated figures.

