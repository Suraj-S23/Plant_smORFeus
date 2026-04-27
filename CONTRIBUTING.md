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
  (`gene`, `five_prime_UTR`, `three_prime_UTR`, `exon`, `mRNA`).

FASTA and GFF3 files for the eight species used in the original experiments
can be obtained from [Phytozome](https://phytozome-next.jgi.doe.gov/) or
[Ensembl Plants](https://plants.ensembl.org/).

### 2. Prepare NPZ label files

```bash
python run_prepare_data.py \
    --dataset_root plant_processed_data_np \
    --fasta_dir    plant_data/fasta \
    --gff_dir      plant_data/gff
```

Additional flags:

```bash
# Dry run - print commands without executing
python run_prepare_data.py \
    --dataset_root plant_processed_data_np \
    --fasta_dir    plant_data/fasta \
    --gff_dir      plant_data/gff \
    --dry_run

# Process a single species only
python run_prepare_data.py \
    --dataset_root plant_processed_data_np \
    --fasta_dir    plant_data/fasta \
    --gff_dir      plant_data/gff \
    --species      <species_stem>
```

If your FASTA and GFF3 filenames do not share a matching biological stem,
add a manual override entry to the `MANUAL_PAIRS` dictionary at the top of
`run_prepare_data.py`. Alternatively, process the species directly with
`prepare_data.py`:

```bash
python prepare_data.py \
    --fasta      path/to/genome.fasta \
    --gff        path/to/annotation.gff3 \
    --output_dir plant_processed_data_np/<SpeciesName>/output_labels/
```

### 3. Convert to HDF5

```bash
mkdir -p plant_finetune
python convert_plant_npz_to_hdf5.py \
    --dataset_root plant_processed_data_np \
    --output_hdf5  plant_finetune/plant_data.hdf5
```

Add `--verbose` to print per-chromosome statistics during conversion.

### 4. Train/val/test split assignment

Splits are assigned at the species level (`organism_split=True`, the
default). The assignment is fully hardcoded: *Arabidopsis thaliana* is
always held out as the test species and *Oryza sativa* as the validation
species; all remaining species go to training. The split is printed to
stdout at the start of each training run.

To assign a new species to a specific split, add it to `FIXED_TEST` or
`FIXED_VAL` inside `setup()` in `helpers/plant_datamodule.py`. Also add
the assembly prefix to the `BIOLOGICAL_SPECIES` mapping at the top of that
file. Species whose prefix is not recognised are automatically assigned to
training with a warning printed to stdout.

## Running experiments

### Quick development run

```bash
python plant_train.py --preset dev
```

### Full training run

```bash
python plant_train.py --config configs/base_config.yaml
```

### Fine-tuning from a pretrained smORFeus backbone

```bash
python plant_train.py \
    --config     configs/base_config.yaml \
    --checkpoint path/to/smorfeus_pretrained.ckpt
```

### Resuming an interrupted plant training run

```bash
python plant_train.py \
    --config            configs/base_config.yaml \
    --resume_checkpoint path/to/plant_checkpoint.ckpt
```

### Overriding specific parameters

Only flags defined in `plant_train.py`'s argument parser can be overridden
at the CLI: `--checkpoint`, `--data_path`, `--max_epochs`, `--batch_size`,
`--resume_checkpoint`, `--test_only`, `--test_checkpoint`. For all other
fields, edit the YAML config or a preset in `plant_config.py`.

```bash
python plant_train.py \
    --config     configs/base_config.yaml \
    --batch_size 2 \
    --max_epochs 10
```

## Evaluation and inference

See the **Checkpoints** section of the README for how to obtain the pretrained
and fine-tuned checkpoints before running the commands below.

```bash
# Evaluate a checkpoint - writes per-label figures to plant_eval_figures/
python plant_eval.py \
    --checkpoint checkpoints/best_plant_smorfeus.ckpt \
    --data_path  plant_finetune/plant_data.hdf5

# Run inference on a new FASTA - writes GFF3 predictions
python plant_inference.py \
    --checkpoint checkpoints/best_plant_smorfeus.ckpt \
    --fasta      new_genome.fasta \
    --outdir     inference_out/

# Inference + evaluation - also writes per-label metrics
python plant_inference.py \
    --checkpoint checkpoints/best_plant_smorfeus.ckpt \
    --fasta      new_genome.fasta \
    --gff        new_genome.gff3 \
    --outdir     inference_out/
```

## Pull request guidelines

1. Fork the repository and create a feature branch from `main`.
2. Keep PRs focused: one logical change per PR.
3. Document new hyperparameters in `plant_config.py` and
   `configs/base_config.yaml`.
4. If you add a new species, include a brief note in the PR description
   with the genome source, annotation version, and which split it falls
   into.
5. Do not commit data files, checkpoints, or generated figures.

