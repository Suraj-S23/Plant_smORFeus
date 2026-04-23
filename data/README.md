# Data

This directory is a convenience location for data artefacts. No data
files are tracked in the repository.

The canonical data layout used by the scripts is:

```
plant_data/
├── fasta/        # per-species FASTA files
└── gff/          # per-species GFF3 annotation files

plant_processed_data_np/
└── <SpeciesName>/output_labels/<seqid>.npz

plant_finetune/
└── plant_data.hdf5   # sequential HDF5 used by the DataModule
```

See the top-level `README.md` and `CONTRIBUTING.md` for how to generate
each stage (`prepare_data.py`, `run_prepare_data.py`,
`convert_plant_npz_to_hdf5.py`).
