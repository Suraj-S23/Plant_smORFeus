"""Helper modules for the Plant_smORFeus data pipeline.

This package provides the dataset, datamodule, collator, and batch sampler
used during training, evaluation, and inference. All components are
designed to work together with the caduceus/ Mamba backbone and the
Plant_smORFeus training loop in plant_train.py.

Exported names:
    GenomicHDF5Dataset         -- Per-chunk HDF5 dataset loader
    PlantDataModule            -- Lightning DataModule with species-level splits
    PlantCollator              -- Padded batch collator with frame-phase encoding
    SequenceAwareBatchSampler  -- Batch sampler that prevents same-sequence collisions
"""

from .plant_dataset import GenomicHDF5Dataset
from .plant_datamodule import PlantDataModule
from .plant_collator import PlantCollator
from .plant_sampling import SequenceAwareBatchSampler

__all__ = [
    "GenomicHDF5Dataset",
    "PlantDataModule",
    "PlantCollator",
    "SequenceAwareBatchSampler",
]
