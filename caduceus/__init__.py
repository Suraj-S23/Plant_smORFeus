"""Caduceus: custom bidirectional Mamba backbone for Plant_smORFeus.

This package contains the modified Caduceus architecture used for
per-nucleotide plant genome annotation in the Plant_smORFeus pipeline.
It is a self-contained, locally modified version of the Caduceus model,
not an external dependency.

Exported names:
    CaduceusConfig      -- Model and training configuration dataclass
    CaduceusMixerModel  -- Full bidirectional Mamba backbone
    Caduceus            -- Backbone with per-nucleotide annotation heads
"""

from .configuration_caduceus import CaduceusConfig
from .modeling_caduceus2h import CaduceusMixerModel, Caduceus

__all__ = ["CaduceusConfig", "CaduceusMixerModel", "Caduceus"]