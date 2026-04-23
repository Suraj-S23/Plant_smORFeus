"""PlantConfig: hyperparameters shared by training, eval, and inference.

Also exposes named presets and a load_config() helper that accepts either a
preset name or a YAML path.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PlantConfig:
    """Fine-tuning config. Serialisable via from_yaml()/to_yaml()."""

    # === EXPERIMENT METADATA ===
    experiment_name: str = "plant_annotation"
    description: str = "Plant genome annotation with smORFeus backbone"
    tags: List[str] = field(default_factory=list)
    seed: int = 42

    # === TASK ===
    num_labels: int = 7
    label_names: List[str] = field(default_factory=lambda: [
        "protein_coding_gene",
        "five_prime_UTR",
        "three_prime_UTR",
        "exon",
        "intron",
        "splice_donor",
        "splice_acceptor",
    ])
    label_weights: List[float] = field(default_factory=lambda: [
        1.0,   # protein_coding_gene
        1.5,   # five_prime_UTR
        1.5,   # three_prime_UTR
        1.0,   # exon
        1.0,   # intron
        5.0,   # splice_donor   (rare, ~2 bp per intron)
        5.0,   # splice_acceptor
    ])

    # === MODEL ARCHITECTURE ===
    d_model: int = 256
    n_layer: int = 8
    vocab_size: int = 6
    bidirectional: bool = True
    bidirectional_strategy: str = "add"
    bidirectional_weight_tie: bool = True
    use_frame_positional_encoding: bool = True
    frame_pos_mode: str = "global"

    # Continuity decoder
    use_orf_decoder: bool = True
    decoder_type: str = "conv_smooth"
    smoothing_kernel_size: int = 31
    smoothing_kernel_sizes: List[int] = field(default_factory=lambda: [
        31,  # protein_coding_gene: spans thousands of bp
        15,  # five_prime_UTR: ~100-500 bp
        15,  # three_prime_UTR: ~100-500 bp
        21,  # exon: ~150 bp median
        31,  # intron: 100 bp to 10 kb+
        3,   # splice_donor: 2 bp (GT dinucleotide)
        3,   # splice_acceptor: 2 bp (AG dinucleotide)
    ])
    decoder_hidden_dim: Optional[int] = None

    # === TRAINING HYPERPARAMETERS ===
    learning_rate: float = 5e-5
    batch_size: int = 4
    max_epochs: int = 20
    warmup_epochs: int = 2
    cosine_t_max: Optional[int] = None
    use_scheduler: bool = True
    scheduler_type: str = "warmup_cosine"

    # Optimizer
    optimizer_type: str = "adamw"
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    gradient_clip_val: float = 0.5
    accumulate_grad_batches: int = 4

    # === LOSS CONFIGURATION ===
    continuity_weight: float = 0.5
    transition_penalty_type: str = "smooth_l1"
    continuity_warmup_epochs: int = 3
    max_continuity_weight: float = 1.0
    continuity_ramp_epochs: int = 2

    # Class imbalance handling
    use_dynamic_weights: bool = False
    weight_ema_beta: float = 0.95
    min_class_weight: float = 0.1
    max_class_weight: float = 10.0

    # === PATHS ===
    # Kept for checkpoint backward compatibility; caduceus/ is self-contained now.
    caduceus_path: str = "caduceus"
    pretrained_checkpoint: Optional[str] = None

    # === DATA CONFIGURATION ===
    data_path: str = "plant_finetune/plant_data.hdf5"
    chunk_size: int = 10_000
    overlap: int = 0
    max_sequences: Optional[int] = None
    test_size: float = 0.15
    val_size: float = 0.15
    num_workers: int = 4
    rc_prob: float = 0.0

    # Data filtering
    min_chunks_per_seq: int = 5
    skip_validation: bool = True
    max_active_sequences: int = 8
    organism_split: bool = True

    # === CACHING ===
    use_hidden_cache: bool = True
    max_cache_size: int = 1000

    # === LOGGING ===
    project_name: str = "plant-annotation"
    log_every_steps: int = 100
    detailed_logging: bool = True
    save_top_k: int = 3
    patience: int = 10
    check_val_every_n_epoch: int = 1

    # Directories
    experiment_dir: str = "experiments"
    checkpoint_dir: str = "checkpoints"
    evaluation_save_dir: str = "evaluation"
    max_eval_batches: int = 50

    # === SYSTEM ===
    precision: str = "16-mixed"
    devices: str = "auto"
    accelerator: str = "auto"
    strategy: str = "auto"
    enable_progress_bar: bool = True

    # === SLURM ===
    slurm_partition: str = "gpu"
    slurm_time: str = "08:00:00"
    slurm_mem: str = "64G"
    slurm_cpus_per_task: int = 8
    slurm_gpus: int = 1
    slurm_job_name: str = "plant_annot"

    # === PRETRAINING ===
    freeze_backbone_epochs: int = 2
    resume_checkpoint: Optional[str] = None
    test_only: bool = False
    test_checkpoint: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PlantConfig":
        """Load from YAML; missing keys fall back to defaults."""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: str) -> None:
        """Dump to YAML. Parent directory must exist."""
        data = {}
        for key, value in self.__dict__.items():
            data[key] = list(value) if isinstance(value, tuple) else value
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def get_model_params(self) -> Dict[str, Any]:
        """Subset consumed by CaduceusConfig."""
        return {
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "vocab_size": self.vocab_size,
            "num_labels": self.num_labels,
            "bidirectional": self.bidirectional,
            "bidirectional_strategy": self.bidirectional_strategy,
            "bidirectional_weight_tie": self.bidirectional_weight_tie,
            "use_frame_positional_encoding": self.use_frame_positional_encoding,
            "frame_pos_mode": self.frame_pos_mode,
        }

    def validate(self) -> List[str]:
        """Return a list of config issues (empty if valid)."""
        issues = []
        if self.num_labels != len(self.label_names):
            issues.append(
                f"num_labels={self.num_labels} but label_names has "
                f"{len(self.label_names)} entries"
            )
        if self.num_labels != len(self.label_weights):
            issues.append(
                f"num_labels={self.num_labels} but label_weights has "
                f"{len(self.label_weights)} entries"
            )
        if self.num_labels != len(self.smoothing_kernel_sizes):
            issues.append(
                f"num_labels={self.num_labels} but smoothing_kernel_sizes has "
                f"{len(self.smoothing_kernel_sizes)} entries"
            )
        if any(k % 2 == 0 or k < 1 for k in self.smoothing_kernel_sizes):
            issues.append("smoothing_kernel_sizes must all be positive odd integers")
        if self.test_size + self.val_size >= 1.0:
            issues.append("test_size + val_size must be < 1.0")
        if self.learning_rate <= 0:
            issues.append("learning_rate must be positive")
        if not Path(self.data_path).exists():
            issues.append(f"data_path does not exist: {self.data_path}")
        return issues

    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Apply overrides in-place; unknown keys are warned and skipped."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: unknown config parameter '{key}'; ignored.")


# ---------------------------------------------------------------------------
# Quick-start presets
# ---------------------------------------------------------------------------

PRESETS = {
    "dev": PlantConfig(
        experiment_name="plant_dev",
        d_model=256,
        n_layer=4,
        batch_size=4,
        max_epochs=3,
        max_sequences=None,
        detailed_logging=False,
        use_hidden_cache=True,
    ),
    "default": PlantConfig(),
    "large": PlantConfig(
        experiment_name="plant_large",
        d_model=384,
        n_layer=12,
        batch_size=2,
        accumulate_grad_batches=8,
        max_epochs=25,
    ),
}


def load_config(source: str = "default") -> PlantConfig:
    """Load by preset name ("dev", "default", "large") or YAML path."""
    if source in PRESETS:
        return PRESETS[source]
    elif Path(source).exists():
        return PlantConfig.from_yaml(source)
    else:
        raise ValueError(
            f"Unknown config source: {source!r}. "
            f"Choose one of {list(PRESETS.keys())} or provide a YAML file path."
        )


if __name__ == "__main__":
    cfg = PlantConfig()
    issues = cfg.validate()
    if issues:
        print("Validation issues:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("Config valid")
    print(f"Labels ({cfg.num_labels}): {cfg.label_names}")
    print(f"Weights: {cfg.label_weights}")
