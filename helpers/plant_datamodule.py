"""Lightning DataModule: species-level splits + SequenceAwareBatchSampler to
keep hidden-state caches valid across consecutive chunks."""

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from torch.utils.data import DataLoader, Subset
from pytorch_lightning import LightningDataModule

from .plant_dataset import GenomicHDF5Dataset
from .plant_sampling import SequenceAwareBatchSampler
from .plant_collator import PlantCollator


class PlantDataModule(LightningDataModule):
    """Species-level train/val/test splits.

    Fixed: Arabidopsis thaliana → test, Oryza sativa → val, rest → train.
    """

    def __init__(
        self,
        data_path: str,
        num_labels: int = 7,
        batch_size: int = 4,
        num_workers: int = 4,
        chunk_size: int = 10_000,
        overlap: int = 0,
        rc_prob: float = 0.0,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42,
        max_active_sequences: int = 8,
        max_sequences: int = None,
        skip_validation: bool = True,
        min_chunks_per_seq: int = 5,
        organism_split: bool = True,
        length_bins: list = None,
    ):
        super().__init__()
        self.data_path = data_path
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.rc_prob = rc_prob
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed
        self.max_active_sequences = max_active_sequences
        self.max_sequences = max_sequences
        self.skip_validation = skip_validation
        self.min_chunks_per_seq = min_chunks_per_seq
        self.organism_split = organism_split
        self.length_bins = length_bins

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    # Assembly ID → biological species. Keeps all assemblies of one organism
    # in the same split to prevent leakage.
    BIOLOGICAL_SPECIES = {
        "ZEAMA": "Zea_mays",
        "MARPO": "Marchantia_polymorpha",
        "POPTR": "Populus_trichocarpa",
        "SOLTU": "Solanum_tuberosum",
        "AAGR": "Amborella_trichopoda",  # covers AagrBONN, AagrOXF
        "ARATH": "Arabidopsis_thaliana",
        "ORYSAJA": "Oryza_sativa",
        "BRADI": "Brachypodium_distachyon",
    }

    @classmethod
    def get_biological_species(cls, assembly_species_id: str) -> str:
        """Assembly ID → canonical species name."""
        for prefix, bio_name in cls.BIOLOGICAL_SPECIES.items():
            if assembly_species_id.upper().startswith(prefix):
                return bio_name
        return assembly_species_id

    @staticmethod
    def get_species_id(seq_id: str) -> str:
        """Strip the chromosome suffix from a sequence ID."""
        import re

        chrom_pattern = re.compile(
            r"_(?:Chr|chr|CHR|scaffold|Scaffold|SCAFFOLD|contig|Contig|"
            r"CONTIG|LG|lg|\d)[A-Za-z0-9]*$"
        )
        m = chrom_pattern.search(seq_id)
        if m:
            return seq_id[: m.start()]
        parts = seq_id.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return seq_id

    def setup(self, stage=None):
        if self.train_dataset is not None:
            print(f"Setup already completed, skipping (stage={stage})")
            return

        t0 = time.time()
        print("Loading plant HDF5 dataset...")
        full = GenomicHDF5Dataset(
            self.data_path,
            overlap=self.overlap,
            length_bins=self.length_bins,
        )
        print(
            f"  Loaded in {time.time() - t0:.1f}s, contains "
            f"{len(full)} chunks, {len(full.sequence_map)} sequences"
        )

        valid_seq_ids = []
        for seq_id in sorted(full.sequence_map.keys()):
            if len(full.sequence_map[seq_id]) >= self.min_chunks_per_seq:
                valid_seq_ids.append(seq_id)
            if self.max_sequences and len(valid_seq_ids) >= self.max_sequences:
                break
        print(
            f"  Using {len(valid_seq_ids)} sequences "
            f"(>={self.min_chunks_per_seq} chunks each)"
        )

        valid_set = set(
            gidx
            for seq_id in valid_seq_ids
            for gidx, _ in sorted(full.sequence_map[seq_id], key=lambda x: x[1])
        )

        t_split = time.time()

        if self.organism_split:
            assembly_to_seqs = defaultdict(list)
            for seq_id in valid_seq_ids:
                assembly_to_seqs[self.get_species_id(seq_id)].append(seq_id)

            bio_to_seqs = defaultdict(list)
            bio_to_assemblies = defaultdict(list)
            for assembly_id, seqs in assembly_to_seqs.items():
                bio_name = self.get_biological_species(assembly_id)
                bio_to_seqs[bio_name].extend(seqs)
                bio_to_assemblies[bio_name].append(assembly_id)

            all_species = sorted(bio_to_seqs.keys())
            n_species = len(all_species)
            print(f"\n  Biological species split across {n_species} species:")
            for sp in all_species:
                assemblies = bio_to_assemblies[sp]
                print(
                    f"    {sp}: {len(bio_to_seqs[sp])} sequences "
                    f"(assemblies: {assemblies})"
                )

            # Best-annotated genomes reserved for eval; rest go to train.
            FIXED_TEST = {"Arabidopsis_thaliana"}
            FIXED_VAL = {"Oryza_sativa"}

            test_species = [s for s in all_species if s in FIXED_TEST]
            val_species = [s for s in all_species if s in FIXED_VAL]
            train_species = [
                s for s in all_species if s not in FIXED_TEST and s not in FIXED_VAL
            ]

            assigned = set(test_species) | set(val_species) | set(train_species)
            unassigned = set(all_species) - assigned
            if unassigned:
                print(f"  [WARN] Unrecognised species assigned to train: {unassigned}")
                train_species.extend(sorted(unassigned))

            if set(train_species) & set(val_species):
                raise ValueError("Train/val species overlap detected after split!")
            if set(train_species) & set(test_species):
                raise ValueError("Train/test species overlap detected after split!")
            if set(val_species) & set(test_species):
                raise ValueError("Val/test species overlap detected after split!")

            print(f"\n  Train species ({len(train_species)}): {train_species}")
            print(f"  Val   species ({len(val_species)}):   {val_species}")
            print(f"  Test  species ({len(test_species)}):  {test_species}")

            def seqs_for(sp_list):
                return [s for sp in sp_list for s in bio_to_seqs[sp]]

            train_seq_ids = seqs_for(train_species)
            val_seq_ids = seqs_for(val_species)
            test_seq_ids = seqs_for(test_species)

        else:
            print("  Using sequence-level split (risk of species leakage!)")
            random.seed(self.seed)
            shuffled = valid_seq_ids.copy()
            random.shuffle(shuffled)
            n_test = max(1, int(len(shuffled) * self.test_size))
            n_val = max(1, int(len(shuffled) * self.val_size))
            n_train = len(shuffled) - n_test - n_val
            train_seq_ids = shuffled[:n_train]
            val_seq_ids = shuffled[n_train : n_train + n_val]
            test_seq_ids = shuffled[n_train + n_val :]

        def get_ordered_indices(seq_ids):
            indices = []
            for seq_id in seq_ids:
                if seq_id not in full.sequence_map:
                    continue
                ordered = sorted(full.sequence_map[seq_id], key=lambda x: x[1])
                indices.extend(gidx for gidx, _ in ordered if gidx in valid_set)
            return indices

        train_idx = get_ordered_indices(train_seq_ids)
        val_idx = get_ordered_indices(val_seq_ids)
        test_idx = get_ordered_indices(test_seq_ids)

        print(f"\n  Split in {time.time() - t_split:.1f}s")
        total = len(train_idx) + len(val_idx) + len(test_idx)
        print(
            f"  Train: {len(train_idx)} chunks  ({len(train_idx) / total * 100:.1f}%)"
        )
        print(f"  Val:   {len(val_idx)} chunks  ({len(val_idx) / total * 100:.1f}%)")
        print(f"  Test:  {len(test_idx)} chunks  ({len(test_idx) / total * 100:.1f}%)")

        self.train_dataset = Subset(full, train_idx)
        self.val_dataset = Subset(full, val_idx)
        self.test_dataset = Subset(full, test_idx)

        self._analyze_cache_potential(full, train_idx, "Train")
        self._analyze_cache_potential(full, val_idx, "Val")
        print(f"  Total setup time: {time.time() - t0:.1f}s")

    def _make_collator(self, augment: bool = False) -> PlantCollator:
        return PlantCollator(
            overlap=self.overlap,
            upstream_window=200,
            reverse_complement_prob=self.rc_prob if augment else 0.0,
            chunk_size=self.chunk_size,
        )

    def _dataloader(self, dataset, is_train: bool = True) -> DataLoader:
        if is_train and len(dataset) > self.batch_size:
            sampler = SequenceAwareBatchSampler(
                dataset=dataset,
                batch_size=self.batch_size,
                seed=self.seed,
                max_active_sequences=self.max_active_sequences,
                fixed_first_batch=True,
                drop_last=True,
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=self._make_collator(augment=True),
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=True,
                prefetch_factor=2,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._make_collator(augment=False),
                num_workers=min(2, self.num_workers),
                persistent_workers=False,
                pin_memory=False,
            )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, False)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, False)

    def save_split_metadata(self, experiment_name: str):
        metadata_dir = Path("training_metadata")
        metadata_dir.mkdir(exist_ok=True)

        metadata: dict = {"experiment_name": experiment_name, "splits": {}}
        for split_name, ds in [
            ("train", self.train_dataset),
            ("val", self.val_dataset),
            ("test", self.test_dataset),
        ]:
            if ds is None:
                continue
            seq_ids = set()
            if hasattr(ds, "dataset") and hasattr(ds.dataset, "all_chunks"):
                for idx in ds.indices:
                    seq_id, _ = ds.dataset.all_chunks[idx]
                    seq_ids.add(seq_id)
            metadata["splits"][split_name] = {
                "sequence_ids": sorted(seq_ids),
                "species": sorted({self.get_species_id(s) for s in seq_ids}),
                "num_sequences": len(seq_ids),
                "num_chunks": len(ds),
            }

        filepath = metadata_dir / f"{experiment_name}_plant_splits.json"
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved split metadata -> {filepath}")
        return filepath

    def _analyze_cache_potential(self, dataset, indices, split_name: str):
        if len(indices) < 100:
            return
        sample = indices[:: max(1, len(indices) // 100)]
        hits = total = 0
        for idx in sample:
            seq_id, chunk_id = dataset.all_chunks[idx]
            prev = dataset.previous_chunk_map.get((seq_id, chunk_id), "")
            if prev:
                total += 1
                prev_idx = dataset.chunk_index.get((seq_id, prev))
                if prev_idx is not None and prev_idx in indices:
                    hits += 1
        if total:
            print(
                f"  {split_name} cache potential: {hits / total:.1%} "
                f"({hits}/{total} chunks can reuse hidden states)"
            )
