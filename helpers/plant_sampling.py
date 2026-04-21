"""Sequence-aware batch sampler for plant genome annotation.

Provides SequenceAwareBatchSampler, a PyTorch BatchSampler that guarantees
no two chunks from the same genomic sequence appear in the same batch. This
property is required for the Mamba hidden-state caching scheme to remain
valid: each sequence maintains its own SSM state across consecutive chunks.

Classes exposed:
    SequenceAwareBatchSampler -- BatchSampler that prevents same-sequence
                                 collisions within a batch.
"""

import random
from collections import deque
from typing import List, Optional
from torch.utils.data import BatchSampler, Subset, Sampler


class SequenceAwareBatchSampler(BatchSampler):
    """BatchSampler that prevents chunks from the same sequence sharing a batch."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        seed: int,
        sampler=None,
        drop_last: bool = True,
        fixed_first_batch: bool = False,
        max_active_sequences: Optional[int] = None,
        **_,
    ):
        super().__init__(sampler, batch_size, drop_last)

        self.dataset    = dataset
        self.batch_size = batch_size
        self.drop_last  = drop_last

        base_seed = seed if seed is not None else random.randint(0, 2 ** 32 - 1)
        self.rng = random.Random(base_seed)

        self.fixed_first_batch   = bool(fixed_first_batch)
        self._cached_first_batch = None
        self.max_active_sequences = (
            int(max_active_sequences) if max_active_sequences is not None else None
        )

        # Map global -> local indices when dataset is a Subset
        if isinstance(dataset, Subset):
            self.indices      = dataset.indices
            self.global2local = {g: i for i, g in enumerate(self.indices)}
            full              = dataset.dataset
        else:
            self.indices      = list(range(len(dataset)))
            self.global2local = {i: i for i in self.indices}
            full              = dataset

        # Build per-sequence chunk queues from the underlying dataset
        self._base_sequence_queues: dict = {}
        for seq_id, chunks in full.sequence_map.items():
            q = deque(gidx for gidx, _ in chunks if gidx in self.global2local)
            if q:
                self._base_sequence_queues[seq_id] = q

        self.sequences       = list(self._base_sequence_queues.keys())
        self.sequence_queues = None
        self._queues_shuffled = False

    def __iter__(self):
        queues      = {s: deque(self._base_sequence_queues[s]) for s in self.sequences}
        all_seq_ids = list(self.sequences)
        wave_size   = self.max_active_sequences or len(all_seq_ids)
        wave_start  = 0
        active      = all_seq_ids[wave_start : wave_start + wave_size]
        first_batch_done = False

        if not self.fixed_first_batch:
            self.rng.shuffle(active)

        if self.fixed_first_batch and self._cached_first_batch is None:
            batch = []
            for _ in range(min(self.batch_size, len(active))):
                seq  = active.pop(0)
                gidx = queues[seq].popleft()
                batch.append(self.global2local[gidx])
                if queues[seq]:
                    active.append(seq)
            self._cached_first_batch = batch

        if self.fixed_first_batch and self._cached_first_batch:
            yield list(self._cached_first_batch)

        while active:
            batch = []

            if self.fixed_first_batch and not first_batch_done:
                for seq in active[: self.batch_size]:
                    if queues[seq]:
                        gidx = queues[seq].popleft()
                        batch.append(self.global2local[gidx])
                first_batch_done = True
                self.rng.shuffle(active)
                if not self._queues_shuffled:
                    for s in active:
                        if len(queues[s]) > 1:
                            tmp = list(queues[s])
                            self.rng.shuffle(tmp)
                            queues[s] = deque(tmp)
                    self._queues_shuffled = True
            else:
                take = min(self.batch_size, len(active))
                for _ in range(take):
                    seq  = active.pop(0)
                    if not queues[seq]:
                        continue
                    gidx = queues[seq].popleft()
                    batch.append(self.global2local[gidx])
                    if queues[seq]:
                        active.append(seq)

            if batch:
                yield batch

            if not self.fixed_first_batch or first_batch_done:
                if not any(len(queues[s]) for s in active):
                    wave_start += wave_size
                    if wave_start >= len(all_seq_ids):
                        break
                    active = all_seq_ids[wave_start : wave_start + wave_size]
                    self.rng.shuffle(active)
                elif len(active) >= self.batch_size:
                    self.rng.shuffle(active)

    def __len__(self):
        total = sum(len(q) for q in self._base_sequence_queues.values())
        n, r  = divmod(total, self.batch_size)
        return n if self.drop_last or r == 0 else n + 1
