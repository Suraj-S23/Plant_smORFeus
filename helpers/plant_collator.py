"""Pads raw dataset items into tensor batches. num_labels is inferred from
the first 2-D target in the batch, so any channel count works."""

import random
import numpy as np
import torch


class PlantCollator:
    """Pads batches, masks non-core regions, emits frame phase."""

    def __init__(
        self,
        token_map=None,
        reverse_complement_prob: float = 0.0,
        overlap: int = 0,
        upstream_window: int = 250,
        chunk_size: int = 10_000,
    ):
        self.token_map = token_map or {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
        self.pad_token_id = 5
        self.reverse_complement_prob = reverse_complement_prob
        self.overlap = overlap
        self.upstream_window = upstream_window
        self.chunk_size = chunk_size

        # ASCII lookup for vectorised tokenisation
        self._lookup = np.full(256, self.pad_token_id, dtype=np.int64)
        for char, idx in self.token_map.items():
            self._lookup[ord(char)] = idx
            self._lookup[ord(char.lower())] = idx

        self.batch_counter = 0
        self.empty_seq_count = 0

    def _get_actual_core_region(self, item, seq_len):
        target_core_len = item["target"].shape[0]
        overlap = min(self.overlap, seq_len // 6)
        core_start = overlap
        core_end = core_start + target_core_len
        if core_end > seq_len - overlap:
            core_end = seq_len - overlap
            core_start = max(0, core_end - target_core_len)
        actual_core_len = max(0, core_end - core_start)
        if actual_core_len < target_core_len * 0.8:
            print(f"SKIPPING {item['chunk_id']}: core too small "
                  f"{actual_core_len}/{target_core_len}")
            return 0, 0, 0
        return core_start, core_end, actual_core_len

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return self._empty_batch()

        self.batch_counter += 1
        B = len(batch)
        max_len = max(len(b["sequence"]) for b in batch)

        num_labels = None
        for item in batch:
            if item.get("target") is not None and hasattr(item["target"], "shape"):
                if item["target"].ndim == 2:
                    num_labels = item["target"].shape[1]
                    break
        if num_labels is None:
            num_labels = 7  # plant default

        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        targets = torch.full((B, max_len, num_labels), -100.0, dtype=torch.float32)
        length_weights = torch.ones((B, max_len, num_labels), dtype=torch.float32)

        sequence_ids = []
        chunk_ids = []
        previous_chunk_ids = []
        requires_hidden_state = []
        sampling_regions = []
        core_positions = []
        frame_phase = []

        for i, item in enumerate(batch):
            if not item["sequence"]:
                self.empty_seq_count += 1
                continue

            seq = item["sequence"]
            L_seq = len(seq)
            tgt = item["target"]

            if tgt.ndim != 2 or tgt.shape[1] != num_labels:
                raise ValueError(
                    f"Collator expected target shape [L, {num_labels}], "
                    f"got {tgt.shape} for {item['chunk_id']}"
                )

            core_start, core_end, core_len = self._get_actual_core_region(item, L_seq)
            if core_len == 0:
                continue

            if core_len <= tgt.shape[0]:
                tgt_core = tgt[:core_len]
            else:
                tgt_core = tgt
                core_len = tgt.shape[0]

            sampling_regions.append({
                "before_start": np.array([], dtype=int),
                "after_start": np.array([], dtype=int),
                "non_orf": np.array([], dtype=int),
            })

            seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            ids = self._lookup[seq_bytes]
            input_ids[i, :len(ids)] = torch.from_numpy(ids)
            attention_mask[i, :L_seq] = 1

            if core_len > 0:
                end_pos = min(core_start + core_len, max_len)
                actual_len = end_pos - core_start
                if actual_len > 0:
                    targets[i, core_start:end_pos] = torch.tensor(
                        tgt_core[:actual_len], dtype=torch.float32
                    )
                    if "length_weights" in item:
                        w = item["length_weights"][:core_len]
                        length_weights[i, core_start:end_pos] = torch.tensor(
                            w[:actual_len], dtype=torch.float32
                        )

            sequence_ids.append(item["sequence_id"])
            chunk_ids.append(item["chunk_id"])
            previous_chunk_ids.append(item["previous_chunk_id"])
            requires_hidden_state.append(item["previous_chunk_id"] != "")
            core_positions.append((core_start, core_start + core_len))

            try:
                n = int(str(item["chunk_id"]).split("_chunk_")[-1])
            except Exception:
                n = 0
            frame_phase.append(int((n * self.chunk_size) % 3))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
            "length_weights": length_weights,
            "sequence_ids": sequence_ids,
            "chunk_ids": chunk_ids,
            "previous_chunk_ids": previous_chunk_ids,
            "requires_hidden_state": requires_hidden_state,
            "sampling_regions": sampling_regions,
            "core_positions": core_positions,
            "frame_phase": torch.tensor(frame_phase, dtype=torch.long),
        }

    def _empty_batch(self):
        num_labels = 7  # plant default
        return {
            "input_ids": torch.empty((0, 1), dtype=torch.long),
            "attention_mask": torch.empty((0, 1), dtype=torch.long),
            "targets": torch.empty((0, 1, num_labels), dtype=torch.float32),
            "length_weights": torch.ones((0, 1, num_labels), dtype=torch.float32),
            "sequence_ids": [],
            "chunk_ids": [],
            "previous_chunk_ids": [],
            "requires_hidden_state": [],
            "sampling_regions": [],
            "core_positions": [],
            "frame_phase": torch.empty((0,), dtype=torch.long),
        }