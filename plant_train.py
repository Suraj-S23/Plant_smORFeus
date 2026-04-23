"""Fine-tune smORFeus on 7-label plant genome annotation.

PlantTrainer (LightningModule) = CaduceusMixerModel backbone + a
ConvSmoothingDecoder (or StandardDecoder) head.

  python plant_train.py                    # default config
  python plant_train.py --preset dev       # smoke test
  python plant_train.py --config foo.yaml
"""
from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)

try:
    from mamba_ssm.utils.generation import InferenceParams
except ImportError:
    InferenceParams = None

from caduceus import CaduceusConfig, CaduceusMixerModel
from helpers import PlantCollator, PlantDataModule
from plant_config import PlantConfig, load_config

os.environ.setdefault("TRITON_DISABLE", "1")


# =============================================================================
# Decoder modules
# =============================================================================

class StandardDecoder(nn.Module):
    """Per-label MLP heads with no temporal smoothing."""

    def __init__(self, d_model: int, num_labels: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or d_model // 2
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_labels)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(h) for head in self.heads], dim=-1)


class ConvSmoothingDecoder(nn.Module):
    """Per-label MLP + learnable 1-D Gaussian smoother. Raw and smoothed
    streams are mixed with learned per-label weights."""

    def __init__(
        self,
        d_model: int,
        num_labels: int,
        kernel_size: int = 31,
        kernel_sizes: List[int] = None,
        hidden_dim: int = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or d_model // 2
        self.num_labels = num_labels

        if kernel_sizes is not None and len(kernel_sizes) == num_labels:
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = [kernel_size] * num_labels

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_labels)
        ])
        self.smoothing_convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=ks, padding=ks // 2, bias=False)
            for ks in self.kernel_sizes
        ])

        # Narrow-kernel labels (splice sites) favour raw predictions.
        raw_init = []
        smooth_init = []
        for ks in self.kernel_sizes:
            if ks <= 5:
                raw_init.append(1.5)    # sigmoid(1.5) ≈ 0.82 → mostly raw
                smooth_init.append(-0.5)
            else:
                raw_init.append(0.7)    # sigmoid(0.7) ≈ 0.67
                smooth_init.append(0.3)
        self.raw_weights = nn.Parameter(torch.tensor(raw_init))
        self.smooth_weights = nn.Parameter(torch.tensor(smooth_init))
        self._init_smoothing()

    def _init_smoothing(self) -> None:
        """Initialise each smoothing kernel to a discrete Gaussian."""
        for conv in self.smoothing_convs:
            with torch.no_grad():
                k = conv.weight.shape[-1]
                t = torch.linspace(-2, 2, k)
                w = torch.exp(-0.5 * t ** 2)
                conv.weight.data = (w / w.sum()).view(1, 1, -1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raw = torch.cat([head(h) for head in self.heads], dim=-1)  # [B, L, N]
        smooth = []
        for i, conv in enumerate(self.smoothing_convs):
            s = conv(raw[..., i:i + 1].transpose(1, 2)).transpose(1, 2)
            smooth.append(s)
        smooth = torch.cat(smooth, dim=-1)  # [B, L, N]
        rw = torch.sigmoid(self.raw_weights)     # [N]
        sw = torch.sigmoid(self.smooth_weights)  # [N]
        return (rw * raw + sw * smooth) / (rw + sw + 1e-8)


def _make_decoder(decoder_type: str, d_model: int, num_labels: int, **kw) -> nn.Module:
    if decoder_type == "conv_smooth":
        return ConvSmoothingDecoder(d_model, num_labels, **kw)
    return StandardDecoder(d_model, num_labels, **kw)


# =============================================================================
# Plant model wrapper
# =============================================================================

class PlantAnnotationModel(nn.Module):
    """Caduceus backbone + multi-label plant annotation decoder."""

    def __init__(self, caduceus_cfg: CaduceusConfig, plant_cfg: PlantConfig):
        super().__init__()
        self.backbone = CaduceusMixerModel(caduceus_cfg)
        self.decoder = _make_decoder(
            plant_cfg.decoder_type,
            plant_cfg.d_model,
            plant_cfg.num_labels,
            kernel_size=plant_cfg.smoothing_kernel_size,
            kernel_sizes=plant_cfg.smoothing_kernel_sizes,
            hidden_dim=plant_cfg.decoder_hidden_dim,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_params=None,
        frame_phase: Optional[torch.Tensor] = None,
        **kw,
    ) -> SimpleNamespace:
        out = self.backbone(
            input_ids=input_ids,
            cache_params=cache_params,
            frame_phase=frame_phase,
            **kw,
        )
        h = out.last_hidden_state  # [B, L, D]
        logits = self.decoder(h)   # [B, L, N]
        result = SimpleNamespace()
        result.frame_logits = logits
        result.last_hidden_state = h
        return result


# =============================================================================
# Lightning module
# =============================================================================

class PlantTrainer(LightningModule):
    """Lightning module for plant annotation fine-tuning."""

    def __init__(self, config: PlantConfig):
        super().__init__()
        if InferenceParams is None and config.use_hidden_cache:
            raise ImportError(
                "mamba_ssm is not installed but use_hidden_cache=True. "
                "Either install mamba_ssm from source "
                "(https://github.com/state-spaces/mamba) or set "
                "use_hidden_cache=False in your config."
            )
        self.save_hyperparameters()
        self.config = config
        nl = config.num_labels

        caduceus_cfg = CaduceusConfig(**config.get_model_params())
        self.model = PlantAnnotationModel(caduceus_cfg, config)

        self.register_buffer(
            "label_weights",
            torch.tensor(config.label_weights, dtype=torch.float32),
        )

        # EMA class-balance buffers: shape [num_labels, 2] (neg, pos counts)
        self.register_buffer("label_class_counts_ema", torch.zeros(nl, 2))
        self.register_buffer("label_total_counts_ema", torch.zeros(nl))

        self.hidden_cache: Optional[OrderedDict] = (
            OrderedDict() if config.use_hidden_cache else None
        )
        self.cache_hits = 0
        self.cache_misses = 0

        self.test_outputs: List = []
        self.grad_norms: List[float] = []
        self.continuity_losses: List[float] = []
        self.current_continuity_weight: float = 0.0

        self._test_tp = torch.zeros(config.num_labels, dtype=torch.long)
        self._test_fp = torch.zeros(config.num_labels, dtype=torch.long)
        self._test_fn = torch.zeros(config.num_labels, dtype=torch.long)

    # -------------------------------------------------------------------------
    # Checkpoint loading
    # -------------------------------------------------------------------------

    @classmethod
    def from_smorfeus_checkpoint(cls, config: PlantConfig) -> "PlantTrainer":
        """Load only the backbone from a smORFeus checkpoint. The old head is
        dropped; the new 7-label head starts random."""
        module = cls(config)
        ckpt_path = config.pretrained_checkpoint
        if not ckpt_path:
            print("[INIT] No pretrained_checkpoint specified - training from scratch.")
            return module

        print(f"[INIT] Loading smORFeus backbone from: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)

        backbone_state = {
            k.replace("model.backbone.", ""): v
            for k, v in state.items()
            if k.startswith("model.backbone.")
        }

        missing, unexpected = module.model.backbone.load_state_dict(
            backbone_state, strict=False
        )
        print(f"  Backbone loaded; missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")

        return module

    # -------------------------------------------------------------------------
    # EMA class balance
    # -------------------------------------------------------------------------

    def _update_label_statistics(
        self, targets: torch.Tensor, core_mask: torch.Tensor
    ) -> None:
        """EMA update of per-label pos/neg counts."""
        if not core_mask.any():
            return
        tgt_core = targets[core_mask].clamp_min(0)  # [N_valid, num_labels]
        beta = self.config.weight_ema_beta
        for i in range(self.config.num_labels):
            col = tgt_core[:, i]
            pos = (col == 1).sum().float()
            neg = (col == 0).sum().float()
            tot = pos + neg
            if tot > 0:
                new_counts = torch.stack([neg, pos])
                self.label_class_counts_ema[i] = (
                    beta * self.label_class_counts_ema[i] + (1 - beta) * new_counts
                )
                self.label_total_counts_ema[i] = (
                    beta * self.label_total_counts_ema[i] + (1 - beta) * tot
                )

    def _get_dynamic_label_weights(self) -> List[Optional[torch.Tensor]]:
        """Per-label pos/neg weights from EMA counts, or [None, ...] if disabled."""
        if not self.config.use_dynamic_weights:
            return [None] * self.config.num_labels
        eps = 1e-6
        weights = []
        for i in range(self.config.num_labels):
            tot = self.label_total_counts_ema[i] + eps
            counts = self.label_class_counts_ema[i] + eps
            w = 1.0 / (counts / tot)
            w = w / w.mean()
            w = torch.clamp(w, self.config.min_class_weight, self.config.max_class_weight)
            weights.append(w)
        return weights

    # -------------------------------------------------------------------------
    # Continuity weight schedule
    # -------------------------------------------------------------------------

    def _get_continuity_weight(self) -> float:
        """Continuity weight for this epoch (0 during warm-up)."""
        if self.current_epoch < self.config.continuity_warmup_epochs:
            return 0.0
        ramp = min(
            1.0,
            (self.current_epoch - self.config.continuity_warmup_epochs)
            / max(1, self.config.continuity_ramp_epochs),
        )
        return ramp * min(self.config.continuity_weight, self.config.max_continuity_weight)

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Per-label BCE + ramped continuity penalty."""
        core_mask = (targets != -100).all(dim=-1)  # [B, L]
        if not core_mask.any():
            return torch.zeros(1, device=logits.device, requires_grad=True), {}

        tgt = targets.clamp_min(0.0)

        if self.config.use_dynamic_weights:
            self._update_label_statistics(targets, core_mask)

        static_w = self.label_weights.to(logits.device)
        dynamic_w = self._get_dynamic_label_weights()

        total_label_loss = torch.zeros(1, device=logits.device)
        components: Dict[str, torch.Tensor] = {}

        for i in range(self.config.num_labels):
            logits_i = logits[..., i]   # [B, L]
            targets_i = tgt[..., i]     # [B, L]

            pos_weight = None
            if dynamic_w[i] is not None:
                neg_w, pos_w = dynamic_w[i]
                pos_weight = (pos_w / neg_w).unsqueeze(0)

            loss_i = F.binary_cross_entropy_with_logits(
                logits_i[core_mask],
                targets_i[core_mask],
                pos_weight=pos_weight,
                reduction="mean",
            )
            total_label_loss = total_label_loss + static_w[i] * loss_i
            components[f"loss_{self.config.label_names[i]}"] = loss_i

        components["total_label_loss"] = total_label_loss

        cw = self._get_continuity_weight()
        self.current_continuity_weight = cw
        if cw > 0:
            cont_loss = self._continuity_loss(logits, targets, core_mask)
            self.continuity_losses.append(cont_loss.item())
            total = total_label_loss + cw * cont_loss
        else:
            cont_loss = torch.zeros(1, device=logits.device)
            total = total_label_loss

        components["continuity_loss"] = cont_loss
        components["continuity_weight"] = torch.tensor(cw, device=logits.device)
        return total, components

    def _continuity_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        core_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean smooth-L1 continuity penalty across labels."""
        probs = torch.sigmoid(logits)
        tgt = targets.clamp_min(0.0)
        total_loss = 0.0
        for i in range(self.config.num_labels):
            total_loss = total_loss + self._single_label_continuity(
                probs[..., i], tgt[..., i], core_mask
            )
        return total_loss / self.config.num_labels

    def _single_label_continuity(
        self,
        probs: torch.Tensor,
        tgt: torch.Tensor,
        core_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth-L1 on adjacent prob differences, one label."""
        p = probs * core_mask.float()
        t = tgt * core_mask.float()
        pd = torch.abs(p[:, 1:] - p[:, :-1])
        td = torch.abs(t[:, 1:] - t[:, :-1])
        valid = core_mask[:, 1:] & core_mask[:, :-1]
        if not valid.any():
            return torch.zeros(1, device=probs.device, requires_grad=True)
        return F.smooth_l1_loss(pd[valid], td[valid], reduction="mean")

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def _compute_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        prefix: str,
    ) -> None:
        """Log per-label + aggregate accuracy/F1."""
        core_mask = (targets != -100).all(dim=-1)
        if not core_mask.any():
            return
        tgt = targets.clamp_min(0.0)
        probs = torch.sigmoid(logits)  # [B, L, N]

        all_correct = torch.ones(core_mask.sum(), device=logits.device, dtype=torch.bool)
        for i in range(self.config.num_labels):
            lname = self.config.label_names[i]
            preds_i = (probs[..., i][core_mask] > 0.5).float()
            tgts_i = tgt[..., i][core_mask]
            acc_i = (preds_i == tgts_i).float().mean()
            self.log(f"{prefix}/{lname}_accuracy", acc_i)

            all_correct &= (preds_i == tgts_i)

            with torch.no_grad():
                p_np = preds_i.cpu().numpy().astype(int)
                t_np = tgts_i.cpu().numpy().astype(int)
                if len(np.unique(t_np)) > 1:
                    self.log(
                        f"{prefix}/{lname}_precision",
                        precision_score(t_np, p_np, zero_division=0),
                    )
                    self.log(
                        f"{prefix}/{lname}_recall",
                        recall_score(t_np, p_np, zero_division=0),
                    )
                    self.log(
                        f"{prefix}/{lname}_f1",
                        f1_score(t_np, p_np, zero_division=0),
                    )

        self.log(
            f"{prefix}/all_labels_accuracy",
            all_correct.float().mean(),
            prog_bar=True,
        )

    # -------------------------------------------------------------------------
    # Lightning steps
    # -------------------------------------------------------------------------

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        cache = self._load_hidden_state(batch) if self.config.use_hidden_cache else None
        out = self.model(
            input_ids=batch["input_ids"],
            cache_params=cache,
            frame_phase=batch.get("frame_phase"),
        )
        logits = out.frame_logits

        loss, components = self._compute_loss(logits, batch["targets"], "train")
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.config.detailed_logging:
            for name, val in components.items():
                if name != "total_label_loss":
                    self.log(f"train/{name}", val, on_step=False, on_epoch=True)

        if self.config.use_hidden_cache and cache is not None:
            self._update_hidden_cache(batch, cache)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        out = self.model(
            input_ids=batch["input_ids"],
            frame_phase=batch.get("frame_phase"),
        )
        logits = out.frame_logits
        loss, _ = self._compute_loss(logits, batch["targets"], "val")
        self.log("val/loss", loss, prog_bar=True)
        with torch.no_grad():
            self._compute_metrics(logits, batch["targets"], "val")
        return loss

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Accumulates TP/FP/FN for epoch-level F1."""
        out = self.model(
            input_ids=batch["input_ids"],
            frame_phase=batch.get("frame_phase"),
        )
        logits = out.frame_logits
        loss, _ = self._compute_loss(logits, batch["targets"], "test")
        self.log("test/loss", loss)
        with torch.no_grad():
            self._compute_metrics(logits, batch["targets"], "test")
            core_mask = (batch["targets"] != -100).all(dim=-1)
            if core_mask.any():
                probs = torch.sigmoid(logits[core_mask]).cpu()
                targets = batch["targets"][core_mask].clamp_min(0).cpu().int()
                for i in range(self.config.num_labels):
                    preds_i = (probs[:, i] > 0.5).int()
                    tgts_i = targets[:, i]
                    self._test_tp[i] += (preds_i & tgts_i).sum()
                    self._test_fp[i] += (preds_i & (1 - tgts_i)).sum()
                    self._test_fn[i] += ((1 - preds_i) & tgts_i).sum()
        return loss

    def on_test_epoch_end(self) -> None:
        """Finalise per-label precision/recall/F1 from accumulated TP/FP/FN."""
        eps = 1e-8
        for i, lname in enumerate(self.config.label_names):
            tp = self._test_tp[i].float()
            fp = self._test_fp[i].float()
            fn = self._test_fn[i].float()
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            self.log(f"test/{lname}_final_f1", f1)
            self.log(f"test/{lname}_final_precision", precision)
            self.log(f"test/{lname}_final_recall", recall)
        self._test_tp.zero_()
        self._test_fp.zero_()
        self._test_fn.zero_()

    # -------------------------------------------------------------------------
    # Backbone freeze / unfreeze schedule
    # -------------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Freeze/unfreeze backbone per freeze_backbone_epochs."""
        freeze_until = self.config.freeze_backbone_epochs
        if freeze_until > 0:
            if self.current_epoch < freeze_until:
                for p in self.model.backbone.parameters():
                    p.requires_grad = False
                if self.current_epoch == 0:
                    print(f"[FREEZE] Backbone frozen for first {freeze_until} epochs.")
            elif self.current_epoch == freeze_until:
                for p in self.model.backbone.parameters():
                    p.requires_grad = True
                print(f"[FREEZE] Backbone unfrozen at epoch {self.current_epoch}.")

    def on_train_epoch_end(self) -> None:
        pass

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log the global grad norm."""
        total_norm = (
            sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        self.grad_norms.append(total_norm)
        self.log("train/grad_norm", total_norm, on_step=False, on_epoch=True)

    # -------------------------------------------------------------------------
    # Optimiser
    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        """AdamW with decoder at 2× backbone LR."""
        backbone_params = list(self.model.backbone.parameters())
        decoder_params = list(self.model.decoder.parameters())
        param_groups = [
            {"params": backbone_params, "lr": self.config.learning_rate},
            {"params": decoder_params, "lr": self.config.learning_rate * 2.0},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
            betas=tuple(self.config.betas),
            eps=self.config.eps,
        )

        if self.config.warmup_epochs > 0:
            t_max = self.config.cosine_t_max or max(
                1, self.trainer.max_epochs - self.config.warmup_epochs
            )
            warmup = LinearLR(
                optimizer, start_factor=0.1, total_iters=self.config.warmup_epochs
            )
            cosine = CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=self.config.learning_rate * 0.01
            )
            sched = SequentialLR(optimizer, [warmup, cosine], [self.config.warmup_epochs])
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
            }

        if self.config.use_scheduler:
            sched = ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=3)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }
        return optimizer

    # -------------------------------------------------------------------------
    # Hidden-state cache
    # -------------------------------------------------------------------------

    def _load_hidden_state(self, batch: Dict) -> Optional[Any]:
        """Assemble batched InferenceParams from the LRU cache."""
        requires = torch.as_tensor(
            batch.get("requires_hidden_state", [False]), dtype=torch.bool
        )
        B = int(batch["input_ids"].size(0))
        if not requires.any():
            return None
        dev = self.device
        ip = InferenceParams(max_seqlen=self.config.chunk_size, max_batch_size=B)
        ip.key_value_memory_dict = {}

        def ensure_layer(k, cv, sv):
            if k not in ip.key_value_memory_dict:
                ip.key_value_memory_dict[k] = (
                    torch.zeros((B,) + cv.shape[1:], device=dev, dtype=cv.dtype),
                    torch.zeros((B,) + sv.shape[1:], device=dev, dtype=sv.dtype),
                )
            return ip.key_value_memory_dict[k]

        for b in range(B):
            if not bool(requires[b]):
                continue
            key = f"{batch['sequence_ids'][b]}_{batch['previous_chunk_ids'][b]}"
            saved = self.hidden_cache.get(key)
            if saved is None:
                continue
            for k, (cv, sv) in saved.key_value_memory_dict.items():
                cb, sb = ensure_layer(k, cv, sv)
                cb[b].copy_(cv.to(dev).squeeze(0))
                sb[b].copy_(sv.to(dev).squeeze(0))
        return ip

    def _update_hidden_cache(self, batch: Dict, cache_params: Any) -> None:
        """Store per-sample hidden states into the LRU cache."""
        if cache_params is None or not cache_params.key_value_memory_dict:
            return
        B = int(batch["input_ids"].size(0))
        for b in range(B):
            key = f"{batch['sequence_ids'][b]}_{batch['chunk_ids'][b]}"
            single = InferenceParams(
                max_seqlen=self.config.chunk_size, max_batch_size=1
            )
            single.key_value_memory_dict = {
                k: (
                    cv[b : b + 1].detach().cpu().contiguous(),
                    sv[b : b + 1].detach().cpu().contiguous(),
                )
                for k, (cv, sv) in cache_params.key_value_memory_dict.items()
            }
            if (
                key not in self.hidden_cache
                and len(self.hidden_cache) >= self.config.max_cache_size
            ):
                self.hidden_cache.popitem(last=False)
            self.hidden_cache[key] = single
            self.hidden_cache.move_to_end(key)


# =============================================================================
# Train function
# =============================================================================

def train(config: PlantConfig) -> str:
    """Run training; returns the best checkpoint path."""
    from plant_eval import PlantEvalCallback
    from pytorch_lightning import seed_everything

    # Seed Python, NumPy, and PyTorch (plus CUDA) for reproducibility.
    seed_everything(config.seed, workers=True)

    dm = PlantDataModule(
        data_path=config.data_path,
        num_labels=config.num_labels,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        rc_prob=config.rc_prob,
        test_size=config.test_size,
        val_size=config.val_size,
        seed=config.seed,
        max_active_sequences=config.max_active_sequences,
        max_sequences=config.max_sequences,
        min_chunks_per_seq=config.min_chunks_per_seq,
        skip_validation=config.skip_validation,
        organism_split=config.organism_split,
    )

    if config.pretrained_checkpoint:
        model = PlantTrainer.from_smorfeus_checkpoint(config)
    else:
        model = PlantTrainer(config)

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_p:,} total, {trainable_p:,} trainable")
    print(f"Labels ({config.num_labels}): {config.label_names}")
    print(f"Label weights: {config.label_weights}")

    ckpt_cb = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=config.save_top_k,
        filename="plant_{epoch:02d}-{val/loss:.4f}-{val/all_labels_accuracy:.4f}",
    )
    early_stop = EarlyStopping(
        monitor="val/loss", patience=config.patience, mode="min", verbose=True
    )
    eval_cb = PlantEvalCallback(
        outdir=config.evaluation_save_dir,
        label_names=config.label_names,
    )

    trainer = Trainer(
        max_epochs=config.max_epochs,
        precision=config.precision,
        enable_progress_bar=config.enable_progress_bar,
        accelerator="gpu",
        log_every_n_steps=50,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[ckpt_cb, early_stop, eval_cb],
        check_val_every_n_epoch=config.check_val_every_n_epoch,
    )

    if config.test_only:
        if not config.test_checkpoint:
            raise ValueError("--test_checkpoint must be set when using --test_only")
        print(f"[TEST ONLY] Loading checkpoint: {config.test_checkpoint}")
        model = PlantTrainer.load_from_checkpoint(
            config.test_checkpoint, config=config
        )
        print("\n=== Test evaluation ===")
        trainer.test(model, dm)
        return config.test_checkpoint

    if config.pretrained_checkpoint:
        model = PlantTrainer.from_smorfeus_checkpoint(config)
    else:
        model = PlantTrainer(config)

    print("\n=== Starting plant fine-tuning ===")
    trainer.fit(model, dm, ckpt_path=config.resume_checkpoint)

    print("\n=== Test evaluation ===")
    trainer.test(model, dm)
    return ckpt_cb.best_model_path


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Plant genome annotation fine-tuning")
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--preset", default="default", help="Config preset name")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="smORFeus checkpoint to fine-tune from",
    )
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--max_epochs", default=None, type=int)
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--test_checkpoint", type=str, default=None)
    parser.add_argument(
        "--resume_checkpoint",
        default=None,
        help="Plant checkpoint to resume training from",
    )
    args = parser.parse_args()

    config = load_config(args.config or args.preset)

    if args.checkpoint:
        config.pretrained_checkpoint = args.checkpoint
    if args.data_path:
        config.data_path = args.data_path
    if args.max_epochs:
        config.max_epochs = args.max_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.test_only:
        config.test_only = args.test_only
    if args.test_checkpoint:
        config.test_checkpoint = args.test_checkpoint
    if args.resume_checkpoint:
        config.resume_checkpoint = args.resume_checkpoint

    issues = [
        i
        for i in config.validate()
        if "data_path" not in i or Path(config.data_path).exists()
    ]
    if issues:
        print("Config issues:")
        for i in issues:
            print(f"  {i}")

    train(config)


if __name__ == "__main__":
    main()
