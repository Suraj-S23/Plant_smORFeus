"""Plant eval: PlantEvalCallback collects test-epoch probabilities and writes
per-label figures (ROC/AUC, PR/AP, row-normalised confusion, genomic overlay).
evaluate_from_checkpoint() runs the same pipeline standalone.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import (
    auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve,
)

try:
    plt.rcParams["text.usetex"] = True
except Exception:
    plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 8
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["text.usetex"] = False

# Colour palette, one colour per label (7 total)
_LABEL_COLOURS = [
    "#4e79a7",  # protein_coding_gene  — blue
    "#f28e2b",  # five_prime_UTR       — orange
    "#e15759",  # three_prime_UTR      — red
    "#76b7b2",  # exon                 — teal
    "#59a14f",  # intron               — green
    "#edc948",  # splice_donor         — yellow
    "#b07aa1",  # splice_acceptor      — purple
]


def _save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", dpi=150)


def _extract_runs(x: np.ndarray) -> List[tuple]:
    """(start, end) pairs for runs of 1s."""
    runs, s = [], None
    for i, v in enumerate(x.astype(int)):
        if v == 1 and s is None:
            s = i
        elif v == 0 and s is not None:
            runs.append((s, i))
            s = None
    if s is not None:
        runs.append((s, len(x)))
    return runs


class PlantEvalCallback(Callback):
    """Accumulates test-epoch probabilities and writes per-label figures."""

    def __init__(
        self,
        outdir: str = "plant_eval_figures",
        threshold: float = 0.5,
        overlay_panels: int = 3,
        label_names: List[str] = None,
    ):
        super().__init__()
        self.outdir = Path(outdir)
        self.threshold = threshold
        self.overlay_panels = overlay_panels
        self.label_names = label_names or [f"label_{i}" for i in range(7)]
        self._nl = len(self.label_names)
        self._reset()

    def _reset(self):
        self.probs: List[List[np.ndarray]] = [[] for _ in range(self._nl)]
        self.tgts: List[List[np.ndarray]] = [[] for _ in range(self._nl)]
        self.overlay_examples: List[Dict] = []

    def on_test_start(self, trainer, pl_module):
        self._nl = len(pl_module.config.label_names)
        self.label_names = pl_module.config.label_names
        self._reset()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch,
                          batch_idx, dataloader_idx=0):
        with torch.no_grad():
            out = pl_module.model(
                input_ids=batch["input_ids"].to(pl_module.device),
                frame_phase=batch.get("frame_phase"),
            )
            logits = out.frame_logits       # [B,L,N]
            targets = batch["targets"].to(pl_module.device)  # [B,L,N]
            core = (targets != -100).all(dim=-1)  # [B,L]
            if not core.any():
                return

            probs_all = torch.sigmoid(logits)
            tgt_all = targets.clamp_min(0.0)

            core_flat = core.view(-1)
            p_flat = probs_all.view(-1, self._nl)[core_flat]
            t_flat = tgt_all.view(-1, self._nl)[core_flat]

            for i in range(self._nl):
                self.probs[i].append(p_flat[:, i].cpu().numpy())
                self.tgts[i].append(t_flat[:, i].cpu().numpy())

            if len(self.overlay_examples) < self.overlay_panels:
                idx = core.sum(dim=1).argmax().item()
                mask = core[idx].nonzero(as_tuple=False).squeeze(1)
                s, e = int(mask[0]), int(mask[-1]) + 1
                self.overlay_examples.append({
                    "true": tgt_all[idx, s:e].cpu().numpy(),  # [L_core, N]
                    "pred": (probs_all[idx, s:e] > self.threshold).cpu().numpy(),
                    "seq_id": str(batch["sequence_ids"][idx])
                              if "sequence_ids" in batch else f"seq{batch_idx}",
                    "chunk_id": str(batch["chunk_ids"][idx])
                                if "chunk_ids" in batch else f"chunk{batch_idx}",
                })

    def on_test_epoch_end(self, trainer, pl_module):
        probs = [np.concatenate(x) if x else np.array([]) for x in self.probs]
        tgts = [np.concatenate(x) if x else np.array([]) for x in self.tgts]

        self._plot_roc(probs, tgts)
        self._plot_pr(probs, tgts)
        self._plot_confusion(probs, tgts)
        self._plot_overlays()

        print(f"[EVAL] Figures saved to {self.outdir}/")

    def _plot_roc(self, probs, tgts):
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        for i, name in enumerate(self.label_names):
            y, p = tgts[i], probs[i]
            if p.size == 0 or len(np.unique(y)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=_LABEL_COLOURS[i % len(_LABEL_COLOURS)],
                    label=f"{name} (AUC={auc(fpr,tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC - Plant Annotation")
        ax.legend(fontsize=6, loc="lower right")
        _save(fig, self.outdir, "plant_roc")
        plt.close(fig)

    def _plot_pr(self, probs, tgts):
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
        for i, name in enumerate(self.label_names):
            y, p = tgts[i], probs[i]
            if p.size == 0 or len(np.unique(y)) < 2:
                continue
            pr, rc, _ = precision_recall_curve(y, p)
            ax.plot(rc, pr, color=_LABEL_COLOURS[i % len(_LABEL_COLOURS)],
                    label=f"{name} (AP={auc(rc,pr):.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall - Plant Annotation")
        ax.legend(fontsize=6, loc="upper right")
        _save(fig, self.outdir, "plant_pr")
        plt.close(fig)

    def _plot_confusion(self, probs, tgts):
        nl = self._nl
        cols = min(4, nl)
        rows = (nl + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 2.8, rows * 2.6))
        axes = np.array(axes).flatten()
        for i, name in enumerate(self.label_names):
            ax = axes[i]
            y, p = tgts[i], probs[i]
            if p.size == 0:
                ax.set_visible(False)
                continue
            yhat = (p >= self.threshold).astype(int)
            cm = confusion_matrix(y, yhat, labels=[0, 1], normalize="true")
            disp = ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"])
            disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
            ax.set_title(name, fontsize=7)
        for j in range(nl, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Confusion Matrices (row-normalised)", fontsize=9)
        fig.tight_layout()
        _save(fig, self.outdir, "plant_confusion")
        plt.close(fig)

    def _plot_overlays(self):
        for ex_idx, ex in enumerate(self.overlay_examples):
            true_mat = ex["true"].astype(int)  # [L_core, N]
            pred_mat = ex["pred"].astype(int)
            L = true_mat.shape[0]
            nl = min(self._nl, true_mat.shape[1])

            fig, axes = plt.subplots(nl, 1, figsize=(6.0, nl * 0.9 + 0.6),
                                     sharex=True)
            if nl == 1:
                axes = [axes]

            for i in range(nl):
                ax = axes[i]
                name = self.label_names[i]
                t_r = _extract_runs(true_mat[:, i])
                p_r = _extract_runs(pred_mat[:, i])

                ax.set_xlim(0, L)
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.set_ylabel(name, fontsize=6, rotation=0,
                              ha="right", va="center")
                for s, e in t_r:
                    ax.add_patch(Rectangle((s, 0.55), e-s, 0.38,
                                           fc="#a0d99b", ec="none",
                                           alpha=0.95, zorder=3))
                for s, e in p_r:
                    ax.add_patch(Rectangle((s, 0.07), e-s, 0.38,
                                           fc="#f2b766", ec="none",
                                           alpha=0.90, zorder=2))
                ax.grid(axis="x", lw=0.3, alpha=0.4, zorder=1)

            axes[-1].set_xlabel("Position in chunk")
            true_h = Rectangle((0, 0), 1, 1, fc="#a0d99b", ec="none", alpha=0.95)
            pred_h = Rectangle((0, 0), 1, 1, fc="#f2b766", ec="none", alpha=0.90)
            axes[0].legend([true_h, pred_h], ["True", "Predicted"],
                           fontsize=7, frameon=False, loc="upper right",
                           ncol=2, handlelength=1.0)
            seq_id = ex.get("seq_id", f"seq{ex_idx}")
            chunk_id = ex.get("chunk_id", f"chunk{ex_idx}")
            fig.suptitle(f"{seq_id}  {chunk_id}", fontsize=8)
            fig.tight_layout()
            fname = f"plant_overlay_{seq_id}_{chunk_id}".replace("/", "_")
            _save(fig, self.outdir, fname)
            plt.close(fig)


def evaluate_from_checkpoint(
    checkpoint_path: str,
    data_path: str,
    outdir: str = "plant_eval_figures",
    batch_size: int = 4,
    num_workers: int = 4,
):
    from plant_config import PlantConfig
    from helpers import PlantDataModule
    from plant_train import PlantTrainer

    print(f"Loading checkpoint: {checkpoint_path}")
    model = PlantTrainer.load_from_checkpoint(checkpoint_path)
    model.eval()

    cfg = model.config
    cfg.data_path = data_path

    dm = PlantDataModule(
        data_path=data_path,
        num_labels=cfg.num_labels,
        batch_size=batch_size,
        num_workers=num_workers,
        chunk_size=cfg.chunk_size,
        test_size=cfg.test_size,
        val_size=cfg.val_size,
        seed=cfg.seed,
        organism_split=cfg.organism_split,
    )

    from pytorch_lightning import Trainer as PLTrainer
    eval_cb = PlantEvalCallback(outdir=outdir, label_names=cfg.label_names)
    trainer = PLTrainer(
        accelerator="auto",
        callbacks=[eval_cb],
        enable_progress_bar=True,
    )
    trainer.test(model, dm)
    print(f"Evaluation figures saved to: {outdir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate plant annotation checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--outdir", default="plant_eval_figures")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    args = parser.parse_args()
    evaluate_from_checkpoint(
        args.checkpoint, args.data_path, args.outdir,
        args.batch_size, args.num_workers,
    )