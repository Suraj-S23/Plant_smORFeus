# Checkpoints

Place model checkpoint files in this directory.

Expected files (not included in the repository; distributed separately):

- `smorfeus_pretrained.ckpt`: pretrained smORFeus backbone used as the
  starting point for fine-tuning. Pass to `plant_train.py` via
  `--checkpoint path/to/smorfeus_pretrained.ckpt`.
- `best.ckpt`: the fine-tuned plant annotation checkpoint produced by
  `plant_train.py`. Consumed by `plant_eval.py` and `plant_inference.py`
  via their `--checkpoint` flags.

Checkpoint files (`*.ckpt`, `*.pt`, `*.pth`) are git-ignored and should
never be committed.
