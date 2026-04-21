---
name: ml-run
description: Use this skill when the user is in the middle of an ML experiment and needs help debugging, tuning, or understanding results. Trigger on: loss not converging, training instability, GPU/CUDA errors, asking for Hydra sweep configs, W&B or Trackio logging issues, OOM errors, slow training, NaN loss, overfitting/underfitting, checkpoint issues, or any question about an ongoing training run. Also trigger when the user shares training curves, loss values, or metric results and wants interpretation or next steps.
---

# ML Run — Experiment Debugging & Guidance

Help the user debug, tune, and understand ongoing ML experiments (Hydra + PyTorch + W&B/Trackio stack).

## Approach

First, understand the context. Ask for or look at:
- The Hydra config (or the relevant parts)
- The error message or symptom
- Recent W&B/Trackio logs or loss curves if available

Then diagnose systematically. Don't guess — identify the most likely cause first.

---

## Common issues and how to handle them

### NaN loss
Likely causes in order: exploding gradients → bad learning rate → bad data (inf/nan in inputs) → fp16 overflow.

Check: Is `grad_clip` set? What's the lr? Is there a scheduler?
Fix suggestions: add/lower grad_clip, lower lr, add `torch.nn.utils.clip_grad_norm_`, check data pipeline with `assert not torch.isnan(batch).any()`.

### Loss not decreasing
Ask: Is val loss also flat, or just train? How many steps before judging?

Likely causes: lr too low or too high, model not reaching data (check dataloader shuffle), optimizer not stepped, wrong loss function for the task.

Quick diagnostic: suggest logging gradient norms to W&B/Trackio to see if gradients are flowing.

### OOM (CUDA out of memory)
Systematically reduce memory:
1. Halve batch_size first
2. Enable gradient checkpointing: `torch.utils.checkpoint`
3. Switch to `bfloat16`: `model = model.to(torch.bfloat16)` (RTX 5090 / Blackwell handles bfloat16 natively)
4. Check for tensor accumulation in loops (detach before appending to lists)
5. `torch.cuda.empty_cache()` between val and train (but not in the loop)

### Slow training
On RTX 5090 (sm_100 / Blackwell arch):
- `torch.compile(model)` — biggest single win, requires PyTorch 2.4+
- `torch.set_float32_matmul_precision('high')` — enables TF32
- `pin_memory=True, persistent_workers=True` in DataLoader
- `num_workers` = number of CPU cores / 2 (good starting point in WSL)
- Check if bottleneck is data loading: log batch load time vs forward pass time

### Overfitting
Suggest in order: more dropout, weight decay (add to optimizer), data augmentation, reduce model size, early stopping based on val metric.

### Hydra sweep help
When user wants to sweep hyperparams, generate the multirun command:
```bash
uv run python train.py --multirun \
  train.lr=1e-3,3e-4,1e-4 \
  train.batch_size=32,64 \
  model.dropout=0.1,0.3
```
Or suggest Hydra Optuna sweeper if they want Bayesian search (add `hydra-optuna-sweeper` dep).

### W&B / Trackio logging issues
- Run not appearing: check `wandb.init()` / `trackio.init()` is called before any `log()`, and that `finish()` is called at end
- Metrics not updating: confirm `step` arg is consistent and incrementing
- Config not logged: use `OmegaConf.to_container(cfg, resolve=True)` when passing to init
- WSL-specific: W&B might need `WANDB_MODE=offline` if network is flaky, then `wandb sync` later

### Checkpoint issues
- Loading on different device: always use `map_location` in `torch.load(..., map_location=device)`
- Partial load (architecture changed): use `strict=False` and log missing/unexpected keys
- Resuming training: restore optimizer state too, not just model weights

---

## WSL-specific gotchas

- CUDA in WSL2: make sure using Windows NVIDIA driver (not Linux driver inside WSL). Check: `nvidia-smi` should show GPU.
- File I/O: datasets on `/mnt/c/` are slow — move to `~/` (Linux filesystem) for training data
- num_workers > 0 with WSL: if getting DataLoader worker crashes, set `multiprocessing_start_method='spawn'` or reduce workers

---

## Response format

Give the diagnosis first (one sentence), then the fix. If multiple causes are possible, rank them. Always include the exact code change or command, not just a description. Keep it tight — the user is in the middle of a run.
