---
name: torch
description: Use this skill for PyTorch-specific patterns, idioms, and advanced usage with nn.Module. Trigger on: writing or debugging nn.Module subclasses, custom loss functions, custom optimizers, learning rate schedulers, gradient manipulation, mixed precision training, torch.compile, model checkpointing, weight initialization, parameter groups, hooks, model surgery (loading partial weights, freezing layers), inference optimization, or any PyTorch-specific question. Also trigger when the user wants to implement a specific architecture pattern (residual, attention, etc.) or debug tensor shape errors.
---

# PyTorch — Patterns & Reference

Stack context: PyTorch 2.4+, nn.Module, RTX 5090 (sm_100 Blackwell), Python 3.11+.

---

## nn.Module patterns

### Clean module definition

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

Prefer `nn.Sequential` for simple stacks. Use `nn.ModuleList` / `nn.ModuleDict` when you need dynamic indexing — never use plain Python lists for submodules (they won't be registered).

### Residual block

```python
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # skip connection
```

---

## Weight initialization

PyTorch defaults are often fine, but for deep networks:

```python
def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

model.apply(init_weights)
```

---

## Optimizer & scheduler

```python
# parameter groups — different lr for different parts
optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-4},
    {"params": model.head.parameters(), "lr": 1e-3},
], weight_decay=0.01)

# cosine with warmup (common in DL)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.train.warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs - cfg.train.warmup_steps)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.train.warmup_steps])
```

Call `scheduler.step()` after each epoch (or each step for step-level schedulers).

---

## Mixed precision (bfloat16)

bfloat16 is preferred over float16 on Blackwell (no loss scaling needed):

```python
# simplest approach — autocast
from torch.amp import autocast

for batch_x, batch_y in loader:
    batch_x = batch_x.to(device)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch_x)
        loss = criterion(output, batch_y)
    loss.backward()  # no GradScaler needed for bfloat16
    optimizer.step()
    optimizer.zero_grad()
```

For float16 (older GPUs), add `GradScaler`:
```python
scaler = torch.amp.GradScaler()
with autocast(device_type="cuda", dtype=torch.float16):
    loss = ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## torch.compile

Biggest single throughput win on RTX 5090. Profile before/after to confirm:

```python
model = MLP(...)
model = torch.compile(model, mode="reduce-overhead")  # or "max-autotune" for longer compile
# rest of training unchanged — compile is transparent
```

Modes:
- `"default"` — balanced compile time vs speedup
- `"reduce-overhead"` — faster, good for training
- `"max-autotune"` — slow compile, max runtime perf (use for long runs)

Known gotchas: first few steps are slow (compilation happening). Don't benchmark the first 5 steps. Not compatible with some dynamic shapes — if it errors, try `torch.compile(model, dynamic=True)`.

---

## Checkpointing

```python
def save_checkpoint(model, optimizer, scheduler, epoch, metric, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metric": metric,
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["metric"]
```

`weights_only=True` is safer (no arbitrary code execution from checkpoint).

### Partial weight loading (architecture changed)

```python
ckpt = torch.load(path, map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"Missing: {missing}")      # keys in model but not in checkpoint
print(f"Unexpected: {unexpected}") # keys in checkpoint but not in model
```

---

## Freezing / unfreezing layers

```python
# freeze encoder, train head only
for param in model.encoder.parameters():
    param.requires_grad = False

# unfreeze all
for param in model.parameters():
    param.requires_grad = True

# check what's trainable
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
```

---

## Custom loss functions

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()
```

---

## Debugging tensor shapes

```python
# quick shape trace — add temporarily, remove after
class ShapeDebug(nn.Module):
    def __init__(self, module, name=""):
        super().__init__()
        self.module = module
        self.name = name
    
    def forward(self, x):
        print(f"{self.name} input:  {x.shape}")
        out = self.module(x)
        print(f"{self.name} output: {out.shape}")
        return out

# wrap any layer temporarily
model.encoder = ShapeDebug(model.encoder, "encoder")
```

Common shape errors:
- `RuntimeError: mat1 and mat2 shapes cannot be multiplied` → Linear input dim mismatch. Print shapes before the layer.
- `Expected input batch_size to match target batch_size` → likely a missing `squeeze()` or wrong dim in loss.
- `CUDA error: device-side assert triggered` → usually invalid class indices in loss (check `targets.max()` vs `num_classes`). Run with `CUDA_LAUNCH_BLOCKING=1` to get the actual line.

---

## Useful one-liners

```python
# count parameters
sum(p.numel() for p in model.parameters()) / 1e6  # in millions

# gradient norm (for monitoring, not clipping)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
# or just compute it:
total_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None).sqrt()

# move entire model to bfloat16
model = model.to(torch.bfloat16)

# set all seeds for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np, random
    np.random.seed(seed)
    random.seed(seed)

# TF32 for faster matmul on Ampere/Blackwell (negligible accuracy loss)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True  # autotuned kernels (fixed input size only)
```
