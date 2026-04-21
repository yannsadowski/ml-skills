---
name: python-practices
description: Use this skill for Python best practices, code quality, and idiomatic Python in a ML/data science context. Trigger on: type hints, dataclasses, pathlib, logging setup, argparse vs Hydra, context managers, generators for large data, list comprehensions, f-strings, exception handling, project structure, __init__.py patterns, avoiding common Python antipatterns, writing testable ML code, or any question about how to write cleaner/more Pythonic code. Also trigger when reviewing or refactoring Python code for readability and correctness.
---

# Python Best Practices — ML Context

Stack context: Python 3.11+, uv, projects structured with src/ layout, Hydra for config.

---

## Type hints

Use them everywhere — they make refactoring safe and IDEs useful:

```python
from typing import Optional
from pathlib import Path
import torch
import torch.nn as nn

def build_model(hidden_dim: int, num_layers: int, dropout: float = 0.1) -> nn.Module:
    ...

def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, torch.Tensor]:
    ...

# for complex types, use type aliases
Batch = tuple[torch.Tensor, torch.Tensor]
MetricsDict = dict[str, float]

def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> MetricsDict:
    ...
```

Python 3.10+ union syntax: `int | None` instead of `Optional[int]`.

---

## Dataclasses for structured data (non-Hydra contexts)

```python
from dataclasses import dataclass, field

@dataclass
class TrainingResult:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    
    def update(self, train_loss: float, val_loss: float, epoch: int) -> None:
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
```

---

## Pathlib over os.path

```python
from pathlib import Path

# bad
import os
checkpoint_path = os.path.join(output_dir, "checkpoints", "best.pt")
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# good
checkpoint_path = Path(output_dir) / "checkpoints" / "best.pt"
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

# other useful methods
config_files = list(Path("conf").glob("**/*.yaml"))
data_path.exists()
data_path.stem          # filename without extension
data_path.suffix        # ".pt"
```

---

## Logging (not print)

```python
import logging

log = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# usage
log.info("Starting training: epochs=%d, lr=%.4f", cfg.train.epochs, cfg.train.lr)
log.warning("Checkpoint not found at %s, starting fresh", checkpoint_path)
log.debug("Batch shape: %s", batch.shape)  # only printed at DEBUG level
```

Use `log.info(...)` instead of `print(...)`. Plays well with Hydra's logging setup.

---

## Context managers for resource cleanup

```python
# bad — file might not close on exception
f = open("data.txt")
data = f.read()
f.close()

# good
with open("data.txt") as f:
    data = f.read()

# custom context manager for e.g. timing a block
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000
    log.info("%s took %.1f ms", name, elapsed)

with timer("data loading"):
    batch = next(iter(loader))
```

---

## Generators for large data

Don't load everything into memory when you can stream:

```python
# bad — loads all predictions into RAM
def predict_all(model, loader):
    all_preds = []
    for batch in loader:
        all_preds.extend(model(batch).tolist())
    return all_preds  # could be millions of items

# good — generator, process one batch at a time
def predict_stream(model, loader):
    model.eval()
    with torch.no_grad():
        for batch in loader:
            yield model(batch).cpu()

# usage
for preds in predict_stream(model, loader):
    save_predictions(preds)
```

---

## Common antipatterns to avoid

**Mutable default arguments:**
```python
# bad
def add_metric(name, values=[]):  # same list shared across all calls!
    values.append(name)

# good
def add_metric(name, values=None):
    if values is None:
        values = []
    values.append(name)
```

**Bare except:**
```python
# bad — catches KeyboardInterrupt, SystemExit, etc.
try:
    model.load_state_dict(torch.load(path))
except:
    pass

# good
try:
    model.load_state_dict(torch.load(path, map_location="cpu"))
except FileNotFoundError:
    log.warning("No checkpoint at %s", path)
except RuntimeError as e:
    log.error("Checkpoint incompatible: %s", e)
```

**String building in loops:**
```python
# bad
msg = ""
for metric, val in metrics.items():
    msg += f"{metric}: {val:.4f}, "

# good
msg = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
```

---

## Project structure (src layout)

```
my-project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── model.py       # nn.Module definitions
│       ├── dataset.py     # Dataset classes
│       ├── trainer.py     # Trainer class
│       └── utils.py       # shared helpers
├── tests/
│   ├── test_model.py
│   └── test_dataset.py
├── conf/                  # Hydra configs
├── train.py               # entry point
└── pyproject.toml
```

With src layout, install the package in dev mode: `uv add -e .` (or it's handled by `uv sync` if you have a build backend).

---

## Writing testable ML code

Separate the logic from the training loop so it can be tested without GPUs:

```python
# testable — takes tensors, returns tensors, no side effects
def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)

# test — runs on CPU, fast
def test_compute_loss():
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    loss = compute_loss(logits, targets)
    assert loss.shape == ()
    assert loss.item() > 0
```

```bash
uv run pytest tests/ -v
```

---

## Useful stdlib modules often overlooked

```python
from functools import partial, lru_cache
from itertools import islice, chain
from collections import defaultdict, Counter
import json, csv         # prefer over manual string parsing
import contextlib        # suppress, contextmanager
import tempfile          # for tests that write files
```

`lru_cache` for caching expensive pure functions (e.g., tokenization, feature extraction):
```python
@lru_cache(maxsize=10_000)
def tokenize(text: str) -> tuple[int, ...]:
    return tuple(tokenizer.encode(text))
```
