---
name: uv-ml
description: Use this skill for anything related to managing Python environments and dependencies with uv (astral-uv) in a ML project context. Trigger on: creating a new uv project, adding/removing packages, syncing environments, managing Python versions, resolving dependency conflicts, setting up CUDA-specific torch versions, creating a pyproject.toml for a ML project, running scripts with uv, uv workspace setups, or any question about uv commands and workflows. Also trigger when the user has pip/conda commands they want to convert to uv equivalents.
---

# uv — ML Project Package Management

Stack context: astral-uv 0.4+, Python 3.11+, PyTorch with CUDA, WSL environment.

---

## New ML project

```bash
uv init my-project          # creates pyproject.toml + .python-version + hello.py
cd my-project
uv python pin 3.11          # pins Python version in .python-version
uv sync                     # creates .venv and installs deps
```

Or init without the example file: `uv init --no-readme --no-pin-python my-project`

## pyproject.toml for ML

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.4",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "wandb>=0.18",
    "trackio>=0.1",
    "numpy>=2.0",
    "scikit-learn>=1.5",
    "pandas>=2.2",
    "matplotlib>=3.8",
    "seaborn>=0.13",
]

[project.optional-dependencies]
dev = [
    "ipykernel",
    "jupyter",
    "pytest>=8",
]
sweep = [
    "hydra-optuna-sweeper>=1.2",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Install with dev extras: `uv sync --extra dev`

## PyTorch with CUDA (WSL / RTX 5090)

uv doesn't natively handle PyTorch index URLs the same way pip does. Two approaches:

**Approach 1 — extra index in pyproject.toml (recommended):**

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu124" }

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true
```

Then just `torch>=2.4` in dependencies, uv will pull from the CUDA index.

**Approach 2 — direct URL:**
```toml
[tool.uv.sources]
torch = { url = "https://download.pytorch.org/whl/cu124/torch-2.4.0%2Bcu124-cp311-cp311-linux_x86_64.whl" }
```

After modifying sources, run `uv sync` to reinstall.

**Verify CUDA torch:**
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## Daily commands

```bash
uv add numpy pandas           # add packages
uv add --dev pytest ipykernel # add dev dependencies
uv remove old-package         # remove
uv sync                       # sync env to lockfile (after git pull, etc.)
uv lock --upgrade-package wandb  # upgrade a specific package
uv lock --upgrade             # upgrade all (regenerates uv.lock)

uv run python train.py        # run in project env
uv run python train.py train.lr=1e-4  # with Hydra override
uv run pytest                 # run tests

uv run jupyter notebook       # launch notebook in project env
```

## Running scripts without activating env

You don't need `source .venv/bin/activate`. Use `uv run` prefix for everything.

If you do want to activate (e.g. for interactive shell):
```bash
source .venv/bin/activate
# then plain: python train.py, pytest, etc.
```

## Python version management

```bash
uv python install 3.11        # install a Python version
uv python list                # list available/installed versions
uv python pin 3.11            # pin for this project (writes .python-version)
```

uv downloads Python automatically — no system Python needed.

## Inline scripts (one-off analysis)

For quick scripts that don't need a full project:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "matplotlib"]
# ///

import pandas as pd
import matplotlib.pyplot as plt
# ...
```

Run with: `uv run my_analysis.py` — uv creates an isolated env on the fly.

## Common issues

**`uv sync` fails after adding torch with CUDA index**
Make sure the `[[tool.uv.index]]` block and `[tool.uv.sources]` are both present. Run `uv sync --reinstall-package torch` to force reinstall.

**Package not found / wrong version**
Check `uv.lock` to see what's actually resolved. Run `uv tree` to see the full dependency tree.

**WSL: uv not found after install**
Add to `~/.bashrc`: `export PATH="$HOME/.local/bin:$PATH"`, then `source ~/.bashrc`.

**Conflict between two packages**
Run `uv add package --verbose` to see the resolution process. Often means pinning one dep: `uv add "numpy<2"`.

## pip / conda equivalents

| pip/conda | uv |
|-----------|-----|
| `pip install X` | `uv add X` |
| `pip install -r requirements.txt` | `uv add -r requirements.txt` |
| `conda create -n env python=3.11` | `uv init + uv python pin 3.11` |
| `conda activate env` | `source .venv/bin/activate` |
| `pip freeze > requirements.txt` | `uv.lock` (auto-managed) |
| `python -m venv .venv` | handled automatically by uv |

## Project structure with uv

```
my-project/
├── .python-version     # e.g. "3.11"
├── pyproject.toml
├── uv.lock             # lockfile — commit this to git
├── .venv/              # local env — gitignore this
├── src/
│   └── my_project/
└── train.py
```

`.gitignore` should include: `.venv/`, `__pycache__/`, `outputs/`, `multirun/`, `*.pyc`
