---
name: ml-setup
description: Use this skill when the user wants to start, scaffold, or bootstrap a new ML experiment or project. Trigger on phrases like "nouveau projet ML", "setup experiment", "créer un projet PyTorch", "scaffold ML", "nouveau training", "init experiment", or whenever the user wants to create a new machine learning project structure from scratch. Also trigger when setting up Hydra configs, integrating W&B or Trackio into a project, or creating a PyTorch training loop for the first time.
argument-hint: [project-name] [tracker: wandb|trackio]
---

# ML Experiment Scaffold

Scaffold a complete ML experiment using Hydra + PyTorch (nn.Module) + W&B or Trackio.

## What to generate

When invoked, collect any missing info (project name, tracker choice) then generate:

1. **Project structure** (see below)
2. **Hydra config files** — typed, composable
3. **`train.py`** — clean training loop
4. **`pyproject.toml`** — uv-ready

### Project structure

```
<project>/
├── conf/
│   ├── config.yaml          # root config (defaults list)
│   ├── model/
│   │   └── base.yaml
│   ├── train/
│   │   └── base.yaml
│   └── data/
│       └── base.yaml
├── src/
│   └── <project>/
│       ├── __init__.py
│       ├── model.py
│       ├── dataset.py
│       └── trainer.py
├── train.py
└── pyproject.toml
```

## Config files

`conf/config.yaml`:
```yaml
defaults:
  - model: base
  - train: base
  - data: base
  - _self_

project_name: <project>
seed: 42
tracker: wandb  # or trackio
```

`conf/model/base.yaml` — architecture hyperparams (hidden_dim, num_layers, dropout, etc.)

`conf/train/base.yaml` — lr, batch_size, epochs, optimizer (adam), scheduler (cosine), grad_clip

`conf/data/base.yaml` — data_path, num_workers, val_split

## train.py

Use `@hydra.main(config_path="conf", config_name="config", version_base=None)`.

Structure:
```python
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    init_tracker(cfg)      # W&B or Trackio based on cfg.tracker
    
    model = build_model(cfg.model).to(device)
    train_loader, val_loader = build_dataloaders(cfg.data)
    optimizer, scheduler = build_optimizer(model, cfg.train)
    
    trainer = Trainer(model, optimizer, scheduler, cfg.train)
    trainer.fit(train_loader, val_loader)
    
    finish_tracker()
```

## Tracker integration

Support both W&B and Trackio. Abstract behind thin helpers so the rest of the code is tracker-agnostic:

```python
def init_tracker(cfg):
    if cfg.tracker == "wandb":
        wandb.init(project=cfg.project_name, config=OmegaConf.to_container(cfg))
    elif cfg.tracker == "trackio":
        trackio.init(project=cfg.project_name)
        trackio.config(OmegaConf.to_container(cfg))

def log_metrics(metrics: dict, step: int):
    if tracker == "wandb":
        wandb.log(metrics, step=step)
    elif tracker == "trackio":
        trackio.log(metrics, step=step)

def finish_tracker():
    if tracker == "wandb": wandb.finish()
    elif tracker == "trackio": trackio.finish()
```

Log: train_loss, val_loss, val_metric, lr — every epoch minimum, train_loss every N steps.

## Trainer class (src/<project>/trainer.py)

```python
class Trainer:
    def fit(self, train_loader, val_loader):
        for epoch in range(self.cfg.epochs):
            train_loss = self._train_epoch(train_loader, epoch)
            val_loss, val_metric = self._val_epoch(val_loader)
            self.scheduler.step()
            log_metrics({...}, step=epoch)
            self._save_checkpoint(epoch, val_metric)
    
    def _train_epoch(self, loader, epoch):
        self.model.train()
        # gradient accumulation + optional grad clipping
        ...
    
    def _val_epoch(self, loader):
        self.model.eval()
        with torch.no_grad():
            ...
```

Include: gradient accumulation support, grad clipping from cfg, checkpoint saving (best + latest).

## Model (src/<project>/model.py)

Generate a clean `nn.Module` placeholder matching the cfg. Include `build_model(cfg)` factory function.

## pyproject.toml

```toml
[project]
name = "<project>"
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
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## RTX 5090 / CUDA considerations

- Default dtype: `torch.float32`, but add a note about `torch.bfloat16` for Blackwell
- Add `torch.compile(model)` as an optional line (commented out) — remind user it helps on sm_100
- Device detection: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`

## After generating

Tell the user:
- How to create the uv env: `uv sync`
- How to run: `uv run python train.py`
- How to override from CLI: `uv run python train.py train.lr=1e-3 model.hidden_dim=512`
- How to do a quick sweep: `uv run python train.py --multirun train.lr=1e-3,1e-4 model.hidden_dim=256,512`
