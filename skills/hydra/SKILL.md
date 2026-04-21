---
name: hydra
description: Use this skill for anything Hydra-config related. Trigger on: structuring Hydra configs, config composition, config groups, overrides, multirun/sweeps, Hydra instantiate, structured configs with dataclasses, debugging Hydra errors, interpolations, resolvers, or any question about managing experiment configurations. Also trigger when the user wants to add a new config group, refactor their conf/ directory, or set up Optuna/Ax sweepers.
---

# Hydra Config — Reference & Patterns

Stack context: Hydra 1.3+, OmegaConf 2.3+, used alongside PyTorch training loops.

---

## Config composition

Root `conf/config.yaml` uses a defaults list — this is the backbone of composition:

```yaml
defaults:
  - model: base          # loads conf/model/base.yaml
  - train: base
  - data: base
  - _self_               # _self_ last = root keys override group keys

project_name: my_exp
seed: 42
```

Adding a new variant (e.g. a transformer model):
- Create `conf/model/transformer.yaml`
- Override at CLI: `python train.py model=transformer`

## Config groups

Organize by what varies across experiments:

```
conf/
├── config.yaml
├── model/
│   ├── base.yaml        # MLP or simple baseline
│   └── transformer.yaml
├── train/
│   ├── base.yaml        # default training hparams
│   └── fast.yaml        # quick debug run (1 epoch, small batch)
└── data/
    ├── base.yaml
    └── tiny.yaml        # small dataset for dev
```

**tip:** always keep a `fast.yaml` / `debug.yaml` variant in train/ for quick iteration without touching base configs.

## Structured configs (typed, autocomplete-friendly)

Define config schema with dataclasses — catches typos at startup, enables IDE autocomplete:

```python
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1

@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    grad_clip: float = 1.0

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    project_name: str = "my_exp"
    seed: int = 42

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
```

Then in train.py: `@hydra.main(config_path="conf", config_name="config", version_base=None)`

## CLI overrides

```bash
# single override
python train.py train.lr=3e-4

# swap config group
python train.py model=transformer

# append to a list
python train.py +train.extra_flag=true

# override a nested key that doesn't exist yet
python train.py ++new.key=value

# delete a key
python train.py ~train.scheduler=null
```

## Multirun / sweeps

```bash
# grid search
python train.py --multirun train.lr=1e-3,3e-4,1e-4 model.hidden_dim=128,256

# with Optuna sweeper (add hydra-optuna-sweeper to deps)
python train.py --multirun \
  hydra/sweeper=optuna \
  hydra.sweeper.n_trials=20 \
  'train.lr=tag(log, interval(1e-5, 1e-2))' \
  'model.hidden_dim=choice(128, 256, 512)'
```

Each run gets its own output dir: `outputs/YYYY-MM-DD/HH-MM-SS/` (or `multirun/` for sweeps).

## Hydra instantiate

Avoid hardcoding class names — let the config pick the implementation:

```yaml
# conf/model/base.yaml
_target_: src.myproject.model.MLP
hidden_dim: 256
num_layers: 4
```

```python
from hydra.utils import instantiate
model = instantiate(cfg.model)  # calls MLP(hidden_dim=256, num_layers=4)
```

Works for optimizers, schedulers, datasets too — keeps train.py free of if/else chains.

## Interpolations & resolvers

```yaml
# reference another key
output_dir: ${hydra:runtime.output_dir}
run_name: ${project_name}_lr${train.lr}

# custom resolver
run_id: ${now:%Y%m%d_%H%M%S}
```

Register custom resolver:
```python
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
# then in yaml: total_steps: ${mul:${train.epochs},${train.steps_per_epoch}}
```

## Common errors

**`MissingMandatoryValue`** — key is `???` in yaml and wasn't overridden. Either set a default or pass it on CLI.

**`ConfigCompositionException`** — usually a missing file in defaults list or a typo in group name. Check the path in `conf/`.

**`ValidationError` (structured configs)** — type mismatch (e.g. passing a float where int expected). Check dataclass field types.

**Output dir issues** — by default Hydra changes cwd to the output dir. If relative paths break, use `hydra.utils.get_original_cwd()` or set `hydra.job.chdir=False` (Hydra 1.2+).

## Accessing config outside @hydra.main

```python
from hydra import compose, initialize

# useful for notebooks or tests
with initialize(config_path="conf", version_base=None):
    cfg = compose(config_name="config", overrides=["train.lr=1e-4"])
```

## Tips for this stack

- Log `OmegaConf.to_yaml(cfg)` at the start of each run so W&B/Trackio captures the full resolved config
- Use `hydra.job.name` as the W&B run name for easy cross-referencing
- Keep `conf/` in git, keep `outputs/` and `multirun/` in `.gitignore`
