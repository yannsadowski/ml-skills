---
name: viz
description: Use this skill when the user wants to visualize ML experiment results, training curves, metric comparisons, or data distributions. Trigger on: plotting loss curves, comparing runs, confusion matrices, feature importance, learning rate schedules, attention maps, embedding visualization (UMAP/t-SNE), seaborn heatmaps, matplotlib figures for papers/reports, or any visualization of W&B/Trackio exported data. Also trigger when the user asks how to make a plot look better, publication-ready, or wants to compare multiple experiments visually.
---

# ML Visualization — matplotlib & seaborn

Stack context: matplotlib 3.8+, seaborn 0.13+, results from W&B/Trackio exports or in-memory dicts. WSL environment (no display by default — use `Agg` backend or save to file).

---

## WSL setup (important)

```python
import matplotlib
matplotlib.use('Agg')  # must be before pyplot import in WSL
import matplotlib.pyplot as plt
```

Or set env var: `export MPLBACKEND=Agg` in `.bashrc`.

Always save figures with `fig.savefig(...)` instead of `plt.show()` in WSL. Use `plt.show()` only if running a Jupyter server with browser.

---

## Training curves

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")

def plot_training_curves(history: dict, save_path="training_curves.png"):
    """history = {"train_loss": [...], "val_loss": [...], "val_metric": [...]}"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    
    axes[1].plot(epochs, history["val_metric"], color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation Metric")
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## Comparing multiple runs

When the user has results from multiple Hydra configs or W&B runs:

```python
def plot_run_comparison(runs: dict[str, dict], metric="val_loss", save_path="comparison.png"):
    """runs = {"run_name": {"val_loss": [...], ...}, ...}"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    palette = sns.color_palette("tab10", n_colors=len(runs))
    
    for (name, history), color in zip(runs.items(), palette):
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], label=name, color=color)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## Loading from W&B export

```python
import wandb

api = wandb.Api()
runs = api.runs("entity/project")

data = {}
for run in runs:
    history = run.history(keys=["train_loss", "val_loss", "val_metric"])
    data[run.name] = {
        "train_loss": history["train_loss"].dropna().tolist(),
        "val_loss": history["val_loss"].dropna().tolist(),
        "val_metric": history["val_metric"].dropna().tolist(),
    }
```

## Hyperparameter sweep results

Visualize the effect of a hyperparameter across runs:

```python
def plot_hparam_sweep(results: list[dict], x_param: str, y_metric: str, save_path="sweep.png"):
    """results = [{"lr": 1e-3, "val_loss": 0.45}, ...]"""
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_param, y=y_metric, ax=ax, s=80)
    
    if df[x_param].nunique() > 5:  # continuous param — add trend line
        sns.regplot(data=df, x=x_param, y=y_metric, ax=ax, scatter=False, color="red", line_kws={"linewidth": 1})
    
    ax.set_title(f"{y_metric} vs {x_param}")
    if df[x_param].min() > 0 and df[x_param].max() / df[x_param].min() > 10:
        ax.set_xscale("log")  # log scale if range spans orders of magnitude
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## Confusion matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## Embedding visualization (UMAP / t-SNE)

```python
def plot_embeddings(embeddings, labels, method="umap", save_path="embeddings.png"):
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    
    coords = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels,
                         cmap="tab10", alpha=0.7, s=10)
    fig.colorbar(scatter, ax=ax)
    ax.set_title(f"{method.upper()} projection")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## Publication-ready style

For papers/reports — clean, minimal, LaTeX-friendly:

```python
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
```

For NeurIPS/ICML column width: `figsize=(3.5, 2.5)` for single column, `(7, 2.5)` for double.

## Logging figures to W&B / Trackio

```python
# W&B
wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

# or directly from fig
wandb.log({"training_curves": wandb.Image(fig)})

# Trackio
trackio.log({"confusion_matrix": trackio.Image("confusion_matrix.png")})
```

Log figures at end of training (not every epoch) to avoid bloating the run.
