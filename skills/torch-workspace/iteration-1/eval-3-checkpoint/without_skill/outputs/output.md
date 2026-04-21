# Sauvegarde et rechargement de checkpoint PyTorch

## Fonctions `save_checkpoint` et `load_checkpoint`

```python
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path
from typing import Optional


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    loss: Optional[float] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Sauvegarde un checkpoint complet pour reprendre l'entraînement.

    Args:
        path      : chemin du fichier .pt à écrire
        model     : le modèle (entraîné sur GPU ou CPU)
        optimizer : l'optimiseur (état des moments, etc.)
        scheduler : le scheduler de lr (peut être None)
        epoch     : numéro de la dernière epoch terminée
        loss      : valeur de la loss (pour référence / early stopping)
        extra     : dict arbitraire pour stocker d'autres infos (metrics, config…)
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "loss": loss,
    }
    if extra is not None:
        checkpoint.update(extra)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint sauvegardé → {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Recharge un checkpoint et restaure les états du modèle, de l'optimiseur
    et du scheduler.

    Args:
        path      : chemin du fichier .pt à lire
        model     : instance du modèle (architecture identique à celle sauvegardée)
        optimizer : l'optimiseur à restaurer (optionnel si inference seulement)
        scheduler : le scheduler à restaurer (optionnel)
        device    : device cible (ex. torch.device("cuda:0") ou "cpu")
                    Si None, utilise le device courant du checkpoint.

    Returns:
        Le dictionnaire complet du checkpoint (contient epoch, loss, etc.)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # map_location garantit le bon chargement même si GPU/CPU change
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"] is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss  = checkpoint.get("loss", None)
    print(f"Checkpoint rechargé ← {path}  (epoch {epoch}, loss {loss})")

    return checkpoint
```

---

## Exemple d'utilisation complet

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- Définition du modèle (exemple minimal) ---
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.GELU(),
    nn.Linear(256, 10),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.CrossEntropyLoss()

# -----------------------------------------------
# BOUCLE D'ENTRAÎNEMENT avec sauvegarde
# -----------------------------------------------
CHECKPOINT_DIR = "./checkpoints"
NUM_EPOCHS = 50
SAVE_EVERY = 5  # sauvegarder tous les N epochs

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()

    # --- ton code d'entraînement ici ---
    inputs = torch.randn(32, 128, device=device)
    labels = torch.randint(0, 10, (32,), device=device)
    logits = model(inputs)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    # -----------------------------------

    if epoch % SAVE_EVERY == 0:
        save_checkpoint(
            path=f"{CHECKPOINT_DIR}/ckpt_epoch_{epoch:03d}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=loss.item(),
            extra={"lr": scheduler.get_last_lr()[0]},
        )

# Sauvegarde du dernier état
save_checkpoint(
    path=f"{CHECKPOINT_DIR}/ckpt_last.pt",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=NUM_EPOCHS,
    loss=loss.item(),
)

# -----------------------------------------------
# REPRISE DE L'ENTRAÎNEMENT depuis un checkpoint
# -----------------------------------------------
model_resumed = nn.Sequential(
    nn.Linear(128, 256),
    nn.GELU(),
    nn.Linear(256, 10),
)
optimizer_resumed = AdamW(model_resumed.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler_resumed = CosineAnnealingLR(optimizer_resumed, T_max=50)

ckpt = load_checkpoint(
    path=f"{CHECKPOINT_DIR}/ckpt_last.pt",
    model=model_resumed,
    optimizer=optimizer_resumed,
    scheduler=scheduler_resumed,
    device=device,
)

start_epoch = ckpt["epoch"] + 1  # reprendre à l'epoch suivante
print(f"Reprise depuis l'epoch {start_epoch}")

for epoch in range(start_epoch, NUM_EPOCHS + 10 + 1):
    model_resumed.train()
    # ... ton code d'entraînement ...
    pass
```

---

## Points importants

### `map_location` — portabilité GPU → CPU et GPU → autre GPU

```python
# Modèle entraîné sur GPU, rechargé sur CPU
ckpt = torch.load("ckpt.pt", map_location=torch.device("cpu"))

# Modèle entraîné sur GPU:0, rechargé sur GPU:1
ckpt = torch.load("ckpt.pt", map_location=torch.device("cuda:1"))

# Laisser PyTorch choisir automatiquement (recommandé)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("ckpt.pt", map_location=device)
```

### Ce qu'il faut TOUJOURS sauvegarder pour reprendre l'entraînement

| Élément | Pourquoi |
|---------|----------|
| `model.state_dict()` | Poids et biais du réseau |
| `optimizer.state_dict()` | Moments (m, v pour Adam), pas de gradient |
| `scheduler.state_dict()` | Compteur interne de steps, last_epoch |
| `epoch` | Pour reprendre à la bonne epoch |
| `loss` | Pour l'early stopping, les logs |

### Checkpoint "best model" (early stopping)

```python
best_val_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    val_loss = evaluate(model, val_loader)  # ta fonction d'évaluation

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            path="checkpoints/best_model.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=val_loss,
        )
        print(f"  -> Nouveau meilleur modèle (val_loss={val_loss:.4f})")
```

### Inférence seule (pas besoin de l'optimizer)

```python
model_inf = MyModel()
ckpt = load_checkpoint("checkpoints/best_model.pt", model=model_inf, device=device)
model_inf.eval()

with torch.no_grad():
    preds = model_inf(test_inputs)
```
