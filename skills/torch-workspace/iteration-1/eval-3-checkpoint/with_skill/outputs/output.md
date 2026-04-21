# Tâche 3 — Checkpointing complet : save & load pour reprendre l'entraînement

## Fonctions `save_checkpoint` et `load_checkpoint`

```python
import torch
import torch.nn as nn
from pathlib import Path


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,          # tout objet avec .state_dict() — LRScheduler, SequentialLR, etc.
    epoch: int,
    metric: float,      # ex. val_loss ou val_acc — pour savoir quel ckpt est le meilleur
) -> None:
    """
    Sauvegarde un checkpoint complet.

    Le modèle est toujours sauvegardé en CPU (map implicite via state_dict()),
    ce qui rend le fichier portable peu importe le GPU de destination.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metric": metric,
        },
        path,
    )
    print(f"Checkpoint sauvegardé : {path}  (epoch={epoch}, metric={metric:.4f})")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    device: str | torch.device = "cpu",
) -> tuple[int, float]:
    """
    Recharge un checkpoint et retourne (epoch, metric).

    Args:
        path:       chemin vers le fichier .pt/.pth
        model:      instance du modèle (architecture déjà instanciée)
        optimizer:  optionnel — à passer pour reprendre l'entraînement
        scheduler:  optionnel — à passer pour reprendre le scheduler
        device:     device cible pour charger les tenseurs
                    Utiliser "cpu" d'abord est plus sûr, puis .to(device) ensuite.

    Returns:
        (epoch, metric) du checkpoint
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    epoch  = ckpt["epoch"]
    metric = ckpt["metric"]
    print(f"Checkpoint chargé : {path}  (epoch={epoch}, metric={metric:.4f})")
    return epoch, metric
```

---

## Exemple d'utilisation complet dans une boucle d'entraînement

```python
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# ── Setup ──────────────────────────────────────────────────────────────────────
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR   = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

model     = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

warmup   = LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine   = CosineAnnealingLR(optimizer, T_max=95)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

# ── Reprendre depuis un checkpoint existant ────────────────────────────────────
RESUME_FROM = CKPT_DIR / "latest.pt"
start_epoch = 0
best_metric = float("inf")  # ou -inf si métrique à maximiser

if RESUME_FROM.exists():
    start_epoch, best_metric = load_checkpoint(
        path=RESUME_FROM,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    start_epoch += 1  # reprendre à l'epoch suivante

# ── Boucle d'entraînement ──────────────────────────────────────────────────────
NUM_EPOCHS = 100

for epoch in range(start_epoch, NUM_EPOCHS):
    model.train()
    train_loss = run_train_epoch(model, optimizer, device)  # ta fonction

    model.eval()
    val_loss = run_val_epoch(model, device)                 # ta fonction

    scheduler.step()

    # Sauvegarde systématique du dernier état
    save_checkpoint(
        path=CKPT_DIR / "latest.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metric=val_loss,
    )

    # Sauvegarde du meilleur modèle séparément
    if val_loss < best_metric:
        best_metric = val_loss
        save_checkpoint(
            path=CKPT_DIR / "best.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metric=val_loss,
        )
        print(f"  -> Nouveau meilleur modèle (val_loss={val_loss:.4f})")

    print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
```

---

## Charger uniquement les poids pour l'inférence

Si tu n'as pas besoin de reprendre l'entraînement — juste faire de l'inférence :

```python
# Pas besoin d'optimizer ni de scheduler
model = MyModel()
epoch, metric = load_checkpoint("checkpoints/best.pt", model, device="cpu")

model.eval()
model = model.to(device)

with torch.no_grad():
    output = model(x.to(device))
```

---

## Chargement partiel (architecture modifiée)

Si tu as modifié l'architecture depuis le dernier checkpoint :

```python
ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"Clés manquantes  (dans modèle, absentes du ckpt) : {missing}")
print(f"Clés inattendues (dans ckpt, absentes du modèle) : {unexpected}")
# Les couches manquantes gardent leur initialisation par défaut.
```

---

## Points clés à retenir

| Point | Pourquoi |
|---|---|
| `weights_only=True` dans `torch.load` | Sécurité : évite l'exécution de code arbitraire depuis le fichier `.pt` |
| `map_location="cpu"` au chargement | Portabilité : fonctionne même si le GPU de sauvegarde n'est plus dispo |
| `.to(device)` après `load_state_dict` | Les poids sont chargés en CPU, il faut les déplacer manuellement |
| Sauvegarder `epoch + 1` ou retourner `epoch` | Cohérence : `start_epoch += 1` dans la boucle de reprise |
| Deux fichiers : `latest.pt` + `best.pt` | `latest` pour reprendre, `best` pour le déploiement |
| Sauvegarder `scheduler.state_dict()` | Sans ça, le scheduler repart de zéro — le LR sera faux à la reprise |
