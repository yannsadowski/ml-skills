# Debug : CUDA error: device-side assert triggered — CrossEntropyLoss

## Diagnostic immédiat

L'erreur `CUDA error: device-side assert triggered` sur une cross-entropy loss est presque **toujours** causée par des **labels hors de la plage valide** pour le nombre de classes du modèle.

### Ton problème concret

- Tes labels vont de **1 à 10** (inclus)
- Ton modèle sort **10 logits** → indices valides : **0 à 9**
- `nn.CrossEntropyLoss` attend des labels dans `[0, num_classes - 1]`
- Le label `10` est **hors limites** → assert CUDA

**Correction : remapper tes labels de `[1..10]` vers `[0..9]`**

```python
labels = labels - 1  # [1..10] → [0..9]
```

---

## Protocole de débogage complet

### Étape 1 : Forcer l'exécution synchrone (CPU) pour obtenir la vraie stacktrace

Par défaut, les erreurs CUDA sont asynchrones — la vraie erreur se produit plusieurs lignes avant celle affichée.

```python
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# OU via la ligne de commande :
# CUDA_LAUNCH_BLOCKING=1 python train.py
```

Relance ton script : tu obtiendras la ligne exacte qui déclenche l'assert.

### Étape 2 : Valider les labels avant la loss

```python
def check_labels(labels: torch.Tensor, num_classes: int):
    print(f"Labels min: {labels.min().item()}, max: {labels.max().item()}")
    print(f"Plage attendue: [0, {num_classes - 1}]")
    assert labels.min() >= 0, f"Label négatif détecté: {labels.min().item()}"
    assert labels.max() < num_classes, (
        f"Label trop grand: {labels.max().item()} >= {num_classes}"
    )
    assert labels.dtype == torch.long, f"dtype incorrect: {labels.dtype} (attendu: torch.long)"

# Utilisation dans la boucle d'entraînement :
check_labels(labels, num_classes=10)
loss = criterion(logits, labels)
```

### Étape 3 : Checklist complète des causes fréquentes

| Cause | Symptôme | Fix |
|-------|----------|-----|
| Labels commencent à 1 | `max(labels) == num_classes` | `labels -= 1` |
| Label `-1` (valeur sentinelle) | `min(labels) < 0` | Filtrer ou utiliser `ignore_index=-1` |
| Label == `num_classes` | `max(labels) >= num_classes` | Vérifier le pipeline de données |
| dtype incorrect | `labels.dtype == torch.float32` | `labels = labels.long()` |
| NaN dans les logits | Loss → NaN → assert CUDA | Vérifier l'init, le lr, les normes |

### Étape 4 : Débogage sur CPU

Reproduire l'erreur sur CPU donne des messages d'erreur beaucoup plus lisibles :

```python
model_cpu = model.cpu()
logits_cpu = model_cpu(inputs.cpu())
loss = criterion(logits_cpu, labels.cpu())  # Erreur claire ici
```

### Étape 5 : Utiliser `ignore_index` si nécessaire

Si tu veux garder des labels à `0` comme classe "inconnue" ou `-1` comme "ignore" :

```python
criterion = nn.CrossEntropyLoss(ignore_index=-1)
# ou
criterion = nn.CrossEntropyLoss(ignore_index=0)
```

---

## Solution finale pour ton cas

```python
import torch
import torch.nn as nn

# TON PROBLÈME : labels dans [1..10]
labels_incorrects = torch.randint(1, 11, (32,))  # [1, 10]

# CORRECTION
labels_corrects = labels_incorrects - 1           # [0, 9]

# Vérification
assert labels_corrects.min() == 0
assert labels_corrects.max() == 9

logits = torch.randn(32, 10)
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels_corrects)         # Fonctionne
print(f"Loss : {loss.item():.4f}")
```

---

## Résumé

1. Mettre `CUDA_LAUNCH_BLOCKING=1` pour localiser l'erreur exacte
2. Vérifier que les labels sont dans `[0, num_classes - 1]` avec le bon dtype (`torch.long`)
3. Dans ton cas : faire `labels = labels - 1` pour passer de `[1..10]` à `[0..9]`
