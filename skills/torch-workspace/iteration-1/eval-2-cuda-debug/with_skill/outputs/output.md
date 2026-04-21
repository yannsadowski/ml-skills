# Tâche 2 — Déboguer "CUDA error: device-side assert triggered" sur CrossEntropyLoss

## Diagnostic immédiat : le vrai problème

Tes labels vont de **1 à 10**, mais `nn.CrossEntropyLoss` attend des labels dans **[0, num_classes - 1]**, soit **[0, 9]** pour 10 classes.

L'index `10` dépasse la borne — PyTorch déclenche un assert côté GPU, d'où l'erreur cryptique.

---

## Étape 1 — Obtenir la vraie stacktrace

Par défaut, les erreurs CUDA sont asynchrones : le message ne pointe pas vers la ligne fautive. Active le mode synchrone **avant** de lancer ton script :

```bash
CUDA_LAUNCH_BLOCKING=1 python train.py
```

Ou dans le code Python, tout au début :

```python
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

Relance — tu obtiendras maintenant une vraie traceback Python avec le fichier et la ligne exacte.

---

## Étape 2 — Vérifier les indices de classes

Ajoute ces assertions **avant** le calcul de la loss, dans ta boucle d'entraînement :

```python
# Juste avant : loss = criterion(logits, labels)
print(f"labels min: {labels.min().item()}, max: {labels.max().item()}")
print(f"logits shape: {logits.shape}")  # attendu : (batch, num_classes)
assert labels.min() >= 0, f"Labels négatifs détectés : {labels.min().item()}"
assert labels.max() < logits.shape[1], (
    f"Label {labels.max().item()} >= num_classes {logits.shape[1]}. "
    f"CrossEntropyLoss attend des indices dans [0, {logits.shape[1] - 1}]."
)
```

Dans ton cas, `labels.max()` retournera `10`, et `logits.shape[1]` est `10` → assert déclenché, message clair.

---

## Étape 3 — Corriger les labels

### Option A — Décaler les labels à la source (recommandé)

Si tes labels proviennent d'un dataset avec convention 1-indexée, convertis-les au chargement :

```python
labels = labels - 1  # 1..10  →  0..9
```

Fais-le dans ton `Dataset.__getitem__` ou dans le collate, pas dans la boucle d'entraînement, pour éviter de l'oublier à l'inférence.

### Option B — Utiliser `ignore_index` pour un label hors-borne spécifique

Si le label `0` a une signification particulière (ex. "ignore" / padding) et que tes vrais labels commencent à 1 :

```python
criterion = nn.CrossEntropyLoss(ignore_index=0)
# puis décale quand même : labels = labels - 1  →  0..9 (et 0 sera ignoré)
```

### Option C — Vérifier le dtype des labels

`CrossEntropyLoss` attend `torch.long` (int64). Un cast silencieux depuis float peut produire des valeurs aberrantes :

```python
labels = labels.long()  # s'assurer du bon dtype
```

---

## Récapitulatif : checklist de débogage

```python
# 1. Mode synchrone pour la vraie stacktrace
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 2. Vérifier min/max des labels
assert 0 <= labels.min() and labels.max() < num_classes, \
    f"Labels invalides : [{labels.min()}, {labels.max()}], attendu [0, {num_classes - 1}]"

# 3. Vérifier le dtype
assert labels.dtype == torch.long, f"dtype attendu torch.long, got {labels.dtype}"

# 4. Vérifier la shape des logits
assert logits.ndim == 2 and logits.shape[1] == num_classes, \
    f"Logits shape incorrecte : {logits.shape}, attendu (batch, {num_classes})"

# 5. Corriger les labels si 1-indexés
labels = labels - 1  # 1..10 → 0..9
```

---

## Autres causes courantes du même message d'erreur

| Cause | Symptôme | Fix |
|---|---|---|
| Labels 1-indexés | `labels.max() == num_classes` | `labels -= 1` |
| Labels négatifs | `labels.min() < 0` | Vérifier le pipeline de données |
| `num_classes` incorrect dans le modèle | `logits.shape[1] != vrai_nombre_classes` | Corriger `out_features` de la dernière couche |
| Dtype float au lieu de long | Silencieux jusqu'à la loss | `.long()` sur les labels |
| NaN dans les logits | Loss retourne NaN avant l'assert | `torch.isnan(logits).any()` |

---

## Vérification que tout est bon

```python
# Test unitaire minimal — à faire tourner avant l'entraînement complet
import torch
import torch.nn as nn

num_classes = 10
batch_size  = 4

logits = torch.randn(batch_size, num_classes)
labels = torch.randint(0, num_classes, (batch_size,))  # [0, 9] inclus

criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
print(f"Loss : {loss.item():.4f}")  # doit afficher un float, pas exploser
```
