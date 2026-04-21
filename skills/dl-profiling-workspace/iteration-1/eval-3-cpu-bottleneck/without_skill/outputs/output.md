# 60% CPU / 40% GPU sur un MLP : Causes et Corrections

## Diagnostic : pourquoi un MLP est particulièrement vulnérable

Un MLP (Multi-Layer Perceptron) est composé essentiellement de `nn.Linear` + activations. Ces opérations sont **très rapides sur GPU** (quelques microsecondes), ce qui signifie que le GPU "attend" souvent que le CPU prépare le prochain batch. Les 60% de temps CPU révèlent un déséquilibre : le CPU est le goulot d'étranglement.

---

## Causes les plus fréquentes (par ordre de probabilité)

### 1. Batch size trop petit

**Pourquoi :** Un MLP avec un batch size de 32 ou 64 exécute des opérations GPU de ~0.1ms. Le CPU a besoin de ~1ms pour préparer le batch suivant, transférer les données, et lancer les kernels CUDA. Le GPU est idle la plupart du temps.

**Correction :**
```python
# Avant : batch_size=32 ou 64
# Après : augmenter drastiquement
loader = DataLoader(dataset, batch_size=2048, ...)  # Essayer 512, 1024, 2048, 4096
```

Règle empirique : augmenter le batch size jusqu'à atteindre ~80% GPU-Util ou until OOM.

---

### 2. Data loading sur CPU sans parallélisation

**Pourquoi :** Avec `num_workers=0` (défaut), le chargement des données se fait dans le thread principal, bloquant l'entraînement.

**Correction :**
```python
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,           # Au moins 4, idéalement = nombre de cores CPU / 2
    pin_memory=True,         # Alloue la mémoire CPU en page-locked pour transfers plus rapides
    prefetch_factor=4,       # Prefetch N batches par worker
    persistent_workers=True, # Garde les workers entre les epochs
)
```

---

### 3. Transfers de données CPU->GPU synchrones

**Pourquoi :** Sans `non_blocking=True`, chaque `.to('cuda')` attend la fin du transfer avant de continuer.

**Correction :**
```python
# Avant (bloquant)
inputs = inputs.to('cuda')
targets = targets.to('cuda')

# Après (non-bloquant)
inputs = inputs.to('cuda', non_blocking=True)
targets = targets.to('cuda', non_blocking=True)
# Requiert pin_memory=True dans le DataLoader pour être efficace
```

---

### 4. Pas de mixed precision (float32 partout)

**Pourquoi :** Sur un MLP, les opérations float32 sont 2-8x plus lentes que float16/bfloat16 sur les GPU récents (RTX 5090 dispose de tensor cores optimisés pour fp16/bf16). Le CPU doit gérer plus de synchronisations et de conversions.

**Correction :**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in loader:
    inputs = inputs.to('cuda', non_blocking=True)
    targets = targets.to('cuda', non_blocking=True)

    optimizer.zero_grad()

    with autocast(dtype=torch.bfloat16):  # ou torch.float16
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### 5. Trop d'opérations Python dans la boucle

**Pourquoi :** Chaque appel Python (logging, `.item()`, métriques calculées à chaque step) force une synchronisation CPU-GPU.

**Correction :**
```python
# Avant (force une sync CPU-GPU à chaque step)
print(f"Loss: {loss.item():.4f}")
accuracy = (outputs.argmax(1) == targets).float().mean().item()

# Après (calculer les métriques moins fréquemment)
if batch_idx % 100 == 0:
    print(f"Loss: {loss.item():.4f}")  # .item() seulement toutes les 100 étapes
```

Éviter aussi :
```python
# Mauvais : force une synchronisation
if loss < best_loss:  # Comparaison d'un tensor GPU avec une valeur Python
    best_loss = loss

# Bon : comparer les floats Python
if loss.item() < best_loss:  # Appeler .item() explicitement, pas implicitement
    best_loss = loss.item()
```

---

### 6. `optimizer.zero_grad()` sans `set_to_none=True`

**Pourquoi :** Par défaut, `zero_grad()` remplit les gradients de zéros, ce qui prend du temps CPU et mémoire.

**Correction :**
```python
# Avant
optimizer.zero_grad()

# Après (plus rapide : met les gradients à None au lieu de zéro)
optimizer.zero_grad(set_to_none=True)
```

---

### 7. Le modèle n'est pas compilé (PyTorch 2.0+)

**Pourquoi :** Sans compilation, PyTorch exécute chaque opération `nn.Linear` séparément via le dispatch Python, ce qui génère un overhead CPU significatif.

**Correction :**
```python
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Compiler le modèle pour fusionner les opérations et réduire l'overhead Python
model = torch.compile(model)
```

`torch.compile` est particulièrement efficace pour les MLP car il fusionne les Linear+ReLU en un seul kernel.

---

## Plan d'action priorisé

Appliquer dans cet ordre (du plus impactant au moins) :

| Priorité | Action | Gain attendu |
|----------|--------|--------------|
| 1 | Augmenter le `batch_size` | Très élevé |
| 2 | `num_workers >= 4` + `pin_memory=True` | Élevé |
| 3 | `non_blocking=True` sur les `.to('cuda')` | Moyen |
| 4 | Mixed precision avec `autocast` | Moyen-élevé |
| 5 | `torch.compile(model)` | Moyen |
| 6 | `zero_grad(set_to_none=True)` | Faible |
| 7 | Réduire les `.item()` dans la boucle | Faible-moyen |

---

## Code complet optimisé

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Modèle
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
).to('cuda')

model = torch.compile(model)  # PyTorch 2.0+

# DataLoader optimisé
loader = DataLoader(
    dataset,
    batch_size=2048,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to('cuda', non_blocking=True)
        targets = targets.to('cuda', non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")
```

Avec ces optimisations, un MLP sur RTX 5090 devrait atteindre 80-95% GPU-Util au lieu de 40%.
