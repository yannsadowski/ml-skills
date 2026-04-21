# Tâche 3 — Augmentation qui prend 80% du temps de chargement

## Diagnostic

80% du temps dans l'augmentation CPU (`__getitem__`) est le signal classique pour déplacer les transforms sur GPU. Les opérations `RandomCrop`, `RandomHorizontalFlip` et `ColorJitter` sont toutes accélérables sur GPU via `torchvision.transforms.v2`.

---

## Solution principale — Déplacer l'augmentation sur GPU

### Étape 1 — Retirer l'augmentation de `__getitem__`

Dans ton Dataset, garder uniquement le chargement et la normalisation de base :

```python
class MyDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels
        # Transforms minimaux — juste décodage + conversion tensor
        self.base_transform = T.Compose([
            T.Resize(256),
            T.ToTensor(),   # PIL/numpy → FloatTensor [0, 1]
        ])

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        x = self.base_transform(img)  # pas d'augmentation ici
        y = self.labels[idx]
        return x, y
```

### Étape 2 — Créer un module d'augmentation GPU

```python
import torch.nn as nn
import torchvision.transforms.v2 as T

class GPUAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            T.RandomHorizontalFlip(p=0.5),
            T.RandomCrop(224, padding=16),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

    def forward(self, x):  # x est déjà sur GPU, shape: (B, C, H, W)
        return self.aug(x)
```

### Étape 3 — Intégrer dans la boucle d'entraînement

```python
gpu_aug = GPUAugmentation().to(device)

for x, y in train_loader:
    x = x.to(device, non_blocking=True)  # transfert CPU → GPU
    y = y.to(device, non_blocking=True)
    x = gpu_aug(x)  # augmentation sur GPU — pas de CPU overhead
    
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

`non_blocking=True` permet au transfert CPU→GPU de se faire en parallèle du reste grâce à `pin_memory=True` dans le DataLoader.

---

## Pourquoi c'est beaucoup plus rapide

| Situation | Où s'exécute l'aug | Parallélisme |
|-----------|-------------------|--------------|
| Avant (CPU dans `__getitem__`) | Worker CPU | Limité à num_workers cœurs |
| Après (GPU dans training loop) | GPU | Opérations matricielles native GPU |

Sur une RTX 5090, ColorJitter et RandomCrop sur un batch entier (32-64 images) prennent < 1ms sur GPU, contre 10-50ms par image sur CPU.

---

## Optimisation complémentaire — Augmentation semi-offline pour les transforms fixes

Si certaines transforms ne dépendent pas de l'aléatoire (ex: resize, normalisation), les pré-cacher :

```python
import os, torch
from tqdm import tqdm

class PreprocessedDataset(Dataset):
    """Pré-applique les transforms déterministes, garde l'augmentation pour le runtime."""
    def __init__(self, raw_dataset, cache_path="preprocessed.pt"):
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path)
        else:
            fixed_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
            self.data = []
            for i in tqdm(range(len(raw_dataset))):
                img, label = raw_dataset[i]
                self.data.append((fixed_transform(img), label))
            torch.save(self.data, cache_path)

    def __getitem__(self, idx):
        return self.data[idx]  # retourne le tensor déjà preprocessé

    def __len__(self):
        return len(self.data)
```

---

## Configuration DataLoader pour maximiser le débit avec aug sur GPU

```python
DataLoader(
    dataset,          # Dataset sans augmentation
    batch_size=64,
    num_workers=8,
    pin_memory=True,          # obligatoire pour que non_blocking=True soit efficace
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=True,
)
```

Avec `pin_memory=True`, le transfert CPU→GPU et l'augmentation GPU peuvent se pipeliner avec le step précédent.

---

## Checklist

```
[x] Retirer RandomCrop / RandomFlip / ColorJitter de Dataset.__getitem__
[x] Créer GPUAugmentation(nn.Module) avec torchvision.transforms.v2
[x] Appliquer gpu_aug(x) après x.to(device) dans la boucle d'entraînement
[x] pin_memory=True + non_blocking=True pour le transfert
[ ] Si dataset petit (<10GB en RAM) → CachedDataset + aug GPU
[ ] GPU util encore faible ? → profiler avec dl-profiling skill
```

---

## Résultat attendu

En déplaçant l'augmentation sur GPU, le `__getitem__` se réduit au pur chargement disque + décodage JPEG. Les workers CPU sont libérés pour charger le prochain batch pendant que le GPU augmente et entraîne le batch courant. Gain typique : 3-5x sur le throughput data, GPU utilisation passant de ~50% à 85-95%.
