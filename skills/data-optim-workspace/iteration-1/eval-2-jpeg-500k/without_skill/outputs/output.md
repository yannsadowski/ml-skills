# Optimisation du data loading pour 500k images JPEG (GPU à 50%)

## Diagnostic : pourquoi le GPU est à 50%

Un GPU à 50% signifie qu'il attend les données — le bottleneck est le data pipeline, pas le compute. Avec 500k JPEG sur disque, les causes typiques sont :
- Décodage JPEG CPU-bound (lent)
- I/O disque séquentiel ou aléatoire lent
- Pas assez de préchargement
- Augmentations trop lourdes dans le pipeline

## Optimisations par ordre d'impact

### 1. Convertir les JPEG en format optimisé pour le streaming

Le JPEG impose un décodage CPU à chaque accès. Convertir en formats plus rapides à lire :

**Option A : WebDataset (format TAR)**
```python
import webdataset as wds

# Conversion une fois :
# tar -cf shard-000.tar img000001.jpg img000002.jpg ...
# (ou utiliser le script webdataset)

dataset = wds.WebDataset("shards/shard-{000..099}.tar") \
    .decode("pil") \
    .to_tuple("jpg", "cls") \
    .map_tuple(transforms, None)
```
Les TARs sont lus séquentiellement → optimal pour le buffer disque.

**Option B : LMDB**
```python
import lmdb
# Écriture : encoder les images en bytes dans LMDB
# Lecture : accès O(1) sans décompression des métadonnées
env = lmdb.open('dataset.lmdb', readonly=True, lock=False)
```

**Option C : TFRecord / HDF5** (moins courant en PyTorch mais efficace)

### 2. Activer le décodage JPEG GPU (turbo-jpeg)

```python
# Utiliser torchvision avec libjpeg-turbo (plus rapide que PIL)
# Installation : pip install Pillow-SIMD ou utiliser le backend libjpeg-turbo

# DALI (NVIDIA Data Loading Library) — décode le JPEG sur GPU directement
import nvidia.dali.pipeline as pipeline
# DALI pipeline bypasse entièrement le CPU pour le décodage
```

DALI peut multiplier le débit de décodage JPEG par 5-10x en le faisant sur GPU.

### 3. Optimiser le DataLoader

```python
DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,          # Augmenter jusqu'à nombre de coeurs CPU
    pin_memory=True,         # Transfert CPU->GPU plus rapide
    persistent_workers=True, # Pas de recréation des workers
    prefetch_factor=4,       # Précharger 4 batches par worker
)
```

**Règle pratique** : `num_workers = 4 * nombre_de_GPU` comme point de départ.

### 4. Mise en cache partielle en RAM

Si le dataset tient en RAM (ou partiellement) :

```python
from torch.utils.data import Dataset
import io

class CachedDataset(Dataset):
    def __init__(self, paths, cache_size=10000):
        self.paths = paths
        self.cache = {}
        self.cache_size = cache_size

    def __getitem__(self, idx):
        if idx not in self.cache:
            img = Image.open(self.paths[idx])
            if len(self.cache) < self.cache_size:
                self.cache[idx] = img.copy()
        else:
            img = self.cache[idx]
        return img
```

Ou utiliser un ramdisk pour les images les plus accédées :
```bash
# Linux : monter un tmpfs
sudo mount -t tmpfs -o size=20G tmpfs /mnt/ramdisk
cp dataset/train/*.jpg /mnt/ramdisk/
```

### 5. Paralléliser les I/O avec prefetch explicite

```python
# Utiliser un prefetcher asynchrone
class Prefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target
```

### 6. Réduire la taille des images avant stockage

Si les images font 2000x2000 et que le modèle prend du 224x224 :
```bash
# Redimensionner tout le dataset une fois
mogrify -resize 256x256 -quality 85 -path resized/ *.jpg
```
Réduire la taille des JPEG réduit le temps de lecture I/O et de décodage.

### 7. Profiler pour identifier le vrai bottleneck

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for i, (inputs, labels) in enumerate(train_loader):
        if i == 10:
            break
        inputs = inputs.cuda()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

## Plan d'action recommandé

1. **Immédiat** : Augmenter `num_workers` + `pin_memory=True` + `prefetch_factor=4` → gain rapide
2. **Court terme** : Redimensionner les images au format cible sur disque
3. **Moyen terme** : Convertir en WebDataset (TAR shards) pour I/O séquentiel optimal
4. **Si GPU critique** : Intégrer NVIDIA DALI pour décodage JPEG sur GPU

## Gains attendus

| Optimisation | GPU utilisation attendue |
|---|---|
| Baseline (actuel) | ~50% |
| num_workers optimal + prefetch | ~65-75% |
| Images redimensionnées | ~75-85% |
| WebDataset / LMDB | ~85-90% |
| NVIDIA DALI | ~90-95% |
