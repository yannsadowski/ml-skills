# Tâche 2 — 500k images JPEG, GPU à 50%, entraînement lent

## Diagnostic

GPU à 50% = le GPU attend les données. Avec 500k JPEG sur disque, le goulot d'étranglement est quasi-certain dans le pipeline de chargement. Avant d'optimiser, mesurer pour confirmer :

```python
import time, torch

def benchmark_dataloader(loader, n_batches=50):
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= n_batches: break
    load_time = (time.perf_counter() - start) / n_batches * 1000
    print(f"Avg batch load time: {load_time:.1f} ms")

# Comparer load_time avec le temps d'un step GPU — si load_time > step_time, data est le bottleneck
```

---

## Optimisation 1 — Paramètres DataLoader (ROI maximal, zéro changement de code)

```python
DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,           # essayer 4, 8, 16 — benchmarker chaque valeur
    pin_memory=True,         # copie en mémoire paginée → transfert GPU plus rapide
    persistent_workers=True, # évite de relancer les workers à chaque epoch
    prefetch_factor=4,       # 4-8 batches en avance par worker (défaut : 2)
    drop_last=True,          # évite un batch partiel lent en fin d'epoch
)
```

Règle : `num_workers = CPU_cores / 2` sous WSL2. Sur un système natif, tester jusqu'à `CPU_cores`.

---

## Optimisation 2 — Données sur le filesystem Linux (WSL2 critique)

Si les JPEG sont sous `/mnt/c/` (NTFS Windows), chaque lecture est 5-10x plus lente qu'en ext4 :

```bash
# Vérifier l'emplacement actuel
ls /chemin/dataset

# Déplacer vers le filesystem Linux si nécessaire
cp -r /mnt/c/mon_dataset ~/data/mon_dataset
```

Mettre à jour `conf/data/base.yaml: data_path`.

---

## Optimisation 3 — Convertir JPEG en format binaire optimisé

500k fichiers JPEG = 500k appels système `open()`. Les formats binaires compacts éliminent cet overhead.

### Option A — HDF5 (recommandé pour images de taille fixe)

```python
import h5py, numpy as np
from PIL import Image
from pathlib import Path

# Conversion one-shot
def jpeg_to_hdf5(image_paths, labels, output_path, img_size=(224, 224)):
    with h5py.File(output_path, 'w') as f:
        n = len(image_paths)
        images = f.create_dataset('images', shape=(n, 3, *img_size), dtype='uint8')
        lbls   = f.create_dataset('labels', shape=(n,), dtype='int64')
        for i, (p, lbl) in enumerate(zip(image_paths, labels)):
            img = Image.open(p).resize(img_size)
            images[i] = np.array(img).transpose(2, 0, 1)  # HWC → CHW
            lbls[i] = lbl

# Dataset HDF5
class HDF5Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        with h5py.File(path, 'r') as f:
            self.length = len(f['images'])

    def __getitem__(self, idx):
        with h5py.File(self.path, 'r') as f:
            x = torch.from_numpy(f['images'][idx][()].astype('float32') / 255.0)
            y = torch.tensor(int(f['labels'][idx][()]))
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.length
```

### Option B — numpy memmap (pour données déjà en array)

```python
import numpy as np

# Conversion one-shot
# np.save("images.npy", all_images_array)  # shape: (500000, 3, 224, 224)

class MemmapDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = np.load(images_path, mmap_mode='r')  # paginé à la demande
        self.labels = np.load(labels_path)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.images[idx].copy()).float() / 255.0
        y = torch.tensor(int(self.labels[idx]))
        return x, y

    def __len__(self):
        return len(self.labels)
```

---

## Optimisation 4 — Augmentation sur GPU (si augmentation CPU = bottleneck)

Avec 500k images, faire l'augmentation sur CPU est souvent le facteur limitant. La déplacer sur GPU :

```python
import torchvision.transforms.v2 as T
import torch.nn as nn

class GPUAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(224),
            T.ColorJitter(0.4, 0.4, 0.4),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

    def forward(self, x):
        return self.aug(x)

gpu_aug = GPUAugmentation().to(device)

# Dans la boucle d'entraînement
for x, y in train_loader:
    x = x.to(device)
    x = gpu_aug(x)   # augmentation sur GPU, pas de transfert CPU↔GPU pour l'aug
    loss = model(x, y)
    ...
```

---

## Plan d'action priorisé

```
[1] Benchmarker le DataLoader actuel → confirmer que data est le bottleneck
[2] Vérifier que les JPEG sont sur ~/data/ (pas /mnt/c/)
[3] Tuner num_workers (tester 4, 8, 16) + pin_memory + persistent_workers + prefetch_factor=4
[4] Convertir les 500k JPEG en HDF5 ou memmap numpy
[5] Déplacer l'augmentation sur GPU si encore bottleneck
```

Avec ces 5 étapes, GPU utilisation devrait monter de 50% à 90%+.
