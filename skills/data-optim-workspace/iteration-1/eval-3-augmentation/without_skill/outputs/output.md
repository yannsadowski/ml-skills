# Optimisation des augmentations de données (80% du temps de chargement)

## Pourquoi les augmentations sont lentes dans __getitem__

Par défaut, `Dataset.__getitem__` s'exécute dans les workers du DataLoader, sur CPU, de manière synchrone. Chaque image passe par une pipeline séquentielle PIL → crop → flip → color jitter → tensor. Ces opérations sont souvent mal vectorisées et sous-utilisent le CPU.

## Stratégies d'optimisation

### 1. Déplacer les augmentations sur GPU (batch-level)

C'est la transformation la plus impactante. Au lieu d'augmenter image par image dans `__getitem__`, augmenter le batch entier sur GPU après transfert.

```python
import torchvision.transforms.v2 as T

# Augmentations GPU-compatibles (torchvision v2)
gpu_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomCrop(224, padding=4),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

# Dans la boucle d'entraînement
for images, labels in train_loader:
    images = images.cuda()
    images = gpu_transforms(images)  # Appliqué sur le batch GPU entier
    outputs = model(images)
```

**Gain typique** : 3-5x sur les augmentations, car elles s'appliquent sur un batch vectorisé sur GPU.

### 2. Utiliser Albumentations au lieu de torchvision.transforms

Albumentations est significativement plus rapide que torchvision pour les augmentations CPU :

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

transform = A.Compose([
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

class MyDataset(Dataset):
    def __getitem__(self, idx):
        image = np.array(Image.open(self.paths[idx]))  # numpy, pas PIL
        result = self.transform(image=image)
        return result['image'], self.labels[idx]
```

Albumentations utilise OpenCV sous le capot (optimisé C++/SIMD) et est 2-4x plus rapide que PIL pour la plupart des opérations.

### 3. Utiliser NVIDIA DALI pour un pipeline entièrement GPU

DALI (Data Loading Library) effectue lecture, décodage et augmentation directement sur GPU :

```python
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

@pipeline_def(batch_size=64, num_threads=4, device_id=0)
def training_pipeline(data_dir):
    images, labels = fn.readers.file(file_root=data_dir, random_shuffle=True)
    images = fn.decoders.image(images, device="mixed")  # Décodage GPU
    images = fn.random_resized_crop(images, device="gpu", size=224)
    images = fn.flip(images, device="gpu", horizontal=fn.random.coin_flip())
    images = fn.color_twist(images, device="gpu",
                            brightness=fn.random.uniform(range=[0.8, 1.2]),
                            contrast=fn.random.uniform(range=[0.8, 1.2]))
    return images, labels
```

DALI élimine entièrement le CPU du pipeline d'augmentation.

### 4. Appliquer les augmentations offline (pré-calcul)

Pour les augmentations déterministes ou si la diversité est suffisante avec un petit nombre de variantes :

```python
# Pré-générer N versions augmentées de chaque image
# et les sauvegarder sur disque
from tqdm import tqdm

N_AUGMENTATIONS = 5
for idx, path in enumerate(tqdm(image_paths)):
    img = Image.open(path)
    for k in range(N_AUGMENTATIONS):
        augmented = transform(img)
        save_path = f"augmented/{idx}_{k}.jpg"
        # Sauvegarder...
```

**Avantage** : `__getitem__` ne fait plus qu'un `load` sans transformation.
**Inconvénient** : Moins de diversité, espace disque multiplié par N.

### 5. Optimiser la pipeline existante dans __getitem__

#### Éviter les conversions de format redondantes

```python
# Mauvais : PIL -> numpy -> PIL -> tensor (conversions inutiles)
img = Image.open(path)
img = np.array(img)  # inutile si on reste en PIL
img = Image.fromarray(img)  # inutile
tensor = transforms(img)

# Bon : PIL -> tensor directement
img = Image.open(path)
tensor = transforms(img)  # transforms.ToTensor() fait la conversion
```

#### Utiliser des transforms v2 de torchvision (plus rapides)

```python
# Ancien (lent)
import torchvision.transforms as T

# Nouveau (plus rapide, supporte batch et GPU)
import torchvision.transforms.v2 as T

transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
])
```

#### Réduire la résolution avant les augmentations

```python
# Charger en résolution réduite dès le début
img = Image.open(path)
img = img.resize((256, 256), Image.BILINEAR)  # Resize avant toute augmentation
# Ensuite RandomCrop(224) sur 256x256, pas sur 2000x2000
```

### 6. Profiler précisément pour identifier la transformation lente

```python
import time
from collections import defaultdict

times = defaultdict(float)

class TimedCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            start = time.perf_counter()
            img = t(img)
            times[type(t).__name__] += time.perf_counter() - start
        return img

# Après quelques batches :
for name, t in sorted(times.items(), key=lambda x: -x[1]):
    print(f"{name}: {t:.3f}s")
```

ColorJitter est souvent l'opération la plus lente (4 opérations colorimétriques). La désactiver temporairement ou réduire sa probabilité peut avoir un fort impact.

## Stratégie recommandée selon le contexte

### Si GPU disponible (cas le plus courant en entraînement)

```python
# Dataset : charge et decode seulement
class SimpleDataset(Dataset):
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.to_tensor(img), self.labels[idx]  # Juste resize + ToTensor

# Augmentations GPU dans la boucle
gpu_aug = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(224, padding=4),
    T.ColorJitter(0.2, 0.2, 0.2, 0.1),
])

for imgs, labels in loader:
    imgs = gpu_aug(imgs.cuda())
    # Forward pass...
```

### Si CPU seulement

Remplacer torchvision.transforms par Albumentations + augmenter `num_workers`.

## Résumé des gains attendus

| Approche | Réduction du temps d'augmentation |
|---|---|
| Baseline (torchvision CPU dans __getitem__) | 100% (référence) |
| Albumentations CPU | ~40-60% |
| torchvision.transforms.v2 | ~20-30% |
| Augmentations GPU (batch-level) | ~70-85% |
| NVIDIA DALI | ~90-95% |
| Offline pre-augmentation | ~95-100% |
