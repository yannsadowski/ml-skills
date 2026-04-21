# Tâche 1 — DataLoader qui freeze sous WSL2 avec num_workers=4

## Diagnostic du problème

Les workers DataLoader qui se bloquent sous WSL2 après quelques epochs sont un problème connu. La cause racine est la méthode de démarrage des processus Python (`fork`) qui est incompatible avec certaines bibliothèques sous WSL2 — les workers enfants héritent d'un état corrompu et se bloquent en attente de ressources qui ne seront jamais libérées.

## Solution 1 — Changer la méthode de démarrage des processus (fix principal)

Ajouter cette ligne **en haut de ton `train.py`**, avant toute autre import PyTorch :

```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

`spawn` crée chaque worker dans un processus vierge (pas de fork), ce qui évite l'héritage d'état problématique. C'est la solution recommandée pour WSL2.

## Solution 2 — Activer `persistent_workers`

Si le freeze survient entre les epochs, c'est souvent lié au redémarrage des workers à chaque epoch. Désactiver ce comportement :

```python
DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=True,  # les workers restent en vie entre les epochs
    prefetch_factor=4,
    pin_memory=True,
)
```

## Solution 3 — Vérifier où se trouvent tes données

Un freeze apparent peut être causé par des I/O extrêmement lents. Si tes données sont sur `/mnt/c/` (NTFS Windows), les accès sont 5 à 10x plus lents qu'en ext4 Linux et peuvent provoquer des timeouts dans les workers.

```bash
# Vérifier où sont tes données
ls -la /chemin/vers/dataset

# Si c'est sous /mnt/c/, les déplacer dans le filesystem Linux
cp -r /mnt/c/mon_dataset ~/data/mon_dataset
```

Mettre à jour le chemin dans la config (`conf/data/base.yaml: data_path`).

## Solution de secours — Passer à num_workers=0

Si les solutions ci-dessus ne suffisent pas, revenir à du chargement synchrone et compenser avec du caching :

```python
# DataLoader sans workers multiprocesses
DataLoader(dataset, num_workers=0, ...)
```

Pour éviter que ça devienne le goulot d'étranglement, pré-cacher les données preprocessées :

```python
import os, torch
from tqdm import tqdm

class CachedDataset(Dataset):
    def __init__(self, raw_dataset, cache_path="cache.pt"):
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path)
        else:
            self.data = [raw_dataset[i] for i in tqdm(range(len(raw_dataset)))]
            torch.save(self.data, cache_path)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
```

## Checklist de résolution

```
[x] Ajouter mp.set_start_method('spawn', force=True) en début de train.py
[x] Activer persistent_workers=True dans le DataLoader
[x] Vérifier que les données sont sur ~/data/ et non sur /mnt/c/
[ ] Si freeze persiste → num_workers=0 + CachedDataset
```

## Configuration DataLoader recommandée pour WSL2

```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # en haut du fichier

loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,           # (CPU cores / 2) en WSL2
    pin_memory=True,
    persistent_workers=True, # clé pour éviter le freeze entre epochs
    prefetch_factor=4,
    drop_last=True,
)
```
