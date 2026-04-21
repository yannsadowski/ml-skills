# DataLoader freeze sous WSL2 avec num_workers > 0

## Cause principale

WSL2 utilise un noyau Linux virtualisé avec une implémentation partielle de `/dev/shm` (shared memory). PyTorch utilise par défaut la mémoire partagée POSIX pour la communication entre le processus principal et les workers. Sous WSL2, cela provoque des deadlocks intermittents, surtout après plusieurs epochs quand les workers tentent de recycler leurs buffers.

## Solutions immédiates

### 1. Changer la stratégie de multiprocessing

```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

Ou directement dans le DataLoader :

```python
DataLoader(
    dataset,
    num_workers=4,
    multiprocessing_context='spawn'
)
```

WSL2 est instable avec `fork` (défaut sur Linux) car l'état du processus parent peut être corrompu après le fork. `spawn` recrée un processus propre.

### 2. Passer à persistent_workers=True

```python
DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=True,
    multiprocessing_context='spawn'
)
```

Sans `persistent_workers`, les workers sont recréés à chaque epoch — c'est lors de cette recréation que les freezes surviennent le plus souvent sous WSL2.

### 3. Réduire ou désactiver le pinned memory

```python
DataLoader(
    dataset,
    num_workers=4,
    pin_memory=False  # Désactiver si pin_memory=True était utilisé
)
```

`pin_memory=True` alloue de la mémoire paginée verrouillée, ce qui peut interagir mal avec la gestion mémoire de WSL2.

### 4. Réduire num_workers

```python
DataLoader(dataset, num_workers=2)
```

WSL2 partage les ressources CPU avec Windows. Un nombre élevé de workers peut saturer les sémaphores ou les descripteurs de fichiers du kernel virtualisé.

## Configuration recommandée pour WSL2

```python
import torch.multiprocessing as mp

# En début de script, avant tout DataLoader
mp.set_start_method('spawn', force=True)

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=2,          # Commencer à 2, tester jusqu'à 4
    persistent_workers=True,
    pin_memory=False,
    multiprocessing_context='spawn',
    prefetch_factor=2
)
```

## Diagnostic pour identifier la cause précise

```python
import os
# Vérifier la taille de /dev/shm
os.system("df -h /dev/shm")

# Si < 1GB, augmenter dans WSL2 :
# Créer/éditer %USERPROFILE%\.wslconfig sous Windows :
# [wsl2]
# memory=8GB
```

## Alternative : num_workers=0

Si rien d'autre ne fonctionne, passer à `num_workers=0` (chargement synchrone dans le process principal) élimine tous les problèmes inter-process. La perte de performance est souvent compensée par la suppression des freezes qui bloquaient l'entraînement.

```python
DataLoader(dataset, num_workers=0)
```

## Résumé des priorités

| Action | Impact | Difficulté |
|--------|--------|-----------|
| `multiprocessing_context='spawn'` | Corrige le freeze dans ~80% des cas | Faible |
| `persistent_workers=True` | Évite la recréation des workers entre epochs | Faible |
| Réduire `num_workers` à 2 | Réduit la contention sur les ressources WSL2 | Faible |
| `pin_memory=False` | Évite les conflits mémoire | Faible |
| `num_workers=0` | Solution de dernier recours, 100% stable | Faible |
