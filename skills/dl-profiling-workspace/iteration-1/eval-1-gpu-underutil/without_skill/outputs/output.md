# Diagnostiquer un bottleneck : Data Loading vs Modèle (GPU à 40%)

## Contexte

Un GPU RTX 5090 à 40% d'utilisation pendant l'entraînement est un signal clair de sous-utilisation. La cause principale est presque toujours un bottleneck en data loading (le CPU ne fournit pas les données assez vite), mais il peut aussi s'agir d'un modèle trop petit ou de transfers CPU-GPU fréquents.

---

## Étape 1 : Observer nvidia-smi en temps réel

Lance cette commande dans un terminal séparé pendant l'entraînement :

```bash
watch -n 0.5 nvidia-smi
```

Observe :
- **GPU-Util** : si elle oscille entre 0% et 80% par rafales, c'est typiquement du data loading.
- Si elle est stable autour de 40%, le modèle est trop petit ou le batch size trop faible.

Pour un monitoring plus fin :

```bash
nvidia-smi dmon -s u -d 1
```

---

## Étape 2 : Désactiver temporairement le data loading

Remplace ton DataLoader par des données factices (tenseurs aléatoires pré-générés en GPU) :

```python
import torch

# Données factices pré-allouées en GPU
dummy_input = torch.randn(batch_size, *input_shape, device='cuda')
dummy_target = torch.randint(0, num_classes, (batch_size,), device='cuda')

# Remplace ta boucle normale par :
for i in range(num_steps):
    optimizer.zero_grad()
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()
```

**Interprétation :**
- Si le GPU monte à 90%+ avec les données factices → **bottleneck = data loading**
- Si le GPU reste à ~40% avec les données factices → **bottleneck = modèle trop petit ou batch size trop faible**

---

## Étape 3 : Mesurer le temps de chaque phase

Instrumente ta boucle d'entraînement pour mesurer explicitement :

```python
import time
import torch

data_times = []
compute_times = []

for batch in dataloader:
    # Mesure du data loading
    t0 = time.perf_counter()
    inputs, targets = batch
    inputs = inputs.to('cuda', non_blocking=True)
    targets = targets.to('cuda', non_blocking=True)
    torch.cuda.synchronize()  # Attendre que le transfer soit terminé
    t1 = time.perf_counter()
    data_times.append(t1 - t0)

    # Mesure du compute
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()  # Attendre la fin du calcul GPU
    t2 = time.perf_counter()
    compute_times.append(t2 - t1)

avg_data = sum(data_times) / len(data_times)
avg_compute = sum(compute_times) / len(compute_times)
print(f"Data loading moyen : {avg_data*1000:.1f} ms")
print(f"Compute moyen      : {avg_compute*1000:.1f} ms")
print(f"Ratio data/compute : {avg_data/avg_compute:.2f}x")
```

**Interprétation :**
- Ratio > 1 → data loading est le bottleneck
- Ratio < 0.1 → le modèle est probablement trop simple pour justifier le GPU

---

## Étape 4 : Utiliser PyTorch Profiler pour confirmer

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 10:
            break
        inputs, targets = batch
        inputs = inputs.to('cuda')
        with record_function("forward"):
            output = model(inputs)
        with record_function("loss"):
            loss = criterion(output, targets)
        with record_function("backward"):
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Cherche dans la sortie :
- `DataLoader` ou `_worker_loop` avec beaucoup de temps CPU → bottleneck data loading
- Gaps entre les opérations CUDA → idle time dû à l'attente de données

---

## Étape 5 : Solutions selon le diagnostic

### Si c'est le data loading :

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,          # Augmenter (règle : 4x nombre de GPU)
    pin_memory=True,        # Copies CPU->GPU plus rapides
    prefetch_factor=4,      # Prefetch plus de batches en avance
    persistent_workers=True # Évite de recréer les workers à chaque epoch
)
```

Aussi envisager :
- Pré-calculer et cacher les augmentations si elles sont coûteuses
- Utiliser NVIDIA DALI pour les pipelines d'augmentation sur GPU
- Stocker les données en format binaire (HDF5, LMDB, webdataset) plutôt que des images JPEG individuelles

### Si c'est le modèle trop petit :

- Augmenter le `batch_size` (tant que ça tient en VRAM)
- Utiliser `torch.compile()` (PyTorch 2.0+) pour fusionner les opérations
- Passer en mixed precision : `torch.autocast('cuda', dtype=torch.float16)`

---

## Checklist de diagnostic rapide

| Test | GPU-Util résultant | Conclusion |
|------|--------------------|------------|
| Données factices GPU | 90%+ | Bottleneck = data loading |
| Données factices GPU | ~40% | Bottleneck = modèle/batch |
| Temps data >> temps compute | - | Augmenter num_workers, pin_memory |
| Temps data << temps compute | - | Augmenter batch size ou compiler le modèle |
