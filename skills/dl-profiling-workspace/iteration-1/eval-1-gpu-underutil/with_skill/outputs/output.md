# Tâche 1 — GPU à 40% d'utilisation : identifier le bottleneck

## Règle de base

Toujours profiler avant d'optimiser. Un GPU à 40% signifie que quelque chose starve le GPU — data loading ou preprocessing CPU sont les coupables les plus fréquents.

---

## Étape 1 — Confirmer et qualifier l'utilisation GPU

Dans un terminal séparé pendant l'entraînement :

```bash
# Vue basique, rafraîchissement toutes les 0.5s
watch -n 0.5 nvidia-smi

# Vue streaming avec détail d'utilisation
nvidia-smi dmon -s u
```

Observer le pattern :
- **Utilisation qui oscille entre ~0% et ~80%** → la GPU attend des données entre chaque step (data loading bottleneck).
- **Utilisation stable à ~40%** → la GPU travaille mais les kernels sont sous-utilisés (petits batches, ops peu parallélisables, overhead CPU).

---

## Étape 2 — Test de smoke : désactiver le data loading

Remplacer temporairement le DataLoader par des tenseurs synthétiques en mémoire GPU :

```python
# Créer un batch factice directement sur GPU
dummy_input = torch.randn(batch_size, *input_shape, device="cuda")
dummy_target = torch.randint(0, n_classes, (batch_size,), device="cuda")

# Mesurer le throughput pur (sans I/O)
import time
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"Throughput sans data loading : {100 / (t1 - t0):.1f} steps/s")
```

Comparer avec le throughput réel (avec DataLoader). Si le throughput monte significativement avec les données synthétiques → **le data loading est le bottleneck**. Sinon → le bottleneck est dans le modèle/compute.

---

## Étape 3 — PyTorch Profiler pour confirmer

Intégrer le profiler sur quelques steps pour voir la répartition CPU vs CUDA :

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=5),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof_logs"),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        with record_function("forward"):
            output = model(inputs)
        with record_function("loss"):
            loss = criterion(output, targets)
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad()
        prof.step()
        if step >= 8:
            break

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

Lire la trace dans Chrome : ouvrir `chrome://tracing` et charger le `.json` depuis `./prof_logs/`.

### Ce qu'on cherche dans la trace

- **Grands blancs sur la timeline CUDA** entre les steps → la GPU attend le CPU (data loading).
- **`cpu_time_total` >> `cuda_time_total`** sur les ops → overhead CPU ou kernel launch excessif.
- **Beaucoup de `aten::copy_`** → transfers host→device inutiles ou répétés.

---

## Étape 4 — Diagnostic data loading spécifique

Si l'étape 2 confirme que le data loading est en cause :

```python
# Mesurer le temps de chargement pur
import time
loader_iter = iter(dataloader)
times = []
for _ in range(50):
    t0 = time.perf_counter()
    batch = next(loader_iter)
    t1 = time.perf_counter()
    times.append(t1 - t0)

import statistics
print(f"Temps moyen par batch : {statistics.mean(times)*1000:.1f} ms")
print(f"Médiane               : {statistics.median(times)*1000:.1f} ms")
print(f"Max                   : {max(times)*1000:.1f} ms")
```

Si le temps de chargement est > au temps de forward+backward → data pipeline est le bottleneck (voir `data-optim` skill pour les solutions).

---

## Résumé décisionnel

| Observation | Cause probable | Action |
|---|---|---|
| GPU oscille 0%↔80% | Data loading trop lent | Augmenter `num_workers`, `pin_memory=True`, prefetch |
| GPU stable ~40%, dummy batch = ~40% | Modèle sous-utilisé | Augmenter batch size, torch.compile, bfloat16 |
| GPU stable ~40%, dummy batch = ~100% | Data loading bottleneck | Optimiser DataLoader |
| Beaucoup de `aten::copy_` dans le profiler | Transfers D→H inutiles | Vérifier `.cpu()` ou `.numpy()` dans la loop |

---

## Notes RTX 5090 (Blackwell sm_100)

- `torch.set_float32_matmul_precision('high')` active TF32 — gain ~2x sur matmul, à activer dès le début.
- `torch.compile(model, mode="reduce-overhead")` peut donner 2-4x de speedup en fusionnant les kernels.
- `bfloat16` est natif sur Blackwell, pas d'émulation. Tester avec `model.to(torch.bfloat16)` peut doubler le throughput.
