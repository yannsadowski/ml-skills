# Intégrer PyTorch Profiler dans une boucle d'entraînement

## Vue d'ensemble

PyTorch Profiler permet de mesurer précisément le temps passé sur chaque opération, tant côté CPU que GPU, et de visualiser les résultats dans TensorBoard ou en console.

---

## Intégration minimale dans une boucle existante

Voici une boucle d'entraînement typique **avant** instrumentation :

```python
for epoch in range(num_epochs):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### Version instrumentée avec PyTorch Profiler :

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# Définir le schedule : quand capturer les traces
profiler_schedule = schedule(
    wait=1,     # Étapes d'attente initiale (warm-up)
    warmup=1,   # Étapes de warm-up (capturées mais ignorées)
    active=3,   # Étapes effectivement profilées
    repeat=1    # Nombre de cycles (0 = répéter indéfiniment)
)

def on_trace_ready(prof):
    """Callback appelé après chaque cycle actif."""
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15
    ))
    # Optionnel : exporter pour TensorBoard
    prof.export_chrome_trace(f"trace_step_{prof.step_num}.json")

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    schedule=profiler_schedule,
    on_trace_ready=on_trace_ready,
    record_shapes=True,       # Capture les shapes des tenseurs
    profile_memory=True,      # Mesure l'utilisation mémoire GPU
    with_stack=False,         # Stack traces (coûteux, activer si besoin)
) as prof:

    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):

            # Annoter les phases pour plus de lisibilité
            with record_function("data_transfer"):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            with record_function("forward_pass"):
                optimizer.zero_grad()
                outputs = model(inputs)

            with record_function("loss_computation"):
                loss = criterion(outputs, targets)

            with record_function("backward_pass"):
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()

            # IMPORTANT : appeler prof.step() à chaque itération
            prof.step()
```

---

## Lire les résultats en console

La sortie de `key_averages().table()` ressemble à :

```
---------------------------------  ------------  ------------  ------------
                             Name    CPU total %    CPU total    CUDA total
---------------------------------  ------------  ------------  ------------
                      aten::addmm        12.34%       5.678ms       3.210ms
                forward_pass           45.67%      21.000ms      18.500ms
             aten::_batch_norm_impl     8.90%       4.100ms       3.800ms
                   backward_pass       38.00%      17.500ms      15.200ms
                    data_transfer        2.10%       0.967ms       0.450ms
---------------------------------  ------------  ------------  ------------
```

**Colonnes importantes :**
- `CPU total` : temps passé à orchestrer l'opération côté CPU
- `CUDA total` : temps réel d'exécution sur le GPU
- Si `CPU total >> CUDA total` : overhead CPU excessif
- Si `CUDA total` est dominant sur les opérations attendues : normal

---

## Visualiser avec TensorBoard

```python
from torch.profiler import tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=5),
    on_trace_ready=tensorboard_trace_handler('./profiler_logs'),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # ... boucle normale ...
        prof.step()
```

Puis lancer TensorBoard :

```bash
tensorboard --logdir=./profiler_logs
```

Naviguer vers l'onglet **"PyTorch Profiler"** pour voir :
- La timeline GPU/CPU (Trace View)
- Le classement des opérations les plus coûteuses
- L'utilisation mémoire dans le temps

---

## Version légère pour un profiling ponctuel (sans schedule)

Pour profiler juste quelques batches rapidement, sans schedule :

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx >= 5:  # Profiler seulement 5 batches
            break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Afficher les 20 opérations les plus coûteuses en temps CUDA
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Exporter la trace Chrome (ouvrir dans chrome://tracing ou Perfetto)
prof.export_chrome_trace("training_trace.json")
```

---

## Annoter des sections personnalisées avec record_function

`record_function` crée des spans nommés visibles dans la trace :

```python
from torch.profiler import record_function

# Utilisation comme context manager
with record_function("my_custom_forward"):
    output = model(input)

# Peut être imbriqué
with record_function("augmentation"):
    x = augment(x)
    with record_function("normalize"):
        x = normalize(x)
```

---

## Bonnes pratiques

1. **Toujours utiliser `prof.step()`** à la fin de chaque itération de boucle, sinon le schedule ne fonctionne pas.
2. **Limiter le profiling** à quelques dizaines de steps : le profiler lui-même ajoute un overhead non négligeable.
3. **`torch.cuda.synchronize()`** n'est pas nécessaire avec le profiler car il synchronise automatiquement.
4. **Désactiver en production** : entourer le bloc `profile` d'un flag `if args.profile:`.
5. **`profile_memory=True`** est utile pour diagnostiquer les OOM mais ralentit davantage le profiling.

---

## Interpréter les résultats : que chercher ?

| Observation | Cause probable | Action |
|-------------|----------------|--------|
| `data_transfer` élevé | Transfers CPU->GPU fréquents | `pin_memory=True`, `non_blocking=True` |
| Gaps dans la timeline CUDA | Idle time, data starvation | Augmenter `num_workers` |
| `aten::copy_` omniprésent | Copies mémoire excessives | Réduire les `.clone()` et `.contiguous()` inutiles |
| Opérations CPU longues | Pas de mixed precision | Activer `torch.autocast` |
| `cudnn_convolution` dominant | Normal pour CNN | Vérifier que cuDNN benchmark est activé |
