# Tâche 2 — Intégrer PyTorch Profiler dans une boucle d'entraînement

## Principe

PyTorch Profiler enregistre les opérations CPU et CUDA sur une fenêtre de steps configurable. On utilise un `schedule` pour éviter de profiler le warmup JIT/CUDA et ne capturer que des steps représentatifs.

---

## Intégration complète dans une boucle existante

### Avant (boucle sans profiler)

```python
for epoch in range(num_epochs):
    for step, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
```

### Après (avec PyTorch Profiler)

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Wrapper de profiling — à appeler sur quelques steps seulement
def profile_training_step(model, dataloader, optimizer, criterion,
                           n_warmup=2, n_active=5):
    """
    Profile n_warmup + n_active steps.
    wait=1 : skip 1 step (let CUDA warm up)
    warmup=n_warmup : collect but discard (JIT tracing)
    active=n_active : steps réellement enregistrés
    """
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=n_warmup,
            active=n_active,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        loader_iter = iter(dataloader)
        for step in range(1 + n_warmup + n_active):
            inputs, targets = next(loader_iter)
            inputs, targets = inputs.cuda(), targets.cuda()

            with record_function("forward"):
                output = model(inputs)

            with record_function("loss_compute"):
                loss = criterion(output, targets)

            with record_function("backward"):
                loss.backward()

            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

            prof.step()  # IMPORTANT : appeler à la fin de chaque step

    # Afficher le top 15 des ops par temps CUDA
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
    ))
    return prof


# Dans la boucle d'entraînement principale
for epoch in range(num_epochs):
    for step, (inputs, targets) in enumerate(dataloader):
        
        # Profiler uniquement au step 0 du premier epoch
        if epoch == 0 and step == 0:
            prof = profile_training_step(model, dataloader, optimizer, criterion)
            continue  # les steps profilés sont déjà consommés
        
        # Boucle normale
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
```

---

## Lire les résultats

### Table console

```
-----------------------------------------  -----------  -----------  -----------
Name                                       CPU total    CUDA total   # of Calls
-----------------------------------------  -----------  -----------  -----------
forward                                    12.345 ms    45.678 ms    5
aten::linear                               2.100 ms     38.200 ms    25
aten::relu                                 0.050 ms     1.200 ms     25
backward                                   8.200 ms     42.100 ms    5
aten::copy_                                5.100 ms     0.300 ms     150   ← RED FLAG
```

### Colonnes clés à surveiller

| Colonne | Ce qu'elle mesure | Red flag |
|---|---|---|
| `self_cuda_time_total` | Temps CUDA de cet op, hors enfants | Op qui domine = cible d'optimisation |
| `cuda_time_total` | Temps CUDA en incluant les enfants | Pour comparer forward/backward |
| `cpu_time_total` | Temps côté host (scheduling) | Élevé = kernel launch overhead |
| `# of Calls` | Nombre d'appels | Très élevé = fusing potentiel |

### Red flags courants

- **Beaucoup de `aten::copy_`** → transfers host↔device inutiles (vérifier `.cpu()`, `.numpy()`, ou création de tenseurs hors GPU).
- **`cpu_time_total` >> `cuda_time_total`** sur les ops GPU → overhead de lancement de kernels (trop de petits ops : envisager `torch.compile`).
- **`aten::to` répété** → cast de dtype à chaque step (caster le modèle une fois pour toutes).

---

## Visualiser la trace Chrome

```bash
# Les logs sont dans ./prof_logs/
# Ouvrir Chrome et aller sur :
chrome://tracing

# Cliquer "Load" et charger le fichier .json généré
```

La timeline montre les streams CPU et CUDA côte à côte. Les trous sur le stream CUDA indiquent que la GPU attend.

## Visualiser avec TensorBoard

```bash
uv run tensorboard --logdir ./prof_logs
# Ouvrir http://localhost:6006 → onglet "PyTorch Profiler"
```

---

## Profiling léger (forward pass uniquement)

Pour un timing rapide sans le profiler complet :

```python
import time

def time_forward(model, x, n_runs=100, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()  # attendre la fin des kernels CUDA

        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        torch.cuda.synchronize()  # OBLIGATOIRE avant de stopper le timer
        t1 = time.perf_counter()

    ms = (t1 - t0) / n_runs * 1000
    print(f"Forward pass : {ms:.2f} ms")
    return ms

x = torch.randn(batch_size, *input_shape, device="cuda")
time_forward(model, x)
```

**Attention :** sans `torch.cuda.synchronize()` avant d'arrêter le timer, on mesure le lancement des kernels (quelques µs), pas leur exécution réelle.

---

## Checklist

```
[ ] prof.step() appelé à la fin de chaque step (pas au début)
[ ] torch.cuda.synchronize() avant tout timing manuel
[ ] record_function() autour de forward / backward / optimizer pour granularité
[ ] Inspecter self_cuda_time_total en priorité (temps réel du kernel)
[ ] Chercher aten::copy_ en grand nombre → data movement inutile
[ ] Comparer avec torch.compile(model) — souvent 2-4x sur RTX 5090
```
