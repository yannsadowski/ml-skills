# Tâche 3 — 60% CPU / 40% GPU sur un MLP : causes et corrections

## Contexte

Un MLP simple sur RTX 5090 avec 60% du temps en CPU et seulement 40% en GPU est un signe clair de **kernel launch overhead** ou de **data movement excessif**. Un MLP n'est pas compute-intensive par nature — les ops sont rapides, ce qui rend l'overhead CPU relativement plus important.

---

## Causes probables

### 1. Trop de petites opérations (kernel launch overhead)

Un MLP applique successivement Linear → Activation → Linear → ... Chaque couche génère plusieurs kernels CUDA séparés. Si le MLP est petit (ex. 512→256→128), chaque kernel prend quelques µs mais le lancement CPU en prend autant — le ratio overhead/compute est défavorable.

**Signal dans le profiler :** `cpu_time_total` élevé sur des ops GPU comme `aten::addmm`, `aten::relu_`.

### 2. Transfers host→device à chaque step

Si les inputs ou les targets sont recréés ou castés côté CPU à chaque step, PyTorch génère des `aten::copy_` (memcpy H→D) qui bloquent le pipeline.

**Signal dans le profiler :** beaucoup d'appels à `aten::copy_`, `aten::to`, ou `cudaMemcpyAsync` avec un `cpu_time_total` élevé.

### 3. Données non épinglées (non pinned memory)

Sans `pin_memory=True` dans le DataLoader, les transfers vers GPU passent par de la mémoire paginable — plus lents et bloquants pour le CPU.

### 4. `optimizer.zero_grad()` avec `set_to_none=False`

Par défaut avant PyTorch 2.0, `zero_grad()` écrit des zéros dans les gradients (mémoire write sur GPU), puis les réalloue au backward. Avec `set_to_none=True`, on évite ce write inutile.

### 5. Création de tenseurs dans la loop

```python
# MAUVAIS : crée un tenseur CPU à chaque step
mask = torch.ones(batch_size, hidden_size)  # sur CPU par défaut !
output = output * mask.cuda()               # copy H→D inutile
```

### 6. Appels à `.item()`, `.numpy()`, ou `.cpu()` dans la boucle

Ces appels forcent une **synchronisation CPU-GPU** (la CPU attend que le GPU termine) et un transfer de données. Un seul `.item()` par step peut dominer le temps total si le modèle est rapide.

---

## Diagnostic : confirmer avec le profiler

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
            optimizer.zero_grad(set_to_none=True)
        prof.step()
        if step >= 8:
            break

# Trier par cpu_time pour trouver l'overhead
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
# Puis trier par cuda_time pour voir les vrais kernels
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```

Chercher dans la sortie :
- `aten::copy_` avec un grand `cpu_time_total` → transfers inutiles.
- `cudaLaunchKernel` ou `aten::addmm` avec `# of Calls` très élevé → kernel launch overhead.
- `cudaStreamSynchronize` ou `cudaDeviceSynchronize` → quelque chose force une sync (souvent `.item()`).

---

## Corrections

### Fix 1 — torch.compile (le plus impactant sur RTX 5090)

`torch.compile` fusionne les kernels successifs du MLP en un seul kernel custom, éliminant le kernel launch overhead :

```python
model = torch.compile(model, mode="reduce-overhead")
# "reduce-overhead" est optimisé pour les petits modèles avec beaucoup de kernels
```

Sur Blackwell (sm_100), gain typique 2-4x sur un MLP.

### Fix 2 — Passer en bfloat16

```python
model = model.to(torch.bfloat16)

# Dans la loop :
inputs = inputs.to(torch.bfloat16)
```

bfloat16 est natif sur Blackwell, pas d'émulation. Divise par 2 la taille des tenseurs, réduit le temps des matmuls.

### Fix 3 — Activer TF32

```python
# À mettre une fois au début du script, avant tout calcul
torch.set_float32_matmul_precision('high')  # active TF32
```

~2x sur les matmuls float32, impact négligeable sur la précision pour la plupart des modèles.

### Fix 4 — pin_memory et num_workers dans le DataLoader

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,         # charger en parallèle
    pin_memory=True,       # mémoire épinglée → transfers GPU plus rapides
    persistent_workers=True,  # évite de recréer les workers à chaque epoch
)
```

### Fix 5 — zero_grad avec set_to_none

```python
optimizer.zero_grad(set_to_none=True)  # évite l'écriture de zéros
```

### Fix 6 — Éliminer les syncs accidentelles

```python
# MAUVAIS
loss_value = loss.item()  # force sync CPU-GPU
print(f"Loss: {loss_value:.4f}")  # à l'intérieur de la loop

# BON : accumuler sur GPU, print hors de la loop hot
total_loss = 0.0
for step, (inputs, targets) in enumerate(dataloader):
    ...
    total_loss += loss.detach()  # pas de sync

if step % 100 == 0:
    print(f"Loss: {total_loss.item() / 100:.4f}")  # sync seulement tous les 100 steps
    total_loss = 0.0
```

### Fix 7 — Créer les tenseurs directement sur GPU

```python
# MAUVAIS
noise = torch.randn(batch_size, hidden_size)  # CPU par défaut
output = output + noise.cuda()               # copy H→D

# BON
noise = torch.randn(batch_size, hidden_size, device="cuda")
output = output + noise
```

---

## py-spy pour flamegraph CPU

Si le CPU time reste élevé après les fixes ci-dessus, utiliser py-spy pour voir exactement quel code Python consomme du temps :

```bash
uv add --dev py-spy

# Pendant que le training tourne
uv run py-spy record -o flamegraph.svg --pid <PID>
# Ouvrir flamegraph.svg dans le navigateur
```

---

## Ordre de priorité pour un MLP

1. `torch.compile(model, mode="reduce-overhead")` — fix principal pour kernel launch overhead
2. `torch.set_float32_matmul_precision('high')` — gratuit, toujours activer
3. Vérifier `.item()` / `.cpu()` dans la loop — sync accidentelle très fréquente
4. `pin_memory=True` + `num_workers` dans le DataLoader
5. Passer en `bfloat16` si la précision le permet
6. `zero_grad(set_to_none=True)`

Après chaque changement, **re-profiler** pour confirmer l'impact réel avant de continuer.
