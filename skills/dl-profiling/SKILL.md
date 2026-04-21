---
name: dl-profiling
description: Use this skill for profiling deep learning training or inference. Trigger on: slow training, GPU underutilization, identifying bottlenecks (CPU vs GPU vs data), PyTorch Profiler, torch.profiler, nsight, py-spy, memory profiling, finding which layer is slow, profiling a training step, kernel-level profiling, profiling with RTX 5090, or any question about where time is being spent in a PyTorch model. Also trigger when the user wants to know if they're GPU-bound or CPU-bound, or wants to optimize before knowing where the bottleneck is.
---

# Deep Learning Profiling

Stack context: PyTorch 2.4+, RTX 5090 (Blackwell sm_100), WSL2, training loop built around nn.Module.

**Rule:** always profile before optimizing. Guessing the bottleneck wastes time.

---

## Step 1 — Quick GPU utilization check

Before any profiler, check if the GPU is even busy:

```bash
# in a separate terminal while training
watch -n 0.5 nvidia-smi

# or for more detail
nvidia-smi dmon -s u  # streaming utilization every second
```

- **GPU util ~100%** → you're GPU-bound, good. Optimize model/kernels.
- **GPU util fluctuating / <80%** → CPU or data pipeline is starving the GPU.
- **GPU util ~0% between steps** → data loading is the bottleneck (see `data-optim` skill).

---

## Step 2 — PyTorch Profiler (main tool)

Wraps a few training steps and gives a Chrome trace + table:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_training_step(model, batch, optimizer, n_warmup=2, n_active=5):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=n_warmup, active=n_active),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./prof_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(1 + n_warmup + n_active):
            with record_function("forward"):
                output = model(batch)
            with record_function("loss"):
                loss = criterion(output, target)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer"):
                optimizer.step()
                optimizer.zero_grad()
            prof.step()
    
    # print top ops by CUDA time
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    return prof
```

View trace in Chrome: open `chrome://tracing` → load the `.json` from `./prof_logs/`.

Or use TensorBoard: `uv run tensorboard --logdir ./prof_logs`

### Reading the output table

Key columns:
- `self_cuda_time_total` — time in this op excluding children (actual kernel time)
- `cuda_time_total` — including children
- `cpu_time_total` — host-side scheduling overhead
- `# of Calls` — if an op is called 1000x, fusing/batching it matters

**Red flags:** lots of small `aten::copy_` or `cudaMemcpyAsync` → unnecessary data movement. High `self_cpu_time` on GPU ops → kernel launch overhead (many small ops).

---

## Step 3 — Memory profiling

```python
# snapshot memory at a specific point
torch.cuda.memory._record_memory_history(max_entries=100_000)

# ... run a few training steps ...

torch.cuda.memory._dump_snapshot("memory_snapshot.pkl")
torch.cuda.memory._record_memory_history(enabled=None)  # stop
```

View: upload `memory_snapshot.pkl` to [pytorch.org/memory_viz](https://pytorch.org/memory_viz)

Quick stats during training:
```python
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Peak:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()
```

---

## Step 4 — py-spy (CPU-side profiling)

Useful when CPU time is high and you want a flamegraph of Python code:

```bash
uv add --dev py-spy

# attach to running process
uv run py-spy top --pid <PID>

# generate flamegraph
uv run py-spy record -o flamegraph.svg --pid <PID>
# then open flamegraph.svg in browser
```

---

## Profiling a single forward pass (lightweight)

For quick layer-by-layer timing without the full profiler:

```python
import time

def time_forward(model, x, n_runs=100, warmup=10):
    model.eval()
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(x)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(x)
        torch.cuda.synchronize()  # must sync before timing CUDA
        
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    print(f"Forward pass: {elapsed:.2f} ms")
```

**Critical:** always call `torch.cuda.synchronize()` before stopping the timer. CUDA ops are async — without sync you're timing kernel launch, not execution.

---

## RTX 5090 / Blackwell-specific notes

- Use `torch.compile(model, mode="reduce-overhead")` — on sm_100 this can give 2-4x speedup by fusing kernels. Profile with and without to confirm.
- `torch.set_float32_matmul_precision('high')` enables TF32 (free ~2x on matmul, negligible accuracy impact for most models).
- `bfloat16` is native on Blackwell — no emulation cost. Profile with `model.to(torch.bfloat16)` to compare throughput vs float32.
- Nsight Systems for kernel-level: `nsys profile --trace cuda,nvtx uv run python train.py` (outputs `.nsys-rep`, open in Nsight GUI).

---

## Profiling checklist

```
[ ] Check nvidia-smi utilization first
[ ] GPU util < 80%? → data pipeline bottleneck → see data-optim skill
[ ] GPU util ~100% but slow? → run PyTorch Profiler
[ ] Which op dominates self_cuda_time? → focus optimization there
[ ] Memory growing? → run memory snapshot
[ ] CPU time high? → py-spy flamegraph
[ ] Tried torch.compile yet? → profile before/after
[ ] Using bfloat16? → profile float32 vs bfloat16 throughput
```
