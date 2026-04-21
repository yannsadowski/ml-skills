---
name: data-optim
description: Use this skill when the data pipeline is a bottleneck in ML training, or when optimizing how data is loaded, preprocessed, or fed to the model. Trigger on: slow DataLoader, GPU starving, workers freezing, slow disk reads, HDF5/numpy/CSV loading optimization, dataset caching, prefetching, memory-mapped files, collate functions, augmentation on GPU vs CPU, WSL filesystem slowness, large datasets that don't fit in RAM, or any question about making data loading faster. Also trigger proactively when the user's GPU utilization is low and training is slow — the data pipeline is the most common cause.
---

# Data Pipeline Optimization

Stack context: PyTorch DataLoader, WSL2 (filesystem perf matters), RTX 5090, training loop with nn.Module.

**The core insight:** the GPU can process a batch in milliseconds. If loading/preprocessing takes longer, the GPU sits idle. Every optimization here directly translates to faster training.

---

## Diagnose first

```python
import time, torch

def benchmark_dataloader(loader, n_batches=50):
    """Measure pure data loading time vs training step time."""
    # data loading only
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= n_batches: break
    load_time = (time.perf_counter() - start) / n_batches * 1000
    
    print(f"Avg batch load time: {load_time:.1f} ms")
    print(f"Max theoretical throughput: {1000/load_time:.0f} batches/sec")
```

Rule of thumb: if load time > GPU step time, data is the bottleneck.

---

## DataLoader settings

Start here — highest ROI, zero code change:

```python
DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,          # rule: (CPU cores / 2) in WSL, try 4-16
    pin_memory=True,        # copies to pinned memory → faster GPU transfer
    persistent_workers=True, # keep workers alive between epochs (avoid fork overhead)
    prefetch_factor=4,      # batches to prefetch per worker (default 2, try 4-8)
    drop_last=True,         # avoids a slow partial batch at epoch end
)
```

**WSL-specific:** `num_workers > 0` with WSL2 can cause hangs. If workers freeze:
```python
# option 1: set start method (do this once at top of train.py)
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

# option 2: if still broken, drop to num_workers=0 and compensate with caching
```

---

## WSL filesystem: the hidden bottleneck

Reading training data from `/mnt/c/` (Windows NTFS) is 5-10x slower than reading from `~/` (Linux ext4).

**Fix:** move datasets to the Linux filesystem:
```bash
cp -r /mnt/c/my_dataset ~/data/my_dataset
```

Check where your data lives in the config (`conf/data/base.yaml: data_path`) and update.

---

## Dataset-level optimizations

### Pre-cache everything that can be pre-cached

If preprocessing is deterministic and expensive, do it once:

```python
class CachedDataset(Dataset):
    def __init__(self, raw_dataset, cache_path="cache.pt"):
        if os.path.exists(cache_path):
            self.data = torch.load(cache_path)
        else:
            self.data = [self._preprocess(raw_dataset[i]) for i in tqdm(range(len(raw_dataset)))]
            torch.save(self.data, cache_path)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

For large datasets: use memory-mapped files instead of loading everything into RAM.

### Memory-mapped numpy arrays

Faster than `torch.load` for large arrays — data is paged in on demand:

```python
import numpy as np

# save once
np.save("data.npy", array)

# load as memmap — doesn't load into RAM
data = np.load("data.npy", mmap_mode='r')

class MemmapDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path, mmap_mode='r')
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())  # .copy() to avoid negative strides
```

### HDF5 for structured data

Better than many small files (avoids filesystem overhead):

```python
import h5py

class HDF5Dataset(Dataset):
    def __init__(self, path):
        # open lazily — h5py handles seeking efficiently
        self.path = path
        with h5py.File(path, 'r') as f:
            self.length = len(f['data'])
    
    def __getitem__(self, idx):
        with h5py.File(self.path, 'r') as f:  # thread-safe with separate opens
            x = torch.from_numpy(f['data'][idx][()])
            y = torch.tensor(f['labels'][idx][()])
        return x, y
```

---

## Move augmentation to GPU

CPU augmentation is often the bottleneck for image/tensor data. Run it on GPU instead:

```python
import torch.nn as nn
import torchvision.transforms.v2 as T

class GPUAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ColorJitter(0.4, 0.4, 0.4),
        )
    
    def forward(self, x):  # x already on GPU
        return self.aug(x)

# in training loop — augment after moving to device
batch = batch.to(device)
batch = gpu_aug(batch)  # runs on GPU, no CPU↔GPU transfer for aug
```

---

## Custom collate for variable-length data

Default collate is slow for sequences — write your own:

```python
def collate_fn(batch):
    # batch is a list of (x, y) tuples
    xs, ys = zip(*batch)
    # pad sequences to max length in batch (dynamic padding)
    xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.stack(ys)
    return xs_padded, ys

DataLoader(dataset, collate_fn=collate_fn, ...)
```

---

## Gradient accumulation to use larger effective batch size

If you're limited by GPU memory (small batch_size) but want large effective batch:

```python
accumulation_steps = 4  # effective batch = batch_size * 4

optimizer.zero_grad()
for i, (x, y) in enumerate(train_loader):
    loss = model(x, y) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## Checklist

```
[ ] Data on Linux filesystem (~/), not /mnt/c/?
[ ] num_workers tuned (try 4, 8, 16 — benchmark each)?
[ ] pin_memory=True?
[ ] persistent_workers=True?
[ ] prefetch_factor >= 4?
[ ] Preprocessing deterministic? → pre-cache it
[ ] Large array data? → use memmap numpy or HDF5
[ ] Image/tensor augmentation? → move to GPU
[ ] GPU util still low after above? → profile with dl-profiling skill
```
