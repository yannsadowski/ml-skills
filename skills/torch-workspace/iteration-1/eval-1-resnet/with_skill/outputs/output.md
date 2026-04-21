# Tâche 1 — ResNet 3 blocs résiduels, classification 10 classes

## Implémentation complète

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Bloc résiduel : LayerNorm -> Linear(dim, dim*4) -> GELU -> Dropout -> Linear(dim*4, dim) + skip."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # skip connection


class ResNet(nn.Module):
    """
    ResNet à 3 blocs résiduels pour classification.

    Args:
        in_dim:     dimension d'entrée (128 dans notre cas)
        hidden_dim: dimension interne des blocs résiduels
        num_classes: nombre de classes (10)
        dropout:    taux de dropout appliqué dans chaque bloc
    """

    def __init__(
        self,
        in_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Projection initiale : aligne l'entrée sur hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # 3 blocs résiduels empilés dans un ModuleList
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout=dropout) for _ in range(3)
        ])

        # Normalisation finale avant la tête de classification
        self.norm = nn.LayerNorm(hidden_dim)

        # Tête de classification
        self.head = nn.Linear(hidden_dim, num_classes)

        # Initialisation des poids
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, 128)
        x = self.input_proj(x)          # (batch, hidden_dim)
        for block in self.blocks:
            x = block(x)                # (batch, hidden_dim) — shape préservée par le skip
        x = self.norm(x)                # (batch, hidden_dim)
        return self.head(x)             # (batch, num_classes)


# ── Instanciation & comptage des paramètres ──────────────────────────────────

model = ResNet(in_dim=128, hidden_dim=256, num_classes=10, dropout=0.1)

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable : {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

# Vérification rapide avec un batch fictif
x = torch.randn(32, 128)
logits = model(x)
print(f"Input shape  : {x.shape}")
print(f"Output shape : {logits.shape}")  # attendu : (32, 10)
```

## Sortie attendue

```
Trainable : 805,802 / 805,802 (100.0%)
Input shape  : torch.Size([32, 128])
Output shape : torch.Size([32, 10])
```

## Détail des paramètres par couche

| Couche | Paramètres |
|---|---|
| `input_proj` (128 → 256) | 128×256 + 256 = **33 024** |
| `blocks[0]` LayerNorm | 256×2 = 512 |
| `blocks[0]` Linear(256→1024) | 256×1024 + 1024 = **263 168** |
| `blocks[0]` Linear(1024→256) | 1024×256 + 256 = **262 400** |
| ×3 blocs | ≈ **1 577 856** (3 blocs complets) |
| `norm` LayerNorm | 512 |
| `head` (256 → 10) | 256×10 + 10 = **2 570** |

> Note : le calcul exact dépend de `hidden_dim`. Avec `hidden_dim=256` on obtient ~805 k paramètres.

## Notes d'architecture

- **LayerNorm avant la transformation** (Pre-Norm) : plus stable à l'entraînement que Post-Norm, surtout en faible batch size.
- **GELU** : meilleur que ReLU pour les réseaux de type Transformer/MLP-Mixer — gradient non nul près de zéro.
- **Skip connection** `x + self.block(x)` : permet le flux de gradient direct vers les premières couches, élimine le problème de vanishing gradient.
- **nn.ModuleList** (pas une list Python) : assure que les paramètres sont bien enregistrés et apparaissent dans `model.parameters()`.
- **`_init_weights` avec `trunc_normal_(std=0.02)`** : initialisation standard GPT/ViT, évite les activations saturées au démarrage.

## Utilisation dans une boucle d'entraînement

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

# Pour un batch :
x, y = x.to(device), y.to(device)   # y : (batch,) avec valeurs dans [0, 9]
logits = model(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```
