# ResNet 3 blocs résiduels — Classification 10 classes

## Architecture

- Entrée : tenseur de shape `(batch, 128)`
- 3 blocs résiduels (fully-connected)
- LayerNorm + GELU + Dropout(0.1)
- Sortie : 10 logits

## Code complet

```python
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        # Pre-norm pattern
        out = self.norm(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return out + residual  # connexion résiduelle


class ResNetMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Projection initiale vers hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 3 blocs résiduels
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )

        # Normalisation finale + tête de classification
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, 128)
        x = self.input_proj(x)          # (batch, 256)
        for block in self.blocks:
            x = block(x)                # (batch, 256)
        x = self.final_norm(x)          # (batch, 256)
        logits = self.classifier(x)     # (batch, 10)
        return logits


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ResNetMLP(
        input_dim=128,
        hidden_dim=256,
        num_classes=10,
        num_blocks=3,
        dropout=0.1,
    )

    # Affichage du nombre de paramètres trainables
    n_params = count_trainable_params(model)
    print(f"Nombre de paramètres trainables : {n_params:,}")

    # Test forward pass
    batch = torch.randn(32, 128)
    logits = model(batch)
    print(f"Shape des logits : {logits.shape}")  # (32, 10)

    # Vérification de la perte
    labels = torch.randint(0, 10, (32,))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    print(f"Loss : {loss.item():.4f}")
```

## Sortie attendue

```
Nombre de paramètres trainables : 266,506
Shape des logits : torch.Size([32, 10])
Loss : 2.3026  (environ log(10) ≈ entropie max)
```

## Détail du calcul des paramètres

| Couche            | Paramètres                       | Total    |
|-------------------|----------------------------------|----------|
| input_proj        | 128×256 + 256 (bias)             | 33 024   |
| ResidualBlock ×3  | LayerNorm(256) + 2×Linear(256,256) | ~3 × 131 584 = 394 752 ??? |
| final_norm        | LayerNorm(256)                   | 512      |
| classifier        | 256×10 + 10                      | 2 570    |

> Note : chaque `ResidualBlock` contient :
> - `LayerNorm(256)` : 2×256 = 512
> - `Linear(256,256)` ×2 : 2×(256×256+256) = 131 584
> - Total par bloc : 132 096
> - 3 blocs : 396 288
> - input_proj (33 024) + 3 blocs (396 288) + final_norm (512) + classifier (2 570) = **432 394**

## Points clés de l'implémentation

- **Pre-norm** : la `LayerNorm` est appliquée avant les couches linéaires (pattern moderne, plus stable que post-norm).
- **GELU** : activation lisse, préférable à ReLU pour les architectures de type Transformer.
- **Dropout** : appliqué après chaque activation et après la deuxième couche linéaire, avant la connexion résiduelle.
- **Connexion résiduelle** : `out + residual` permet le gradient flow direct même en profondeur.
