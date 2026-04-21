# ml-skills

Plugin Claude Code contenant un ensemble de skills spécialisés pour le développement ML. Chaque skill injecte du contexte et des patterns dans Claude au moment où c'est pertinent, évitant d'avoir un contexte global surchargé.

## Stack couverte

PyTorch 2.4+ · Hydra · uv · W&B / Trackio · RTX 5090 (Blackwell sm_100) · WSL2 · Python 3.11+

---

## Structure du projet

```
ml-skills/
├── skills/
│   ├── <skill-name>/
│   │   ├── SKILL.md          # Le skill lui-même (frontmatter + contenu)
│   │   └── evals/
│   │       └── evals.json    # Tâches de benchmark pour ce skill
└── README.md
```

---

## Skills disponibles

| Skill | Rôle | Se déclenche sur |
|---|---|---|
| `torch` | Patterns et idiomes PyTorch avancés | nn.Module, loss, optimizer, mixed precision, torch.compile, checkpointing, model surgery |
| `ml-setup` | Scaffolding d'un nouveau projet ML | "nouveau projet", "setup experiment", création d'une structure PyTorch from scratch |
| `ml-run` | Debug et tuning d'un run en cours | Loss qui diverge, NaN, OOM, overfitting, interprétation de courbes, logs W&B/Trackio |
| `dl-profiling` | Profiling du training et de l'inférence | GPU underutilization, PyTorch Profiler, bottleneck CPU vs GPU vs data |
| `data-optim` | Optimisation du pipeline de données | DataLoader lent, GPU qui attend, workers WSL2, HDF5, augmentation CPU vs GPU |
| `hydra` | Configuration avec Hydra | Config groups, overrides, multirun/sweeps, Hydra instantiate, Optuna sweeper |
| `uv-ml` | Gestion des envs avec uv | Créer/syncer un env, ajouter des packages, versions CUDA de torch, pyproject.toml |
| `viz` | Visualisation des résultats | Loss curves, confusion matrix, UMAP/t-SNE, figures publication-ready, comparaison de runs |
| `python-practices` | Code Python idiomatique | Type hints, dataclasses, pathlib, logging, structure de projet, refactoring |

---

## Comment les skills fonctionnent

Un skill est un fichier `SKILL.md` avec deux parties :

**Frontmatter** — métadonnées lues par le harness Claude Code :
```yaml
---
name: torch
description: >
  Texte décrivant quand déclencher ce skill.
  C'est ce texte qui est utilisé pour le matching automatique.
---
```

**Corps** — contenu injecté dans le contexte de Claude quand le skill est actif. C'est du Markdown libre : patterns de code, règles, exemples, gotchas spécifiques à la stack.

Le skill est chargé automatiquement quand la description matche la requête de l'utilisateur. Il peut aussi être invoqué manuellement avec `/skill-name`.

---

## Benchmarks (evals)

Chaque skill peut avoir un dossier `evals/` avec un fichier `evals.json`. Ces evals permettent de mesurer objectivement l'apport du skill via le `skill-creator` — en lançant la même tâche **avec** et **sans** le skill actif, puis en comparant les résultats.

### Format d'un eval

```json
{
  "skill_name": "nom-du-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "La tâche donnée à l'agent",
      "expected_output": "Description humaine de ce qu'on attend",
      "files": ["evals/files/fichier_optionnel.csv"],
      "expectations": [
        "Assertion vérifiable 1",
        "Assertion vérifiable 2"
      ]
    }
  ]
}
```

- **`prompt`** : la requête envoyée à l'agent, telle qu'un utilisateur la formulerait
- **`expected_output`** : description en prose de la réponse idéale (pour le grader)
- **`files`** : fichiers à fournir à l'agent (chemin relatif à la racine du skill)
- **`expectations`** : liste d'assertions booléennes — le grader vérifie chacune et calcule un pass rate

### Lancer un benchmark

Via le skill `skill-creator` :

```
/skill-creator benchmark skills/torch
```

Cela exécute chaque eval 3 fois en configuration `with_skill` et `without_skill`, puis génère un rapport comparatif dans `benchmarks/<timestamp>/benchmark.json`.

### Evals disponibles

| Skill | Evals | Thèmes couverts |
|---|---|---|
| `torch` | 3 | ResNet + paramètres, debug CUDA assert, save/load checkpoint |
| `dl-profiling` | 3 | GPU underutilization, intégration PyTorch Profiler, bottleneck CPU/GPU |
| `data-optim` | 3 | Workers WSL2 freeze, 500k images JPEG, augmentation lente |

---

## Ajouter un skill

1. Créer `skills/<nom>/SKILL.md` avec le frontmatter `name` et `description`
2. Écrire le contenu : patterns, règles, exemples de code, gotchas
3. (Optionnel) Créer `skills/<nom>/evals/evals.json` avec au moins 3 evals

La `description` est critique : c'est elle qui détermine si le skill se déclenche au bon moment. Elle doit lister explicitement les mots-clés et situations qui doivent l'activer.

## Ajouter des evals à un skill existant

1. Créer `skills/<nom>/evals/evals.json`
2. Écrire des prompts réalistes, proches de ce qu'un utilisateur demanderait vraiment
3. Les `expectations` doivent être des assertions vérifiables sans ambiguïté (éviter "la réponse est bonne" — préférer "X est mentionné", "le code utilise Y")
4. Viser 3 à 5 evals par skill, couvrant des cas différents (cas nominal, cas limite, debug)
