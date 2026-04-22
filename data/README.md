# Dataset — Instructions de Reconstruction

Les données brutes (CSV oscilloscope) et les fichiers prétraités (`.npy`) ne sont **pas inclus** dans ce dépôt en raison de leur taille (~2 Go).

---

## Structure attendue des données brutes

Placer les fichiers CSV dans un dossier `OthmaneSalim11032026/` à la racine du projet :

```
PFE/
└── OthmaneSalim11032026/
    ├── C1EE GraphCu Arc_<charge>00000.csv   ← V_ligne
    ├── C2EE GraphCu Arc_<charge>00000.csv   ← V_arc (oracle)
    ├── C3EE GraphCu Arc_<charge>00000.csv   ← Courant I
    └── ...  (26 triplets C1/C2/C3)
```

**Format CSV :** Export Teledyne LeCroy, 5 lignes d'en-tête, colonnes `Time, Ampl`.  
**Fréquence d'échantillonnage :** 1 MHz — 20 000 samples par cycle à 50 Hz.

---

## Reconstruction du dataset

```bash
# Étape 1 : Segmentation + calibration des seuils + labels (C3 only)
python scripts/step1_build_labeled_matrix.py

# Étape 2 : Dataset 2 canaux pour Arc-FaultNet (C1 + C3)
python scripts/step2_build_multichannel.py
```

### Fichiers produits dans `labeled_dataset/`

| Fichier | Shape | Description |
|---------|-------|-------------|
| `X_multi.npy` | `(N, 2, 20000)` | Signaux normalisés [V_ligne, I] |
| `y.npy` | `(N,)` | Labels binaires {0=normal, 1=arc} |
| `charges.npy` | `(N,)` | Index de configuration de charge |
| `charge_map.json` | — | Mapping nom → index |
| `metadata.csv` | `(N, ...)` | Métadonnées par échantillon |
| `config_multi.json` | — | Paramètres du pipeline |

---

## Paramètres clés du pipeline

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `V_TH` | 10 V | Seuil de tension d'arc sur C2 |
| `R_LOW` | 0.05 | Ratio max → label normal |
| `R_HIGH` | 0.95 | Ratio min → label arc |
| `FS` | 1 000 000 Hz | Fréquence d'échantillonnage |
| `SAMPLES_PER_CYCLE` | 20 000 | Samples par cycle 50 Hz |

---

## Note

`charge_map.json` est inclus dans le dépôt — il contient uniquement le mapping textuel des 26 configurations de charge, sans données de signal.
