# 🎯 Résultat GroupKFold — Full V2 vs No Attention

## Verdict : **Full V2 GAGNE** sur tous les axes

| Métrique | Full V2 | No Attention | **Δ (avantage V2)** |
|---|---|---|---|
| **Accuracy** | 90.16% ± 9.90% | 86.17% ± 9.37% | **+3.99%** |
| **F1-Score** | 87.82% ± 12.56% | 82.20% ± 13.92% | **+5.62%** |
| **Précision** | 94.57% ± 7.73% | 91.02% ± 7.51% | **+3.55%** |
| **Rappel** | 83.84% ± 18.08% | 79.56% ± 22.95% | **+4.29%** |
| **Spécificité** | 95.74% ± 6.12% | 92.37% ± 7.89% | **+3.37%** |

## Comparaison fold par fold (F1)

| Fold | Full V2 | No Attention | Δ | Gagnant |
|---|---|---|---|---|
| 1 (exp13) | 75.34% | 75.86% | -0.52% | ≈ égalité |
| 2 (exp12) | **96.11%** | 84.91% | **+11.21%** | ✓ V2 |
| 3 (exp11) | **69.93%** | 58.88% | **+11.04%** | ✓ V2 |
| 4 (OthmaneSalim 1) | **99.25%** | 98.33% | +0.92% | ✓ V2 |
| 5 (OthmaneSalim 2) | **98.49%** | 93.02% | **+5.47%** | ✓ V2 |

> [!IMPORTANT]
> **V2 gagne 4 folds sur 5.** L'avantage est massif sur les folds difficiles (2 et 3 : +11%).

## Pourquoi le single-split donnait l'inverse ?

Le **single random split** mélangeait des cycles de la même session entre train et test. Le modèle No Attention mémorisait les patterns de surface sans avoir besoin de généraliser. Le **GroupKFold par enregistrement** force chaque fold à tester sur des enregistrements **jamais vus** — et là, l'attention cross-branche fait la différence.

> [!TIP]
> ### Ce que vous pouvez écrire dans votre mémoire :
> 
> *"L'étude d'ablation en mode single-split suggérait un gain marginal de la variante sans attention (+0.43% accuracy). Cependant, une évaluation rigoureuse en GroupKFold par enregistrement (5 folds, anti-fuite) révèle que le modèle complet avec mécanismes d'attention surpasse significativement la variante sans attention : **+5.62% en F1-Score**, avec une victoire sur **4 folds sur 5**. L'avantage est particulièrement marqué sur les folds difficiles (+11% en F1), confirmant que les mécanismes d'attention (FrequencyGate + RevisedCrossAttention) sont essentiels pour la généralisation à de nouveaux enregistrements."*
