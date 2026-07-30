# Présentation de suivi — Arc-FaultNet (juillet 2026)

25 slides PNG 1920 × 1080 destinées à l'industriel. Style repris de
[`artifacts/arcssm_explained.html`](../artifacts/arcssm_explained.html).

## Régénérer

```bash
cd presentation_suivi
../venv/bin/python build.py          # -> slides/slide_NN_*.png
```

Le PDF (`Arc-FaultNet_point_avancement_juillet2026.pdf`) se reconstruit depuis
les PNG avec `matplotlib.backends.backend_pdf`.

## Organisation

| Fichier | Rôle |
|---|---|
| `style.py` | palette, typographie, primitives (`card`, `table`, `callout`, `kpi`…) |
| `data.py` | **tous** les chiffres, avec le chemin du run qui les produit |
| `s01_05.py` … `s21_25.py` | une fonction par slide |
| `build.py` | vide `slides/` et régénère les 25 slides dans l'ordre |

Un chiffre à corriger se change **uniquement dans `data.py`**.

## Plan

**Partie 1 — la fusion repensée (04 → 15)**

| # | Slide |
|---|---|
| 01–03 | Titre, sommaire, quatre chiffres clés |
| 05 | Point de départ : la version de juin |
| 06 | Le diagnostic : on résumait avant de comparer |
| 07–08 | L'ancien mécanisme (porte) vs le nouveau (cross-attention séquentielle) |
| 09 | L'intuition : l'alignement temps ↔ fréquence |
| 10 | −48 768 paramètres |
| 11–13 | Radar, tableau des runs, matrices de confusion |
| 14 | Ablation des briques |
| 15 | Lecture industrielle |

**Partie 2 — robustesse et ArcSSM (16 → 25)**

| # | Slide |
|---|---|
| 17–18 | Pourquoi le découpage aléatoire trompe ; le protocole *leave-one-campaign-out* |
| 19 | Résultats campagne par campagne |
| 20 | Le diagnostic : 8 pts de calibration + 9 pts de représentation |
| 21–23 | ArcSSM : l'idée, la motivation, les résultats |
| 24 | Les sept actions du plan |
| 25 | À retenir en cinq phrases |

## Sources des chiffres

| Bloc | Source |
|---|---|
| Paramètres (358 601 / 309 833) | `model.ArcFaultNetV2(fusion_mode='gated' \| 'cross_attention')` |
| Runs split aléatoire | `runs/arcfaultnet_v2_single_2026072{9,30}_*/{results,eval/metrics}.json` |
| Ancienne version (4 seeds) | deck précédent — `gen_presentation_slides.py` |
| Ablation | `ablation_attention_results/results.json` |
| Cross-campagne Arc-FaultNet | `runs/arcfaultnet_v2_groupkfold_campaign_20260729_182029` |
| Cross-campagne ArcSSM | `runs/arcssm_groupkfold_campaign_20260729_123709` |
| Décomposition de l'écart | [`generalization_plan.md`](../generalization_plan.md) §1 |
| Plan en sept actions | [`generalization_plan.md`](../generalization_plan.md) §2 |

## Points à valider avant envoi

- **Slide 12** — les runs 3 et 4 (graine 42) donnent des métriques de test
  *identiques* sous `cosine` et `warm_restarts`. C'est signalé sur la slide ;
  la moyenne porte donc sur trois configurations distinctes.
- **Slide 20** — la décomposition 8 / 9 points vient des diagnostics de la
  baseline ArcSSM B1, pas du run Arc-FaultNet mis en avant slide 19.
- **Slide 5** — les fausses alarmes de l'ancienne version reprennent les
  valeurs du deck précédent (3 / 7 / 5 / 22) ; la slide « matrices » de ce
  même deck affichait 8 au lieu de 5 pour le troisième run.
