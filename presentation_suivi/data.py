#!/usr/bin/env python3
"""
Toutes les valeurs chiffrées de la présentation, avec leur source.
Aucun chiffre ne doit être écrit ailleurs que dans ce fichier.
"""

# ===================================================================
# PARAMÈTRES  — comptés par model.ArcFaultNetV2(fusion_mode=...)
# ===================================================================
PARAMS = {
    "gated":      {"total": 358_601, "fusion": 131_712},
    "sequential": {"total": 309_833, "fusion":  82_944},
}
PARAMS_SHARED = {"temporal": 57_540, "spectral": 158_788, "classifier": 10_561}
PARAMS_DELTA = PARAMS["gated"]["total"] - PARAMS["sequential"]["total"]      # 48 768
PARAMS_DELTA_PCT = 100 * PARAMS_DELTA / PARAMS["gated"]["total"]            # 13,60 %
FUSION_DELTA_PCT = 100 * (PARAMS["gated"]["fusion"] - PARAMS["sequential"]["fusion"]) \
    / PARAMS["gated"]["fusion"]                                             # 37,03 %

# ===================================================================
# SPLIT ALÉATOIRE  (« plafond » — 1 630 cycles de test)
# ===================================================================
N_TEST = 1630

# --- Ancienne version : fusion « gated » (deck précédent, 4 seeds)
V1_RUNS = [
    # nom,            acc,     fp, f1,      prec,    rec
    ("Seed 42",       97.61,   3,  97.37,   99.59,   95.24),
    ("Seed 3",        96.99,   7,  96.77,   99.05,   94.58),
    ("Seed 42 (bis)", 97.48,   5,  97.25,   98.77,   95.77),
    ("Seed 4",        97.24,  22,  96.82,   96.89,   96.75),
]
V1_MEAN = {"acc": 97.33, "f1": 97.05, "prec": 98.61, "rec": 95.55,
           "spec": 98.88, "fp": 9.25, "acc_std": 0.26, "f1_std": 0.27}

# --- Nouvelle version : cross-attention séquentielle (runs 29-30 juillet)
#     runs/arcfaultnet_v2_single_2026072{9,30}_*  →  results.json + eval/metrics.json
V2_RUNS = [
    # étiquette,               seed, planif.,        acc,   f1,    prec,  rec,   spec, tn,  fp, fn, tp
    ("Run 1",                  3, "constant",      98.71, 98.63, 99.61, 97.68, 99.65, 852, 3, 18, 757),
    ("Run 2",                  3, "cosine",        99.20, 99.16, 99.74, 98.58, 99.77, 853, 2, 11, 764),
    ("Run 3",                 42, "cosine",        97.48, 97.23, 99.58, 94.98, 99.66, 870, 3, 38, 719),
    ("Run 4",                 42, "warm restarts", 97.48, 97.23, 99.58, 94.98, 99.66, 870, 3, 38, 719),
]
V2_MEAN = {"acc": 98.22, "f1": 98.06, "prec": 99.63, "rec": 96.55,
           "spec": 99.68, "fp": 2.75}
V2_MEAN_CM = {"tn": 861, "fp": 2.75, "fn": 26, "tp": 740}       # moyenne des 4 runs
V2_BEST = {"label": "Run 2  ·  seed 3  ·  cosine", "acc": 99.20, "f1": 99.16,
           "prec": 99.74, "rec": 98.58, "spec": 99.77, "auc": 0.9979,
           "tn": 853, "fp": 2, "fn": 11, "tp": 764}
V2_BEST_RUN = "runs/arcfaultnet_v2_single_20260729_234636"

# gains nouvelle vs ancienne (moyennes)
GAIN = {k: V2_MEAN[k] - V1_MEAN[k] for k in ("acc", "f1", "prec", "rec", "spec")}
GAIN_FP = V1_MEAN["fp"] - V2_MEAN["fp"]          # 6,5 FP en moins

RADAR_AXES = ["Exactitude", "F1", "Précision", "Rappel", "Spécificité"]
RADAR_V1 = [V1_MEAN[k] for k in ("acc", "f1", "prec", "rec", "spec")]
RADAR_V2 = [V2_MEAN[k] for k in ("acc", "f1", "prec", "rec", "spec")]

# ===================================================================
# ABLATION DES BRIQUES DE FUSION  — ablation_attention_results/results.json
#   (seed 3, même split, 1 630 cycles)
# ===================================================================
ABLATION = [
    # variante affichée,             clé,            acc,   f1,    fp, params
    ("Modèle complet",               "full",         98.34, 98.28, 23, 309_833),
    ("sans cross-attention",         "no_xattn",     98.22, 98.11,  7, 259_785),
    ("sans porte fréquentielle",     "no_freqgate",  97.61, 97.47, 14, 309_829),
    ("sans attention canaux (DCA)",  "no_dca",       97.12, 96.98, 27, 304_165),
    ("aucun mécanisme",              "none",         95.03, 94.62, 18, 254_113),
]
ABLATION_RECALL = {"full": 99.48, "no_xattn": 97.16, "no_freqgate": 96.77,
                   "no_dca": 97.42, "none": 91.87}

# ===================================================================
# LEAVE-ONE-CAMPAIGN-OUT  (4 campagnes, 10 860 cycles au total)
# ===================================================================
CAMPAIGNS = ["15 juillet", "22 juillet", "8 juillet", "Banc 2026"]

# --- Arc-FaultNet V2 durci
#     runs/arcfaultnet_v2_groupkfold_campaign_20260729_182029
#     (GroupDRO + CORAL 0.5 + augmentation forte + channel-dropout 0.2)
V2_CV = [
    # campagne testée, n,    acc,   f1,    prec,  rec,   spec,  auc
    ("15 juillet", 2820, 91.77, 90.64, 96.89, 85.14, 97.60, 0.975),
    ("22 juillet", 3820, 82.62, 77.09, 99.64, 62.86, 99.80, 0.924),
    ("8 juillet",  2746, 77.71, 75.34, 79.04, 71.98, 82.86, 0.883),
    ("Banc 2026",  1474, 91.25, 89.77, 81.56, 99.82, 85.89, 0.997),
]
V2_CV_MEAN = {"acc": 85.84, "acc_std": 5.93, "f1": 83.21, "f1_std": 7.03,
              "auc": 0.945, "auc_std": 0.044}
V2_CV_POOLED = {"n": 10_860, "acc": 84.93, "f1": 82.05, "prec": 90.00,
                "rec": 75.39, "spec": 92.95, "auc": 0.909,
                "tp": 3741, "fp": 416, "fn": 1221, "tn": 5482}
V2_CV_RUN = "runs/arcfaultnet_v2_groupkfold_campaign_20260729_182029"

# --- Même protocole, sans les techniques de robustesse (run de contrôle)
#     runs/arcfaultnet_v2_groupkfold_campaign_20260729_184723
V2_CV_PLAIN = {"acc": 83.36, "acc_std": 3.41, "f1": 82.15, "f1_std": 4.85,
               "auc": 0.918, "pooled_acc": 84.02, "pooled_f1": 83.27}

# --- ArcSSM  —  runs/arcssm_groupkfold_campaign_20260729_123709
SSM_CV = [
    ("15 juillet", 2820, 63.87, 71.81, 56.53, 98.41, 33.51),
    ("22 juillet", 3820, 86.99, 86.93, 81.59, 93.02, 81.74),
    ("8 juillet",  2746, 79.06, 72.86, 94.15, 59.43, 96.68),
    ("Banc 2026",  1474, 73.61, 74.32, 59.39, 99.29, 57.55),
]
SSM_CV_MEAN = {"acc": 75.88, "acc_std": 8.41, "f1": 76.48, "f1_std": 6.10}
SSM_CV_POOLED = {"n": 10_860, "acc": 77.16, "f1": 77.56, "prec": 70.38,
                 "rec": 86.38, "spec": 69.41,
                 "tp": 4286, "fp": 1804, "fn": 676, "tn": 4094}
SSM_CV_RUN = "runs/arcssm_groupkfold_campaign_20260729_123709"

# ===================================================================
# DIAGNOSTIC DE L'ÉCART  — generalization_plan.md §1 (baseline ArcSSM B1)
# ===================================================================
DIAG = {
    "auc_par_campagne": [0.912, 0.908, 0.880, 0.996],
    "p_arc_min": 0.497, "p_arc_max": 0.973,      # p(arc) moyen sur cycles ARC
    "p_norm_min": 0.047, "p_norm_max": 0.450,    # p(arc) moyen sur cycles NORMAUX
    "pooled_fixe": 81.28,                        # seuil 0,5 pour tous
    "pooled_seuil_local": 89.55,                 # seuil ajusté par campagne
    "ceiling": 98.5,                             # split aléatoire
    "gap_total": 17,
    "gap_calibration": 8,
    "gap_representation": 9,
}

# ===================================================================
# PROCHAINES ÉTAPES  — generalization_plan.md §2
# ===================================================================
NEXT_STEPS = [
    ("A", "Mesurer le bruit de mesure",
     "Rejouer le protocole sur 2–3 graines pour savoir quel écart est réel.",
     "Fait / en cours", "done"),
    ("B", "Choisir le modèle hors banc",
     "L'arrêt de l'entraînement ne doit plus s'appuyer sur des données du même banc.",
     "Prochain run", "next"),
    ("C", "Augmentation physique du signal",
     "Bruit rose, bande passante variable, jitter 50 Hz, mélange de charges.",
     "En place", "done"),
    ("D", "Invariance au banc d'essai",
     "Lots équilibrés par campagne, GroupDRO, alignement CORAL des représentations.",
     "En place", "done"),
    ("E", "Calibration à la mise en service",
     "Régler le seuil sur des cycles non étiquetés du site + décision sur k cycles.",
     "À industrialiser", "next"),
    ("F", "Réduire la capacité du modèle",
     "Moins de couches, plus de régularisation : un modèle plus petit transfère souvent mieux.",
     "À tester", "todo"),
    ("G", "Plus de bancs d'essai",
     "Le seul vrai levier restant : 3 des 4 campagnes viennent du même banc.",
     "Besoin de vous", "ask"),
]
