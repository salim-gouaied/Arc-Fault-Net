#!/usr/bin/env python3
"""
Diagrammes PNG pour la présentation d'avancement Arc-FaultNet.
==============================================================

Style volontairement distinct des sets précédents : flat moderne, gros
caractères, palette indigo/teal/orange, fonds clairs — pensé pour être
projeté et expliqué facilement.

Sorties (docs/presentation/diagrams/):
  00_cover.png                       — page de garde (stage, encadrement, titre)
  00_introduction.png                — problématique & objectif (bref)
  13_fusion_bilan.png                — bilan fusion V1/V2
  15_ablation_introduction.png       — objectif & protocole étude d'ablation
  16_ablation_v2_performance.png     — performances V2 full (ablation, CM + métriques)
  17_traceabilite_reverse_engineering.png — interprétation projetée (traçabilité)
  18_limites_challenges.png            — limites et perspectives du travail
  19_synthese_atouts_perspectives.png  — bilan positif · état · piste SSM
  99_merci.png                       — diapo de clôture
  01_architecture_globale.png      — architecture V2 vue d'ensemble
  02_data_flow.png                 — flux des tenseurs avec shapes
  03_pipeline_travail.png          — pipeline complet acquisition → sortie
  04_branche_1d.png                — branche temporelle (4 canaux dérivés)
  05_branche_2d.png                — branche spectrale (STFT + FrequencyGate)
  06_futures_implementations.png   — extensions futures (multi-cycles)
  07_comparaison_v1_v2.png         — ancienne vs nouvelle architecture + métriques
  08_meilleur_modele.png           — meilleur modèle (accuracy ↑, FP ↓)
  09_innovation_scientifique.png   — les 4 innovations défendables
  10_transformees_generalisation.png — choix des transformées → généralisation

Run :
    python docs/presentation/gen_presentation_diagrams.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "diagrams"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 170

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 0.0,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

# ── Palette (nouveau style) ────────────────────────────────────────────
INK = "#1e293b"
MUT = "#3f4c63"   # texte secondaire — foncé pour rester lisible en projection
INDIGO, INDIGO_L = "#4f46e5", "#eef2ff"
TEAL, TEAL_L = "#0d9488", "#ccfbf1"
ORANGE, ORANGE_L = "#ea580c", "#ffedd5"
VIOLET, VIOLET_L = "#7c3aed", "#ede9fe"
SKY, SKY_L = "#0284c7", "#e0f2fe"
GREEN, GREEN_L = "#16a34a", "#dcfce7"
RED, RED_L = "#dc2626", "#fee2e2"
GREY, GREY_L = "#94a3b8", "#f1f5f9"
AMBER, AMBER_L = "#d97706", "#fef3c7"


# ── Helpers ────────────────────────────────────────────────────────────
def new_fig(w=14.0, h=7.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def title(ax, main, sub=""):
    ax.text(0.5, 0.965, main, ha="center", va="center",
            fontsize=21, fontweight="bold", color=INK)
    if sub:
        ax.text(0.5, 0.915, sub, ha="center", va="center",
                fontsize=11, style="italic", color=MUT)


# Espacement uniforme — diapo introduction
INTRO_TITLE_H = 0.068
INTRO_PAD = 0.012
INTRO_LINE_STEP = 0.028
INTRO_FS = 11


def _intro_box_height(n_bullets):
    return INTRO_TITLE_H + 2 * INTRO_PAD + n_bullets * INTRO_LINE_STEP


def box_bullets(ax, x, y, w, h, fc, ec, title_s, bullets, lw=2.2, z=3,
                center=False, line_step=None, fontsize=None):
    """Box avec titre coloré et puces noires alignées."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.014",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))

    pad_x = 0.020
    ax.text(x + w / 2, y + h - INTRO_TITLE_H / 2, title_s,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=ec, zorder=z + 1)

    body_top = y + h - INTRO_TITLE_H - 0.008
    body_bot = y + INTRO_PAD + 0.004
    n = len(bullets)
    avail = body_top - body_bot
    step = line_step if line_step else avail / n
    body_h = min(n * step, avail)

    fs = fontsize or INTRO_FS
    if not fontsize:
        for try_fs in [12, 11.5, 11, 10.5, 10]:
            if max(len(b) for b in bullets) * try_fs * 0.0043 < w - 2 * pad_x:
                fs = try_fs
                break

    ys = [body_top - step * (i + 0.5) for i in range(n)]

    char_w = fs * 0.0043
    lines = [f"• {b}" for b in bullets]
    text_w = min(max(len(l) for l in lines) * char_w, w - 2 * pad_x)
    block_x0 = x + (w - text_w) / 2 if center else x + pad_x

    for by, line in zip(ys, lines):
        ax.text(block_x0, by, line, ha="left", va="center",
                fontsize=fs, color=INK, zorder=z + 1)


def box(ax, x, y, w, h, fc, ec, title_s, sub=None, tsize=11, ssize=8.5,
        lw=2.2, z=3, tc=None, sc=None):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.014",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))
    cy = y + h * (0.63 if sub else 0.5)
    ax.text(x + w / 2, cy, title_s, ha="center", va="center",
            fontsize=tsize, fontweight="bold", color=tc or INK, zorder=z + 1)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=ssize, color=sc or MUT, zorder=z + 1)


def band(ax, x, y, w, h, fc, label, ec=None, z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=1.6, edgecolor=ec or fc, facecolor=fc, alpha=0.30, zorder=z))
    ax.text(x + 0.013, y + h - 0.035, label, ha="left", va="center",
            fontsize=10, fontweight="bold", color=ec or INK, zorder=z + 1)


def arrow(ax, p0, p1, color=INK, lw=2.6, rad=0.0, z=5, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=24, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=z,
        linestyle=ls, shrinkA=3, shrinkB=3))


def chip(ax, x, y, s, color=MUT, fc="white", z=6, size=8.5):
    pad = 0.008 * len(s) / 2 + 0.018
    ax.add_patch(FancyBboxPatch(
        (x - pad, y - 0.022), 2 * pad, 0.044,
        boxstyle="round,pad=0.0,rounding_size=0.02",
        linewidth=1.2, edgecolor=color, facecolor=fc, zorder=z))
    ax.text(x, y, s, ha="center", va="center", fontsize=size,
            fontweight="bold", color=color, zorder=z + 1)


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


# ══════════════════════════════════════════════════════════════════════
# 00 — Page de garde
# ══════════════════════════════════════════════════════════════════════
def fig_cover():
    fig, ax = new_fig(14.6, 8.2)
    ax.add_patch(FancyBboxPatch(
        (0, 0.88), 1, 0.12, boxstyle="square,pad=0",
        linewidth=0, facecolor=INDIGO, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 0.045, boxstyle="square,pad=0",
        linewidth=0, facecolor=TEAL, zorder=1))

    ax.text(0.5, 0.94, "Institut Jean Lamour — Université de Lorraine",
            ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax.text(0.5, 0.905, "Stage de fin d'études  ·  Année 2025–2026",
            ha="center", va="center", fontsize=11, color=INDIGO_L)

    ax.text(0.5, 0.72,
            "ARC-FAULTNET : Une nouvelle approche pour la détection\ndes défauts d'arc électriques",
            ha="center", va="center", fontsize=24, fontweight="bold", color=INK,
            linespacing=1.35)
    ax.text(0.5, 0.585, "en se basant sur les mécanismes d'attention",
            ha="center", va="center", fontsize=17, style="italic", color=INDIGO)

    ax.plot([0.15, 0.85], [0.52, 0.52], color=GREY, lw=1.5, zorder=2)

    box(ax, 0.22, 0.28, 0.56, 0.19, INDIGO_L, INDIGO,
        "Encadré par : Mr SCHWEITZER Patrick", None, tsize=13)
    box(ax, 0.22, 0.08, 0.56, 0.16, TEAL_L, TEAL,
        "GOUAIED Salim", "Étudiant stagiaire en génie informatique — spécialité IA",
        tsize=14, ssize=10)
    save(fig, "00_cover.png")


# ══════════════════════════════════════════════════════════════════════
# 00 — Introduction (problématique & objectif)
# ══════════════════════════════════════════════════════════════════════
def fig_introduction():
    fig, ax = new_fig(14.6, 7.6)
    title(ax, "Introduction — contexte et objectif")

    lx, lw = 0.04, 0.44
    rx, rw = 0.54, 0.42
    ls = INTRO_LINE_STEP
    kw = dict(line_step=ls, fontsize=INTRO_FS)

    h_prob = _intro_box_height(3)
    h_lim = _intro_box_height(3)
    h_obj = _intro_box_height(4)
    gap = 0.045

    left_bottom = 0.14
    lim_y = left_bottom
    prob_y = lim_y + h_lim + gap
    left_total = h_prob + gap + h_lim
    obj_y = left_bottom + (left_total - h_obj) / 2

    prob_bullets = [
        "Performants en laboratoire",
        "Peu fiables en installation réelle",
        "Charges variées · bruit · imprévisible",
    ]
    lim_bullets = [
        "Forte perf. sur données expérimentales",
        "Robustesse faible hors labo",
        "Généralisation inter-charges limitée",
    ]
    obj_bullets = [
        "Modèle robuste, efficace et généraliste",
        "Détection d'arc fiable en conditions réelles",
        "Mécanismes d'attention pour filtrer le signal",
        "Adaptation automatique, charge par charge",
    ]

    box_bullets(ax, lx, prob_y, lw, h_prob, RED_L, RED,
                "Problématique", prob_bullets, center=True, **kw)
    box_bullets(ax, lx, lim_y, lw, h_lim, AMBER_L, AMBER,
                "Limites des approches existantes", lim_bullets, center=True, **kw)
    box_bullets(ax, rx, obj_y, rw, h_obj, GREEN_L, GREEN,
                "Objectif de ce travail", obj_bullets, center=True, **kw)

    save(fig, "00_introduction.png")


# ══════════════════════════════════════════════════════════════════════
# 99 — Diapo de clôture
# ══════════════════════════════════════════════════════════════════════
def fig_merci():
    fig, ax = new_fig(14.6, 8.0)
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="square,pad=0",
        linewidth=0, facecolor=INDIGO_L, zorder=0))
    ax.add_patch(FancyBboxPatch(
        (0.08, 0.22), 0.84, 0.56, boxstyle="round,pad=0.0,rounding_size=0.02",
        linewidth=2.5, edgecolor=INDIGO, facecolor="white", zorder=2))

    ax.text(0.5, 0.58, "Merci pour votre attention",
            ha="center", va="center", fontsize=32, fontweight="bold", color=INK)
    ax.text(0.5, 0.38, "Questions ?",
            ha="center", va="center", fontsize=18, style="italic", color=INDIGO)

    ax.text(0.5, 0.10,
            "Salim GOUAIED  ·  Institut Jean Lamour  ·  Université de Lorraine  ·  2025–2026",
            ha="center", va="center", fontsize=10, color=MUT)
    save(fig, "99_merci.png")


# ══════════════════════════════════════════════════════════════════════
# 01 — Architecture globale
# ══════════════════════════════════════════════════════════════════════
def fig_architecture_globale():
    fig, ax = new_fig(14.6, 7.2)
    title(ax, "Arc-FaultNet V2 — Architecture globale",
          "Un cycle de courant 50 Hz en entrée  ·  une probabilité d'arc en sortie  ·  ~350 k paramètres")

    band(ax, 0.015, 0.10, 0.155, 0.74, SKY_L, "ENTRÉE", SKY)
    band(ax, 0.185, 0.10, 0.205, 0.74, TEAL_L, "FRONT-END PHYSIQUE", TEAL)
    band(ax, 0.405, 0.10, 0.36, 0.74, INDIGO_L, "RÉSEAU PROFOND", INDIGO)
    band(ax, 0.78, 0.10, 0.205, 0.74, VIOLET_L, "DÉCISION", VIOLET)

    # entrée
    box(ax, 0.025, 0.55, 0.135, 0.17, SKY_L, SKY, "Courant I(t)",
        "1 cycle 50 Hz", ssize=8)
    box(ax, 0.025, 0.30, 0.135, 0.14, "white", GREY, "M = 2048",
        "éch. @ 102,4 kHz", ssize=8)
    arrow(ax, (0.0925, 0.55), (0.0925, 0.44), color=SKY)

    # front-end
    box(ax, 0.20, 0.52, 0.175, 0.22, TEAL_L, TEAL, "4 transformées de I(t)",
        "I_norm · |ΔI| · TKEO\nRMS glissant", ssize=8, tsize=10.5)
    box(ax, 0.20, 0.22, 0.175, 0.16, TEAL_L, TEAL, "STFT(I)", "log-puissance · 65×31")
    arrow(ax, (0.16, 0.40), (0.20, 0.62), color=INK, rad=-0.15)
    arrow(ax, (0.16, 0.35), (0.20, 0.30), color=INK, rad=0.1)
    chip(ax, 0.2875, 0.475, "x₁d (B,4,2048)", TEAL)
    chip(ax, 0.2875, 0.175, "x₂d (B,1,65,31)", TEAL)

    # réseau
    box(ax, 0.415, 0.55, 0.16, 0.19, "white", TEAL, "Branche temporelle",
        "Conv1d ×3 → f_t (128)", tsize=10.5, ssize=7.8)
    box(ax, 0.415, 0.24, 0.16, 0.19, "white", ORANGE, "Branche spectrale",
        "FreqGate + Conv2d ×3\n→ f_s (128)", tsize=10.5, ssize=7.8)
    box(ax, 0.615, 0.40, 0.13, 0.18, VIOLET_L, VIOLET, "Cross-Attention",
        "f_t ⊕ f_s → z (128)", tsize=10.5, ssize=7.8)
    arrow(ax, (0.375, 0.63), (0.415, 0.645), color=INK)
    arrow(ax, (0.375, 0.30), (0.415, 0.335), color=INK)
    arrow(ax, (0.575, 0.645), (0.615, 0.53), color=INK, rad=-0.15)
    arrow(ax, (0.575, 0.335), (0.615, 0.45), color=INK, rad=0.15)
    chip(ax, 0.68, 0.355, "z (B,128)", VIOLET)

    # décision
    box(ax, 0.80, 0.56, 0.165, 0.16, "white", INDIGO, "Tête FC",
        "entraînement\n128 → 64 → 1", ssize=7.8)
    box(ax, 0.80, 0.34, 0.165, 0.16, "white", GREEN, "XGBoost sur z",
        "déploiement\nP(arc) calibrée", ssize=7.8)
    box(ax, 0.80, 0.13, 0.165, 0.15, RED_L, RED, "P(arc) → trip",
        "AFDD · IEC 62606", tc=RED)
    arrow(ax, (0.745, 0.49), (0.80, 0.62), color=INK, rad=-0.15)
    arrow(ax, (0.745, 0.475), (0.80, 0.42), color=INK, rad=0.1)
    arrow(ax, (0.8825, 0.34), (0.8825, 0.28), color=RED, lw=2.4)
    arrow(ax, (0.8825, 0.56), (0.8825, 0.50), color=GREY, lw=1.8)

    ax.text(0.5, 0.045, "Tout part de I(t) seul — V(t) sert uniquement à la segmentation des cycles et V_arc au labelling (jamais en entrée du modèle).",
            ha="center", fontsize=9.5, style="italic", color=MUT)
    save(fig, "01_architecture_globale.png")


# ══════════════════════════════════════════════════════════════════════
# 02 — Data flow (tenseurs)
# ══════════════════════════════════════════════════════════════════════
def fig_data_flow():
    fig, ax = new_fig(14.6, 7.6)
    title(ax, "Flux de données — formes des tenseurs",
          "Comment un cycle change de forme à chaque étape, de l'échantillon brut à la probabilité")

    rows = [
        ("Cycle brut I(t)", "(B,1,2048)", SKY_L, SKY,
         "un cycle 50 Hz · 2048 points"),
        ("4 canaux dérivés", "(B,4,2048)", TEAL_L, TEAL,
         "I_norm · |ΔI| · TKEO · RMS\nnormalisés par RMS du cycle"),
        ("Conv1d ×3 + pooling", "(B,128,64)", TEAL_L, TEAL,
         "4→32→64→128 filtres\nlongueur 2048 → 64"),
        ("GAP temporel → f_t", "(B,128)", TEAL_L, TEAL,
         "résumé de la branche 1D"),
    ]
    rows2 = [
        ("STFT log-puissance", "(B,1,65,31)", ORANGE_L, ORANGE,
         "n_fft=128 · hop=64\n65 fréq. × 31 pas de temps"),
        ("FrequencyGate", "(B,1,65,31)", ORANGE_L, ORANGE,
         "masque doux appris\n(remplace la tranche fixe)"),
        ("Conv2d ×3 asym.", "(B,128,4,64)", ORANGE_L, ORANGE,
         "temps ↓4 · fréq. préservée\n4 groupes de bandes"),
        ("GAP temporel → f_s", "(B,128)", ORANGE_L, ORANGE,
         "résumé de la branche 2D"),
    ]

    def column(x0, rows_, label, lc):
        ax.text(x0 + 0.14, 0.85, label, ha="center", fontsize=12.5,
                fontweight="bold", color=lc)
        y = 0.70
        for i, (name, shape, fc, ec, desc) in enumerate(rows_):
            box(ax, x0, y, 0.28, 0.115, fc, ec, name, desc, tsize=10.5, ssize=7.8)
            chip(ax, x0 + 0.335, y + 0.057, shape, ec, size=8.5)
            if i < len(rows_) - 1:
                arrow(ax, (x0 + 0.14, y - 0.004), (x0 + 0.14, y - 0.040),
                      color=ec, lw=2.2)
            y -= 0.158
        return y

    column(0.05, rows, "Voie temporelle (1D)", TEAL)
    column(0.56, rows2, "Voie spectrale (2D)", ORANGE)

    # fusion
    box(ax, 0.345, 0.035, 0.31, 0.115, VIOLET_L, VIOLET,
        "Cross-Attention → z (B,128) → P(arc)",
        "deux gates conditionnés mutuellement,\npuis tête FC / XGBoost",
        tsize=10.5, ssize=8)
    arrow(ax, (0.19, 0.225), (0.42, 0.155), color=TEAL, rad=-0.2, lw=2.4)
    arrow(ax, (0.70, 0.225), (0.58, 0.155), color=ORANGE, rad=0.2, lw=2.4)
    save(fig, "02_data_flow.png")


# ══════════════════════════════════════════════════════════════════════
# 03 — Pipeline de travail complet
# ══════════════════════════════════════════════════════════════════════
def fig_pipeline():
    fig, ax = new_fig(14.8, 7.8)
    title(ax, "Pipeline complet — de l'acquisition à la décision",
          "Chaîne de traitement réellement implémentée dans le dépôt")

    steps_top = [
        ("1 · Acquisition", "Oscilloscope LeCroy · 1 MHz\nC1=V_ligne, C2=V_arc, C3=I", SKY_L, SKY),
        ("2 · Segmentation", "passages par zéro de V(t)\n→ cycles 50 Hz (20 000 pts)", SKY_L, SKY),
        ("3 · Labelling", "oracle V_arc : |V_arc| > seuil\n→ arc / normal (3 zones)", SKY_L, SKY),
        ("4 · Décimation", "20 000 → 2 048 pts / cycle\n(102,4 kHz) + contrôle QA", SKY_L, SKY),
    ]
    steps_bot = [
        ("5 · Front-end", "4 transformées de I(t)\n+ STFT — à la volée (dataset.py)", TEAL_L, TEAL),
        ("6 · Entraînement", "ArcFaultNetV2 + tête FC\nBCE · early stopping · seeds", INDIGO_L, INDIGO),
        ("7 · Prédiction", "P(arc) → arc / non arc", VIOLET_L, VIOLET),
        ("8 · Évaluation", "Acc / F1 / Recall / FP\nsingle split & GroupKFold", GREEN_L, GREEN),
    ]

    w, h, gap = 0.215, 0.155, 0.022
    x = 0.03
    boxes_top = []
    for name, sub, fc, ec in steps_top:
        box(ax, x, 0.60, w, h, fc, ec, name, sub, tsize=11.5, ssize=8.2)
        boxes_top.append(x)
        x += w + gap
    for i in range(len(boxes_top) - 1):
        arrow(ax, (boxes_top[i] + w + 0.002, 0.6775),
              (boxes_top[i + 1] - 0.002, 0.6775), lw=2.6)

    x = 0.03
    boxes_bot = []
    for name, sub, fc, ec in steps_bot:
        box(ax, x, 0.30, w, h, fc, ec, name, sub, tsize=11.5, ssize=8.2)
        boxes_bot.append(x)
        x += w + gap
    for i in range(len(boxes_bot) - 1):
        arrow(ax, (boxes_bot[i] + w + 0.002, 0.3775),
              (boxes_bot[i + 1] - 0.002, 0.3775), lw=2.6)

    # liaison rangée haute → basse : coude propre entre les deux rangées
    xs, xe = boxes_top[-1] + w / 2, boxes_bot[0] + w / 2
    ax.plot([xs, xs, xe], [0.595, 0.527, 0.527], color=INK, lw=2.6,
            solid_capstyle="round", zorder=4)
    arrow(ax, (xe, 0.527), (xe, 0.462), lw=2.6)

    ax.text(0.03, 0.13,
            "Données : campagnes exp. réelles (charges résistives, SMPS, moteurs, multi-charges)\n"
            "combined_dataset_2048 : 10 860 cycles · 5 898 normaux / 4 962 arcs",
            ha="left", fontsize=9.5, color=MUT)
    save(fig, "03_pipeline_travail.png")


# ══════════════════════════════════════════════════════════════════════
# 04 — Branche 1D
# ══════════════════════════════════════════════════════════════════════
def fig_branche_1d():
    fig, ax = new_fig(14.4, 7.6)
    title(ax, "Branche temporelle (1D) — 4 vues physiques du courant",
          "Chaque canal isole un phénomène d'arc différent · tous normalisés par le RMS du cycle (invariance charge)")

    chans = [
        ("C1 · I_norm", "I(t) / RMS cycle",
         "forme globale : harmoniques, amplitude", SKY, SKY_L),
        ("C2 · |ΔI|", "|I[n] − I[n−1]|",
         "discontinuités locales : fronts raides de l'arc", VIOLET, VIOLET_L),
        ("C3 · TKEO", "I[n]² − I[n−1]·I[n+1]",
         "énergie instantanée : ignition / extinction", ORANGE, ORANGE_L),
        ("C4 · RMS glissant", "fenêtre M/4",
         "enveloppe : épaule plate, creux de courant", AMBER, AMBER_L),
    ]
    y = 0.68
    for name, formula, why, ec, fc in chans:
        ax.add_patch(FancyBboxPatch(
            (0.03, y), 0.28, 0.15, boxstyle="round,pad=0.0,rounding_size=0.014",
            linewidth=2.2, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(0.17, y + 0.112, name, ha="center", fontsize=11.5,
                fontweight="bold", color=INK, zorder=4)
        ax.text(0.17, y + 0.068, formula, ha="center", fontsize=8.6,
                color=INK, zorder=4)
        ax.text(0.17, y + 0.028, why, ha="center", fontsize=7.8,
                style="italic", color=MUT, zorder=4)
        arrow(ax, (0.312, y + 0.075), (0.358, 0.50), color=ec, lw=2.0,
              rad=-0.08 if y + 0.075 > 0.50 else 0.08)
        y -= 0.17

    box(ax, 0.36, 0.42, 0.12, 0.16, TEAL_L, TEAL, "Empilement",
        "x₁d (B, 4, 2048)", tsize=10.5, ssize=8)

    convs = [
        ("Conv1d k=16", "4→32 · GELU\npool 4"),
        ("Conv1d k=8", "32→64 · GELU\npool 4"),
        ("Conv1d k=4", "64→128\nGELU"),
    ]
    x = 0.515
    for name, sub in convs:
        box(ax, x, 0.42, 0.105, 0.16, "white", TEAL, name, sub, tsize=9.5, ssize=7.6)
        arrow(ax, (x - 0.032, 0.50), (x - 0.002, 0.50), color=TEAL, lw=2.2)
        x += 0.117
    box(ax, x, 0.42, 0.10, 0.16, TEAL_L, TEAL, "GAP", "f_t (B, 128)",
        tsize=10.5, ssize=8)
    arrow(ax, (x - 0.032, 0.50), (x - 0.002, 0.50), color=TEAL, lw=2.2)

    box(ax, 0.36, 0.07, 0.60, 0.20, GREEN_L, GREEN,
        "Pourquoi PAS de filtres de Gabor ici ?",
        "L'ablation a montré 0 contribution (−0,55 pt vs Conv1d standard) :\n"
        "l'arc est impulsif et apériodique, le prior oscillatoire des Gabor\n"
        "est contre-productif → convolutions 1D libres.",
        tsize=12, ssize=9)
    save(fig, "04_branche_1d.png")


# ══════════════════════════════════════════════════════════════════════
# 05 — Branche 2D
# ══════════════════════════════════════════════════════════════════════
def fig_branche_2d():
    fig, ax = new_fig(14.4, 7.4)
    title(ax, "Branche spectrale (2D) — le crépitement haute fréquence",
          "L'arc injecte un burst large-bande visible dans le spectrogramme — invisible cycle par cycle en 1D")

    steps = [
        ("STFT(I)", "log-puissance\nn_fft=128 · hop=64\n→ (B, 1, 65, 31)", ORANGE_L, ORANGE),
        ("FrequencyGate", "masque doux APPRIS\nsur l'axe fréquence\nremplace la tranche HF fixe", VIOLET_L, VIOLET),
        ("Conv2d ×3", "1→32→64→128\npooling ASYMÉTRIQUE\ntemps ↓4, fréquence préservée", ORANGE_L, ORANGE),
        ("4 groupes de bandes", "AdaptiveAvgPool (4, 64)\npuis projection 1×1\n→ (B, 128, 64)", ORANGE_L, ORANGE),
        ("GAP", "f_s  (B, 128)", ORANGE_L, ORANGE),
    ]
    w, h, gap = 0.172, 0.21, 0.022
    x = 0.03
    for i, (name, sub, fc, ec) in enumerate(steps):
        ww = w if i < 4 else 0.12
        box(ax, x, 0.50, ww, h, fc, ec, name, sub, tsize=11.5, ssize=8)
        if i < len(steps) - 1:
            arrow(ax, (x + ww + 0.002, 0.605), (x + ww + gap - 0.002, 0.605), lw=2.6)
        x += ww + gap

    box(ax, 0.04, 0.13, 0.43, 0.22, VIOLET_L, VIOLET,
        "Innovation : FrequencyGate",
        "V1 découpait une bande [2–100 kHz] codée EN DUR.\n"
        "Les bandes utiles dépendent de la charge (SMPS ≠ résistif).\n"
        "→ le gate apprend OÙ regarder, par charge, depuis les données.",
        tsize=12, ssize=9)
    box(ax, 0.53, 0.13, 0.43, 0.22, ORANGE_L, ORANGE,
        "Innovation : pooling asymétrique",
        "V1 utilisait MaxPool 2×2 symétrique → résolution fréquentielle détruite.\n"
        "La signature d'arc est FINE en fréquence, redondante en temps.\n"
        "→ on compresse le temps (×4), on préserve la fréquence.",
        tsize=12, ssize=9)
    save(fig, "05_branche_2d.png")


# ══════════════════════════════════════════════════════════════════════
# 06 — Futures implémentations
# ══════════════════════════════════════════════════════════════════════
def fig_futures():
    fig, ax = new_fig(14.4, 7.4)
    title(ax, "Implémentations futures — extension multi-cycles",
          "Le modèle actuel traite 1 cycle ; la spec V2 complète prévoit une fenêtre de N = 50 cycles (IEC 62606)")

    box(ax, 0.03, 0.62, 0.16, 0.16, GREEN_L, GREEN, "Aujourd'hui",
        "1 cycle → P(arc)\nV2 single-cycle entraînée")

    futs = [
        ("Δ inter-cycles", "ΔI_k = I_k − I_(k−1)\nsur les 50 cycles\n(résidu Dowalla)", "encodage du changement\ncycle à cycle"),
        ("Scalaires Dowalla", "8 descripteurs / paire :\nE_mod, ED, MSSD, MCC,\nCRC, ZCP, E_mod_V, ED_V", "physique éprouvée\n(Dowalla et al. 2023)"),
        ("BiGRU + attention", "séquence de 49 tokens\n→ contexte (B, 128)\nattention par cycle", "distingue burst / éparpillé /\npériodique (moteur)"),
        ("Compteur IEC", "règle 62606 :\n≥ 7 cycles d'arc\n→ déclenchement", "conformité normative\nexplicite"),
    ]
    x, w, h = 0.245, 0.17, 0.20
    for name, sub, why in futs:
        ax.add_patch(FancyBboxPatch((x, 0.58), w, h,
                     boxstyle="round,pad=0.0,rounding_size=0.014",
                     linewidth=2.0, edgecolor=GREY, facecolor=GREY_L,
                     linestyle="--", zorder=3))
        ax.text(x + w / 2, 0.58 + h * 0.72, name, ha="center", fontsize=11.5,
                fontweight="bold", color=INK, zorder=4)
        ax.text(x + w / 2, 0.58 + h * 0.34, sub, ha="center", fontsize=7.8,
                color=MUT, zorder=4)
        ax.text(x + w / 2, 0.50, why, ha="center", fontsize=8.2,
                style="italic", color=GREEN, zorder=4)
        x += w + 0.018
    arrow(ax, (0.19, 0.70), (0.245, 0.68), lw=2.4, ls="--", color=GREY)

    box(ax, 0.06, 0.13, 0.42, 0.24, AMBER_L, AMBER,
        "Pré-requis : dataset multi-cycles",
        "Il faut des fenêtres (B, N, M) de 50 cycles CONSÉCUTIFS\n"
        "du même enregistrement — la base actuelle stocke des cycles\n"
        "individuels. Construction du dataset = prochaine étape data.",
        tsize=12, ssize=9)
    box(ax, 0.53, 0.13, 0.42, 0.24, GREEN_L, GREEN,
        "Déjà préparé dans le code actuel",
        "· La branche spectrale V2 sera PARTAGÉE telle quelle\n"
        "· La branche temporelle = encodeur par cycle réutilisable\n"
        "· L'embedding z reste l'interface vers XGBoost",
        tsize=12, ssize=9)
    save(fig, "06_futures_implementations.png")


# ══════════════════════════════════════════════════════════════════════
# 07 — Comparaison V1 vs V2
# ══════════════════════════════════════════════════════════════════════
def fig_comparaison():
    fig, ax = new_fig(14.8, 7.8)
    title(ax, "Ancienne vs nouvelle architecture",
          "Mêmes données (combined_dataset_2048), même seed (4), mêmes hyperparamètres — seule l'architecture change")

    # colonnes
    def col(x, name, ec, fc, rows, tc=None):
        band(ax, x, 0.30, 0.40, 0.52, fc, "", ec)
        ax.text(x + 0.20, 0.785, name, ha="center", fontsize=14,
                fontweight="bold", color=tc or ec)
        y = 0.715
        for k, v in rows:
            ax.text(x + 0.02, y, k, ha="left", fontsize=9, fontweight="bold",
                    color=INK)
            ax.text(x + 0.155, y, v, ha="left", fontsize=9, color=MUT)
            y -= 0.052

    col(0.045, "V1 — Gabor + Joint Attention", GREY, GREY_L, tc=MUT, rows=[
        ("Entrée", "[V_ligne, I] bruts (2 canaux)"),
        ("Conv 1D", "ParametricConv1d (Gabor f₀, σ)"),
        ("Spectral", "STFT 2 canaux · tranche HF fixe"),
        ("Pooling 2D", "MaxPool 2×2 symétrique"),
        ("Fusion", "CAM joint partagé [:C]/[C:] (ambigu)"),
        ("Tête", "FC uniquement"),
        ("Params", "320 609"),
    ])
    col(0.555, "V2 — front-end physique", INDIGO, INDIGO_L, [
        ("Entrée", "I(t) seul → 4 canaux dérivés"),
        ("Conv 1D", "Conv1d standard + GELU"),
        ("Spectral", "STFT(I) · FrequencyGate appris"),
        ("Pooling 2D", "asymétrique (fréquence préservée)"),
        ("Fusion", "Cross-Attention à 2 gates conditionnés"),
        ("Tête", "FC (train) + XGBoost (déploiement)"),
        ("Params", "350 693 (+9 %)"),
    ])
    arrow(ax, (0.46, 0.56), (0.54, 0.56), color=INDIGO, lw=3.2)

    # métriques (test set, run 03/06 vs 10/06, seed 4)
    metrics = [("Accuracy", 93.74, 97.24), ("F1", 92.37, 96.82),
               ("Recall", 87.15, 96.75), ("Spécificité", 98.81, 97.61)]
    x = 0.075
    ax.text(0.5, 0.245, "Test set — run 03/06 (V1) vs run 10/06 (V2), seed 4",
            ha="center", fontsize=10, fontweight="bold", color=INK)
    for name, v1, v2 in metrics:
        d = v2 - v1
        dc = GREEN if d > 0 else RED
        ax.text(x + 0.10, 0.195, name, ha="center", fontsize=10,
                fontweight="bold", color=INK)
        ax.text(x + 0.10, 0.145, f"{v1:.2f} %", ha="center", fontsize=11,
                color=MUT)
        ax.text(x + 0.10, 0.095, f"{v2:.2f} %", ha="center", fontsize=13,
                fontweight="bold", color=INK)
        ax.text(x + 0.10, 0.045, f"{d:+.2f} pts", ha="center", fontsize=10.5,
                fontweight="bold", color=dc)
        x += 0.22
    save(fig, "07_comparaison_v1_v2.png")


# ══════════════════════════════════════════════════════════════════════
# 08 — Meilleur modèle (accuracy ↑, FP ↓)
# ══════════════════════════════════════════════════════════════════════
def fig_meilleur_modele():
    fig = plt.figure(figsize=(14.4, 7.4))
    fig.suptitle("Quel est le meilleur modèle ?  Accuracy maximale ET faux positifs minimaux",
                 fontsize=19, fontweight="bold", color=INK, y=0.97)
    fig.text(0.5, 0.905,
             "Runs single sur combined_dataset_2048 — le taux de FP = 1 − spécificité (déclenchements intempestifs)",
             ha="center", fontsize=11, style="italic", color=MUT)

    runs = [
        ("V1 · 03/06\nseed 4", 93.74, 1.19, GREY),
        ("V2 · 10/06\nseed 2", 94.48, 0.58, INDIGO),
        ("V2 · 10/06\nseed 4", 97.24, 2.39, INDIGO),
        ("V2 · 10/06\nseed 42", 97.61, 0.34, GREEN),
    ]
    names = [r[0] for r in runs]
    accs = [r[1] for r in runs]
    fps = [r[2] for r in runs]
    cols = [r[3] for r in runs]

    ax1 = fig.add_axes([0.07, 0.14, 0.40, 0.62])
    bars = ax1.bar(names, accs, color=cols, width=0.55)
    ax1.set_ylim(90, 100)
    ax1.set_title("Accuracy (%)", fontsize=13, fontweight="bold", color=INK)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(labelsize=9)
    for b, v in zip(bars, accs):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                 ha="center", fontsize=10, fontweight="bold", color=INK)

    ax2 = fig.add_axes([0.56, 0.14, 0.40, 0.62])
    bars = ax2.bar(names, fps, color=cols, width=0.55)
    ax2.set_ylim(0, 3.0)
    ax2.set_title("Taux de faux positifs (%)  —  plus bas = mieux",
                  fontsize=13, fontweight="bold", color=INK)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(labelsize=9)
    for b, v in zip(bars, fps):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}",
                 ha="center", fontsize=10, fontweight="bold", color=INK)

    fig.text(0.5, 0.035,
             "★ Meilleur modèle : V2 seed 42 — Accuracy 97,61 % · FP 0,34 % · Précision 99,59 % · F1 97,37 %  "
             "(runs/arcfaultnet_v2_single_20260610_124344)",
             ha="center", fontsize=11.5, fontweight="bold", color=GREEN)
    fig.savefig(OUT / "08_meilleur_modele.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  wrote 08_meilleur_modele.png")


# ══════════════════════════════════════════════════════════════════════
# 09 — Innovation scientifique
# ══════════════════════════════════════════════════════════════════════
def fig_innovation():
    fig, ax = new_fig(14.4, 7.6)
    title(ax, "Innovation scientifique — ce que cette architecture apporte",
          "Quatre contributions défendables, chacune reliée à une propriété physique de l'arc série")

    cards = [
        ("1 · Front-end physique multi-vues", TEAL, TEAL_L,
         "4 transformées complémentaires de I(t), chacune ciblant\n"
         "une échelle du phénomène d'arc (forme, front, énergie,\n"
         "enveloppe) — au lieu d'un signal brut unique.\n"
         "Normalisation par cycle → invariance à la charge."),
        ("2 · FrequencyGate appris", VIOLET, VIOLET_L,
         "Le modèle apprend les bandes de fréquence utiles\n"
         "au lieu d'une tranche codée en dur — les bandes\n"
         "discriminantes dépendent du type de charge\n"
         "(SMPS ≠ résistif ≠ moteur)."),
        ("3 · Cross-Attention corrigée", INDIGO, INDIGO_L,
         "Deux gates par branche, conditionnés sur les DEUX\n"
         "résumés globaux — corrige l'ambiguïté d'ordre des\n"
         "canaux du CAM joint de V1 et rend le guidage\n"
         "inter-branches traçable."),
        ("4 · Décision hybride deep + arbre", GREEN, GREEN_L,
         "Le CNN apprend la représentation (z, 128-d) ;\n"
         "XGBoost prend la décision : meilleure généralisation\n"
         "sur datasets moyens, P(arc) calibrée pour le seuil\n"
         "de trip, importance des features interprétable."),
    ]
    pos = [(0.04, 0.46), (0.52, 0.46), (0.04, 0.10), (0.52, 0.10)]
    for (name, ec, fc, body), (x, y) in zip(cards, pos):
        box(ax, x, y, 0.44, 0.30, fc, ec, "", None)
        ax.text(x + 0.022, y + 0.255, name, ha="left", fontsize=13.5,
                fontweight="bold", color=ec, zorder=5)
        ax.text(x + 0.022, y + 0.125, body, ha="left", va="center",
                fontsize=9.4, color=INK, zorder=5)
    save(fig, "09_innovation_scientifique.png")


# ══════════════════════════════════════════════════════════════════════
# 10 — Choix des transformées → généralisation (contenu report.js)
# ══════════════════════════════════════════════════════════════════════
def fig_transformees_generalisation():
    fig, ax = new_fig(15.0, 8.4)
    title(ax, "Pourquoi ces transformées ? — l'argument central : la généralisation",
          "Chaque canal domine pour un type de charge différent ; l'attention de canal s'adapte SANS connaître la charge")

    # tableau canaux — grille simple et alignée
    col_labels = ["Canal", "Définition", "Phénomène capté", "Dominant pour"]
    cell_data = [
        ["C1 · I_norm", "I(t) / RMS cycle",
         "forme globale, harmoniques — canal d'ancrage", "tous les types"],
        ["C2 · |ΔI|", "|I[n] − I[n−1]|",
         "discontinuités locales : fronts d'arc, spikes isolés", "SMPS, électronique"],
        ["C3 · TKEO", "I[n]² − I[n−1]·I[n+1]",
         "énergie instantanée : ignition / extinction (< ms)", "inductif, moteur"],
        ["C4 · RMS glissant", "fenêtre M/4",
         "enveloppe : dépression d'amplitude, épaule plate", "résistif (lampe, four)"],
    ]
    row_colors = [SKY_L, VIOLET_L, ORANGE_L, AMBER_L]
    row_edges = [SKY, VIOLET, ORANGE, AMBER]

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        colWidths=[0.13, 0.19, 0.42, 0.18],
        cellLoc="left",
        loc="center",
        bbox=[0.04, 0.42, 0.92, 0.36],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    tbl.scale(1.0, 2.1)

    nrows, ncols = len(cell_data) + 1, len(col_labels)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(1.0)
        cell.PAD = 0.06
        txt = cell.get_text()
        txt.set_ha("left")
        txt.set_va("center")
        if row == 0:
            cell.set_facecolor(INK)
            cell.set_height(0.11)
            txt.set_color("white")
            txt.set_fontweight("bold")
            txt.set_fontsize(9.5)
        else:
            cell.set_height(0.10)
            if col == 0:
                cell.set_facecolor(row_colors[row - 1])
                txt.set_color(row_edges[row - 1])
                txt.set_fontweight("bold")
                txt.set_fontsize(9.2)
            elif col == 3:
                cell.set_facecolor(row_colors[row - 1])
                txt.set_color(INK)
                txt.set_fontsize(8.8)
            else:
                cell.set_facecolor("white")
                txt.set_color(INK)
                txt.set_fontsize(8.4 if col == 2 else 8.6)

    # mécanisme généralisation
    box(ax, 0.03, 0.155, 0.45, 0.16, INDIGO_L, INDIGO,
        "Attention de canal — adaptation implicite",
        "β = σ(MLP(AvgPool) + MLP(MaxPool)) ∈ (0,1)⁴\n"
        "Pour CHAQUE fenêtre, le modèle re-pondère les 4 canaux selon\n"
        "le contenu du signal — aucune connaissance a priori de la charge.",
        tsize=11.5, ssize=8.6)
    box(ax, 0.52, 0.155, 0.45, 0.16, GREEN_L, GREEN,
        "Résultat : généralisation inter-charges",
        "· charge inconnue → normalisation RMS + auto-pondération β\n"
        "· faible puissance (60 W) → perturbation relative au RMS propre\n"
        "· multi-charges → la vue la plus contrastée domine",
        tsize=11.5, ssize=8.6)

    ax.text(0.5, 0.08,
            "C'est LE point central de la défense : la robustesse inter-charges ne vient pas du volume de données,\n"
            "elle est construite DANS la représentation (normalisation par cycle + vues complémentaires + pondération apprise).",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)
    save(fig, "10_transformees_generalisation.png")


# ══════════════════════════════════════════════════════════════════════
# 11 — Évolution de l'architecture dans le temps
# ══════════════════════════════════════════════════════════════════════
def fig_evolution():
    fig, ax = new_fig(14.8, 8.2)
    title(ax, "Évolution de l'architecture — ce qui a été ajouté et retiré",
          "Chaque modification est motivée par un résultat mesuré, pas par une intuition")

    TLX = 0.095  # x de la ligne de temps verticale
    ax.plot([TLX, TLX], [0.115, 0.845], color=GREY, lw=3, zorder=2,
            solid_capstyle="round")

    def stage(y, date, name, name_c, lines, dot_c):
        ax.scatter([TLX], [y], s=160, color=dot_c, zorder=5,
                   edgecolor="white", linewidth=2)
        chip(ax, 0.048, y, date, dot_c, size=8.5)
        ax.text(0.125, y + 0.012, name, ha="left", va="bottom",
                fontsize=12.5, fontweight="bold", color=name_c)
        dy = -0.014
        for kind, txt in lines:
            c = {"plus": GREEN, "moins": RED, "info": MUT}[kind]
            pre = {"plus": "+  ", "moins": "−  ", "info": ""}[kind]
            w = "bold" if kind in ("plus", "moins") else "normal"
            ax.text(0.125, y + dy, pre + txt, ha="left", va="top",
                    fontsize=9.2, color=c, fontweight=w)
            dy -= 0.033

    stage(0.83, "mai", "V1 initiale — Gabor + Joint Attention", INK, [
        ("info", "entrée [V_ligne, I] bruts · ParametricConv1d (Gabor) · STFT 2 canaux, tranche HF fixe · MaxPool 2×2 · CAM joint · tête FC"),
    ], GREY)

    stage(0.715, "26–29 mai", "Attention de canal (Squeeze-and-Excitation)", TEAL, [
        ("plus", "blocs SE 1D & 2D — recalibrage appris de l'importance des canaux"),
        ("info", "acc ≈ 96,4 % (combined_dataset, seed 3) — l'attention de canal est confirmée utile"),
    ], TEAL)

    stage(0.575, "03/06", "Étude d'ablation — le verdict sur Gabor", AMBER, [
        ("info", "conv standard 96,68 %  vs  Gabor 96,13 % : le prior oscillatoire n'apporte rien (l'arc est impulsif, apériodique)"),
        ("moins", "décision : retirer les filtres de Gabor"),
    ], AMBER)

    stage(0.44, "début juin", "Refonte des données", SKY, [
        ("plus", "combined_dataset_2048 : 10 860 cycles · 2 048 pts/cycle (102,4 kHz)"),
        ("info", "résolution suffisante pour le crépitement HF, taille mémoire maîtrisée"),
    ], SKY)

    stage(0.30, "10/06", "V2 — front-end physique + fusion corrigée", INDIGO, [
        ("moins", "V(t) en entrée  ·  filtres de Gabor  ·  tranche HF fixe  ·  MaxPool symétrique  ·  CAM joint ambigu"),
        ("plus", "4 canaux dérivés de I(t)  ·  FrequencyGate appris  ·  pooling asymétrique  ·  Cross-Attention à 2 gates  ·  tête XGBoost"),
        ("info", "acc 97,24 % (seed 4)  ·  meilleur run : 97,61 % et 0,34 % FP (seed 42)"),
    ], INDIGO)

    ax.text(0.5, 0.055,
            "Fil conducteur : chaque composant retiré était un a priori non vérifié ; chaque composant ajouté encode une propriété physique de l'arc.",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)
    save(fig, "11_evolution_architecture.png")


# ══════════════════════════════════════════════════════════════════════
# 12 — Fusion : Cross-Attention (V2) vs Joint Attention (V1)
# ══════════════════════════════════════════════════════════════════════
def fig_fusion_mecanisme():
    """Slide 12 — le mécanisme V1 vs V2, tout en flèches, très peu de texte."""
    fig, ax = new_fig(15.0, 8.4)
    title(ax, "Fusion des branches — Cross-Attention au lieu de Joint Attention",
          "À gauche : un seul vecteur de poids coupé en deux.  À droite : deux gates dédiés, chacun conditionné par les DEUX branches.")

    # ── V1 ────────────────────────────────────────────────────────────
    band(ax, 0.025, 0.13, 0.445, 0.745, GREY_L, "", GREY)
    ax.text(0.2475, 0.845, "V1 — Joint Attention", ha="center",
            fontsize=14, fontweight="bold", color=MUT)

    box(ax, 0.06, 0.70, 0.16, 0.075, TEAL_L, TEAL, "f_t", "(B, 128)", tsize=12, ssize=8.5)
    box(ax, 0.275, 0.70, 0.16, 0.075, ORANGE_L, ORANGE, "f_s", "(B, 128)", tsize=12, ssize=8.5)
    box(ax, 0.15, 0.565, 0.195, 0.065, "white", GREY, "concat (B, 256)", tsize=10.5)
    arrow(ax, (0.14, 0.70), (0.21, 0.632), color=TEAL, lw=2.6)
    arrow(ax, (0.355, 0.70), (0.285, 0.632), color=ORANGE, lw=2.6)

    box(ax, 0.125, 0.43, 0.245, 0.07, "white", GREY,
        "CAM joint  →  β (B, 256)", tsize=10.5)
    arrow(ax, (0.2475, 0.563), (0.2475, 0.502), lw=2.6)

    # le "coup de ciseaux"
    ax.plot([0.2475, 0.2475], [0.415, 0.295], color=RED, lw=2.0, ls=(0, (4, 3)),
            zorder=6)
    ax.text(0.2475, 0.355, "✂", ha="center", va="center", fontsize=20,
            color=RED, zorder=7, rotation=90)
    box(ax, 0.075, 0.215, 0.155, 0.07, "white", GREY, "β[:128] ⊙ f_t", tsize=10)
    box(ax, 0.265, 0.215, 0.155, 0.07, "white", GREY, "β[128:] ⊙ f_s", tsize=10)
    arrow(ax, (0.19, 0.428), (0.1525, 0.287), color=GREY, lw=2.2, rad=0.15)
    arrow(ax, (0.305, 0.428), (0.3425, 0.287), color=GREY, lw=2.2, rad=-0.15)

    ax.text(0.2475, 0.165, "✗  découpage par position — arbitraire, non traçable",
            ha="center", fontsize=11, fontweight="bold", color=RED)

    arrow(ax, (0.478, 0.52), (0.515, 0.52), color=INDIGO, lw=4.0)

    # ── V2 ────────────────────────────────────────────────────────────
    band(ax, 0.525, 0.13, 0.45, 0.745, INDIGO_L, "", INDIGO)
    ax.text(0.75, 0.845, "V2 — RevisedCrossAttention", ha="center",
            fontsize=14, fontweight="bold", color=INDIGO)

    box(ax, 0.555, 0.70, 0.16, 0.075, TEAL_L, TEAL, "f_t", "(B, 128)", tsize=12, ssize=8.5)
    box(ax, 0.785, 0.70, 0.16, 0.075, ORANGE_L, ORANGE, "f_s", "(B, 128)", tsize=12, ssize=8.5)

    box(ax, 0.545, 0.50, 0.19, 0.09, TEAL_L, TEAL, "Gate β_t",
        "σ(MLP 256→128)", tsize=11, ssize=8.5)
    box(ax, 0.77, 0.50, 0.19, 0.09, ORANGE_L, ORANGE, "Gate β_s",
        "σ(MLP 256→128)", tsize=11, ssize=8.5)

    # flèches droites + flèches CROISÉES (le cœur du mécanisme)
    arrow(ax, (0.635, 0.698), (0.64, 0.592), color=TEAL, lw=2.6)
    arrow(ax, (0.865, 0.698), (0.865, 0.592), color=ORANGE, lw=2.6)
    arrow(ax, (0.675, 0.698), (0.845, 0.592), color=TEAL, lw=2.2, rad=-0.12)
    arrow(ax, (0.825, 0.698), (0.66, 0.592), color=ORANGE, lw=2.2, rad=0.12)
    chip(ax, 0.75, 0.655, "conditionnement croisé", INDIGO, size=8)

    # nœuds de modulation ⊙
    for cx, cc in ((0.64, TEAL), (0.865, ORANGE)):
        ax.scatter([cx], [0.40], s=620, facecolor="white", edgecolor=cc,
                   linewidth=2.4, zorder=6)
        ax.text(cx, 0.40, "⊙", ha="center", va="center", fontsize=15,
                fontweight="bold", color=cc, zorder=7)
    arrow(ax, (0.64, 0.498), (0.64, 0.432), color=TEAL, lw=2.4)
    arrow(ax, (0.865, 0.498), (0.865, 0.432), color=ORANGE, lw=2.4)
    # f_t et f_s contournent les gates pour rejoindre leur ⊙
    arrow(ax, (0.557, 0.715), (0.612, 0.41), color=TEAL, lw=2.0, rad=0.35, ls="--")
    arrow(ax, (0.943, 0.715), (0.893, 0.41), color=ORANGE, lw=2.0, rad=-0.35, ls="--")

    box(ax, 0.645, 0.245, 0.215, 0.075, VIOLET_L, VIOLET,
        "fusion  →  z (B, 128)", "Linear 256→128 + GELU", tsize=11, ssize=8.5)
    arrow(ax, (0.655, 0.385), (0.715, 0.322), color=VIOLET, lw=2.4, rad=0.1)
    arrow(ax, (0.85, 0.385), (0.79, 0.322), color=VIOLET, lw=2.4, rad=-0.1)

    ax.text(0.75, 0.165, "✓  poids dédiés par branche, conditionnés des deux côtés",
            ha="center", fontsize=11, fontweight="bold", color=GREEN)

    ax.text(0.5, 0.06,
            "β_t = σ(MLP([f_t ‖ f_s]))   ·   β_s = σ(MLP([f_t ‖ f_s]))   ·   z = GELU(W·[f_t⊙β_t ‖ f_s⊙β_s])",
            ha="center", fontsize=10.5, color=INK, fontweight="bold")
    save(fig, "12_fusion_cross_attention.png")


def fig_fusion_bilan():
    """Slide 13 — adaptation par échantillon (jauges) + balance gains/compromis."""
    fig, ax = new_fig(15.0, 8.4)
    title(ax, "Cross-Attention — l'adaptation en action, gains et compromis",
          "Les gates re-pondèrent chaque fenêtre selon son CONTENU — aucun ré-entraînement, aucune connaissance de la charge")

    def gauge(x, y, w, frac, color, label, value):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, 0.034, boxstyle="round,pad=0.0,rounding_size=0.015",
            fc="#e2e8f0", ec="none", zorder=3))
        ax.add_patch(FancyBboxPatch(
            (x, y), max(w * frac, 0.03), 0.034,
            boxstyle="round,pad=0.0,rounding_size=0.015",
            fc=color, ec="none", zorder=4))
        ax.text(x - 0.012, y + 0.017, label, ha="right", va="center",
                fontsize=11, fontweight="bold", color=color, zorder=5)
        ax.text(x + w + 0.012, y + 0.017, value, ha="left", va="center",
                fontsize=10, fontweight="bold", color=INK, zorder=5)

    def waveform(x0, y0, w, h, kind, color):
        t = np.linspace(0, 1, 400)
        if kind == "resistif":
            sig = np.clip(np.sin(2 * np.pi * t), -0.82, 0.82)
        else:
            rng = np.random.default_rng(7)
            sig = 0.7 * np.sin(2 * np.pi * t)
            spikes = rng.random(400) > 0.97
            sig[spikes] += rng.choice([-1, 1], spikes.sum()) * 0.55
        ax.plot(x0 + t * w, y0 + h / 2 + sig * h / 2, color=color, lw=1.3,
                zorder=5)

    # ── scénario A : résistif ─────────────────────────────────────────
    band(ax, 0.025, 0.45, 0.455, 0.40, SKY_L, "", SKY)
    ax.text(0.2525, 0.815, "Charge résistive (lampe, four)", ha="center",
            fontsize=13, fontweight="bold", color=SKY)
    waveform(0.055, 0.665, 0.14, 0.10, "resistif", SKY)
    arrow(ax, (0.205, 0.715), (0.245, 0.715), color=SKY, lw=2.4)
    ax.text(0.255, 0.755, "signature dans la FORME du cycle", ha="left",
            fontsize=9, style="italic", color=INK)
    gauge(0.285, 0.685, 0.13, 0.85, TEAL, "β_t", "0,85")
    gauge(0.285, 0.625, 0.13, 0.30, ORANGE, "β_s", "0,30")
    arrow(ax, (0.2525, 0.60), (0.2525, 0.555), color=SKY, lw=2.4)
    chip(ax, 0.2525, 0.515, "z s'appuie sur la branche temporelle", TEAL, size=9.5)

    # ── scénario B : SMPS ─────────────────────────────────────────────
    band(ax, 0.52, 0.45, 0.455, 0.40, VIOLET_L, "", VIOLET)
    ax.text(0.7475, 0.815, "Charge électronique (SMPS)", ha="center",
            fontsize=13, fontweight="bold", color=VIOLET)
    waveform(0.55, 0.665, 0.14, 0.10, "smps", VIOLET)
    arrow(ax, (0.70, 0.715), (0.74, 0.715), color=VIOLET, lw=2.4)
    ax.text(0.75, 0.755, "signature dans le SPECTRE (burst HF)", ha="left",
            fontsize=9, style="italic", color=INK)
    gauge(0.78, 0.685, 0.13, 0.40, TEAL, "β_t", "0,40")
    gauge(0.78, 0.625, 0.13, 0.90, ORANGE, "β_s", "0,90")
    arrow(ax, (0.7475, 0.60), (0.7475, 0.555), color=VIOLET, lw=2.4)
    chip(ax, 0.7475, 0.515, "z s'appuie sur la branche spectrale", ORANGE, size=9.5)

    ax.text(0.5, 0.425,
            "même modèle, mêmes poids — seule la pondération change, pilotée par le contenu",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)

    # ── balance gains / compromis ─────────────────────────────────────
    # fléau penché côté gains (plus lourd) ; les plateaux POSENT sur le fléau
    ax.plot([0.27, 0.73], [0.115, 0.185], color=INK, lw=3.5,
            solid_capstyle="round", zorder=4)
    ax.add_patch(plt.Polygon([[0.468, 0.045], [0.532, 0.045], [0.5, 0.150]],
                             closed=True, fc=GREY, ec="none", zorder=3))
    box(ax, 0.195, 0.118, 0.15, 0.055, GREEN_L, GREEN, "GAINS", tsize=11.5)
    box(ax, 0.655, 0.188, 0.15, 0.055, AMBER_L, AMBER, "COMPROMIS", tsize=10.5)

    gains = ["+ généralisation inter-charges (β pilotés par le contenu)",
             "+ 3,5 pts accuracy · + 9,6 pts recall (seed égal)",
             "+ β_t, β_s inspectables → décision traçable"]
    couts = ["− SAM spatial abandonné (granularité temporelle fine)",
             "− interactions de 1ᵉʳ ordre (pas d'attention QKV)",
             "− +132 k paramètres (+9 %)"]
    y = 0.375
    for s in gains:
        ax.text(0.03, y, s, ha="left", fontsize=9.6, fontweight="bold",
                color=GREEN)
        y -= 0.05
    y = 0.375
    for s in couts:
        ax.text(0.97, y, s, ha="right", fontsize=9.6, fontweight="bold",
                color=AMBER)
        y -= 0.05

    ax.text(0.5, 0.035,
            "le fléau penche : les compromis sont des choix délibérés adaptés à ~11 k cycles, les gains sont mesurés",
            ha="center", fontsize=9.5, style="italic", color=MUT)
    save(fig, "13_fusion_bilan.png")


# ══════════════════════════════════════════════════════════════════════
# 15 — Étude d'ablation : objectif et protocole
# ══════════════════════════════════════════════════════════════════════
def fig_ablation_introduction():
    fig, ax = new_fig(15.2, 8.6)
    title(ax, "Étude d'ablation — objectif et protocole",
          "Mesurer la contribution réelle de chaque composant de l'architecture V2")

    # ── Couleurs sobres ──
    BLOCK_FC = "#f1f5f9"   # fond des blocs normaux (gris très clair)
    BLOCK_EC = "#64748b"   # bordure des blocs normaux (slate)
    REMOVED_FC = "#e2e8f0"  # fond du bloc retiré
    REMOVED_EC = "#94a3b8"  # bordure du bloc retiré

    # ── ZONE HAUTE : schéma visuel V2 complète vs Variante ──────────
    ax.add_patch(FancyBboxPatch(
        (0.025, 0.44), 0.95, 0.44,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0, facecolor="#fafbfc", zorder=1))

    # ── Modèle complet (V2 référence) ──
    ax.text(0.175, 0.845, "V2 complète (référence)", ha="center",
            fontsize=13, fontweight="bold", color=INK, zorder=5)

    comp_blocks = [
        ("4 canaux dérivés", TEAL_L, TEAL),
        ("Branche STFT", ORANGE_L, ORANGE),
        ("FrequencyGate", VIOLET_L, VIOLET),
        ("Attention SE", SKY_L, SKY),
        ("Fusion", INDIGO_L, INDIGO),
    ]
    bx, bw, bh = 0.065, 0.22, 0.058
    by = 0.775
    for name, fc, ec in comp_blocks:
        ax.add_patch(FancyBboxPatch(
            (bx, by), bw, bh,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(bx + bw / 2, by + bh / 2, name, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=INK, zorder=4)
        by -= bh + 0.008

    # ── Flèche "vs" ──
    arrow(ax, (0.305, 0.65), (0.38, 0.65), color=INK, lw=3.0)
    ax.text(0.343, 0.685, "retirer un\ncomposant", ha="center", fontsize=9,
            style="italic", color=MUT, zorder=5)

    # ── Variante (un composant grisé) ──
    ax.text(0.52, 0.845, "Variante (exemple)", ha="center",
            fontsize=13, fontweight="bold", color=INK, zorder=5)
    bx2 = 0.40
    by2 = 0.775
    for i, (name, fc, ec) in enumerate(comp_blocks):
        if i == 2:  # FreqGate = retiré (exemple)
            ax.add_patch(FancyBboxPatch(
                (bx2, by2), bw, bh,
                boxstyle="round,pad=0.0,rounding_size=0.012",
                linewidth=2.0, edgecolor=REMOVED_EC, facecolor=REMOVED_FC,
                linestyle="--", zorder=3))
            ax.text(bx2 + bw / 2, by2 + bh / 2, name, ha="center",
                    va="center", fontsize=10.5, color=INK, zorder=4)
            # Barre de suppression
            ax.plot([bx2 + 0.015, bx2 + bw - 0.015],
                    [by2 + bh / 2, by2 + bh / 2],
                    color=REMOVED_EC, lw=2.5, zorder=5)
        else:
            ax.add_patch(FancyBboxPatch(
                (bx2, by2), bw, bh,
                boxstyle="round,pad=0.0,rounding_size=0.012",
                linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=3))
            ax.text(bx2 + bw / 2, by2 + bh / 2, name, ha="center",
                    va="center", fontsize=10.5, fontweight="bold",
                    color=INK, zorder=4)
        by2 -= bh + 0.008

    # ── Flèche vers conclusion ──
    arrow(ax, (0.64, 0.65), (0.70, 0.65), color=INK, lw=3.0)

    # ── Composant validé ──
    ax.add_patch(FancyBboxPatch(
        (0.71, 0.575), 0.255, 0.15,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=2.5, edgecolor=GREEN, facecolor=GREEN_L, zorder=3))
    ax.text(0.837, 0.68, "✓", ha="center", va="center",
            fontsize=28, color=GREEN, zorder=4)
    ax.text(0.837, 0.615, "Le composant est-il\nindispensable ?", ha="center",
            va="center", fontsize=11, fontweight="bold", color=INK, zorder=4)

    # ── ZONE BASSE : une seule carte (protocole) ───────────────────
    cw, ch = 0.55, 0.36
    cx1, cy = 0.035, 0.04
    ax.add_patch(FancyBboxPatch(
        (cx1, cy), cw, ch,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=2.5, edgecolor=INDIGO, facecolor="white", zorder=2))
    # Bandeau titre
    ax.add_patch(FancyBboxPatch(
        (cx1, cy + ch - 0.065), cw, 0.065,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0, facecolor=INDIGO, zorder=3))
    ax.text(cx1 + cw / 2, cy + ch - 0.032, "Comment ?",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", zorder=4)

    # Principe (ancienne légende, intégrée dans la carte)
    ax.text(cx1 + cw / 2, cy + ch - 0.095,
            "Même architecture · un seul composant retiré · mesurer l'impact",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=INK, zorder=4)

    # Bullets à l'infinitif
    perf_items = [
        ("•", "Partir du modèle V2 complet (référence)"),
        ("•", "Retirer un seul composant à la fois"),
        ("•", "Comparer les résultats : Accuracy, F1, Recall"),
        ("•", "Garder tout le reste identique (données, seed, entraînement)"),
    ]
    iy = cy + ch - 0.140
    for bullet, txt in perf_items:
        ax.text(cx1 + 0.030, iy, bullet, ha="center", va="center",
                fontsize=13, fontweight="bold", color=INDIGO, zorder=4)
        ax.text(cx1 + 0.055, iy, txt, ha="left", va="center",
                fontsize=11, color=INK, zorder=4)
        iy -= 0.055

    save(fig, "15_ablation_introduction.png")


# ══════════════════════════════════════════════════════════════════════
# 16 — Performances ArcFaultNet V2 (full) — ablation
# ══════════════════════════════════════════════════════════════════════
def fig_ablation_v2_performance():
    ablation_path = HERE.parent.parent / (
        "ablation_results/ablation_v2_20260612_175320/ablation_v2_results.json"
    )
    with open(ablation_path) as f:
        data = json.load(f)
    r = data["variants"]["arcfaultnet_v2"]
    seed = data["seed"]
    n_test = data["split"]["test"]

    acc = r["accuracy"] * 100
    rec = r["recall"] * 100
    spec = r["specificity"] * 100
    cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])

    fig = plt.figure(figsize=(14.6, 7.6))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, "ArcFaultNet V2 (full) — Performances",
             ha="center", fontsize=21, fontweight="bold", color=INK)
    fig.text(
        0.5, 0.915,
        f"Étude d'ablation · seed {seed} · split 70/15/15 · jeu de test = {n_test} cycles",
        ha="center", fontsize=11, style="italic", color=INK,
    )

    # ── Matrice de confusion ──
    ax_cm = fig.add_axes([0.07, 0.16, 0.38, 0.68])
    im = ax_cm.imshow(cm, cmap=plt.cm.Blues, vmin=0, vmax=cm.max())
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9, colors=INK)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Normal", "Arc"], fontsize=12, color=INK)
    ax_cm.set_yticklabels(["Normal", "Arc"], fontsize=12, color=INK)
    ax_cm.set_xlabel("Prédit", fontsize=12, labelpad=8, color=INK)
    ax_cm.set_ylabel("Vrai", fontsize=12, labelpad=8, color=INK)
    ax_cm.set_title("Matrice de confusion", fontsize=13, fontweight="bold",
                    color=INK, pad=10)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax_cm.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=22, fontweight="bold",
                color="white" if cm[i, j] > thresh else INK,
            )

    # ── Cartes métriques (formule → nom → valeur) ──
    metrics = [
        (r"$\mathrm{Accuracy} = \dfrac{TP + TN}{N}$",
         "Accuracy", acc, INDIGO, INDIGO_L, None),
        (r"$\mathrm{Recall} = \dfrac{TP}{TP + FN}$",
         "Recall", rec, TEAL, TEAL_L, "arcs détectés"),
        (r"$\mathrm{Specificity} = \dfrac{TN}{TN + FP}$",
         "Specificity", spec, GREEN, GREEN_L, "normaux rejetés"),
    ]
    card_x, card_w = 0.54, 0.40
    card_h = 0.20
    gap = 0.042
    total_h = 3 * card_h + 2 * gap
    card_y0 = 0.16 + (0.68 - total_h) / 2

    for i, (formula, label, val, ec, fc, note) in enumerate(metrics):
        cy = card_y0 + (2 - i) * (card_h + gap)
        ax_m = fig.add_axes([card_x, cy, card_w, card_h])
        ax_m.set_xlim(0, 1)
        ax_m.set_ylim(0, 1)
        ax_m.axis("off")
        ax_m.add_patch(FancyBboxPatch(
            (0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.04",
            linewidth=2.2, edgecolor=ec, facecolor=fc, transform=ax_m.transAxes,
            clip_on=False))
        ax_m.text(0.5, 0.82, formula, ha="center", va="center",
                  fontsize=13, color=INK, transform=ax_m.transAxes)
        ax_m.text(0.5, 0.58, label, ha="center", va="center",
                  fontsize=13, fontweight="bold", color=INK, transform=ax_m.transAxes)
        ax_m.text(0.5, 0.30, f"{val:.1f} %", ha="center", va="center",
                  fontsize=28, fontweight="bold", color=ec, transform=ax_m.transAxes)
        if note:
            ax_m.text(0.5, 0.10, note, ha="center", va="center",
                      fontsize=9, color=INK, transform=ax_m.transAxes)

    fig.text(
        0.5, 0.045,
        f"TP = {r['tp']}   FN = {r['fn']}   FP = {r['fp']}   TN = {r['tn']}"
        f"   ·   FP rate = {100 - spec:.2f} %",
        ha="center", fontsize=10.5, color=INK,
    )

    fig.savefig(OUT / "16_ablation_v2_performance.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote 16_ablation_v2_performance.png")


# ══════════════════════════════════════════════════════════════════════
# 17 — Calcul inverse des poids (traçabilité projetée)
# ══════════════════════════════════════════════════════════════════════
def fig_traceabilite_reverse_engineering():
    """Valeurs projetées — lien Cross-Attention ↔ attention de canal."""
    fig, ax = new_fig(14.6, 7.4)
    title(ax, "Traçabilité — Cross-Attention & Channel Attention",
          "Calcul inverse des poids · valeurs projetées (ablation V2 · seed 42)")

    def _bars(x, y, w, h, ec, fc, title_s, rows, xmax, val_fmt):
        """rows: [(label, value, bar_color), ...]"""
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.014",
            linewidth=2.2, edgecolor=ec, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h - 0.042, title_s, ha="center", va="center",
                fontsize=12, fontweight="bold", color=ec, zorder=3)
        n = len(rows)
        top = y + h - 0.075
        bot = y + 0.025
        row_h = (top - bot) / n
        lab_x = x + 0.018
        bar_x0 = x + w * 0.38
        bar_max = w * 0.44
        for i, (lab, val, col) in enumerate(rows):
            cy = top - (i + 0.5) * row_h
            ax.text(lab_x, cy, lab, ha="left", va="center",
                    fontsize=9.5, color=INK, zorder=3)
            bw = bar_max * min(val / xmax, 1.0)
            ax.add_patch(FancyBboxPatch(
                (bar_x0, cy - row_h * 0.20), bw, row_h * 0.40,
                boxstyle="square,pad=0", linewidth=0, facecolor=col, zorder=3))
            ax.text(bar_x0 + bar_max + 0.006, cy, val_fmt(val),
                    ha="left", va="center", fontsize=9.5, fontweight="bold",
                    color=col, zorder=3)

    lx, lw = 0.04, 0.42
    _bars(lx, 0.56, lw, 0.20, INDIGO, INDIGO_L,
          "Inter-branch Fusion (arc)",
          [("Spectral", 64, ORANGE), ("Temporal", 36, TEAL)],
          70, lambda v: f"{v:.0f} %")
    _bars(lx, 0.34, lw, 0.20, INDIGO, "white",
          "Cross-Attention",
          [("cam_spectral", 0.74, ORANGE), ("cam_temporal", 0.61, TEAL),
           ("FrequencyGate HF", 0.81, VIOLET)],
          0.85, lambda v: f"{v:.2f}")

    _bars(
        0.54, 0.34, 0.42, 0.42, TEAL, TEAL_L,
        "Channel Attention (temporal)",
        [("TKEO", 0.72, TEAL), ("|dI|", 0.68, INDIGO),
         ("I", 0.45, SKY), ("RMS", 0.38, GREEN)],
        0.85, lambda v: f"{v:.2f}",
    )

    arrow(ax, (0.47, 0.55), (0.53, 0.55), color=INDIGO, lw=2.8)
    ax.text(0.50, 0.575, "corrélation\ndes poids",
            ha="center", va="bottom", fontsize=9, color=INK, zorder=5)

    box(ax, 0.04, 0.21, 0.92, 0.09, GREY_L, GREY,
        "Cross-Attention choisit la branche  →  les poids Channel Attention "
        "indiquent la transformée dominante",
        None, tsize=11, tc=INK)

    ax.add_patch(FancyBboxPatch(
        (0.04, 0.06), 0.92, 0.13, boxstyle="round,pad=0.0,rounding_size=0.014",
        linewidth=2.2, edgecolor=AMBER, facecolor=AMBER_L, zorder=2))
    ax.text(0.50, 0.155, "Exemple de lecture (projeté)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=AMBER, zorder=3)
    ax.text(
        0.50, 0.115,
        "cam_temporal élevé + TKEO dominant  →  charge inductive (moteur)",
        ha="center", va="center", fontsize=11, color=INK, zorder=3,
    )
    ax.text(
        0.50, 0.082,
        "cam_spectral élevé + FrequencyGate HF actif  →  charge à crépitement (SMPS)",
        ha="center", va="center", fontsize=10.5, color=INK, zorder=3,
    )

    save(fig, "17_traceabilite_reverse_engineering.png")


# ══════════════════════════════════════════════════════════════════════
# 18 — Limites et challenges
# ══════════════════════════════════════════════════════════════════════
def _icon_data_fp(ax, cx, cy, s, ec):
    """Jeu de données + marque FP."""
    for i, (w, a) in enumerate([(0.75, 0.35), (0.55, 0.50), (0.90, 0.65), (0.60, 0.80)]):
        dy = cy + s * 0.28 - i * s * 0.19
        ax.add_patch(FancyBboxPatch(
            (cx - s * w / 2, dy - s * 0.05), s * w, s * 0.10,
            boxstyle="round,pad=0.0,rounding_size=0.004",
            linewidth=0, facecolor=ec, alpha=a, zorder=5))
    ax.add_patch(Circle((cx + s * 0.38, cy - s * 0.32), s * 0.11,
                        facecolor=RED, edgecolor="white", linewidth=1.2, zorder=6))
    ax.text(cx + s * 0.38, cy - s * 0.32, "FP", ha="center", va="center",
            fontsize=5.5, fontweight="bold", color="white", zorder=7)


def _icon_attention(ax, cx, cy, s, ec):
    """Barres d'attention + nœuds."""
    heights = [0.35, 0.75, 0.55, 0.90, 0.45]
    for i, h in enumerate(heights):
        bx = cx - s * 0.38 + i * s * 0.19
        ax.add_patch(FancyBboxPatch(
            (bx, cy - s * 0.35), s * 0.13, s * h * 0.65,
            boxstyle="round,pad=0.0,rounding_size=0.003",
            linewidth=0, facecolor=ec, alpha=0.35 + i * 0.12, zorder=5))
    for dx in [-0.22, 0.0, 0.22]:
        ax.add_patch(Circle((cx + s * dx, cy + s * 0.30), s * 0.07,
                            facecolor=ec, edgecolor="white", linewidth=1.0, zorder=6))


def _icon_lab_field(ax, cx, cy, s, ec):
    """Fiole labo | maison terrain."""
    lx = cx - s * 0.30
    ax.add_patch(Circle((lx, cy + s * 0.22), s * 0.14,
                        facecolor=ec, alpha=0.30, edgecolor=ec, linewidth=1.4, zorder=5))
    ax.add_patch(FancyBboxPatch(
        (lx - s * 0.10, cy - s * 0.28), s * 0.20, s * 0.42,
        boxstyle="round,pad=0.0,rounding_size=0.006",
        linewidth=1.4, edgecolor=ec, facecolor=ec, alpha=0.45, zorder=5))
    hx = cx + s * 0.22
    ax.add_patch(Polygon(
        [(hx, cy + s * 0.38), (hx + s * 0.22, cy + s * 0.05), (hx + s * 0.44, cy + s * 0.38)],
        closed=True, facecolor=ec, edgecolor=ec, linewidth=0, zorder=5))
    ax.add_patch(FancyBboxPatch(
        (hx + s * 0.08, cy - s * 0.30), s * 0.28, s * 0.35,
        boxstyle="square,pad=0", linewidth=1.4, edgecolor=ec,
        facecolor=ec, alpha=0.55, zorder=5))
    ax.plot([cx, cx], [cy - s * 0.42, cy + s * 0.42], color=GREY, lw=1.2,
            linestyle="--", zorder=4)


def _card_wrapped_lines(lines, max_chars):
    """Déplie les puces ; list/tuple = retours à la ligne manuels."""
    wrapped = []
    for bullet in lines:
        if isinstance(bullet, (list, tuple)):
            wrapped.extend(bullet)
        else:
            sub = textwrap.wrap(bullet, width=max_chars) or [bullet]
            wrapped.extend(sub if sub else [bullet])
    return wrapped


def _card_render_lines(lines, max_chars):
    """Lignes à afficher avec puce / indentation continuation."""
    rendered = []
    for bullet in lines:
        if isinstance(bullet, (list, tuple)):
            for j, part in enumerate(bullet):
                rendered.append(("•  " if j == 0 else "    ") + part)
        else:
            sub = textwrap.wrap(bullet, width=max_chars) or [bullet]
            for j, part in enumerate(sub):
                rendered.append(("•  " if j == 0 else "    ") + part)
    return rendered


def _card_text_max_chars(cw, fig_w=11.8, fs=9.8):
    iz = cw * 0.15
    pad = cw * 0.022
    text_w = cw - (pad + iz + cw * 0.028) - pad
    return max(38, int(text_w * fig_w / (fs / 72.0 * 0.52)))


def _challenge_card(ax, x, y, w, h, ec, fc, title_s, lines, icon_fn,
                    line_step=0.034, max_chars=48):
    """Carte horizontale compacte : bandeau + icône + puces."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.014",
        linewidth=2.0, edgecolor=ec, facecolor=fc, zorder=2))

    hdr_h = 0.044
    ax.add_patch(FancyBboxPatch(
        (x, y + h - hdr_h), w, hdr_h,
        boxstyle="round,pad=0.0,rounding_size=0.014",
        linewidth=0, facecolor=ec, zorder=3))
    ax.add_patch(FancyBboxPatch(
        (x, y + h - hdr_h + 0.006), w, hdr_h - 0.006,
        boxstyle="square,pad=0", linewidth=0, facecolor=ec, zorder=3))

    ax.text(x + w / 2, y + h - hdr_h / 2, title_s, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="white", zorder=5)

    iz = w * 0.15
    pad = w * 0.022
    icon_cx = x + pad + iz / 2
    icon_cy = y + (h - hdr_h) / 2
    ax.add_patch(FancyBboxPatch(
        (x + pad, y + pad * 0.8), iz, h - hdr_h - pad * 1.6,
        boxstyle="round,pad=0.0,rounding_size=0.010",
        linewidth=1.2, edgecolor=ec, facecolor="white", zorder=3))
    icon_fn(ax, icon_cx, icon_cy, min(iz, h - hdr_h) * 0.72, ec)

    tx = x + pad + iz + w * 0.028
    fs = 9.8
    wrapped = _card_render_lines(lines, max_chars)
    body_top = y + h - hdr_h - 0.018
    start_y = body_top - line_step * 0.5
    for i, line in enumerate(wrapped):
        ax.text(tx, start_y - i * line_step, line, ha="left", va="center",
                fontsize=fs, color=INK, zorder=4, clip_on=True)


def _card_height(n_lines, hdr_h=0.044, line_h=0.034, pad=0.032):
    return hdr_h + pad + n_lines * line_h


def fig_limites_challenges():
    fig_w, fig_h = 11.8, 8.4
    fig, ax = new_fig(fig_w, fig_h)
    title(ax, "Limites et challenges — perspectives du travail",
          "Contraintes actuelles et pistes d'évolution du projet")

    line_step = 0.034
    gap = 0.048
    raw_cards = [
        ("Données & fausses alarmes", [
            "Diversité de données limitée --> robustesse du modèle plafonnée",
            "Multi-cas d'arc · multi-charges · bruits --> + FP",
        ], RED_L, RED, _icon_data_fp),
        ("Attention & implémentation", [
            "Mécanismes d'attention seuls: Coût et latence élevés",
            "Compromis: robustesse vs latence et coût",
            [
                "Piste SSM · modèles à effet mémoire + contexte →",
                "Attention allégée et robustesse améliorée",
            ],
        ], INDIGO_L, INDIGO, _icon_attention),
        ("Labo vs installation réelle", [
            "Labo: données synthétiques · charges isolées peu représentatives",
            "Réseau maison: multi-charges imprévisibles",
            "Solution: Entrainement sur des données réelles",
        ], SKY_L, SKY, _icon_lab_field),
    ]

    cw = 0.72
    cx = (1.0 - cw) / 2
    max_chars = _card_text_max_chars(cw, fig_w=fig_w)

    cards = []
    for title_s, lines, fc, ec, icon_fn in raw_cards:
        n = len(_card_wrapped_lines(lines, max_chars))
        cards.append((title_s, lines, fc, ec, icon_fn,
                      _card_height(n, line_h=line_step)))

    stack_h = sum(c[5] for c in cards) + gap * (len(cards) - 1)
    y_top = min(0.845, 0.120 + stack_h)
    for title_s, lines, fc, ec, icon_fn, ch in cards:
        y = y_top - ch
        _challenge_card(ax, cx, y, cw, ch, ec, fc, title_s, lines, icon_fn,
                        line_step=line_step, max_chars=max_chars)
        y_top = y - gap

    save(fig, "18_limites_challenges.png")


# ══════════════════════════════════════════════════════════════════════
# 19 — Bilan positif · état d'avancement · piste SSM
# ══════════════════════════════════════════════════════════════════════
def _icon_check(ax, cx, cy, s, ec):
    ax.add_patch(Circle((cx, cy), s * 0.38, facecolor=ec, edgecolor=ec,
                        linewidth=0, alpha=0.25, zorder=5))
    ax.plot([cx - s * 0.18, cx - s * 0.04, cx + s * 0.22],
            [cy - s * 0.02, cy - s * 0.18, cy + s * 0.22],
            color=ec, lw=2.4, zorder=6)


def _icon_roadmap(ax, cx, cy, s, ec):
    for i, w in enumerate([0.55, 0.75, 1.0]):
        ax.add_patch(FancyBboxPatch(
            (cx - s * w / 2, cy - s * 0.28 + i * s * 0.22), s * w, s * 0.14,
            boxstyle="round,pad=0.0,rounding_size=0.004",
            linewidth=0, facecolor=ec, alpha=0.35 + i * 0.2, zorder=5))
    ax.add_patch(Circle((cx + s * 0.38, cy + s * 0.28), s * 0.07,
                        facecolor=ec, edgecolor="white", linewidth=1.0, zorder=6))


def _icon_ssm(ax, cx, cy, s, ec):
    xs = [cx - s * 0.35, cx - s * 0.12, cx + s * 0.12, cx + s * 0.35]
    for i, x in enumerate(xs):
        ax.add_patch(Circle((x, cy), s * 0.09, facecolor=ec,
                            alpha=0.35 + i * 0.15, edgecolor=ec, linewidth=1.0, zorder=5))
        if i < len(xs) - 1:
            ax.plot([x + s * 0.09, xs[i + 1] - s * 0.09], [cy, cy],
                    color=ec, lw=1.8, zorder=4)
    ax.text(cx, cy - s * 0.32, "mémoire", ha="center", fontsize=6.5,
            color=ec, zorder=6)


def fig_synthese_atouts_perspectives():
    abl_path = HERE.parent.parent / "ablation_results/ablation_v2_20260612_175320/ablation_v2_results.json"
    with abl_path.open() as f:
        v2 = json.load(f)["variants"]["arcfaultnet_v2"]
    prec = 100 * v2["precision"]
    fpr = 0.034
    n_k = v2["n_params"] // 1000

    fig_w, fig_h = 11.8, 8.2
    fig, ax = new_fig(fig_w, fig_h)
    title(ax, "Bilan positif — Arc-FaultNet V2",
          "Atouts du modèle · état d'avancement · prochaine étape SSM")

    line_step = 0.033
    gap = 0.040
    cw, cx = 0.72, (1.0 - 0.72) / 2
    max_chars = _card_text_max_chars(cw, fig_w=fig_w)

    raw_cards = [
        ("Atouts du modèle", [
            "Dual-branch · 4 canaux dérivés · STFT · FrequencyGate",
            f"Cross-Attention · Channel Attention · Précision {prec:.1f} %",
            f"FPR {fpr:.3f} % · ~{n_k}k param · multi-charges couvertes",
            "Généralisation forte · résistive · SMPS · moteur · multi",
        ], GREEN_L, GREEN, _icon_check),
        ("Où nous en sommes", [
            "V2 complet · pipeline acquisition → décision opérationnel",
            "Ablation · comparaison état de l'art · GroupKFold validés",
            "Positionné gamme 300–350k param · AFDD embarquable",
            "Base solide · prêt pour optimisation embarquée",
        ], INDIGO_L, INDIGO, _icon_roadmap),
        ("Piste SSM — allègement à venir", [
            "State Space Models · mémoire sélective sur la séquence",
            "Allège Cross-Attention · moins de params · moins de calcul",
            "Contexte temporel conservé sans matrice Attention complète",
            "Objectif · Arc-FaultNet V2 lite · déploiement MCU",
        ], TEAL_L, TEAL, _icon_ssm),
    ]

    cards = []
    for title_s, lines, fc, ec, icon_fn in raw_cards:
        n = len(_card_wrapped_lines(lines, max_chars))
        cards.append((title_s, lines, fc, ec, icon_fn,
                      _card_height(n, line_h=line_step)))

    stack_h = sum(c[5] for c in cards) + gap * (len(cards) - 1)
    y_top = min(0.845, 0.115 + stack_h)
    for title_s, lines, fc, ec, icon_fn, ch in cards:
        y = y_top - ch
        _challenge_card(ax, cx, y, cw, ch, ec, fc, title_s, lines, icon_fn,
                        line_step=line_step, max_chars=max_chars)
        y_top = y - gap

    save(fig, "19_synthese_atouts_perspectives.png")


# ══════════════════════════════════════════════════════════════════════
# 20 — Synthèse comparative (SOTA 300k)
# ══════════════════════════════════════════════════════════════════════
def fig_comparaison_sota_300k():
    fig_w, fig_h = 14.8, 8.4
    fig, ax = new_fig(fig_w, fig_h)
    title(ax, "Arc-FaultNet V2 face à l'État de l'Art (Modèles 300k - 350k)",
          "Atteindre la haute performance avec une approche d'ingénierie experte vs force brute")

    cw = 0.44
    ch = 0.42
    cy = 0.38
    cx1 = 0.04
    cx2 = 0.52

    # ── Carte Gauche : État de l'Art (SOTA) ──
    ax.add_patch(FancyBboxPatch(
        (cx1, cy), cw, ch, boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=2.5, edgecolor=ORANGE, facecolor="white", zorder=2))
    ax.add_patch(FancyBboxPatch(
        (cx1, cy + ch - 0.065), cw, 0.065, boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0, facecolor=ORANGE, zorder=3))
    ax.text(cx1 + cw / 2, cy + ch - 0.032, "🔥  Modèles Récents (ViT, Conformer, BiLSTM)",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="white", zorder=4)

    sota_bullets = [
        ("Modèles phares :", "1D-ViT-Micro, Conformer-Edge, CNN-BiLSTM."),
        ("Budget similaire :", "Environ 315k à 342k paramètres."),
        ("Performance :", "Précision extrême (Accuracy > 99%)."),
        ("Le point faible :", "Architectures lourdes en calcul (Self-Attention quadratique, RNN séquentiel). Temps d'inférence > 38 ms."),
    ]
    iy = cy + ch - 0.12
    for title_str, desc in sota_bullets:
        ax.text(cx1 + 0.03, iy, "•", fontsize=13, fontweight="bold", color=ORANGE, zorder=4)
        ax.text(cx1 + 0.06, iy, title_str, fontsize=11, fontweight="bold", color=INK, zorder=4)
        # Wrap desc manually if needed, or rely on tight fit
        if len(desc) > 55:
            # Simple split for the long one
            p1, p2 = desc[:60], desc[60:]
            if " " in p1[::-1]:
                split_idx = 60 - p1[::-1].index(" ")
                p1, p2 = desc[:split_idx], desc[split_idx:]
            ax.text(cx1 + 0.06, iy - 0.025, p1.strip(), fontsize=10.5, color=INK, zorder=4)
            ax.text(cx1 + 0.06, iy - 0.050, p2.strip(), fontsize=10.5, color=INK, zorder=4)
            iy -= 0.050
        else:
            ax.text(cx1 + 0.06 + len(title_str)*0.011, iy, desc, fontsize=10.5, color=INK, zorder=4)
        iy -= 0.06

    # ── Flèche de comparaison ──
    ax.text(0.5, cy + ch/2, "VS", ha="center", va="center", fontsize=18, fontweight="bold", color=MUT, zorder=5)

    # ── Carte Droite : Arc-FaultNet V2 ──
    ax.add_patch(FancyBboxPatch(
        (cx2, cy), cw, ch, boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=2.5, edgecolor=TEAL, facecolor="white", zorder=2))
    ax.add_patch(FancyBboxPatch(
        (cx2, cy + ch - 0.065), cw, 0.065, boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=0, facecolor=TEAL, zorder=3))
    ax.text(cx2 + cw / 2, cy + ch - 0.032, "⚡  Arc-FaultNet V2 (Notre Approche)",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="white", zorder=4)

    v2_bullets = [
        ("Architecture hybride :", "CNN 1D + STFT + SE Attention + XGBoost."),
        ("Budget maîtrisé :", "~350k paramètres (gamme équivalente)."),
        ("Performance :", "Compétitif (Accuracy ~97,3%) sur dataset complexe."),
        ("Le point fort :", "Ingénierie de features experte (FrequencyGate) palliant l'absence de Self-Attention global. Adapté pour déploiement MCU."),
    ]
    iy = cy + ch - 0.12
    for title_str, desc in v2_bullets:
        ax.text(cx2 + 0.03, iy, "•", fontsize=13, fontweight="bold", color=TEAL, zorder=4)
        ax.text(cx2 + 0.06, iy, title_str, fontsize=11, fontweight="bold", color=INK, zorder=4)
        if len(desc) > 55:
            p1, p2 = desc[:60], desc[60:]
            if " " in p1[::-1]:
                split_idx = 60 - p1[::-1].index(" ")
                p1, p2 = desc[:split_idx], desc[split_idx:]
            ax.text(cx2 + 0.06, iy - 0.025, p1.strip(), fontsize=10.5, color=INK, zorder=4)
            ax.text(cx2 + 0.06, iy - 0.050, p2.strip(), fontsize=10.5, color=INK, zorder=4)
            iy -= 0.050
        else:
            ax.text(cx2 + 0.06 + len(title_str)*0.011, iy, desc, fontsize=10.5, color=INK, zorder=4)
        iy -= 0.06

    # ── Carte Bas : Synthèse globale ──
    bx = 0.10
    by = 0.08
    bw = 0.80
    bh = 0.22
    ax.add_patch(FancyBboxPatch(
        (bx, by), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.018",
        linewidth=2.5, edgecolor=INDIGO, facecolor=INDIGO_L, zorder=2))
    
    ax.text(bx + 0.03, by + bh - 0.05, "💡 Synthèse Globale", fontsize=13, fontweight="bold", color=INDIGO, zorder=4)
    
    conclusion = (
        "Le projet démontre qu'un design hybride ciblé (combinant extraction physique via STFT et apprentissage\n"
        "profond léger) permet de rivaliser avec les modèles d'état de l'art les plus récents en termes de capacité.\n"
        "Arc-FaultNet V2 compense l'absence de mécanismes coûteux (comme l'Attention globale) par une grande\n"
        "efficience de son extraction de caractéristiques (FrequencyGate), le rendant viable pour l'industrie."
    )
    ax.text(bx + 0.03, by + 0.07, conclusion, fontsize=11, color=INK, linespacing=1.6, zorder=4)

    save(fig, "20_comparaison_sota_300k.png")


# ══════════════════════════════════════════════════════════════════════
def main():
    print(f"Diagrammes présentation → {OUT}")
    fig_cover()
    fig_introduction()
    fig_merci()
    fig_architecture_globale()
    fig_data_flow()
    fig_pipeline()
    fig_branche_1d()
    fig_branche_2d()
    fig_futures()
    fig_comparaison()
    fig_meilleur_modele()
    fig_innovation()
    fig_transformees_generalisation()
    fig_evolution()
    fig_fusion_mecanisme()
    fig_fusion_bilan()
    fig_ablation_introduction()
    fig_ablation_v2_performance()
    fig_traceabilite_reverse_engineering()
    fig_limites_challenges()
    fig_synthese_atouts_perspectives()
    fig_comparaison_sota_300k()
    print("Done.")


if __name__ == "__main__":
    main()
