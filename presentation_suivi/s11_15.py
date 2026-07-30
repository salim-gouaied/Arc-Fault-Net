#!/usr/bin/env python3
"""Slides 11 à 15 — radar, tableau des runs, matrices de confusion,
ablation des briques, lecture industrielle."""

import numpy as np
from style import *          # noqa: F401,F403
import data as D


# ------------------------------------------------------------------ 11
def s11():
    fig, ax, top = slide(
        eb="Résultats  ·  vue d'ensemble", n=11,
        title="La nouvelle fusion gagne sur les cinq métriques",
        lede="Moyenne de 4 entraînements pour chaque version, "
             "sur le même jeu de test de 1 630 cycles.")

    # ---------- radar
    rax = fig.add_axes([0.078, 0.185, 0.355, 0.400], polar=True)
    rax.set_facecolor(SURFACE)
    n = len(D.RADAR_AXES)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([ang, ang[:1]])
    lo, hi = 93.0, 100.0

    for series, col, lab, lw in (
            (D.RADAR_V1, NOISE, "Avant  ·  fusion « à porte »", 1.8),
            (D.RADAR_V2, TEAL, "Maintenant  ·  cross-attention", 2.4)):
        v = np.array(series + series[:1])
        rax.plot(closed, v, color=col, lw=lw, label=lab, zorder=5,
                 solid_joinstyle="round")
        rax.fill(closed, v, color=col, alpha=0.13, zorder=4, linewidth=0)
        rax.scatter(ang, series, s=26, color=col, zorder=6, edgecolor="white",
                    linewidth=1.2)

    rax.set_ylim(lo, hi)
    rax.set_yticks([94, 96, 98, 100])
    rax.set_yticklabels(["94", "96", "98", "100 %"], fontsize=8, color=FAINT)
    rax.set_xticks(ang)
    rax.set_xticklabels(D.RADAR_AXES, fontsize=10.5, color=INK,
                        fontweight="bold")
    rax.tick_params(axis="x", pad=14)
    rax.grid(color=LINE, lw=0.9)
    rax.spines["polar"].set_color(LINE)
    rax.set_rlabel_position(90)

    # légende dessinée à la main (sous le radar, dans les coordonnées de slide)
    lx = ML + 0.012
    for col, lab in ((NOISE, "Avant  ·  fusion « à porte »"),
                     (TEAL, "Maintenant  ·  cross-attention")):
        ax.plot([lx, lx + 0.026], [0.098, 0.098], color=col, lw=2.6,
                zorder=6, solid_capstyle="round")
        txt(ax, lx + 0.034, 0.098, lab, size=9.8, color=INK, weight="bold")
        lx += 0.046 + 0.0062 * len(lab) * 1.28

    # ---------- tableau des écarts
    xr = 0.505
    wr = MR - xr
    txt(ax, xr, top - 0.018, "Écart moyen, version par version", size=12.5,
        weight="bold", color=INK)

    keys = [("Exactitude", "acc"), ("F1", "f1"), ("Précision", "prec"),
            ("Rappel", "rec"), ("Spécificité", "spec")]
    rows = [[lab, fr(D.V1_MEAN[k], 2, "%"), fr(D.V2_MEAN[k], 2, "%"),
             f"+{fr(D.GAIN[k], 2)} pt"] for lab, k in keys]
    rows.append(["Fausses alarmes", fr(D.V1_MEAN["fp"], 2),
                 fr(D.V2_MEAN["fp"], 2), f"−{fr(D.GAIN_FP, 2)}"])
    rows.append(["Paramètres", frint(D.PARAMS["gated"]["total"]),
                 frint(D.PARAMS["sequential"]["total"]),
                 f"−{frint(D.PARAMS_DELTA)}"])

    def fmt(i, j, v):
        if j == 3:
            return {"color": GOOD, "weight": "bold"}
        if j == 2:
            return {"color": INK, "weight": "bold"}
        return None

    bot = table(ax, ["Métrique", "Avant", "Maintenant", "Écart"], rows,
                xr, top - 0.062,
                [wr * 0.32, wr * 0.22, wr * 0.24, wr * 0.22],
                row_h=0.050, head_h=0.046, fs=10.3, cell_fmt=fmt)

    callout(ax, xr, bot - 0.018 - 0.146, wr, "Comment lire ce graphique",
            "Chaque axe est une métrique : plus la surface est grande,\n"
            "mieux c'est. L'écart se creuse surtout sur le rappel.",
            kind="intuition", fs=10.2, h=0.146)
    return save(fig, 11, "radar")


# ------------------------------------------------------------------ 12
def s12():
    fig, ax, top = slide(
        eb="Résultats  ·  détail", n=12,
        title="Quatre entraînements, quatre fois au-dessus",
        lede="Le plus faible des nouveaux runs (97,48 %) dépasse déjà "
             "la moyenne de l'ancienne version (97,33 %).")

    rows = []
    for lab, seed, sched, acc, f1, prec, rec, spec, tn, fp, fn, tp in D.V2_RUNS:
        rows.append([lab, str(seed), sched, fr(acc, 2, "%"), fr(f1, 2, "%"),
                     fr(prec, 2, "%"), fr(rec, 2, "%"), str(fp)])
    best = max(range(4), key=lambda i: D.V2_RUNS[i][3])

    w = MR - ML
    widths = [w * 0.10, w * 0.08, w * 0.16, w * 0.13, w * 0.12, w * 0.13,
              w * 0.12, w * 0.16]

    def fmt(i, j, v):
        if i == best and j == 3:
            return {"color": GOOD, "weight": "bold"}
        if j == 7:
            return {"color": GOOD, "weight": "bold"} if int(v) <= 3 else \
                   {"color": INK, "weight": "bold"}
        return None

    bot = table(ax, ["Run", "Graine", "Planification du pas", "Exactitude",
                     "F1", "Précision", "Rappel", "F. alarmes"],
                rows, ML, top - 0.020, widths, row_h=0.062, head_h=0.050,
                fs=10.5, cell_fmt=fmt, highlight_row=best)

    # bandeau moyenne
    sy = bot - 0.022 - 0.088
    card(ax, ML, sy, w, 0.088, fc=GREY_TINT, ec=LINE)
    parts = [("Exactitude", fr(D.V2_MEAN["acc"], 2, "%")),
             ("F1", fr(D.V2_MEAN["f1"], 2, "%")),
             ("Précision", fr(D.V2_MEAN["prec"], 2, "%")),
             ("Rappel", fr(D.V2_MEAN["rec"], 2, "%")),
             ("Fausses alarmes", fr(D.V2_MEAN["fp"], 2))]
    txt(ax, ML + 0.024, sy + 0.044, "Moyenne\ndes 4 runs", size=9.5,
        weight="bold", color=TEAL_DEEP, ls=1.4)
    xx = ML + 0.135
    step = (w - 0.160) / len(parts)
    for lab, val in parts:
        txt(ax, xx + step / 2, sy + 0.058, val, size=13, weight="bold",
            color=INK, ha="center")
        txt(ax, xx + step / 2, sy + 0.026, lab, size=8.8, color=FAINT,
            ha="center")
        xx += step

    cy = sy - 0.018 - 0.132
    callout(ax, ML, cy, w * 0.615, "À noter, par honnêteté",
            "Les runs 3 et 4 (graine 42) donnent des résultats identiques : "
            "les deux planifications du pas y\nsélectionnent le même point de "
            "fonctionnement. La moyenne porte sur trois configurations "
            "distinctes.", kind="engi", fs=10.2)

    x2 = ML + w * 0.645
    card(ax, x2, cy, MR - x2, 0.132, fc=SURFACE, ec=LINE)
    txt(ax, x2 + 0.024, cy + 0.102, sp("CE QUE ÇA DIT"), size=8, color=FAINT,
        weight="bold")
    txt(ax, x2 + 0.024, cy + 0.072,
        "L'architecture est stable : le gain se\n"
        "répète sur toutes les graines.",
        size=10.3, color=INK, va="top", family=SERIF, ls=1.55)
    return save(fig, 12, "runs")


# ------------------------------------------------------------------ 13
def _cm_table(ax, x, y_top, w, title, subtitle, cm, accent, decimals=0):
    """Matrice de confusion sous forme de tableau."""
    txt(ax, x, y_top, title, size=12.5, weight="bold", color=INK)
    txt(ax, x, y_top - 0.034, subtitle, size=9.5, color=FAINT, family=SERIF)

    def f(v):
        return fr(v, decimals) if decimals else str(int(round(v)))

    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    rows = [
        ["Cycle sain", f(tn) + "  ✓", f(fp), f"{100 * tn / (tn + fp):.1f} %"
         .replace(".", ",")],
        ["Cycle avec arc", f(fn), f(tp) + "  ✓",
         f"{100 * tp / (tp + fn):.1f} %".replace(".", ",")],
    ]

    def fmt(i, j, v):
        if (i == 0 and j == 1) or (i == 1 and j == 2):
            return {"color": accent, "weight": "bold"}
        if (i == 0 and j == 2) or (i == 1 and j == 1):
            return {"color": NOISE, "weight": "bold"}
        if j == 3:
            return {"color": MUTED, "weight": "bold"}
        return None

    bot = table(ax, ["Vérité \\ Prédiction", "Prédit sain", "Prédit arc",
                     "Taux"], rows, x, y_top - 0.062,
                [w * 0.34, w * 0.22, w * 0.22, w * 0.22],
                row_h=0.072, head_h=0.048, fs=11.5, cell_fmt=fmt, zebra=False)
    return bot


def s13():
    fig, ax, top = slide(
        eb="Résultats  ·  matrices de confusion", n=13,
        title="Où se trompe le modèle, exactement",
        lede="1 630 cycles de test. Les deux erreurs n'ont pas le même coût "
             "pour vous.")

    w = (MR - ML - 0.050) / 2

    b1 = _cm_table(ax, ML, top - 0.020, w, "Modèle moyen",
                   "moyenne des 4 entraînements", D.V2_MEAN_CM, TEAL_DEEP)
    b2 = _cm_table(ax, ML + w + 0.050, top - 0.020, w, "Meilleur modèle",
                   D.V2_BEST["label"], D.V2_BEST, TEAL_DEEP)

    bot = min(b1, b2)

    for i, (x, cm, lab) in enumerate((
            (ML, D.V2_MEAN_CM, "moyen"),
            (ML + w + 0.050, D.V2_BEST, "meilleur"))):
        card(ax, x, bot - 0.020 - 0.100, w, 0.100, fc=GREY_TINT, ec=LINE)
        cells = [("Fausses alarmes", cm["fp"], NOISE),
                 ("Arcs manqués", cm["fn"], NOISE),
                 ("Exactitude", None, INK)]
        acc = 100 * (cm["tn"] + cm["tp"]) / D.N_TEST
        xx = x
        step = w / 3
        for name, val, col in cells:
            v = fr(acc, 2, "%") if val is None else fr(val, 2 if
                                                       isinstance(val, float)
                                                       and val % 1 else 0)
            txt(ax, xx + step / 2, bot - 0.052, v, size=14, weight="bold",
                color=col, ha="center")
            txt(ax, xx + step / 2, bot - 0.088, name, size=8.8, color=FAINT,
                ha="center")
            xx += step

    callout(ax, ML, 0.078, MR - ML, "Les deux erreurs, en clair",
            "Une fausse alarme déclenche un disjoncteur sans raison — c'est une "
            "gêne, mesurable en interventions.\nUn arc manqué laisse un risque "
            "d'incendie — c'est ce qu'il faut supprimer en priorité. "
            f"Le meilleur modèle en laisse {D.V2_BEST['fn']} sur "
            f"{D.V2_BEST['tp'] + D.V2_BEST['fn']}.",
            kind="intuition")
    return save(fig, 13, "confusion")


# ------------------------------------------------------------------ 14
def s14():
    fig, ax, top = slide(
        eb="Vérification", n=14,
        title="Chaque brique de fusion paye sa place",
        lede="On retire une brique à la fois et on réentraîne. "
             "Même graine, même jeu de test.")

    bx, bw = ML + 0.245, 0.400
    lo, hi = 94.5, 99.0
    bh, gap = 0.048, 0.026

    # en-têtes de colonnes
    hdr = top - 0.022
    txt(ax, bx, hdr, sp("EXACTITUDE"), size=8, color=FAINT, weight="bold")
    txt(ax, bx + bw + 0.098, hdr, sp("PARAMÈTRES"), size=8, color=FAINT,
        weight="bold", ha="right")
    txt(ax, MR, hdr, sp("F. ALARMES"), size=8, color=FAINT, weight="bold",
        ha="right")
    rule(ax, hdr - 0.018)

    yy = hdr - 0.030 - bh
    for name, key, acc, f1, fp, prm in D.ABLATION:
        is_full = key == "full"
        col = TEAL if is_full else "#9FB4BD"
        frac = (acc - lo) / (hi - lo)
        txt(ax, ML, yy + bh / 2, name, size=10.5,
            weight="bold" if is_full else "normal",
            color=INK if is_full else MUTED)
        ax.add_patch(plt.Rectangle((bx, yy), bw * frac, bh, facecolor=col,
                                   zorder=4, edgecolor="none"))
        txt(ax, bx + bw * frac + 0.012, yy + bh / 2, fr(acc, 2, "%"),
            size=11, weight="bold", color=TEAL_DEEP if is_full else MUTED)
        txt(ax, bx + bw + 0.098, yy + bh / 2, frint(prm), size=9.8,
            color=MUTED, ha="right")
        txt(ax, MR, yy + bh / 2, str(fp), size=10.5, weight="bold",
            color=NOISE if fp >= 20 else (GOOD if fp <= 7 else INK),
            ha="right")
        yy -= bh + gap

    y2 = yy + gap - 0.014
    rule(ax, y2 + 0.020)

    w3 = (MR - ML - 2 * 0.020) / 3
    facts = [
        ("Retirer tout", f"−{fr(D.ABLATION[0][2] - D.ABLATION[4][2], 2)} pt",
         "Sans mécanisme de fusion, on perd plus de trois points.", NOISE),
        ("Le compromis réel", f"{D.ABLATION[1][4]} fausses alarmes",
         "Moins d'alarmes, mais plus d'arcs ratés : "
         f"{fr(D.ABLATION_RECALL['no_xattn'], 1, '%')} de rappel contre "
         f"{fr(D.ABLATION_RECALL['full'], 1, '%')}.", ARC),
        ("Le meilleur équilibre", "modèle complet",
         "Le rappel le plus élevé, à coût en paramètres identique.", GOOD),
    ]
    for i, (lab, big, body, col) in enumerate(facts):
        x = ML + i * (w3 + 0.020)
        h = y2 - 0.078
        y0 = 0.078
        card(ax, x, y0, w3, h)
        ax.add_patch(plt.Rectangle((x, y0 + h - 0.006), w3, 0.006,
                                   facecolor=col, zorder=3, edgecolor="none"))
        txt(ax, x + 0.022, y0 + h - 0.034, lab, size=9.6, weight="bold",
            color=FAINT)
        txt(ax, x + 0.022, y0 + h - 0.068, big, size=14, weight="bold",
            color=col)
        txt(ax, x + 0.022, y0 + h - 0.092, wrap(body, 46), size=9.5,
            color=MUTED, va="top", family=SERIF, ls=1.5)
    return save(fig, 14, "ablation")


# ------------------------------------------------------------------ 15
def s15():
    fig, ax, top = slide(
        eb="Lecture industrielle", n=15,
        title="Ce que la partie 1 vous apporte concrètement",
        lede="Trois conséquences directes, à protocole d'évaluation constant.")

    items = [
        ("Moins de déclenchements injustifiés",
         f"{fr(D.V1_MEAN['fp'], 2)} → {fr(D.V2_MEAN['fp'], 2)}",
         "fausses alarmes en moyenne sur 855 cycles sains",
         "Chaque fausse alarme est une coupure et un déplacement. "
         "Le nouveau mécanisme en supprime environ deux tiers, et surtout "
         "supprime le run aberrant à 22 fausses alarmes.", GOOD),
        ("Moins d'arcs manqués",
         f"{fr(D.V1_MEAN['rec'], 2, '%')} → {fr(D.V2_MEAN['rec'], 2, '%')}",
         "de rappel : la part des arcs réellement détectés",
         "C'est la métrique de sécurité. Sur le meilleur modèle, "
         f"{D.V2_BEST['fn']} arcs échappent à la détection sur "
         f"{D.V2_BEST['tp'] + D.V2_BEST['fn']}, contre environ 34 avant.",
         TEAL),
        ("Un modèle plus facile à embarquer",
         f"−{D.PARAMS_DELTA_PCT:.1f} %".replace(".", ","),
         "de paramètres, sans toucher aux branches",
         f"{frint(D.PARAMS['sequential']['total'])} paramètres, "
         "soit une empreinte mémoire réduite d'autant sur la cible. "
         "Le gain de précision ne se paye pas en coût matériel.", TEAL_DEEP),
    ]

    y = top - 0.014
    for lab, big, unit, body, col in items:
        h = 0.132
        card(ax, ML, y - h, MR - ML, h)
        ax.add_patch(plt.Rectangle((ML, y - h), 0.005, h, facecolor=col,
                                   zorder=3, edgecolor="none"))
        txt(ax, ML + 0.028, y - 0.038, lab, size=12.5, weight="bold",
            color=INK)
        txt(ax, ML + 0.028, y - 0.079, big, size=19, weight="bold", color=col)
        txt(ax, ML + 0.028, y - 0.112, unit, size=8.8, color=FAINT,
            family=SERIF)
        txt(ax, ML + 0.345, y - h / 2, wrap(body, 76), size=10.4, color=MUTED,
            family=SERIF, ls=1.5)
        y -= h + 0.012

    callout(ax, ML, y - 0.018 - 0.132, MR - ML, "La limite de ces chiffres",
            "Ces chiffres viennent d'un découpage aléatoire des cycles : "
            "le modèle a vu des cycles voisins des mêmes\nenregistrements. "
            "C'est un plafond de performance, pas une garantie sur une "
            "installation neuve — d'où la partie 2.",
            kind="engi")
    return save(fig, 15, "lecture_industrielle")
