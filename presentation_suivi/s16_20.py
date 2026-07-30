#!/usr/bin/env python3
"""Slides 16 à 20 — séparateur Partie 2, le test qui compte, le protocole,
les résultats cross-campagne, le diagnostic de l'écart."""

import numpy as np
from matplotlib.patches import Circle
from style import *          # noqa: F401,F403
import data as D


# ------------------------------------------------------------------ 16
def s16():
    fig, ax, _ = slide(dark=True, n=16)
    return section_divider(
        fig, ax, "02", "PARTIE 2  ·  ROBUSTESSE ET PISTE ARCSSM",
        "Tenir sur une installation\nqu'on n'a jamais vue",
        ["Le protocole : retirer une campagne entière",
         f"{D.V2_CV_MEAN['acc']:.0f} % de bonnes décisions hors banc, "
         f"AUC {fr(D.V2_CV_MEAN['auc'], 2)}",
         "Le diagnostic : ce qui tient, ce qui glisse",
         "ArcSSM et les prochaines étapes"], 16)


# ------------------------------------------------------------------ 17
def s17():
    fig, ax, top = slide(
        eb="La bonne question", n=17,
        title="Un bon score peut cacher une mauvaise nouvelle",
        lede="Tout dépend de la façon dont on découpe les données "
             "entre entraînement et test.")

    w = (MR - ML - 0.045) / 2
    y = top - 0.020
    h = 0.300

    # ---- deux protocoles côte à côte
    protos = [
        (ML, "Découpage aléatoire", NOISE,
         "On mélange tous les cycles, puis on en retire une partie pour tester.",
         "Le modèle a vu des cycles voisins, enregistrés le même jour, "
         "sur le même banc, avec les mêmes charges.",
         "Mesure un plafond.", f"{fr(D.DIAG['ceiling'], 1, '%')}"),
        (ML + w + 0.045, "Campagne entière retirée", TEAL,
         "On retire les quatre campagnes une par une, et on teste sur "
         "celle qui n'a jamais servi.",
         "Le modèle affronte un banc, des électrodes et un mélange de "
         "charges qu'il n'a jamais vus.",
         "Mesure la réalité.", f"{fr(D.V2_CV_MEAN['acc'], 1, '%')}"),
    ]
    for x, title, col, how, why, verdict, score in protos:
        card(ax, x, y - h, w, h)
        ax.add_patch(plt.Rectangle((x, y - 0.007), w, 0.007, facecolor=col,
                                   zorder=3, edgecolor="none"))
        txt(ax, x + 0.026, y - 0.042, title, size=13.5, weight="bold",
            color=col)
        txt(ax, x + 0.026, y - 0.072, wrap(how, 54), size=10.2, color=INK,
            va="top", family=SERIF, ls=1.5)
        txt(ax, x + 0.026, y - 0.146, wrap(why, 54), size=10.2, color=MUTED,
            va="top", family=SERIF, ls=1.5)
        rule(ax, y - h + 0.086, x + 0.026, x + w - 0.026)
        txt(ax, x + 0.026, y - h + 0.050, verdict, size=11.5, weight="bold",
            color=col)
        txt(ax, x + w - 0.026, y - h + 0.050, score, size=20, weight="bold",
            color=col, ha="right")

    # ---- la flèche de l'écart
    ay = y - h - 0.055
    ax.annotate("", xy=(ML + w + 0.045 + w * 0.5, ay),
                xytext=(ML + w * 0.5, ay),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color=ARC,
                                shrinkA=6, shrinkB=6), zorder=6)
    txt(ax, (ML + MR) / 2, ay + 0.030,
        f"l'écart réel : {D.DIAG['gap_total']} points",
        size=12, weight="bold", color=ARC, ha="center")

    callout(ax, ML, 0.078, MR - ML, "Pourquoi nous mesurons ainsi",
            "Un AFDD installé chez un client n'a jamais vu son installation. "
            "Le seul chiffre qui a une valeur d'engagement\nest celui mesuré sur "
            "une campagne inédite. Nous avons donc changé de protocole — "
            "c'est plus exigeant, et c'est\nce qui nous dit où travailler.",
            kind="intuition")
    return save(fig, 17, "protocole_question")


# ------------------------------------------------------------------ 18
def s18():
    fig, ax, top = slide(
        eb="Le protocole", n=18,
        title="Quatre campagnes, quatre entraînements séparés",
        lede="Chaque campagne sert de test une fois, et n'est jamais vue "
             "pendant l'entraînement correspondant.")

    # ---- matrice des plis
    x0 = ML + 0.135
    cw = 0.150
    gapx = 0.020
    rh = 0.056
    gapy = 0.012
    y = top - 0.070

    # en-têtes de colonnes = campagnes
    for j, camp in enumerate(D.CAMPAIGNS):
        n = D.V2_CV[j][1]
        cx = x0 + j * (cw + gapx)
        txt(ax, cx + cw / 2, y + 0.036, camp, size=10.5, weight="bold",
            color=INK, ha="center")
        txt(ax, cx + cw / 2, y + 0.012, f"{frint(n)} cycles", size=8.6,
            color=FAINT, ha="center")

    for i in range(4):
        ry = y - (i + 1) * (rh + gapy)
        txt(ax, ML, ry + rh / 2, f"Entraînement {i + 1}", size=10.2,
            color=MUTED, weight="bold")
        for j in range(4):
            cx = x0 + j * (cw + gapx)
            is_test = (i == j)
            fc = ARC_TINT if is_test else TEAL_TINT
            ec = ARC if is_test else TEAL
            tc = ARC if is_test else TEAL_DEEP
            card(ax, cx, ry, cw, rh, fc=fc, ec=ec, r=0.007, lw=1.1)
            txt(ax, cx + cw / 2, ry + rh / 2,
                "TEST" if is_test else "entraînement", size=9.6,
                weight="bold" if is_test else "normal", color=tc, ha="center")

    ybot = y - 4 * (rh + gapy)

    # légende
    lx = x0
    for fc, ec, lab in ((TEAL_TINT, TEAL, "sert à entraîner"),
                        (ARC_TINT, ARC, "jamais vu — sert à mesurer")):
        card(ax, lx, ybot - 0.048, 0.024, 0.020, fc=fc, ec=ec, r=0.004)
        txt(ax, lx + 0.032, ybot - 0.038, lab, size=9.2, color=MUTED)
        lx += 0.048 + 0.0062 * len(lab) * 1.25

    rule(ax, ybot - 0.078)

    # trois faits
    w3 = (MR - ML - 2 * 0.020) / 3
    facts = [
        (f"{frint(D.V2_CV_POOLED['n'])}", "cycles évalués au total",
         "Chaque cycle est testé une fois, par un modèle qui ne "
         "l'a jamais vu.", TEAL),
        ("3 sur 4", "campagnes issues du même banc",
         "La limite du jeu de données actuel : une seule "
         "installation vraiment différente.", ARC),
        ("1", "campagne perdue pour l'entraînement",
         "Le modèle n'apprend que sur trois campagnes. Moins de "
         "données, protocole plus dur.", MUTED),
    ]
    fy = ybot - 0.100
    for i, (big, unit, body, col) in enumerate(facts):
        x = ML + i * (w3 + 0.020)
        h = fy - 0.078
        card(ax, x, 0.078, w3, h)
        txt(ax, x + 0.022, 0.078 + h - 0.044, big, size=21, weight="bold",
            color=col)
        txt(ax, x + 0.022, 0.078 + h - 0.072, unit, size=9.4, color=FAINT,
            family=SERIF)
        txt(ax, x + 0.022, 0.078 + h - 0.094, wrap(body, 46), size=9.6,
            color=MUTED, va="top", family=SERIF, ls=1.5)
    return save(fig, 18, "protocole")


# ------------------------------------------------------------------ 19
def s19():
    fig, ax, top = slide(
        eb="Résultats hors banc", n=19,
        title="Ce que donne le modèle sur une campagne inédite",
        lede="Arc-FaultNet, protocole leave-one-campaign-out, "
             "avec les techniques de robustesse activées.")

    # ---- tableau par campagne
    w = MR - ML
    rows = []
    for camp, n, acc, f1, prec, rec, spec, auc in D.V2_CV:
        rows.append([camp, frint(n), fr(acc, 2, "%"), fr(f1, 2, "%"),
                     fr(rec, 2, "%"), fr(spec, 2, "%"), fr(auc, 3)])

    def fmt(i, j, v):
        if j == 6:
            return {"color": TEAL_DEEP, "weight": "bold"}
        if j == 2:
            return {"color": INK, "weight": "bold"}
        return None

    widths = [w * 0.19, w * 0.13, w * 0.14, w * 0.13, w * 0.13, w * 0.14,
              w * 0.14]
    bot = table(ax, ["Campagne testée", "Cycles", "Exactitude", "F1",
                     "Rappel", "Spécificité", "AUC"],
                rows, ML, top - 0.020, widths, row_h=0.054, head_h=0.046,
                fs=10.4, cell_fmt=fmt)

    # ---- bandeau moyenne / cumul
    sy = bot - 0.020 - 0.092
    card(ax, ML, sy, w, 0.092, fc=GREY_TINT, ec=LINE)
    txt(ax, ML + 0.024, sy + 0.046, "Sur les\n4 campagnes", size=9.4,
        weight="bold", color=TEAL_DEEP, ls=1.4)
    parts = [
        ("Exactitude moyenne", f"{fr(D.V2_CV_MEAN['acc'], 2, '%')}"),
        ("F1 moyen", f"{fr(D.V2_CV_MEAN['f1'], 2, '%')}"),
        ("AUC moyen", fr(D.V2_CV_MEAN["auc"], 3)),
        ("Exactitude cumulée", fr(D.V2_CV_POOLED["acc"], 2, "%")),
        ("Arcs détectés", fr(D.V2_CV_POOLED["rec"], 2, "%")),
    ]
    xx = ML + 0.140
    step = (w - 0.165) / len(parts)
    for lab, val in parts:
        txt(ax, xx + step / 2, sy + 0.060, val, size=13, weight="bold",
            color=INK, ha="center")
        txt(ax, xx + step / 2, sy + 0.026, lab, size=8.6, color=FAINT,
            ha="center")
        xx += step

    # ---- deux lectures
    cy = sy - 0.020
    wl = w * 0.485
    h = cy - 0.078
    for i, (lab, body, col, kind) in enumerate((
        ("La bonne nouvelle",
         f"AUC moyen {fr(D.V2_CV_MEAN['auc'], 3)} : à l'intérieur de chaque "
         "campagne, le modèle place presque toujours les arcs au-dessus des "
         "cycles sains. Y compris sur le banc 2026, le seul vraiment "
         "différent (AUC 0,997).", GOOD, "good"),
        ("Le point à travailler",
         "Mais l'exactitude va de 77,7 % à 91,8 % selon la campagne. Le "
         "niveau à partir duquel le modèle déclenche n'est pas le même d'un "
         "banc à l'autre : un problème de réglage, pas de reconnaissance.",
         ARC, "warn"),
    )):
        x = ML + i * (wl + 0.030)
        lband(ax, x, 0.078, wl, h, col,
              "#E8F4EE" if kind == "good" else ARC_TINT)
        txt(ax, x + 0.024, 0.078 + h - 0.032, sp(lab.upper()), size=8.4,
            color=col, weight="bold")
        txt(ax, x + 0.024, 0.078 + h - 0.054, wrap(body, 60), size=10.0,
            color=INK, va="top", family=SERIF, ls=1.55)
    return save(fig, 19, "resultats_hors_banc")


# ------------------------------------------------------------------ 20
def s20():
    fig, ax, top = slide(
        eb="Le diagnostic", n=20,
        title="Le classement tient, c'est le seuil qui glisse",
        lede="L'écart de 17 points se décompose en deux causes très "
             "inégales en difficulté.")

    # ---- schéma : distributions décalées d'une campagne à l'autre
    gx0, gx1 = ML + 0.030, ML + 0.470
    gy = 0.382
    card(ax, ML, 0.268, 0.520, 0.300)
    txt(ax, ML + 0.030, 0.545, "Le score du modèle, campagne par campagne",
        size=11.5, weight="bold", color=INK)

    def px(u):
        return gx0 + u * (gx1 - gx0)

    xs = np.linspace(0, 1, 400)

    def bump(c, s):
        return np.exp(-((xs - c) ** 2) / (2 * s ** 2))

    # deux campagnes : distributions sain / arc décalées
    for k, (cn, ca, lab, col, off) in enumerate((
            (0.047, 0.497, "8 juillet", "#8FA6B0", 0.0),
            (0.450, 0.973, "banc 2026", TEAL, 0.090))):
        base = gy + off
        for c, fill, name in ((cn, "#C3CDD4", "sain"), (ca, col, "arc")):
            yv = bump(c, 0.085) * 0.062
            ax.fill_between(px(xs), base, base + yv, color=fill, alpha=0.75,
                            zorder=4, linewidth=0)
            ax.plot(px(xs), base + yv, color=fill, lw=1.3, zorder=5)
        ax.plot([gx0, gx1], [base, base], color=LINE, lw=0.9, zorder=3)
        txt(ax, gx0 - 0.006, base + 0.026, lab, size=9, color=INK,
            weight="bold", ha="right")

    # le seuil unique à 0,5
    ax.plot([px(0.5), px(0.5)], [gy - 0.026, gy + 0.090 + 0.048],
            color=ARC, lw=1.8, ls=(0, (3, 2.5)), zorder=6)
    txt(ax, px(0.5) + 0.007, gy + 0.006, "seuil unique 0,5", size=8.4,
        color=ARC, weight="bold", rotation=90, va="bottom")

    ax.plot([gx0, gx1], [gy - 0.026, gy - 0.026], color=LINE, lw=1.0, zorder=3)
    txt(ax, gx0, gy - 0.048, "0", size=8.4, color=FAINT, ha="center")
    txt(ax, gx1, gy - 0.048, "1", size=8.4, color=FAINT, ha="center")
    txt(ax, (gx0 + gx1) / 2, gy - 0.070,
        "score de sortie du modèle", size=8.8, color=FAINT, ha="center",
        family=SERIF)

    lx = gx0
    for fc, lab in (("#C3CDD4", "cycles sains"), (TEAL, "cycles avec arc")):
        ax.add_patch(plt.Rectangle((lx, 0.286), 0.018, 0.012, facecolor=fc,
                                   zorder=5, edgecolor="none"))
        txt(ax, lx + 0.026, 0.292, lab, size=8.6, color=MUTED)
        lx += 0.042 + 0.0058 * len(lab) * 1.3

    # ---- décomposition de l'écart, à droite
    xr = ML + 0.550
    wr = MR - xr
    txt(ax, xr, top - 0.018, "D'où viennent les 17 points", size=12.5,
        weight="bold", color=INK)

    splits = [
        (D.DIAG["gap_calibration"], "Réglage du seuil", GOOD,
         "Le modèle note correctement, mais toute son échelle de notes se "
         "décale d'un banc à l'autre. Réglable à la mise en service, sans "
         "réentraîner.", "récupérable"),
        (D.DIAG["gap_representation"], "Vraie différence de signal", ARC,
         "Ce que l'arc « ressemble » change réellement entre bancs et mélanges "
         "de charges. Demande plus de données et de robustesse.",
         "chantier de fond"),
    ]
    yy = top - 0.058
    for pts, lab, col, body, tag in splits:
        h = 0.186
        card(ax, xr, yy - h, wr, h)
        ax.add_patch(plt.Rectangle((xr, yy - h), 0.005, h, facecolor=col,
                                   zorder=3, edgecolor="none"))
        txt(ax, xr + 0.026, yy - 0.042, f"{pts} pts", size=19, weight="bold",
            color=col)
        txt(ax, xr + 0.115, yy - 0.042, lab, size=11.5, weight="bold",
            color=INK)
        pill(ax, xr + 0.115, yy - 0.072, tag,
             fc="#E8F4EE" if col == GOOD else ARC_TINT, tc=col, size=8,
             h=0.030)
        txt(ax, xr + 0.026, yy - 0.102, wrap(body, 57), size=9.7, color=MUTED,
            va="top", family=SERIF, ls=1.5)
        yy -= h + 0.016

    callout(ax, ML, 0.078, MR - ML, "La preuve que le seuil est bien le coupable",
            f"En réglant le seuil séparément sur chaque campagne — ce qu'un "
            f"appareil peut faire à l'installation, sans\naucune étiquette — "
            f"l'exactitude cumulée passe de "
            f"{fr(D.DIAG['pooled_fixe'], 2, '%')} à "
            f"{fr(D.DIAG['pooled_seuil_local'], 2, '%')}. "
            "Même modèle, mêmes poids : seul le point de déclenchement change.",
            kind="good")
    return save(fig, 20, "diagnostic_ecart")
