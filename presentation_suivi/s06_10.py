#!/usr/bin/env python3
"""Slides 6 à 10 — le diagnostic, les deux mécanismes, le gain en paramètres."""

import numpy as np
from matplotlib.patches import Circle
from style import *          # noqa: F401,F403
import data as D

# grille commune aux deux schémas
CX = [0.058, 0.238, 0.418, 0.598, 0.778]
CW = 0.126
ROW_T, ROW_S = 0.498, 0.328          # bas des boîtes temporelle / spectrale
BH = 0.090                            # hauteur d'une boîte
MIDY = (ROW_T + ROW_S + BH) / 2       # axe médian


def _lane_label(ax, y, s, col):
    txt(ax, ML - 0.002, y + BH / 2, s, size=8.6, color=col, weight="bold",
        ha="right", rotation=90, va="center")


# ------------------------------------------------------------------ 06
def s06():
    fig, ax, top = slide(
        eb="Le diagnostic", n=6,
        title="Le problème : on résumait avant de comparer",
        lede="Toute la question tient à l'ordre de deux opérations.")

    y = top - 0.030
    w = (MR - ML - 0.040) / 2

    variants = [
        (ML, "Avant  ·  fusion « à porte »", NOISE, ARC_TINT, "×",
         "On compare deux résumés.",
         "Un pic de courant à l'instant t et une bouffée haute-fréquence\n"
         "au même instant t deviennent indiscernables :\nla dimension temps a déjà disparu."),
        (ML + w + 0.040, "Maintenant  ·  cross-attention séquentielle",
         TEAL, TEAL_TINT, "↔",
         "On compare deux déroulés.",
         "Le modèle peut dire : « ce pic de courant et cette bouffée\n"
         "haute-fréquence arrivent au même moment ».\nC'est exactement la signature d'un arc."),
    ]

    for x, title, col, tint, sym, punch, expl in variants:
        h = 0.336
        y0 = y - h
        card(ax, x, y0, w, h)
        ax.add_patch(plt.Rectangle((x, y - 0.007), w, 0.007, facecolor=col,
                                   zorder=3, edgecolor="none"))
        txt(ax, x + 0.024, y - 0.040, title, size=12.5, weight="bold",
            color=col)

        # deux pistes + point de fusion
        gy0, gy1 = y - 0.098, y - 0.146
        xa, xb = x + 0.034, x + w - 0.034
        merge_x = xa + (xb - xa) * (0.32 if col == NOISE else 0.74)
        for gy, lab in ((gy0, "temps"), (gy1, "fréquence")):
            ax.plot([xa, merge_x], [gy, gy], color=col, lw=2.4,
                    solid_capstyle="round", zorder=5)
            ax.plot([merge_x, xb], [gy, gy], color=LINE, lw=1.6,
                    zorder=4, ls=(0, (2, 2)))
            txt(ax, xa, gy + 0.021, lab, size=8.2, color=FAINT)
        ax.plot([merge_x, merge_x], [gy1, gy0], color=col, lw=1.6, zorder=5)
        op_node(ax, merge_x, (gy0 + gy1) / 2, sym, rr=0.016, fc=col, fs=10)
        txt(ax, merge_x, gy0 + 0.021, "fusion", size=8.2, color=col,
            weight="bold", ha="center")
        txt(ax, xb, gy0 + 0.021, "décision", size=8.2, color=FAINT, ha="right")

        bh_ = 0.158
        lband(ax, x + 0.022, y0 + 0.020, w - 0.044, bh_, col, tint)
        txt(ax, x + 0.044, y0 + 0.020 + bh_ - 0.030, punch, size=11.5,
            weight="bold", color=col)
        txt(ax, x + 0.044, y0 + 0.020 + bh_ - 0.056, expl, size=9.6,
            color=INK, va="top", family=SERIF, ls=1.55)

    callout(ax, ML, 0.078, MR - ML, "Pourquoi c'est le bon changement",
            "Un arc ne se reconnaît pas à une moyenne, mais à la coïncidence entre "
            "une déformation de la forme d'onde\net une bouffée haute-fréquence. "
            "Il fallait donc que la fusion voie le temps.", kind="intuition")
    return save(fig, 6, "diagnostic")


# ------------------------------------------------------------------ 07
def s07():
    fig, ax, top = slide(
        eb="Avant", n=7,
        title="L'ancien mécanisme : une porte sur des moyennes",
        lede="Chaque étape est correcte, mais l'étape 2 est irréversible.")

    for lab, y, col in (("Temporel", ROW_T, TEAL), ("Spectral", ROW_S, ARC)):
        _lane_label(ax, y, lab.upper(), col)

    # étapes 1 à 3 : deux pistes parallèles
    stages = [
        ("Séquence", "B × C × T", SURFACE, INK),
        ("Moyenne globale", "on résume tout", ARC_TINT, ARC),
        ("Un vecteur", "B × C", SURFACE, INK),
    ]
    for i, (t_, s_, fc, tc) in enumerate(stages):
        for y in (ROW_T, ROW_S):
            box_node(ax, CX[i], y, CW, BH, t_, s_, fc=fc, tc=tc,
                     ec=ARC if fc == ARC_TINT else LINE, fs=10, sfs=8.4)
            if i:
                arrow(ax, (CX[i - 1] + CW, y + BH / 2), (CX[i] - 0.004,
                      y + BH / 2), color=FAINT, lw=1.4, ms=8)

    # étape 4 : concaténation + MLP → masques
    box_node(ax, CX[3], MIDY - 0.075, CW, 0.150,
             "Concaténation", "puis 2 petits réseaux\ndenses → masques",
             fs=10, sfs=8.4)
    for y in (ROW_T, ROW_S):
        arrow(ax, (CX[2] + CW, y + BH / 2), (CX[3] - 0.004, MIDY),
              color=FAINT, lw=1.4, ms=8, rad=0.12 if y == ROW_T else -0.12)

    # étape 5 : application + décision
    box_node(ax, CX[4], MIDY - 0.075, CW, 0.150,
             "Application", "chaque vecteur est\npondéré, puis fusionné",
             fs=10, sfs=8.4)
    arrow(ax, (CX[3] + CW, MIDY), (CX[4] - 0.004, MIDY), color=FAINT,
          lw=1.4, ms=8)
    arrow(ax, (CX[4] + CW, MIDY), (CX[4] + CW + 0.038, MIDY), color=TEAL_DEEP,
          lw=1.8, ms=10)
    txt(ax, CX[4] + CW + 0.044, MIDY, "p(arc)", size=10.5, weight="bold",
        color=TEAL_DEEP, va="center")

    # annotation du point de perte
    xg = CX[1] + CW / 2
    ax.plot([xg, xg], [ROW_S - 0.026, ROW_S - 0.052], color=ARC, lw=1.4,
            zorder=5)
    txt(ax, xg, ROW_S - 0.072, "la dimension « temps » disparaît ici",
        size=9.6, color=ARC, weight="bold", ha="center")

    callout(ax, ML, 0.088, (MR - ML) * 0.615, "Ce que ça coûte",
            "Après la moyenne il ne reste qu'un niveau par descripteur.\n"
            "On peut encore pondérer une branche, plus les aligner.",
            kind="warn")

    x2 = ML + (MR - ML) * 0.645
    card(ax, x2, 0.088, MR - x2, 0.148, fc=GREY_TINT, ec=LINE)
    txt(ax, (x2 + MR) / 2, 0.196, frint(D.PARAMS["gated"]["fusion"]),
        size=24, weight="bold", color=NOISE, ha="center")
    txt(ax, (x2 + MR) / 2, 0.148, "paramètres pour la fusion",
        size=9.8, color=MUTED, ha="center", family=SERIF)
    txt(ax, (x2 + MR) / 2, 0.115,
        f"sur {frint(D.PARAMS['gated']['total'])} au total",
        size=9, color=FAINT, ha="center", family=SERIF)
    return save(fig, 7, "ancienne_fusion")


# ------------------------------------------------------------------ 08
def s08():
    fig, ax, top = slide(
        eb="Maintenant", n=8,
        title="Le nouveau mécanisme : on compare avant de résumer",
        lede="Même entrée, même sortie. Seul l'ordre des opérations change.")

    for lab, y, col in (("Temporel", ROW_T, TEAL), ("Spectral", ROW_S, ARC)):
        _lane_label(ax, y, lab.upper(), col)

    # 1 — séquences conservées
    for y in (ROW_T, ROW_S):
        box_node(ax, CX[0], y, CW, BH, "Séquence", "B × C × T", fs=10, sfs=8.4)

    # 2 — projections légères
    for y in (ROW_T, ROW_S):
        box_node(ax, CX[1], y, CW, BH, "Projections", "3 convolutions 1×1",
                 fc=TEAL_TINT, ec=TEAL, tc=TEAL_DEEP, fs=10, sfs=8.4)
        arrow(ax, (CX[0] + CW, y + BH / 2), (CX[1] - 0.004, y + BH / 2),
              color=FAINT, lw=1.4, ms=8)

    # 3 — attention croisée bidirectionnelle
    AX0, AW = CX[2] - 0.032, CW + 0.064
    box_node(ax, AX0, MIDY - 0.100, AW, 0.200,
             "Attention croisée", "chaque instant d'une branche\ninterroge chaque "
             "instant\nde l'autre — et l'inverse\n\n4 têtes en parallèle",
             fc=TEAL_TINT, ec=TEAL, tc=TEAL_DEEP, fs=11, sfs=8.6)
    for y, r in ((ROW_T, 0.14), (ROW_S, -0.14)):
        arrow(ax, (CX[1] + CW, y + BH / 2), (AX0 - 0.004, MIDY),
              color=TEAL, lw=1.6, ms=9, rad=r)

    # 4 — résiduel puis moyenne
    for y, r in ((ROW_T, -0.14), (ROW_S, 0.14)):
        box_node(ax, CX[3], y, CW, BH, "Recalé + résumé",
                 "normalisation, puis\nmoyenne globale", fs=10, sfs=8.4)
        arrow(ax, (AX0 + AW, MIDY), (CX[3] - 0.004, y + BH / 2),
              color=TEAL, lw=1.6, ms=9, rad=r)

    # 5 — décision
    box_node(ax, CX[4], MIDY - 0.075, CW, 0.150, "Décision",
             "concaténation puis\nréseau de classification", fs=10, sfs=8.4)
    for y, r in ((ROW_T, 0.12), (ROW_S, -0.12)):
        arrow(ax, (CX[3] + CW, y + BH / 2), (CX[4] - 0.004, MIDY),
              color=FAINT, lw=1.4, ms=8, rad=r)
    arrow(ax, (CX[4] + CW, MIDY), (CX[4] + CW + 0.038, MIDY), color=TEAL_DEEP,
          lw=1.8, ms=10)
    txt(ax, CX[4] + CW + 0.044, MIDY, "p(arc)", size=10.5, weight="bold",
        color=TEAL_DEEP, va="center")

    xg = AX0 + AW / 2
    ax.plot([xg, xg], [MIDY - 0.107, MIDY - 0.137], color=TEAL, lw=1.4, zorder=5)
    txt(ax, xg, MIDY - 0.157, "la dimension « temps » est encore là",
        size=9.6, color=TEAL_DEEP, weight="bold", ha="center")

    callout(ax, ML, 0.088, (MR - ML) * 0.615, "Ce que ça apporte",
            "Des convolutions 1×1 partagées le long du temps\n"
            "remplacent les couches denses : moins de poids,\n"
            "et l'accès à la coïncidence entre les deux branches.",
            kind="good")

    x2 = ML + (MR - ML) * 0.645
    card(ax, x2, 0.088, MR - x2, 0.148, fc=GREY_TINT, ec=LINE)
    txt(ax, (x2 + MR) / 2, 0.196, frint(D.PARAMS["sequential"]["fusion"]),
        size=24, weight="bold", color=GOOD, ha="center")
    txt(ax, (x2 + MR) / 2, 0.148, "paramètres pour la fusion",
        size=9.8, color=MUTED, ha="center", family=SERIF)
    txt(ax, (x2 + MR) / 2, 0.115,
        f"soit {D.FUSION_DELTA_PCT:.0f} % de moins qu'avant",
        size=9, color=GOOD, ha="center", family=SERIF, weight="bold")
    return save(fig, 8, "nouvelle_fusion")


# ------------------------------------------------------------------ 09
def s09():
    fig, ax, top = slide(
        eb="L'intuition", n=9,
        title="Ce que le modèle peut voir maintenant",
        lede="Un arc, c'est une coïncidence entre deux signaux. "
             "Il faut la même horloge pour les deux.")

    # ---- schéma : deux pistes temporelles alignées
    x0, x1 = ML + 0.045, MR - 0.235
    t = np.linspace(0, 1, 700)
    span = x1 - x0
    ev = 0.46                            # position de l'événement d'arc

    def px(u):
        return x0 + u * span

    yb_t, yb_s = 0.520, 0.360
    amp = 0.052
    rng = np.random.default_rng(3)

    # zone d'événement
    ax.add_patch(plt.Rectangle((px(ev - 0.055), 0.318), span * 0.11, 0.262,
                               facecolor=ARC_TINT, zorder=1, edgecolor="none"))

    # piste temporelle : sinusoïde déformée
    wave = np.sin(2 * np.pi * 2 * t)
    dip = np.exp(-((t - ev) ** 2) / (2 * 0.022 ** 2))
    wave_a = wave * (1 - 0.55 * dip)
    ax.plot(px(t), yb_t + wave_a * amp, color=TEAL, lw=2.0, zorder=4)
    txt(ax, ML, yb_t + 0.075, "Branche temporelle  ·  la forme d'onde",
        size=10.5, weight="bold", color=TEAL_DEEP)

    # piste spectrale : énergie haute-fréquence
    hf = 0.16 + 0.80 * np.exp(-((t - ev) ** 2) / (2 * 0.026 ** 2)) \
        + 0.05 * rng.standard_normal(t.size)
    hf = np.clip(hf, 0.05, None)
    ax.fill_between(px(t), yb_s - 0.048, yb_s - 0.048 + hf * 0.085,
                    color=ARC, alpha=0.30, zorder=3, linewidth=0)
    ax.plot(px(t), yb_s - 0.048 + hf * 0.085, color=ARC, lw=1.6, zorder=4)
    txt(ax, ML, yb_s + 0.062, "Branche spectrale  ·  l'énergie haute-fréquence",
        size=10.5, weight="bold", color=ARC)

    # trait d'alignement
    ax.plot([px(ev), px(ev)], [0.318, 0.580], color=ARC, lw=1.6, ls=(0, (3, 3)),
            zorder=5)
    op_node(ax, px(ev), 0.598, "↔", rr=0.018, fc=ARC, fs=11)
    txt(ax, px(ev), 0.638, "même instant", size=9.8, color=ARC, weight="bold",
        ha="center")

    ax.plot([x0, x1], [0.300, 0.300], color=LINE, lw=1.0, zorder=3)
    for u, lab in ((0.0, "0 ms"), (0.5, "10 ms"), (1.0, "20 ms")):
        ax.plot([px(u), px(u)], [0.300, 0.290], color=LINE, lw=1.0, zorder=3)
        txt(ax, px(u), 0.272, lab, size=8.4, color=FAINT, ha="center")
    txt(ax, (x0 + x1) / 2, 0.240, "un cycle du secteur  ·  2 048 points",
        size=9, color=FAINT, ha="center", family=SERIF)

    # ---- encadré de droite
    xr = MR - 0.200
    card(ax, xr, 0.300, 0.200, 0.300)
    txt(ax, xr + 0.100, 0.560, sp("LA QUESTION POSÉE"), size=8, color=FAINT,
        weight="bold", ha="center")
    txt(ax, xr + 0.016, 0.522,
        "« La déformation de\nl'onde et la bouffée\nhaute-fréquence\n"
        "tombent-elles au\nmême instant ? »",
        size=11, color=INK, va="top", family=SERIF, ls=1.65)
    rule(ax, 0.348, xr + 0.016, xr + 0.184)
    txt(ax, xr + 0.100, 0.318, "Avant : hors de portée.", size=10,
        color=NOISE, weight="bold", ha="center")

    callout(ax, ML, 0.080, MR - ML, "Traduction concrète",
            "C'est ce qui distingue un vrai arc d'un simple appel de courant : "
            "un moteur qui démarre déforme l'onde\nsans produire cette bouffée "
            "haute-fréquence au même instant. Le modèle a maintenant les moyens "
            "de faire la différence.", kind="intuition")
    return save(fig, 9, "alignement")


# ------------------------------------------------------------------ 10
def s10():
    fig, ax, top = slide(
        eb="Coût du modèle", n=10,
        title=f"{frint(D.PARAMS_DELTA)} paramètres en moins",
        lede="Le gain vient entièrement du bloc de fusion. "
             "Les deux branches sont inchangées.")

    # ---- barres empilées comparatives
    blocks = [
        ("Branche temporelle", D.PARAMS_SHARED["temporal"], "#9FB4BD"),
        ("Branche spectrale", D.PARAMS_SHARED["spectral"], "#5D8A93"),
        ("Bloc de fusion", None, ARC),
        ("Classifieur", D.PARAMS_SHARED["classifier"], "#C3CDD4"),
    ]
    total_max = D.PARAMS["gated"]["total"]
    x_bar = ML + 0.218
    bar_w = 0.545
    for k, (key, label, y) in enumerate((
            ("gated", "Avant  ·  fusion « à porte »", 0.470),
            ("sequential", "Maintenant  ·  cross-attention", 0.330))):
        h = 0.078
        acc = 0.0
        for name, val, col in blocks:
            v = D.PARAMS[key]["fusion"] if val is None else val
            frac = v / total_max
            ax.add_patch(plt.Rectangle((x_bar + acc * bar_w, y), frac * bar_w,
                                       h, facecolor=col, edgecolor="white",
                                       lw=1.2, zorder=4))
            if frac > 0.10:
                txt(ax, x_bar + (acc + frac / 2) * bar_w, y + h / 2,
                    frint(v), size=9.2, color="white", weight="bold",
                    ha="center", z=6)
            acc += frac
        txt(ax, ML, y + h / 2, label, size=10.5, weight="bold", color=INK)
        txt(ax, x_bar + acc * bar_w + 0.014, y + h / 2,
            frint(D.PARAMS[key]["total"]), size=12, weight="bold",
            color=NOISE if key == "gated" else GOOD, va="center")

    # légende
    lx = x_bar
    for name, val, col in blocks:
        ax.add_patch(plt.Rectangle((lx, 0.268), 0.014, 0.014, facecolor=col,
                                   zorder=4, edgecolor="none"))
        txt(ax, lx + 0.020, 0.275, name, size=8.8, color=MUTED)
        lx += 0.028 + 0.0058 * len(name) * 1.35

    # flèche du gain
    ax.annotate("", xy=(x_bar + bar_w, 0.415),
                xytext=(x_bar + bar_w * D.PARAMS["sequential"]["total"] / total_max,
                        0.415),
                arrowprops=dict(arrowstyle="<->", lw=1.6, color=GOOD),
                zorder=6)
    txt(ax, x_bar + bar_w * 0.955, 0.443,
        f"−{frint(D.PARAMS_DELTA)}", size=10.5, color=GOOD, weight="bold",
        ha="right")

    rule(ax, 0.235)
    w = (MR - ML - 2 * 0.020) / 3
    kpi(ax, ML, 0.078, w, f"−{D.PARAMS_DELTA_PCT:.1f} %".replace(".", ","),
        "Taille totale du modèle",
        f"{frint(D.PARAMS['gated']['total'])} → "
        f"{frint(D.PARAMS['sequential']['total'])}", h=0.142, vcolor=GOOD,
        vsize=24, accent=GOOD)
    kpi(ax, ML + w + 0.020, 0.078, w, f"−{D.FUSION_DELTA_PCT:.0f} %",
        "Bloc de fusion seul",
        f"{frint(D.PARAMS['gated']['fusion'])} → "
        f"{frint(D.PARAMS['sequential']['fusion'])}", h=0.142,
        vcolor=GOOD, vsize=24, accent=GOOD)
    kpi(ax, ML + 2 * (w + 0.020), 0.078, w, "0", "Autre bloc modifié",
        "les branches et le classifieur sont identiques", h=0.142,
        vcolor=TEAL_DEEP, vsize=24, accent=TEAL)
    return save(fig, 10, "parametres")
