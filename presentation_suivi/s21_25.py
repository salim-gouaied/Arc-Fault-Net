#!/usr/bin/env python3
"""Slides 21 à 25 — ArcSSM expliqué, ses résultats, le plan, la clôture."""

import numpy as np
from matplotlib.patches import Circle
from style import *          # noqa: F401,F403
import data as D


# ------------------------------------------------------------------ 21
def s21():
    fig, ax, top = slide(
        eb="Piste ArcSSM", n=21,
        title="Une autre façon de lire le signal : la mémoire",
        lede="Au lieu de découper le cycle en morceaux, le modèle le lit "
             "point par point et retient l'essentiel.")

    # ---- schéma : le SSM déroulé
    card(ax, ML, 0.300, MR - ML, 0.278)
    txt(ax, ML + 0.026, 0.552, "Le principe, déroulé dans le temps",
        size=11.5, weight="bold", color=INK)

    n = 7
    x0, x1 = ML + 0.085, MR - 0.145
    xs = np.linspace(x0, x1, n)
    yin, ymem = 0.480, 0.395

    txt(ax, ML + 0.070, yin, "signal", size=9.4, color=MUTED, ha="right",
        weight="bold")
    txt(ax, ML + 0.070, ymem, "mémoire", size=9.4, color=TEAL_DEEP,
        ha="right", weight="bold")

    rng = np.random.default_rng(11)
    vals = [0.20, 0.28, 0.24, 0.86, 0.78, 0.72, 0.66]
    for i, x in enumerate(xs):
        is_arc = i >= 3
        col = ARC if is_arc else "#9FB4BD"
        # échantillon d'entrée
        ax.add_patch(Circle((x, yin), 0.0095, facecolor=col, edgecolor="white",
                            lw=1.2, zorder=6))
        # état de mémoire, dont la taille suit l'accumulation
        r = 0.013 + 0.016 * vals[i]
        ax.add_patch(Circle((x, ymem), r, facecolor=TEAL_TINT,
                            edgecolor=TEAL, lw=1.4, zorder=6))
        txt(ax, x, ymem, "h", size=8.6, color=TEAL_DEEP, weight="bold",
            ha="center", z=7)
        arrow(ax, (x, yin - 0.012), (x, ymem + r + 0.003), color=col, lw=1.2,
              ms=7)
        if i:
            arrow(ax, (xs[i - 1] + 0.030, ymem), (x - r - 0.004, ymem),
                  color=TEAL, lw=1.6, ms=8)

    arrow(ax, (x1 + 0.030, ymem), (x1 + 0.070, ymem), color=TEAL_DEEP, lw=1.8,
          ms=10)
    txt(ax, x1 + 0.078, ymem, "arc /\nnon-arc", size=9.6, weight="bold",
        color=TEAL_DEEP, ls=1.4)

    ax.add_patch(plt.Rectangle((xs[3] - 0.022, 0.352), xs[-1] - xs[3] + 0.044,
                               0.148, facecolor=ARC_TINT, zorder=1,
                               edgecolor="none"))
    txt(ax, (xs[3] + xs[-1]) / 2, 0.332, "l'arc démarre ici et se maintient",
        size=9.2, color=ARC, weight="bold", ha="center")

    # ---- trois encarts
    w3 = (MR - ML - 2 * 0.020) / 3
    items = [
        ("L'idée en une phrase",
         "Le modèle garde un résumé de tout ce qu'il a lu depuis le début du "
         "cycle, et le met à jour à chaque nouveau point.", TEAL),
        ("Pourquoi c'est adapté",
         "Un arc a une histoire : un motif avant, une signature pendant, et "
         "il se maintient une fois amorcé. Un modèle à mémoire suit ça "
         "naturellement.", TEAL),
        ("Le bonus : pas de Fourier",
         "En rendant cette mémoire oscillante, elle joue le rôle d'un banc de "
         "filtres appris. Le modèle analyse les fréquences lui-même.", ARC),
    ]
    for i, (lab, body, col) in enumerate(items):
        x = ML + i * (w3 + 0.020)
        h = 0.190
        card(ax, x, 0.078, w3, h)
        ax.add_patch(plt.Rectangle((x, 0.078 + h - 0.006), w3, 0.006,
                                   facecolor=col, zorder=3, edgecolor="none"))
        txt(ax, x + 0.024, 0.078 + h - 0.038, lab, size=11.5, weight="bold",
            color=col)
        txt(ax, x + 0.024, 0.078 + h - 0.066, wrap(body, 45), size=9.9,
            color=MUTED, va="top", family=SERIF, ls=1.55)
    return save(fig, 21, "arcssm_idee")


# ------------------------------------------------------------------ 22
def s22():
    fig, ax, top = slide(
        eb="Piste ArcSSM", n=22,
        title="Pourquoi nous explorons cette piste",
        lede="Trois raisons, par ordre d'importance pour nous — "
             "et une honnêteté sur ce qui n'en est pas une.")

    reasons = [
        ("01", "L'effet mémoire",
         "C'est la raison principale. L'arc ne se lit pas dans un instant "
         "isolé mais dans un enchaînement. Le modèle suit cette dynamique "
         "sans qu'on ait à la lui décrire.", TEAL),
        ("02", "Le bon a priori physique",
         "Ces équations décrivent depuis toujours les systèmes électriques "
         "continus, comme un circuit RLC. On utilise l'outil fait pour les "
         "signaux, pas un modèle emprunté au texte.", TEAL),
        ("03", "Léger à l'exécution",
         "À l'inférence, le modèle se réduit à une petite mise à jour de "
         "mémoire par point, à coût constant. C'est intéressant pour une "
         "cible embarquée.", TEAL),
    ]

    y = top - 0.014
    for num, lab, body, col in reasons:
        h = 0.155
        card(ax, ML, y - h, (MR - ML) * 0.615, h)
        ax.add_patch(plt.Rectangle((ML, y - h), 0.005, h, facecolor=col,
                                   zorder=3, edgecolor="none"))
        txt(ax, ML + 0.030, y - h / 2, num, size=22, weight="bold",
            color="#C3CDD4")
        txt(ax, ML + 0.086, y - 0.046, lab, size=12.5, weight="bold",
            color=INK)
        txt(ax, ML + 0.086, y - 0.078, wrap(body, 82), size=10.1, color=MUTED,
            va="top", family=SERIF, ls=1.55)
        y -= h + 0.016

    # ---- colonne droite
    xr = ML + (MR - ML) * 0.645
    wr = MR - xr
    callout(ax, xr, top - 0.014 - 0.200, wr, "L'honnêteté d'ingénieur",
            "On lit parfois que ce type de modèle\n"
            "« passe mieux à l'échelle » que\n"
            "l'attention. Sur nos 2 048 points,\n"
            "c'est faux : l'attention passe très\n"
            "bien. Notre justification tient\n"
            "sur la mémoire, pas sur la vitesse.",
            kind="engi", fs=10.0, h=0.200)

    yy = top - 0.014 - 0.200 - 0.020
    card(ax, xr, y + 0.016, wr, yy - (y + 0.016))
    txt(ax, xr + 0.024, yy - 0.036, "Où en est cette piste", size=11.5,
        weight="bold", color=INK)
    states = [
        ("Le modèle tourne", True),
        ("Évalué sur le même protocole", True),
        ("Encore derrière Arc-FaultNet", False),
        ("Optimisation à venir", False),
    ]
    ys = yy - 0.072
    for lab, done in states:
        col = GOOD if done else FAINT
        ax.add_patch(Circle((xr + 0.032, ys), 0.0062, facecolor=col,
                            edgecolor="none", zorder=5))
        txt(ax, xr + 0.050, ys, lab, size=10, color=INK if done else MUTED,
            family=SERIF)
        ys -= 0.038
    return save(fig, 22, "arcssm_motivation")


# ------------------------------------------------------------------ 23
def s23():
    fig, ax, top = slide(
        eb="Piste ArcSSM  ·  résultats", n=23,
        title="ArcSSM sur le même protocole exigeant",
        lede="Mêmes campagnes, même découpage, même mesure. "
             "Pour l'instant, Arc-FaultNet reste devant.")

    w = MR - ML
    rows = []
    for i, (camp, n, acc, f1, prec, rec, spec) in enumerate(D.SSM_CV):
        v2_acc = D.V2_CV[i][2]
        rows.append([camp, fr(acc, 2, "%"), fr(f1, 2, "%"), fr(rec, 2, "%"),
                     fr(spec, 2, "%"), fr(v2_acc, 2, "%"),
                     f"{'+' if acc > v2_acc else '−'}"
                     f"{fr(abs(acc - v2_acc), 2)} pt"])

    def fmt(i, j, v):
        if j == 6:
            return {"color": GOOD if v.startswith("+") else NOISE,
                    "weight": "bold"}
        if j == 5:
            return {"color": MUTED}
        if j == 1:
            return {"color": INK, "weight": "bold"}
        return None

    widths = [w * 0.17, w * 0.13, w * 0.12, w * 0.12, w * 0.14, w * 0.16,
              w * 0.16]
    bot = table(ax, ["Campagne testée", "Exactitude", "F1", "Rappel",
                     "Spécificité", "Arc-FaultNet", "Écart"],
                rows, ML, top - 0.020, widths, row_h=0.054, head_h=0.046,
                fs=10.3, cell_fmt=fmt)

    sy = bot - 0.020 - 0.084
    card(ax, ML, sy, w, 0.084, fc=GREY_TINT, ec=LINE)
    txt(ax, ML + 0.024, sy + 0.042, "Cumul sur\nles 4 plis", size=9.2,
        weight="bold", color=TEAL_DEEP, ls=1.4)
    parts = [("Exactitude", fr(D.SSM_CV_POOLED["acc"], 2, "%")),
             ("F1", fr(D.SSM_CV_POOLED["f1"], 2, "%")),
             ("Arcs détectés", fr(D.SSM_CV_POOLED["rec"], 2, "%")),
             ("Arc-FaultNet", fr(D.V2_CV_POOLED["acc"], 2, "%")),
             ("Écart", f"−{fr(D.V2_CV_POOLED['acc'] - D.SSM_CV_POOLED['acc'], 2)} pt")]
    xx = ML + 0.135
    step = (w - 0.160) / len(parts)
    for k, (lab, val) in enumerate(parts):
        txt(ax, xx + step / 2, sy + 0.055, val, size=13, weight="bold",
            color=NOISE if k == 4 else INK, ha="center")
        txt(ax, xx + step / 2, sy + 0.024, lab, size=8.6, color=FAINT,
            ha="center")
        xx += step

    cy = sy - 0.020
    wl = w * 0.485
    h = cy - 0.078
    for i, (lab, body, col, tint) in enumerate((
        ("Notre lecture",
         "ArcSSM détecte beaucoup d'arcs (86,4 % contre 75,4 %) mais alarme "
         "trop souvent. Il souffre du même défaut de réglage "
         "qu'Arc-FaultNet, en plus marqué. Ce n'est pas une impasse : c'est "
         "le même problème, à traiter de la même manière.",
         MUTED, GREY_TINT),
        ("Pourquoi nous la gardons ouverte",
         "Les deux modèles se trompent sur des cycles différents. À terme, "
         "les faire voter ensemble est la piste la plus prometteuse pour "
         "gagner sur une installation neuve, sans données supplémentaires.",
         TEAL, TEAL_TINT),
    )):
        x = ML + i * (wl + 0.030)
        lband(ax, x, 0.078, wl, h, col, tint)
        txt(ax, x + 0.024, 0.078 + h - 0.032, sp(lab.upper()), size=8.4,
            color=col if col != MUTED else MUTED, weight="bold")
        txt(ax, x + 0.024, 0.078 + h - 0.054, wrap(body, 60), size=10.0,
            color=INK, va="top", family=SERIF, ls=1.55)
    return save(fig, 23, "arcssm_resultats")


# ------------------------------------------------------------------ 24
def s24():
    fig, ax, top = slide(
        eb="Prochaines étapes", n=24,
        title="Le plan pour combler l'écart",
        lede="Sept actions, classées par rapport entre gain attendu "
             "et effort. Quatre sont déjà engagées.")

    palette = {
        "done": (GOOD, "#E8F4EE", "engagé"),
        "next": (TEAL, TEAL_TINT, "prochain run"),
        "todo": (FAINT, GREY_TINT, "à tester"),
        "ask":  (ARC, ARC_TINT, "besoin de vous"),
    }

    w = (MR - ML - 0.024) / 2
    col_items = [D.NEXT_STEPS[:4], D.NEXT_STEPS[4:]]
    for ci, items in enumerate(col_items):
        x = ML + ci * (w + 0.024)
        y = top - 0.014
        for code, title, body, status_lab, kind in items:
            col, tint, _ = palette[kind]
            h = 0.112
            card(ax, x, y - h, w, h)
            ax.add_patch(plt.Rectangle((x, y - h), 0.005, h, facecolor=col,
                                       zorder=3, edgecolor="none"))
            txt(ax, x + 0.026, y - h / 2, code, size=17, weight="bold",
                color="#C3CDD4")
            txt(ax, x + 0.062, y - 0.032, title, size=11.2, weight="bold",
                color=INK)
            txt(ax, x + 0.062, y - 0.054, wrap(body, 56), size=9.4,
                color=MUTED, va="top", family=SERIF, ls=1.45)
            pill(ax, x + w - 0.132, y - 0.030, status_lab, fc=tint, tc=col,
                 size=7.8, h=0.028)
            y -= h + 0.006

    callout(ax, ML, 0.078, MR - ML,
            "L'action G est la seule que nous ne pouvons pas mener seuls",
            "Trois de nos quatre campagnes viennent du même banc IJL. Deux ou "
            "trois campagnes sur des installations réellement différentes "
            "changeraient tout.", kind="warn", h=0.092, fs=10.0)
    return save(fig, 24, "prochaines_etapes")


# ------------------------------------------------------------------ 25
def s25():
    fig, ax, _ = slide(dark=True, n=25)
    eyebrow(ax, ML, 0.885, "À retenir", color=TEAL)
    txt(ax, ML, 0.815, "Cinq phrases", size=32, weight="bold",
        color="#FFFFFF", va="top")

    points = [
        "La fusion des deux branches se fait maintenant **avant** de résumer "
        "le signal : le modèle peut relier un événement temporel et un "
        "événement fréquentiel au même instant.",
        f"Ce changement apporte **+{fr(D.GAIN['acc'])} point d'exactitude** et "
        f"divise les fausses alarmes par trois, tout en **retirant "
        f"{frint(D.PARAMS_DELTA)} paramètres** au modèle.",
        f"Sur une campagne d'essais jamais vue, le modèle décide correctement "
        f"dans **{fr(D.V2_CV_MEAN['acc'], 1, '%')}** des cas, avec un "
        f"**AUC de {fr(D.V2_CV_MEAN['auc'], 3)}** : il sait reconnaître un arc "
        f"partout.",
        f"L'écart restant vient pour moitié du **réglage du seuil**, pas de la "
        f"reconnaissance : un réglage à la mise en service ramène "
        f"{fr(D.DIAG['pooled_fixe'], 1, '%')} à "
        f"{fr(D.DIAG['pooled_seuil_local'], 1, '%')}.",
        "L'autre moitié demande des **données de bancs différents**. C'est le "
        "point sur lequel nous avons besoin de vous avant la fin du projet.",
    ]

    y = 0.720
    for i, p in enumerate(points):
        card(ax, ML, y - 0.096, MR - ML, 0.096, fc="#1C262E", ec="#2A3742",
             r=0.008)
        ax.add_patch(Circle((ML + 0.032, y - 0.048), 0.015,
                            facecolor="#0E3A40", edgecolor=TEAL, lw=1.2,
                            zorder=5))
        txt(ax, ML + 0.032, y - 0.048, str(i + 1), size=10.5, weight="bold",
            color="#7FD4DB", ha="center", z=6)
        # gras simulé : on découpe sur les marqueurs **
        segs = p.split("**")
        line = "".join(segs)
        txt(ax, ML + 0.062, y - 0.048, wrap(line, 112), size=10.5,
            color="#D5DEE5", family=SERIF, ls=1.5)
        y -= 0.096 + 0.010

    rule(ax, 0.166, ML, MR, color="#2A3742", lw=1.2)
    txt(ax, ML, 0.128, "Prochain point d'avancement", size=9.5,
        color="#5C6773", weight="bold")
    txt(ax, ML, 0.094, "après le run avec sélection sur campagne inédite "
        "(action B) et la calibration à la mise en service (action E).",
        size=10, color="#8FA0AD", family=SERIF)
    txt(ax, MR, 0.111, "Salim Gouaied  ·  Institut Jean Lamour", size=10,
        color="#5C6773", ha="right", family=SERIF)
    return save(fig, 25, "a_retenir")
