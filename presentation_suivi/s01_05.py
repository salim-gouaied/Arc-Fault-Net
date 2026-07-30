#!/usr/bin/env python3
"""Slides 1 à 5 — ouverture, sommaire, résumé, séparateur Partie 1, rappel."""

from matplotlib.patches import Circle
import numpy as np
from style import *          # noqa: F401,F403
import data as D


# ------------------------------------------------------------------ 01
def s01():
    fig, ax, _ = slide(dark=True)
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=INK, zorder=0))

    # filigrane : un cycle de courant avec signature d'arc
    t = np.linspace(0, 2 * np.pi, 1400)
    base = np.sin(t)
    rng = np.random.default_rng(7)
    arcy = base.copy()
    m = (t > 2.55) & (t < 4.05)
    arcy[m] = base[m] * 0.45 + 0.22 * rng.standard_normal(m.sum()) * \
        np.hanning(m.sum())
    ax.plot(0.055 + t / (2 * np.pi) * 0.89, 0.30 + base * 0.115,
            color="#20343B", lw=2.4, zorder=1)
    ax.plot(0.055 + t / (2 * np.pi) * 0.89, 0.30 + arcy * 0.115,
            color="#2A3E33", lw=1.6, zorder=1, alpha=0.9)

    eyebrow(ax, ML, 0.855, "Détection de défaut d'arc série  ·  IJL", color=TEAL)
    txt(ax, ML, 0.775, "Arc-FaultNet", size=54, weight="bold",
        color="#FFFFFF", va="top", ls=1.05)
    txt(ax, ML, 0.635, "Point d'avancement", size=32, weight="bold",
        color="#7FD4DB", va="top")

    rule(ax, 0.575, ML, 0.42, color="#2C3A45", lw=1.4)
    txt(ax, ML, 0.520,
        "Deux avancées depuis la dernière version : une fusion plus efficace\n"
        "et plus légère, et une mesure honnête de la robustesse hors banc.",
        size=14, color="#B7C2CC", family=SERIF, va="top", ls=1.6)

    x = ML
    for lab in ("Juillet 2026", "Version 3", "Confidentiel"):
        x += pill(ax, x, 0.405, lab, fc="#1E2A32", tc="#8FB3B8", size=9.5) + 0.010

    txt(ax, ML, 0.115, sp("PRÉSENTÉ PAR"), size=8, color="#5C6773",
        weight="bold")
    txt(ax, ML, 0.075, "Salim Gouaied", size=13, weight="bold", color="#E6ECF1")
    txt(ax, MR, 0.075, "Institut Jean Lamour", size=11, color="#7E8B96",
        ha="right", family=SERIF)
    return save(fig, 1, "titre")


# ------------------------------------------------------------------ 02
def s02():
    fig, ax, top = slide(
        eb="Sommaire", n=2,
        title="Ce que couvre ce document",
        lede="Deux parties indépendantes. La première est un résultat acquis,\n"
             "la seconde un chantier en cours avec un plan chiffré.")

    parts = [
        ("01", "Une fusion repensée", TEAL, "Slides 04 → 15",
         ["Ce qui limitait la version de juin",
          "Le nouveau mécanisme, sans formalisme",
          f"{frint(D.PARAMS_DELTA)} paramètres en moins",
          "Radar, tableaux et matrices de confusion"]),
        ("02", "Robustesse et piste ArcSSM", ARC, "Slides 16 → 25",
         ["Le test qui compte : un banc jamais vu",
          "Résultats campagne par campagne",
          "ArcSSM : un modèle à mémoire, sans jargon",
          "Les prochaines étapes et nos besoins"]),
    ]

    w = 0.425
    for k, (num, title, col, rng, items) in enumerate(parts):
        x = ML + k * (w + 0.035)
        h = 0.398
        y = top - h - 0.010
        card(ax, x, y, w, h)
        ax.add_patch(plt.Rectangle((x, y + h - 0.007), w, 0.007,
                                   facecolor=col, zorder=3, edgecolor="none"))
        txt(ax, x + 0.028, y + h - 0.060, num, size=25, weight="bold",
            color=col)
        txt(ax, x + w - 0.028, y + h - 0.058, sp(rng.upper()), size=7.6,
            color=FAINT, weight="bold", ha="right")
        txt(ax, x + 0.028, y + h - 0.118, title, size=15.5, weight="bold",
            color=INK, va="top")
        yy = y + h - 0.185
        for it in items:
            ax.add_patch(Circle((x + 0.034, yy + 0.005), 0.0042,
                                facecolor=col, edgecolor="none", zorder=5))
            txt(ax, x + 0.052, yy, wrap(it, 42), size=10.5, color=MUTED,
                family=SERIF, va="top", ls=1.4)
            yy -= 0.050 + 0.028 * wrap(it, 42).count("\n")

    callout(ax, ML, 0.078, MR - ML, "En une phrase",
            "Le modèle est plus précis et plus léger qu'en juin ; le travail "
            "restant porte entièrement sur sa tenue\nface à une installation "
            "qu'il n'a jamais vue.", kind="intuition")
    return save(fig, 2, "sommaire")


# ------------------------------------------------------------------ 03
def s03():
    fig, ax, top = slide(
        eb="Résumé", n=3,
        title="Les quatre chiffres à retenir",
        lede="1 630 cycles de test pour les trois premiers ; "
             "10 860 cycles et 4 campagnes pour le dernier.")

    y = top - 0.190
    w = (MR - ML - 3 * 0.020) / 4
    cards = [
        (fr(D.V2_BEST['acc'], 1, "%"), "Exactitude, meilleur modèle",
         f"{fr(D.V2_MEAN['acc'], 1, '%')} en moyenne sur 4 runs",
         TEAL_DEEP, TEAL),
        (f"{D.V2_BEST['fp']}", "Fausses alarmes",
         f"sur {D.V2_BEST['tn'] + D.V2_BEST['fp']} cycles sains", GOOD, GOOD),
        (f"−{frint(D.PARAMS_DELTA)}", "Paramètres économisés",
         f"{frint(D.PARAMS['gated']['total'])} → "
         f"{frint(D.PARAMS['sequential']['total'])}", TEAL_DEEP, TEAL),
        (f"{D.V2_CV_MEAN['acc']:.0f} %", "Sur un banc jamais vu",
         f"moyenne des 4 campagnes, AUC {fr(D.V2_CV_MEAN['auc'], 2)}",
         ARC, ARC),
    ]
    for i, (v, lab, sub, vc, acc) in enumerate(cards):
        kpi(ax, ML + i * (w + 0.020), y, w, v, lab, sub, h=0.190,
            vcolor=vc, vsize=29, accent=acc)

    rule(ax, y - 0.040)
    txt(ax, ML, y - 0.085, "Ce qui a changé", size=13, weight="bold",
        color=INK)

    items = [
        ("Précision", "La fusion des deux branches se fait maintenant avant "
         "de résumer le signal. Le modèle peut relier un événement temporel "
         "à un événement fréquentiel au même instant.",
         f"+{fr(D.GAIN['acc'])} pt d'exactitude, "
         f"{fr(D.GAIN_FP)} fausses alarmes en moins"),
        ("Coût", "Les grandes couches denses de l'ancienne fusion sont "
         "remplacées par des projections convolutives partagées.",
         f"−{D.PARAMS_DELTA_PCT:.0f} % de paramètres, "
         f"bloc de fusion allégé de {D.FUSION_DELTA_PCT:.0f} %"),
        ("Robustesse", "Nous mesurons désormais la performance sur une "
         "campagne d'essais entièrement retirée de l'entraînement.",
         f"{D.V2_CV_POOLED['acc']:.0f} % hors banc, "
         f"écart identifié et chiffré"),
    ]
    yy = y - 0.125
    for lab, body, gain in items:
        h = 0.098
        card(ax, ML, yy - h, MR - ML, h)
        txt(ax, ML + 0.022, yy - 0.028, lab, size=11.5, weight="bold",
            color=TEAL_DEEP)
        txt(ax, ML + 0.145, yy - h / 2, wrap(body, 92), size=10.1,
            color=MUTED, family=SERIF, ls=1.45)
        txt(ax, MR - 0.022, yy - h / 2, wrap(gain, 26), size=9.8,
            color=INK, weight="bold", ha="right", ls=1.5)
        yy -= h + 0.012
    return save(fig, 3, "resume")


# ------------------------------------------------------------------ 04
def s04():
    fig, ax, _ = slide(dark=True, n=4)
    return section_divider(
        fig, ax, "01", "PARTIE 1  ·  ARCHITECTURE ET RÉSULTATS",
        "Une fusion repensée :\nplus précise, plus légère",
        ["Ce qui limitait la version de juin",
          "Le nouveau mécanisme, sans formalisme",
          f"{frint(D.PARAMS_DELTA)} paramètres économisés",
          "Les résultats, run par run"], 4)


# ------------------------------------------------------------------ 05
def s05():
    fig, ax, top = slide(
        eb="Point de départ", n=5,
        title="Où nous en étions à la dernière version",
        lede="Deux branches complémentaires, et un mécanisme de fusion "
             "« à porte ».")

    # ---- colonne gauche : l'architecture en quatre temps
    x0, wl = ML, 0.40
    txt(ax, x0, top - 0.018, "L'architecture, en quatre temps", size=12.5,
        weight="bold", color=INK)
    steps = [
        ("Le signal", "un cycle de 50 Hz, 2 048 points,\n4 descripteurs du courant", TEAL),
        ("Branche temporelle", "suit la forme d'onde : les déformations\nprès des passages par zéro", TEAL),
        ("Branche spectrale", "regarde le contenu fréquentiel :\nle bruit haute-fréquence de l'arc", TEAL),
        ("Fusion + décision", "combine les deux points de vue,\npuis sort une probabilité d'arc", ARC),
    ]
    yy = top - 0.062
    h = 0.112
    for i, (t_, s_, col) in enumerate(steps):
        card(ax, x0, yy - h, wl, h)
        ax.add_patch(plt.Rectangle((x0, yy - h), 0.0045, h, facecolor=col,
                                   zorder=3, edgecolor="none"))
        txt(ax, x0 + 0.024, yy - 0.030, t_, size=11, weight="bold", color=INK)
        txt(ax, x0 + 0.024, yy - 0.052, s_, size=9.3, color=MUTED,
            family=SERIF, va="top", ls=1.5)
        if i < 3:
            arrow(ax, (x0 + 0.020, yy - h - 0.002), (x0 + 0.020, yy - h - 0.012),
                  color=LINE, lw=1.3, ms=7)
        yy -= h + 0.014

    # ---- colonne droite : les chiffres annoncés
    x1 = ML + wl + 0.045
    wr = MR - x1
    txt(ax, x1, top - 0.018, "Ce que nous avions annoncé", size=12.5,
        weight="bold", color=INK)

    rows = [[n, fr(a, 2, "%"), str(fp), fr(f, 2, "%")]
            for n, a, fp, f, _, _ in D.V1_RUNS]
    bot = table(ax, ["Run", "Exactitude", "Fausses alarmes", "F1"], rows,
                x1, top - 0.062, [wr * 0.28, wr * 0.24, wr * 0.28, wr * 0.20],
                row_h=0.054, head_h=0.046, fs=10.5,
                cell_fmt=lambda i, j, v: (
                    {"color": NOISE, "weight": "bold"} if j == 2 and int(v) >= 10
                    else {"color": GOOD, "weight": "bold"} if j == 2 and int(v) <= 3
                    else None))

    sy = bot - 0.020 - 0.086

    card(ax, x1, sy, wr, 0.086, fc=GREY_TINT, ec=LINE)
    txt(ax, x1 + wr / 2, sy + 0.057,
        f"Moyenne :  {fr(D.V1_MEAN['acc'], 2, '%')}   ·   "
        f"F1 {fr(D.V1_MEAN['f1'], 2, '%')}   ·   "
        f"{fr(D.V1_MEAN['fp'], 2)} fausses alarmes",
        size=10.3, weight="bold", color=INK, ha="center")
    txt(ax, x1 + wr / 2, sy + 0.023,
        f"{frint(D.PARAMS['gated']['total'])} paramètres",
        size=10, color=MUTED, ha="center", family=SERIF)

    callout(ax, x1, 0.085, wr, "Le point qui nous a interpellés",
            "Un run sur quatre produisait 22 fausses alarmes au lieu\n"
            "de 3. Cette dispersion vient du mécanisme de fusion —\n"
            "c'est le point de départ de la suite.",
            kind="warn", fs=10.2)
    return save(fig, 5, "point_de_depart")
