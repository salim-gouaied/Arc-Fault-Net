#!/usr/bin/env python3
"""
Charte graphique + helpers de mise en page pour les slides de suivi.
Reprend la palette et la typographie de artifacts/arcssm_explained.html.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import textwrap
from pathlib import Path

# ---------------------------------------------------------------- palette
GROUND = "#F5F7F9"
SURFACE = "#FFFFFF"
INK = "#141A21"
MUTED = "#586472"
FAINT = "#8A93A0"
LINE = "#DCE2E8"
TEAL = "#0E7C86"
TEAL_DEEP = "#0A5B63"
TEAL_TINT = "#E7F2F3"
ARC = "#C9711F"
ARC_TINT = "#FBEEDD"
NOISE = "#AA4436"
GREY_TINT = "#F3F5F7"
GOOD = "#1B7F5C"

SANS = "DejaVu Sans"
SERIF = "DejaVu Serif"

W, H = 12.8, 7.2          # pouces -> 1920x1080 @ dpi 150
DPI = 150
ASPECT = W / H            # pour l'arrondi des coins

ML, MR = 0.055, 0.945     # marges gauche / droite
OUT = Path(__file__).resolve().parent / "slides"
OUT.mkdir(parents=True, exist_ok=True)

FOOTER = "Arc-FaultNet  ·  Point d'avancement  ·  Juillet 2026"

plt.rcParams.update({
    "font.family": SANS,
    "savefig.facecolor": GROUND,
    "figure.facecolor": GROUND,
})


def sp(text, gap=" "):
    """Interlettrage pour les sur-titres."""
    return gap.join(text)


def wrap(text, n):
    return "\n".join(textwrap.wrap(text, n))


def fr(v, d=1, unit=""):
    """Nombre au format français : virgule décimale, espace fine des milliers."""
    s = f"{v:,.{d}f}".replace(",", " ").replace(".", ",")
    return s + (f" {unit}" if unit else "")


def frint(v):
    return f"{int(round(v)):,}".replace(",", " ")


# ---------------------------------------------------------------- primitives
def card(ax, x, y, w, h, fc=SURFACE, ec=LINE, lw=1.0, r=0.010, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
        mutation_aspect=ASPECT, zorder=z, alpha=alpha))


def lband(ax, x, y, w, h, color, fc, lw_band=0.006, z=2):
    """Carte avec liseré vertical à gauche (callout)."""
    card(ax, x, y, w, h, fc=fc, ec=fc, r=0.008, z=z)
    ax.add_patch(FancyBboxPatch(
        (x, y), lw_band, h,
        boxstyle="round,pad=0,rounding_size=0.002",
        linewidth=0, facecolor=color,
        mutation_aspect=ASPECT, zorder=z + 1))


def rule(ax, y, x0=ML, x1=MR, color=LINE, lw=1.0, z=3):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=z,
            solid_capstyle="butt")


def arrow(ax, p0, p1, color=TEAL, lw=1.6, style="-|>", ms=9, z=5, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=ms, linewidth=lw,
        color=color, zorder=z, shrinkA=0, shrinkB=0,
        connectionstyle=f"arc3,rad={rad}"))


def txt(ax, x, y, s, size=11, color=INK, weight="normal", ha="left",
        va="center", family=SANS, style="normal", ls=1.45, z=6, alpha=1.0,
        rotation=0):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, fontfamily=family, fontstyle=style,
                   linespacing=ls, zorder=z, alpha=alpha, rotation=rotation,
                   rotation_mode="anchor")


def eyebrow(ax, x, y, s, color=TEAL_DEEP, size=8.5):
    txt(ax, x, y, sp(s.upper()), size=size, color=color, weight="bold")


def pill(ax, x, y, s, fc=TEAL_TINT, tc=TEAL_DEEP, size=8.5, pad=0.011,
         h=0.038, weight="bold"):
    """Étiquette arrondie ; renvoie la largeur occupée."""
    w = 0.0062 * len(s) * (size / 8.5) + 2 * pad
    card(ax, x, y - h / 2, w, h, fc=fc, ec=fc, r=h / 2, z=4)
    txt(ax, x + w / 2, y, s, size=size, color=tc, weight=weight,
        ha="center", z=5)
    return w


# ---------------------------------------------------------------- squelette
def slide(eb=None, title=None, lede=None, n=None, dark=False,
          title_size=25, rule_y=None):
    """Crée une slide et renvoie (fig, ax, y_top) où y_top = haut du contenu."""
    bg = INK if dark else GROUND
    fig = plt.figure(figsize=(W, H), facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=bg, zorder=0))

    y = 0.915
    if eb:
        eyebrow(ax, ML, y, eb, color=TEAL if dark else TEAL_DEEP)
        y -= 0.062
    if title:
        txt(ax, ML, y, title, size=title_size, weight="bold",
            color="#FFFFFF" if dark else INK, va="top", ls=1.14)
        y -= 0.075 * (title.count("\n") + 1)
    if lede:
        txt(ax, ML, y - 0.012, lede, size=12.5,
            color="#B7C2CC" if dark else MUTED, family=SERIF, va="top",
            ls=1.5)
        y -= 0.048 * (lede.count("\n") + 1) + 0.016

    top = rule_y if rule_y is not None else y - 0.022
    if not dark:
        rule(ax, top)

    if n is not None:
        txt(ax, ML, 0.046, FOOTER, size=8,
            color=FAINT if not dark else "#5C6773")
        txt(ax, MR, 0.046, f"{n:02d} / 25", size=8.5,
            color=FAINT if not dark else "#5C6773", weight="bold", ha="right")
    return fig, ax, top - 0.030


def save(fig, n, name):
    path = OUT / f"slide_{n:02d}_{name}.png"
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {path.name}")
    return path


# ---------------------------------------------------------------- composants
def kpi(ax, x, y, w, value, label, sub=None, h=0.19, vcolor=TEAL_DEEP,
        vsize=30, accent=None):
    card(ax, x, y, w, h)
    if accent:
        ax.add_patch(FancyBboxPatch(
            (x, y + h - 0.006), w, 0.006,
            boxstyle="round,pad=0,rounding_size=0.002",
            linewidth=0, facecolor=accent, mutation_aspect=ASPECT, zorder=3))
    cx = x + w / 2
    txt(ax, cx, y + h * 0.60, value, size=vsize, weight="bold", color=vcolor,
        ha="center")
    txt(ax, cx, y + h * 0.30, label, size=10, color=INK, ha="center",
        weight="bold")
    if sub:
        txt(ax, cx, y + h * 0.13, sub, size=8.8, color=FAINT, ha="center",
            family=SERIF)


def table(ax, headers, rows, x, y_top, widths, row_h=0.058, head_h=0.050,
          cell_fmt=None, align=None, fs=10.5, hfs=8.2, zebra=True,
          highlight_row=None):
    """
    Tableau propre. widths = fractions relatives (somme normalisée à la largeur).
    cell_fmt(i_row, j_col, value) -> dict(color=, weight=) optionnel.
    align = liste 'l'/'c'/'r' par colonne.
    """
    total_w = MR - ML if x is None else None
    if x is None:
        x = ML
    tw = sum(widths)
    xs, acc = [], x
    for w in widths:
        xs.append(acc)
        acc += w
    n = len(rows)
    body_top = y_top - head_h
    bottom = body_top - n * row_h
    align = align or (["l"] + ["c"] * (len(headers) - 1))

    # en-tête
    card(ax, x, body_top, tw, head_h, fc=GREY_TINT, ec=LINE, r=0.006, z=2)
    for j, hd in enumerate(headers):
        a = align[j]
        cx = xs[j] + 0.012 if a == "l" else (
            xs[j] + widths[j] / 2 if a == "c" else xs[j] + widths[j] - 0.012)
        txt(ax, cx, body_top + head_h / 2, sp(hd.upper()), size=hfs,
            color=FAINT, weight="bold",
            ha={"l": "left", "c": "center", "r": "right"}[a], z=5)

    card(ax, x, bottom, tw, n * row_h, fc=SURFACE, ec=LINE, r=0.006, z=2)
    for i, row in enumerate(rows):
        ry = body_top - (i + 1) * row_h
        if zebra and i % 2 == 1:
            ax.add_patch(plt.Rectangle((x, ry), tw, row_h,
                                       facecolor="#FAFBFC", zorder=3,
                                       edgecolor="none"))
        if highlight_row is not None and i == highlight_row:
            ax.add_patch(plt.Rectangle((x, ry), tw, row_h,
                                       facecolor=TEAL_TINT, zorder=3,
                                       edgecolor="none"))
        if i:
            ax.plot([x, x + tw], [ry + row_h, ry + row_h], color=LINE,
                    lw=0.7, zorder=4)
        for j, val in enumerate(row):
            a = align[j]
            cx = xs[j] + 0.012 if a == "l" else (
                xs[j] + widths[j] / 2 if a == "c"
                else xs[j] + widths[j] - 0.012)
            st = {"color": INK, "weight": "normal"}
            if j == 0:
                st = {"color": TEAL_DEEP, "weight": "bold"}
            if cell_fmt:
                st.update(cell_fmt(i, j, val) or {})
            txt(ax, cx, ry + row_h / 2, str(val), size=fs, z=5,
                ha={"l": "left", "c": "center", "r": "right"}[a], **st)
    return bottom


def callout(ax, x, y, w, label, body, kind="intuition", fs=10.5, lfs=8.0,
            pad=0.018, h=None, lines=None):
    palette = {
        "intuition": (TEAL, TEAL_TINT, TEAL_DEEP),
        "engi": (MUTED, GREY_TINT, MUTED),
        "warn": (ARC, ARC_TINT, ARC),
        "good": (GOOD, "#E8F4EE", GOOD),
    }[kind]
    nl = lines if lines is not None else body.count("\n") + 1
    hh = h if h is not None else 0.052 + 0.040 * nl
    lband(ax, x, y, w, hh, palette[0], palette[1])
    txt(ax, x + pad, y + hh - 0.030, sp(label.upper()), size=lfs,
        color=palette[2], weight="bold")
    txt(ax, x + pad, y + hh - 0.052, body, size=fs, color=INK, va="top",
        family=SERIF, ls=1.5)
    return hh


def box_node(ax, x, y, w, h, label, sub=None, fc=SURFACE, ec=LINE,
             tc=INK, fs=10, sfs=8.2, weight="bold", r=0.008, lw=1.1):
    card(ax, x, y, w, h, fc=fc, ec=ec, r=r, lw=lw, z=4)
    if sub:
        nl = sub.count("\n") + 1
        line = 0.0028 * sfs * 1.55
        blk = 0.0032 * fs + nl * line          # hauteur titre + sous-titre
        ty = y + h / 2 + blk / 2
        txt(ax, x + w / 2, ty, label, size=fs, color=tc,
            weight=weight, ha="center", va="top", z=6)
        txt(ax, x + w / 2, ty - 0.0042 * fs - 0.008, sub, size=sfs, color=tc,
            ha="center", va="top", z=6, alpha=0.75, family=SERIF, ls=1.55)
    else:
        txt(ax, x + w / 2, y + h / 2, label, size=fs, color=tc, weight=weight,
            ha="center", z=6)


def op_node(ax, cx, cy, label, rr=0.019, fc=TEAL, tc="#FFFFFF", fs=11):
    ax.add_patch(Circle((cx, cy), rr, facecolor=fc, edgecolor=fc,
                        zorder=5, transform=ax.transData))
    txt(ax, cx, cy, label, size=fs, color=tc, weight="bold", ha="center", z=6)


def section_divider(fig, ax, num, kicker, title, bullets, n):
    """Slide de séparation (fond encre)."""
    txt(ax, ML, 0.72, num, size=64, weight="bold", color=TEAL, va="center")
    txt(ax, ML, 0.575, kicker, size=10, color="#7FD4DB", weight="bold")
    txt(ax, ML, 0.50, title, size=34, weight="bold", color="#FFFFFF",
        va="top", ls=1.16)
    y = 0.30
    for b in bullets:
        ax.add_patch(Circle((ML + 0.006, y), 0.0055, facecolor=TEAL,
                            edgecolor="none", zorder=5))
        txt(ax, ML + 0.026, y, b, size=12.5, color="#C6D1DA", family=SERIF)
        y -= 0.062
    return save(fig, n, f"partie{num}")
