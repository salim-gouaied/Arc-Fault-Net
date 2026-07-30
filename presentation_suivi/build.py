#!/usr/bin/env python3
"""
Génère les 25 slides PNG (1920 × 1080) de la présentation de suivi.

    ../venv/bin/python build.py

Sortie : presentation_suivi/slides/slide_NN_*.png
Tous les chiffres viennent de data.py ; la charte est dans style.py.
"""

import shutil
from pathlib import Path

import s01_05
import s06_10
import s11_15
import s16_20
import s21_25
from style import OUT

SLIDES = [
    s01_05.s01, s01_05.s02, s01_05.s03, s01_05.s04, s01_05.s05,
    s06_10.s06, s06_10.s07, s06_10.s08, s06_10.s09, s06_10.s10,
    s11_15.s11, s11_15.s12, s11_15.s13, s11_15.s14, s11_15.s15,
    s16_20.s16, s16_20.s17, s16_20.s18, s16_20.s19, s16_20.s20,
    s21_25.s21, s21_25.s22, s21_25.s23, s21_25.s24, s21_25.s25,
]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Génération de {len(SLIDES)} slides -> {OUT}/")
    for fn in SLIDES:
        fn()
    pngs = sorted(OUT.glob("slide_*.png"))
    print(f"\n{len(pngs)} slides générées.")
    assert len(pngs) == 25, f"attendu 25 slides, obtenu {len(pngs)}"


if __name__ == "__main__":
    main()
