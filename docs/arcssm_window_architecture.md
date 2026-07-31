# ArcSSM « fenêtre » (multi-cycles) — architecture & logique, en simple

> ⚠️ **RÉSULTAT : cette architecture a été TESTÉE puis ÉCARTÉE.** Elle atteint
> 80.4 % / 73.6 % de spécificité, moins bien que simplement moyenner les scores de
> B1 sur les mêmes 2 cycles (84.1 % / 83.9 %). Sa branche écart-type se déclenche sur
> *tout* changement, donc une variation normale de charge crée des faux positifs. Le
> code (`train_window.py`) a été retiré au nettoyage (récupérable dans git).
> **La méthode retenue est décrite dans [`multicycle_decision_howto.md`](multicycle_decision_howto.md).**
>
> Ce document reste utile pour comprendre **pourquoi** l'idée multi-cycles est bonne
> (§2, §6 : la mesure de répétitivité est réelle et invariante à l'installation) — c'est elle
> qui a mené à la solution retenue, mais appliquée au niveau *décision* et non
> *features*.

---

## 1. Le problème de départ (en une phrase)

Le modèle B1 regarde **un seul cycle** et le juge par sa **forme**. Or la forme d'un
cycle *normal* dépend de l'installation (type de charge, bruit, harmoniques). Sur une installation
**jamais vue**, un cycle normal a une forme *inhabituelle* → le modèle crie « arc »
à tort → **faux positifs**, donc **spécificité basse**.

## 2. L'idée nouvelle (en une phrase)

Au lieu de « à quoi ressemble ce cycle ? » (dépend de l'installation), on pose une question
qui marche sur **toutes les installations** :

> **« Est-ce que ça se répète d'un cycle à l'autre ? »**

- Un appareil **normal** répète ses cycles **presque à l'identique** (courant périodique).
- Un **arc** est **erratique** : il change d'un cycle à l'autre (ré-allumages aléatoires,
  bruit haute-fréquence qui n'est jamais deux fois pareil).

« Se répète ou pas » est **la même question quelle que soit l'installation** → c'est un signal
**invariant à l'installation**, exactement ce qui manquait.

```mermaid
flowchart LR
    subgraph N["NORMAL — se répète"]
      n1["cycle"]:::g --> n2["cycle"]:::g --> n3["cycle"]:::g
    end
    N --> Nout["variation FAIBLE → normal"]:::ok
    subgraph A["ARC — erratique"]
      a1["cycle"]:::r --> a2["cycle'"]:::r --> a3["cycle''"]:::r
    end
    A --> Aout["variation FORTE → arc"]:::bad
    classDef g fill:#e6f4ea,stroke:#34a853;
    classDef r fill:#fde8e8,stroke:#ea4335;
    classDef ok fill:#d7f0db,stroke:#137333,color:#0b5;
    classDef bad fill:#fbd4d4,stroke:#a50e0e,color:#a00;
```

## 3. Comment le modèle le fait (étapes simples)

1. On prend une **fenêtre de K cycles consécutifs** (on commence à **K = 2**).
2. Chaque cycle passe dans **le même extracteur S4D que B1** — **inchangé**, on garde
   ce qui marche. Il transforme un cycle en un **« résumé »** de 128 nombres (un
   *embedding*).
3. On combine les K résumés de **deux** façons :
   - **la MOYENNE** des résumés → *« de quel type de signal s'agit-il ? »* (la partie
     « est-ce un contenu type-arc », comme B1) ;
   - **l'ÉCART-TYPE** des résumés → *« à quel point ça change d'un cycle à l'autre ? »*
     (**la répétitivité** — le nouveau signal invariant à l'installation).
4. Un petit **classifieur** décide **arc / normal** à partir de ces deux informations.

```mermaid
flowchart TD
    IN["Fenêtre = K cycles consécutifs<br/>(chaque cycle = 4 canaux : I, |dI|, TKEO, RMS_slide)"]
    IN --> C1["cycle 1"]
    IN --> C2["cycle 2 … (K)"]
    C1 --> ENC["Encodeur S4D<br/>(partagé, identique à B1)"]
    C2 --> ENC
    ENC --> E1["résumé cycle 1<br/>(128 nombres)"]
    ENC --> E2["résumé cycle 2<br/>(128 nombres)"]
    E1 --> MEAN["MOYENNE sur les cycles (128)<br/>« type arc ? »"]
    E2 --> MEAN
    E1 --> STD["ÉCART-TYPE sur les cycles (128)<br/>« ça change ? »"]
    E2 --> STD
    MEAN --> CLF["classifieur<br/>(256 → 64 → 1)"]
    STD --> CLF
    CLF --> OUT(["Arc / Normal (pour la fenêtre)"])
```

## 4. Pourquoi garder les DEUX (moyenne **et** écart-type)

L'écart-type détecte **le changement**. Mais **tout changement n'est pas un arc** :
allumer un moteur ou une bouilloire change aussi le courant d'un cycle à l'autre.
Donc :

- l'**écart-type** dit *« ça change »*,
- la **moyenne** dit *« est-ce un changement de type arc »*.

**Ensemble**, ils séparent « arc » de « simple changement de charge ». Un détecteur de
changement *tout seul* déclencherait à tort sur les enclenchements de charge.

## 5. Ce que ça NE change PAS (important)

- L'**extracteur S4D** — le cœur qui marche déjà (AUC 0.9 intra-campagne) — est
  **identique à B1**. On ne casse rien.
- On ajoute seulement une **petite tête** (moyenne+écart-type puis classifieur) :
  **+8 000 paramètres** (367 745 au total vs 359 553 pour B1). Vitesse ~identique.
- La **mémoire du SSM** n'est pas un souci : il traite **un cycle** (ce qu'il fait déjà
  parfaitement) ; la comparaison entre cycles est calculée **explicitement** (l'écart-type),
  pas confiée à la mémoire du SSM.

## 6. Pourquoi on pense que ça peut marcher (c'est mesuré)

On a mesuré, sur le contenu haute-fréquence (`|dI|`), à quel point deux cycles
**consécutifs** se ressemblent, pour le normal vs l'arc, sur chaque campagne
(`ndiff` = à quel point ça change ; plus grand = moins répétitif) :

| Campagne | normal→normal | arc→arc | **Δ (arc − normal)** |
|---|---|---|---|
| 15_juillet | 0.921 | 1.228 | **+0.31** |
| 22_juillet | 0.910 | 1.230 | **+0.32** |
| 8_juillet | 0.924 | 1.112 | **+0.19** |
| 2026 | 0.681 | 1.181 | **+0.50** |

→ **Sur les 4 campagnes**, l'arc est **moins répétitif** que le normal (Δ toujours
positif, y compris sur les folds difficiles). C'est **exactement** le signal que
l'écart-type de l'architecture va capter, et il est **présent partout** (invariant à l'installation).

## 7. B1 (avant) vs Fenêtre (nouveau)

```mermaid
flowchart LR
    subgraph B1["B1 — avant"]
      direction LR
      x1["1 cycle"] --> s1["S4D"] --> d1(["décision"])
    end
    subgraph WIN["Fenêtre — nouveau"]
      direction LR
      x2["K cycles"] --> s2["S4D par cycle<br/>(même qu'avant)"] --> cmp["compare les cycles<br/>moyenne + écart-type"] --> d2(["décision"])
    end
```

| | B1 (par cycle) | Fenêtre (K cycles) |
|---|---|---|
| Entrée | 1 cycle | K cycles consécutifs |
| Question posée | « quelle forme ? » (dépend de l'installation) | « ça se répète ? » (invariant installation) + « type arc ? » |
| Unité de décision | 1 cycle | 1 fenêtre (≈ IEC 62606, plusieurs demi-cycles) |
| Extracteur S4D | — | **identique** |
| Paramètres | 359 553 | 367 745 (+8 k) |

## 8. Honnêteté (attentes)

- Le signal de non-répétitivité est **modéré** (Δ ≈ 0.2–0.3) → viser une **amélioration
  réelle de la spécificité** sur une installation inédite, **pas** un bond à 98 %.
- L'**unité de décision devient la fenêtre** (pas le cycle) — plus proche d'un vrai
  AFDD, mais les chiffres ne se comparent pas *au cycle près* avec B1.
- Le **plafond reste borné par les données** (3 campagnes/4 = même montage IJL). Cette
  architecture attaque la *bonne* cause (le style propre à l'installation), mais elle ne
  fabrique pas de la diversité des installations.

## 9. Protocole d'évaluation

Leave-one-campaign-out **sur les fenêtres** (une fenêtre ne traverse jamais deux
enregistrements ; train/test séparés par campagne → pas de fuite). On lance **K=2**
d'abord (l'événement d'arc est instantané → 2 cycles suffisent), puis **K=4** pour
voir si une estimation de répétitivité plus lissée aide. Comparaison à B1 sur
accuracy / **spécificité** par campagne.
