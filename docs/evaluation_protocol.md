# Evaluation protocol — why leave-one-campaign-out, and what the random split was for

This note fixes the evaluation protocol for the report and records why each option
was chosen or dropped. It applies to every model in the repo (V1, V2, ArcSSM); the
commands use `arcssm` because that is the track being evaluated.

## 1. What the data actually is

`combined_dataset_2048/` holds **10 860 cycles** of 2048 points at 102.4 kHz
(one 20 ms mains cycle each), 5898 normal / 4962 arc, cut from **four acquisition
campaigns**:

| Campaign (`dataset` column) | When | Cycles | Arc | Recordings (`exp_name`) |
|---|---|---|---|---|
| `8_juillet_clean` | 8 July, IJL | 2746 | 1299 | 1 (`exp11--IJL--LR`) |
| `15_juillet_clean` | 15 July, IJL | 2820 | 1319 | 1 (`exp12--IJL--LR`) |
| `22_juillet_clean` | 22 July, IJL | 3820 | 1777 | 1 (`exp13--IJL--LR`) |
| `OthmaneSalim10052026` | 10 May 2026 | 1474 | 567 | 22 (load combinations) |

Two structural facts drive everything below:

1. **There is no per-experiment load information for the three IJL campaigns.**
   `charge_map.json` is `{"combined": 0}` — a single pseudo-charge. The 2026
   campaign *does* name its loads in `exp_name`
   (`AcierCu_Kettle+Halogene+AspiRouge`, `GraphAcier_Kettle+SilenceI+3lampes`, …),
   but since the other three campaigns carry nothing comparable, the whole 2026
   campaign is treated as **one group**, not as 22 load groups.
2. **Cycles are not independent samples.** Consecutive cycles are cut from the same
   *alternance* (one arc burst, or one normal window): 10 860 cycles come from only
   **1622 alternances**, and a single 2024 alternance can contribute up to **88
   cycles** of nearly identical signal. In the 2026 campaign each recording yields a
   single cycle, so there an alternance *is* a cycle.

This gives a leakage hierarchy, from weakest split to strongest:

```
cycle  <  alternance  <  recording  <  campaign
        (same burst)   (same setup)   (same day, bench, sensor, wiring)
```

## 2. Why leave-one-charge-out is not used

`train.py --mode cv` (`LeaveOneChargeOutSplitter`) was written when the working
assumption was that each experiment recorded a known electrical load, so that
holding out one load at a time would measure generalization to unseen loads. That
assumption does not hold for this dataset: the load was not stored per experiment
for the 2024 campaigns. With one charge in `charge_map.json` the splitter
degenerates to a single fold with an empty training set — it cannot be run at all,
which is a property of the data, not a bug in the code.

**Leave-one-campaign-out replaces it.** A campaign differs from the others in day,
bench wiring, electrode material, sensor placement and load mix at once, so
holding one out is a *stronger* shift than holding out one load inside a single
campaign. It measures what the report needs to claim: the detector still works on a
recording session it was never trained on.

## 3. The three protocols in the repo, and what each one is for

| Protocol | Command | What it measures | Status |
|---|---|---|---|
| Random 70/15/15 split at cycle level | `--mode single` | in-distribution fit | development tool, **ceiling number** |
| Recording-level `StratifiedGroupKFold` | `--mode groupkfold --group-level recording` | unseen recordings | legacy, folds are size-pathological here |
| **Leave-one-campaign-out** | `--mode groupkfold --group-level campaign` | **unseen acquisition campaign** | **headline number for the report** |

### 3.1 Why the random single split exists, and why it is not the headline

`--mode single` shuffles the 10 860 cycles and takes 70 % / 15 % / 15 %
(7601 / 1629 / 1630). Cycles of the *same alternance* therefore land in train and
test at once, and the test set contains cycles from every campaign the model
trained on. Its 98.5 % accuracy / 0.983 F1 on ArcSSM is an **in-distribution
ceiling**, not a generalization estimate.

It was the right tool for what it was used for, and that use should be stated
plainly in the report rather than hidden:

- **Architecture iteration.** Choosing between fusion modes, channel attention on/off,
  SE blocks, deep classifier head, and later attention-vs-SSM required dozens of
  runs. One split at ~35 min is affordable; a 4-fold protocol at ~2–3 h per
  configuration is not.
- **Sanity and capacity checks.** It answers "can this architecture fit the task at
  all, and does the training loop converge" — a model that fails here is dead
  without spending CV time on it.
- **Reference ceiling.** The gap between the random split and leave-one-campaign-out
  is itself a result: it quantifies how much of the easy performance came from
  cycle-level correlation.

`train.py` marks the mode accordingly (*"random split — NOT for generalization"*),
and its docstring says *"Does NOT test generalization to unseen charges. Use for
quick smoke tests only."*

### 3.2 Why recording-level GroupKFold is not the headline either

At recording level the groups are wildly unequal: each 2024 campaign is one
monolithic recording of 2746–3820 cycles, while the 2026 campaign splits into 22
recordings of 40–83 cycles. `StratifiedGroupKFold(n_splits=5)` then produces folds
whose test sets range from ~60 to ~7158 cycles, and past runs show fold F1 between
0.61 and 1.00 in the *same* run purely from which giant recording landed where. A
mean ± std over such folds is not a meaningful statistic. The mode is kept for
backwards comparison with earlier V1/V2 runs, not for new claims.

### 3.3 Leave-one-campaign-out — the protocol used

Four folds, each training on three campaigns and testing on the fourth:

| Fold | Test campaign | Test cycles (arc/normal) | Train | Val |
|---|---|---|---|---|
| 1 | `15_juillet_clean` | 2820 (1319 / 1501) | 6891 | 1149 |
| 2 | `22_juillet_clean` | 3820 (1777 / 2043) | 6034 | 1006 |
| 3 | `8_juillet_clean` | 2746 (1299 / 1447) | 6955 | 1159 |
| 4 | `OthmaneSalim10052026` | 1474 (567 / 907) | 8050 | 1336 |

Guarantees enforced in code (`run_groupkfold_cv`, asserted per fold, not assumed):

- test campaign ∩ (train ∪ val) campaigns = ∅;
- train alternances ∩ val alternances = ∅;
- pos_weight, normalization and early stopping are computed on train/val only.

**The validation split is `--val-mode alternance`.** With only four groups, spending
whole groups on validation (`--val-mode group`, the previous behaviour) costs an
entire campaign: training then sees two campaigns out of four and validates on a
third, i.e. 4220–6640 training cycles instead of 6034–8050, with an early-stopping
signal drawn from a single day. Instead, ~1/7 of the **alternances** of the training
campaigns is held out for validation, label-stratified, with every alternance kept
intact. No arc burst is split across train and val, so early stopping is not
selected on near-duplicates of training cycles, and no test campaign is touched.
`--val-mode random` (cycle-level) is available but leaky by construction and should
only be used for smoke tests.

## 4. Reading the results

Two numbers come out of a CV run, and they answer different questions:

- **Pooled out-of-fold** — with leave-one-group-out the four folds partition the
  dataset, so every one of the 10 860 cycles is classified exactly once by a model
  that never saw its campaign. Pooling those predictions gives a single confusion
  matrix over the whole dataset. **This is the number to quote**, with a Wilson 95 %
  interval.
- **Mean ± std across folds** — weights a 1474-cycle fold like a 3820-cycle one, so
  it is not the dataset-level score; read it as *campaign-to-campaign variability*.
  A large std is a finding (sensitivity to acquisition conditions), not noise to
  average away.

Per-campaign rows matter more than the average: the 2026 campaign uses different
electrode materials and household loads, and a model that only fails there is
telling you something specific.

An additional fully-independent check exists: `merge_datasets.py` kept **368 cycles
(20 %) of the 2026 campaign out of `combined_dataset_2048` entirely**
(`config.json: n_holdout`), evaluable with `eval_holdout.py`. It is a hold-out, not
cross-validation — use it as a one-shot confirmation, not as the main protocol.

## 5. Commands

Train (ArcSSM, matching the hyper-parameters of the single run it is compared to):

```bash
python train.py --model arcssm --mode groupkfold --group-level campaign \
  --data-dir combined_dataset_2048/combined_dataset_2048 \
  --output-dir runs --epochs 60 --patience 10 --batch-size 32 \
  --n-fft 512 --hop-length 256 --num-workers 4 --seed 42
```

`--val-mode` defaults to `alternance` whenever there are fewer than six groups, so
it does not need to be passed; pass it explicitly to override.

Evaluate (writes `<run_dir>/eval/results.md`, figures and error CSVs):

```bash
python eval_groupcv.py runs/arcssm_groupkfold_campaign_<timestamp> \
  --data-dir combined_dataset_2048/combined_dataset_2048 \
  --compare-single home/top/Arc-Fault-Net/runs/arcssm_single_20260726_150603/results.json
```

Outputs: pooled and per-fold confusion matrices, per-fold ROC with AUC, per-fold
metric bars, a pooled threshold/operating-point sweep (including the highest recall
reachable at ≤ 1 % false-alarm rate), per-campaign error rates, and FP/FN forensics
joined with `metadata.csv`.

Raw per-fold probabilities are saved (`fold_*/test_predictions.npz`,
`oof_predictions.npz`), so thresholds, ROC and pooled statistics can be recomputed
without retraining.

## 6. One caveat to state in the report

Four campaigns give four folds, so the fold-to-fold spread is estimated from four
points. The pooled matrix is over 10 860 cycles and is tight, but the *variability*
across acquisition conditions is measured coarsely — with three IJL days sharing a
bench, fold 4 (2026) is the only genuinely different setup. Report the pooled number
as the headline, the four per-campaign rows next to it, and do not present the
4-fold std as a confidence interval on the mean.
