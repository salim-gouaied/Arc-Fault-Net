#!/usr/bin/env python3
"""
Full evaluation + forensics for a single ArcSSM run.
Reproduces the exact seed-42 test split, loads the best checkpoint, and produces:
  - confusion matrix, ROC curve, training curves
  - false-positive / false-negative forensics (traced to experiment via metadata)
  - correlation analysis between the 4 derived descriptors and the label
Outputs go to <run_dir>/eval/.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

from train import set_seed
from model import get_model
from dataset import ArcFaultDataset

RUN_DIR = Path("home/top/Arc-Fault-Net/runs/arcssm_single_20260726_150603")
DATA_DIR = "combined_dataset_2048/combined_dataset_2048"
SEED, TRAIN_R, VAL_R = 42, 0.7, 0.15
CH_NAMES = ["I_norm", "|dI|", "TKEO", "RMS_slide"]
device = torch.device("cpu")  # eval on CPU to avoid VRAM contention
out = RUN_DIR / "eval"; out.mkdir(parents=True, exist_ok=True)
cfg = json.load(open(RUN_DIR / "results.json"))

# ---- dataset + exact split ----
ds = ArcFaultDataset(data_dir=DATA_DIR, n_fft=cfg["n_fft"],
                     hop_length=cfg["hop_length"], channel_mode="i_derived4")
N = len(ds)
set_seed(SEED)
idx = np.random.permutation(N)
n_tr, n_va = int(N * TRAIN_R), int(N * VAL_R)
test_idx = idx[n_tr + n_va:]
print(f"Test samples: {len(test_idx)}  (device={device})")

# ---- model ----
model = get_model("arcssm", in_channels=2, use_se=cfg["use_se"],
                  deep_classifier=cfg["deep_classifier"], fusion_mode=cfg["fusion_mode"],
                  fs=cfg["fs"], n_fft=cfg["n_fft"]).to(device)
model.load_state_dict(torch.load(RUN_DIR / "best_single.pt", map_location=device))
model.eval()

# ---- inference on test ----
loader = DataLoader(Subset(ds, test_idx), batch_size=64, shuffle=False)
probs, labels = [], []
with torch.no_grad():
    for x1, x2, y, *_ in loader:
        p = torch.sigmoid(model(x1.to(device), x2.to(device))).cpu().numpy()
        probs.append(p); labels.append(y.numpy())
probs = np.concatenate(probs).ravel()
labels = np.concatenate(labels).ravel().astype(int)
preds = (probs > 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
acc = (tp + tn) / len(labels)
prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
spec = tn / (tn + fp + 1e-9); f1 = 2 * prec * rec / (prec + rec + 1e-9)
print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} spec={spec:.4f}")

# ============ 1) confusion matrix ============
cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                fontsize=16, color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Arc"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Arc"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"Confusion matrix — ArcSSM (test)\nacc={acc:.3f}  F1={f1:.3f}")
plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

# ============ 2) ROC ============
fpr, tpr, _ = roc_curve(labels, probs); roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(4.8, 4.4))
ax.plot(fpr, tpr, lw=2.2, color="#2C6DB5", label=f"AUC = {roc_auc:.4f}")
ax.plot([0, 1], [0, 1], "--", color="0.6")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve — ArcSSM (test)"); ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(out / "roc_curve.png", dpi=150); plt.close()

# ============ 3) training curves ============
h = json.load(open(RUN_DIR / "history_single.json"))
ep = range(1, len(h["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ep, h["train_loss"], label="train"); axes[0].plot(ep, h["val_loss"], label="val")
axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
axes[1].plot(ep, h["val_f1"], color="#1E8C74", label="val F1")
axes[1].plot(ep, h["val_acc"], color="#2C6DB5", label="val acc")
axes[1].axvline(h["best_epoch"], ls="--", color="red", label=f"best (ep {h['best_epoch']})")
axes[1].set_title("Validation metrics"); axes[1].set_xlabel("epoch"); axes[1].legend()
plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=150); plt.close()

# ============ 4) FP / FN forensics ============
meta = pd.read_csv(Path(DATA_DIR) / "metadata.csv")
tmeta = meta.iloc[test_idx].reset_index(drop=True).copy()
tmeta["prob"] = probs; tmeta["pred"] = preds; tmeta["true"] = labels
fp_rows = tmeta[(tmeta.pred == 1) & (tmeta.true == 0)].sort_values("prob", ascending=False)
fn_rows = tmeta[(tmeta.pred == 0) & (tmeta.true == 1)].sort_values("prob")
fp_rows.to_csv(out / "false_positives.csv", index=False)
fn_rows.to_csv(out / "false_negatives.csv", index=False)
print("\n=== FALSE POSITIVES (pred=arc, true=normal) ===")
print(fp_rows[["dataset", "exp_name", "alt_index", "arc_ratio", "prob"]].to_string(index=False))
print("\nFP by experiment:")
print(fp_rows["exp_name"].value_counts().to_string())
print("\n=== FALSE NEGATIVES (missed arcs) — top by lowest prob ===")
print(fn_rows[["dataset", "exp_name", "alt_index", "arc_ratio", "prob"]].head(12).to_string(index=False))
print("\nFN by experiment:")
print(fn_rows["exp_name"].value_counts().head(10).to_string())

# plot FP current waveforms (raw I = channel 1 of X_multi)
X = ds.X  # (N, 2, seq_len)
show = fp_rows.head(6)
if len(show):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, (_, r) in zip(axes.ravel(), show.iterrows()):
        gi = test_idx[r.name]
        ax.plot(X[gi, 1], lw=0.7, color="#C0392B")
        ax.set_title(f"{r.exp_name}\nalt {int(r.alt_index)} | arc_ratio={r.arc_ratio:.2f} | p={r.prob:.2f}",
                     fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("False-positive current cycles (model predicted ARC, oracle said normal)")
    plt.tight_layout(); plt.savefig(out / "fp_waveforms.png", dpi=150); plt.close()

# ============ 5) descriptor correlation ============
# per-cycle scalar summary (RMS) of each of the 4 derived channels
feats = np.zeros((len(test_idx), 4))
for k, gi in enumerate(test_idx):
    x1, _, _, *_ = ds[gi]
    x1 = x1.numpy()
    feats[k] = np.sqrt((x1 ** 2).mean(axis=1))   # RMS per channel
M = np.column_stack([feats, labels])
corr = np.corrcoef(M, rowvar=False)
lab = CH_NAMES + ["label"]
fig, ax = plt.subplots(figsize=(5.6, 5.0))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(corr[i, j]) > 0.6 else "black")
ax.set_xticks(range(5)); ax.set_xticklabels(lab, rotation=45, ha="right")
ax.set_yticks(range(5)); ax.set_yticklabels(lab)
ax.set_title("Correlation: descriptor RMS + label (test)")
plt.colorbar(im, fraction=0.046); plt.tight_layout()
plt.savefig(out / "feature_correlation.png", dpi=150); plt.close()
print("\n=== Correlation matrix (descriptor RMS + label) ===")
print(pd.DataFrame(corr, index=lab, columns=lab).round(3).to_string())

print(f"\nAll figures + CSVs saved to: {out}")
