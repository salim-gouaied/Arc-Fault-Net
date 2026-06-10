import json, glob, statistics
import matplotlib.pyplot as plt
import numpy as np

group_0806 = glob.glob("runs/arcfaultnet_v2_single_20260608_*/results.json")
group_0910 = glob.glob("runs/arcfaultnet_v2_single_20260609_*/results.json") + glob.glob("runs/arcfaultnet_v2_single_20260610_*/results.json")

def get_metrics(group_files):
    metrics = {"accuracy": [], "f1": [], "precision": [], "recall": [], "specificity": []}
    for f in group_files:
        try:
            with open(f) as file:
                data = json.load(file)
                for k in metrics.keys():
                    metrics[k].append(data.get("test_" + k, 0) * 100)
        except Exception:
            pass
    means, stds = {}, {}
    for k in metrics.keys():
        if metrics[k]:
            means[k] = sum(metrics[k]) / len(metrics[k])
            stds[k] = statistics.stdev(metrics[k]) if len(metrics[k]) > 1 else 0
    return means, stds

m1, s1 = get_metrics(group_0806)
m2, s2 = get_metrics(group_0910)

labels = [k.capitalize() for k in m1.keys()]
means1 = [m1[k] for k in m1.keys()]
means2 = [m2[k] for k in m2.keys()]
stds1 = [s1[k] for k in m1.keys()]
stds2 = [s2[k] for k in m2.keys()]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, means1, width, yerr=stds1, label='abs(ΔI) [08/06]', capsize=5, color='#3498db')
rects2 = ax.bar(x + width/2, means2, width, yerr=stds2, label='Dowalla Residual [09/06-10/06]', capsize=5, color='#e74c3c')

ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('Feature Engineering Comparison: abs(ΔI) vs Dowalla Residual', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim(85, 100)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for i in range(len(labels)):
    ax.text(x[i] - width/2, means1[i] + stds1[i] + 0.2, f'{means1[i]:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.text(x[i] + width/2, means2[i] + stds2[i] + 0.2, f'{means2[i]:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('feature_comparison.png', dpi=150)
print("Saved feature_comparison.png")
