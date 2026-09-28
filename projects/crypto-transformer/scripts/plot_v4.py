import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

with open("data/results_v4.json") as f:
    results = json.load(f)

fig_dir = Path("docs/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

histories = results["histories"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

seed_colors = {"seed_42": "#2196F3", "seed_123": "#4CAF50", "seed_456": "#FF9800"}

ax = axes[0, 0]
for seed_name, hist in histories.items():
    epochs = range(1, len(hist["train_loss"]) + 1)
    ax.plot(epochs, hist["train_loss"], label=f"{seed_name} train", color=seed_colors[seed_name], linestyle="-")
    ax.plot(epochs, hist["val_loss"], label=f"{seed_name} val", color=seed_colors[seed_name], linestyle="--", alpha=0.6)
ax.set_title("Loss Curves (All Seeds)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend(fontsize=8)

ax = axes[0, 1]
for seed_name, hist in histories.items():
    epochs = range(1, len(hist["val_ic"]) + 1)
    ax.plot(epochs, hist["val_ic"], label=seed_name, color=seed_colors[seed_name], marker="o", markersize=2)
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Validation IC (All Seeds)")
ax.set_xlabel("Epoch")
ax.set_ylabel("IC")
ax.legend()

ax = axes[1, 0]
for seed_name, hist in histories.items():
    epochs = range(1, len(hist["val_dir_acc"]) + 1)
    ax.plot(epochs, hist["val_dir_acc"], label=seed_name, color=seed_colors[seed_name], marker="o", markersize=2)
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Validation Directional Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Directional Accuracy")
ax.legend()

per_pair = results["per_pair"]
names = sorted(per_pair.keys())
ics = [per_pair[n]["ic"] for n in names]
dir_accs = [per_pair[n]["dir_acc"] for n in names]
colors = ["#4CAF50" if v >= 0 else "#F44336" for v in ics]

ax = axes[1, 1]
ax.barh(range(len(names)), ics, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.replace("/USDT", "") for n in names])
ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax.set_title("Per-Pair IC (Test, Ensemble)")
ax.set_xlabel("Information Coefficient")

fig.suptitle("V4 Experiment: Regression + Pairwise Ranking, 3-Seed Ensemble", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(fig_dir / "v4_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

try:
    with open("data/results_v3.json") as f:
        v3 = json.load(f)
    has_v3 = True
except Exception:
    has_v3 = False

if has_v3:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].bar(["V3\n121K", "V4\n642K"], [v3["overall"]["ic"], results["overall"]["ic"]], color=["#2196F3", "#FF9800"])
    axes[0].set_title("IC (Test)")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    axes[1].bar(["V3", "V4"], [v3["overall"]["dir_acc"], results["overall"]["dir_acc"]], color=["#2196F3", "#FF9800"])
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("Directional Accuracy")
    axes[1].set_ylim(0.4, 0.6)

    axes[2].bar(["V3\n80 epochs", "V4\n100 epochs"], [121393, 642033], color=["#2196F3", "#FF9800"])
    axes[2].set_title("Model Parameters")

    v3_best = max(v3["history"]["val_ic"])
    v4_best = max(max(h["val_ic"]) for h in results["histories"].values())
    axes[3].bar(["V3", "V4"], [v3_best, v4_best], color=["#2196F3", "#FF9800"])
    axes[3].set_title("Best Validation IC")

    fig.suptitle("V3 vs V4 Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "v3_v4_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Figures saved to docs/figures/")
