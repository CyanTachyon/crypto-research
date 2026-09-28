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

with open("data/results_v3.json") as f:
    results = json.load(f)

history = results["history"]
fig_dir = Path("docs/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

epochs = range(1, len(history["train_loss"]) + 1)
axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3")
axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", color="#F44336")
axes[0, 0].set_title("Loss Curves")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Huber Loss")
axes[0, 0].legend()

axes[0, 1].plot(epochs, history["val_ic"], color="#4CAF50", marker="o", markersize=3)
axes[0, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0, 1].set_title("Validation Information Coefficient (IC)")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("IC (Correlation)")

axes[1, 0].plot(epochs, history["val_dir_acc"], color="#FF9800", marker="o", markersize=3)
axes[1, 0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[1, 0].set_title("Validation Directional Accuracy")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Directional Accuracy")

per_pair = results["per_pair"]
names = sorted(per_pair.keys())
ics = [per_pair[n]["ic"] for n in names]
dir_accs = [per_pair[n]["dir_acc"] for n in names]
colors = ["#4CAF50" if v >= 0 else "#F44336" for v in ics]

axes[1, 1].barh(range(len(names)), ics, color=colors, edgecolor="white", linewidth=0.5)
axes[1, 1].set_yticks(range(len(names)))
axes[1, 1].set_yticklabels([n.replace("/USDT", "") for n in names])
axes[1, 1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
axes[1, 1].set_title("Per-Pair IC (Test)")
axes[1, 1].set_xlabel("Information Coefficient")

fig.suptitle("V3 Experiment: Multi-Asset Daily Transformer (Pure Regression)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(fig_dir / "v3_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

try:
    with open("data/results.json") as f:
        v1 = json.load(f)
    with open("data/results_v2.json") as f:
        v2 = json.load(f)
    has_comparison = True
except Exception:
    has_comparison = False

if has_comparison:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    v1_hist = v1["ETH/USDT"]["training_history"]["val_loss"]
    v2_hist = v2["ETH/USDT"]["history"]["val_loss"]
    v1_best = min(v1_hist)
    v2_best = min(v2_hist)
    v3_best = results["best_val_loss"]
    axes[0].bar(["V1", "V2", "V3"], [v1_best, v2_best, v3_best], color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[0].set_title("Best Validation Loss")

    v1_dir = v1["ETH/USDT"]["transformer"]["accuracy"]
    v2_dir = v2["ETH/USDT"]["transformer"]["directional_accuracy"]
    v3_dir = results["overall"]["dir_acc"]
    axes[1].bar(["V1", "V2", "V3"], [v1_dir, v2_dir, v3_dir], color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("Directional Accuracy (Test)")
    axes[1].set_ylim(0.3, 0.8)

    axes[2].bar(["V1\n804K", "V2\n879K", "V3\n121K"], [804000, 879000, 121393], color=["#2196F3", "#4CAF50", "#FF9800"])
    axes[2].set_title("Model Parameters")

    fig.suptitle("V1 vs V2 vs V3 Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "v1_v2_v3_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Figures saved to docs/figures/")
