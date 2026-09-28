import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_v2_training(history: dict, symbol: str, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"V2 Training - {symbol}", fontsize=14, fontweight="bold")

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    axes[1].plot(history["val_acc"], label="Val Accuracy", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["val_f1"], label="Val F1 (weighted)", color="orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1")
    axes[2].set_title("F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_v1_v2_comparison(v1: dict, v2: dict, save_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("V1 vs V2 Comparison", fontsize=14, fontweight="bold")

    metrics = ["accuracy", "f1_weighted"]
    metric_labels = ["Accuracy", "F1 (weighted)"]

    for idx, symbol in enumerate(v1.keys()):
        models = ["V1 Transformer", "V2 Transformer", "V1 XGBoost", "V2 XGBoost"]
        accs = [
            v1[symbol]["transformer"]["accuracy"],
            v2[symbol]["transformer"]["accuracy"],
            v1[symbol]["baselines"]["xgboost"]["accuracy"],
            v2[symbol]["xgboost"]["accuracy"],
        ]
        f1s = [
            v1[symbol]["transformer"]["f1_weighted"],
            v2[symbol]["transformer"]["f1_weighted"],
            v1[symbol]["baselines"]["xgboost"]["f1_weighted"],
            v2[symbol]["xgboost"]["f1_weighted"],
        ]

        x = np.arange(len(models))
        width = 0.35
        axes[idx][0].bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
        axes[idx][0].bar(x + width / 2, f1s, width, label="F1", color="coral")
        axes[idx][0].set_title(symbol)
        axes[idx][0].set_xticks(x)
        axes[idx][0].set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        axes[idx][0].legend()
        axes[idx][0].set_ylim(0, 1.0)
        axes[idx][0].grid(True, alpha=0.3, axis="y")

        # Per-class recall comparison
        v1_cls = v1[symbol]["transformer"]["per_class"]
        v2_cls = v2[symbol]["transformer"]["per_class"]
        classes = ["up", "down", "flat"]
        v1_recalls = [v1_cls[c]["recall"] for c in classes]
        v2_recalls = [v2_cls[c]["recall"] for c in classes]
        x2 = np.arange(len(classes))
        axes[idx][1].bar(x2 - 0.2, v1_recalls, 0.35, label="V1", color="gray")
        axes[idx][1].bar(x2 + 0.2, v2_recalls, 0.35, label="V2", color="teal")
        axes[idx][1].set_title(f"{symbol} - Recall by Class")
        axes[idx][1].set_xticks(x2)
        axes[idx][1].set_xticklabels(["Up", "Down", "Flat"])
        axes[idx][1].legend()
        axes[idx][1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_backtest_comparison(v1: dict, v2: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Backtest Comparison: V1 vs V2", fontsize=14, fontweight="bold")

    for idx, symbol in enumerate(v1.keys()):
        labels = ["V1 Strategy", "V1 Buy&Hold", "V2 Strategy", "V2 Buy&Hold"]
        returns = [
            v1[symbol]["backtest"]["total_return"] * 100,
            v1[symbol]["backtest"]["buy_hold_return"] * 100,
            v2[symbol]["backtest"]["strategy_return"] * 100,
            v2[symbol]["backtest"]["buy_hold_return"] * 100,
        ]
        colors = ["#666666", "#cccccc", "#2196F3", "#90CAF9"]
        axes[idx].bar(labels, returns, color=colors)
        axes[idx].set_title(symbol)
        axes[idx].set_ylabel("Return (%)")
        axes[idx].axhline(y=0, color="black", linewidth=0.5)
        axes[idx].grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(returns):
            axes[idx].annotate(f"{v:.1f}%", xy=(i, v), ha="center", fontsize=9,
                               va="bottom" if v >= 0 else "top")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)

    with open("data/results.json") as f:
        v1 = json.load(f)
    with open("data/results_v2.json") as f:
        v2 = json.load(f)

    for symbol in v2:
        safe = symbol.replace("/", "_")
        plot_v2_training(v2[symbol]["history"], symbol, str(out / f"v2_training_{safe}.png"))

    plot_v1_v2_comparison(v1, v2, str(out / "v1_v2_comparison.png"))
    plot_backtest_comparison(v1, v2, str(out / "v1_v2_backtest.png"))
    print(f"Figures saved to {out}/")


if __name__ == "__main__":
    main()
