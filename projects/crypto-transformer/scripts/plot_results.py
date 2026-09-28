import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def plot_training_curves(history: dict, title: str, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["val_acc"], label="Val Accuracy", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["val_f1"], label="Val F1 (weighted)", color="orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Validation F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: list, title: str, save_path: str):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Up", "Down", "Flat"]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title, ylabel="True label", xlabel="Predicted label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(results: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")

    for idx, symbol in enumerate(results.keys()):
        r = results[symbol]
        models = ["Transformer"] + list(r["baselines"].keys())
        accs = [r["transformer"]["accuracy"]] + [b["accuracy"] for b in r["baselines"].values()]
        f1s = [r["transformer"]["f1_weighted"]] + [b["f1_weighted"] for b in r["baselines"].values()]

        x = np.arange(len(models))
        width = 0.35
        bars1 = axes[idx].bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
        bars2 = axes[idx].bar(x + width / 2, f1s, width, label="F1 (weighted)", color="coral")

        axes[idx].set_xlabel("Model")
        axes[idx].set_ylabel("Score")
        axes[idx].set_title(symbol)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(models, rotation=30, ha="right")
        axes[idx].legend()
        axes[idx].set_ylim(0, 1.0)
        axes[idx].grid(True, alpha=0.3, axis="y")

        for bar in bars1:
            h = bar.get_height()
            axes[idx].annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                               xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            axes[idx].annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                               xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_label_distribution(results: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Label Distribution", fontsize=14, fontweight="bold")
    labels = ["Up", "Down", "Flat"]
    colors = ["green", "red", "gray"]

    for idx, symbol in enumerate(results.keys()):
        r = results[symbol]
        splits = list(r["data_size"]["label_distribution"].keys())
        data = r["data_size"]["label_distribution"]

        x = np.arange(len(labels))
        width = 0.25
        for i, split in enumerate(splits):
            counts = data[split]
            total = sum(counts)
            pcts = [c / total for c in counts]
            axes[idx].bar(x + i * width, pcts, width, label=split.capitalize(), alpha=0.8)

        axes[idx].set_xlabel("Class")
        axes[idx].set_ylabel("Proportion")
        axes[idx].set_title(symbol)
        axes[idx].set_xticks(x + width)
        axes[idx].set_xticklabels(labels)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    out_dir = Path("docs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open("data/results.json") as f:
        results = json.load(f)

    for symbol in results:
        safe = symbol.replace("/", "_")
        r = results[symbol]

        plot_training_curves(
            r["training_history"],
            f"Training History - {symbol}",
            str(out_dir / f"training_curves_{safe}.png"),
        )
        plot_confusion_matrix(
            r["transformer"]["confusion_matrix"],
            f"Confusion Matrix - {symbol}",
            str(out_dir / f"confusion_{safe}.png"),
        )

    plot_model_comparison(results, str(out_dir / "model_comparison.png"))
    plot_label_distribution(results, str(out_dir / "label_distribution.png"))

    print(f"Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
