import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from crypto_transformer.training.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from crypto_transformer.training.metrics import compute_metrics


def train_model(
    model: nn.Module,
    train_dataset,
    val_dataset,
    config: dict,
    checkpoint_dir: str = "data/checkpoints",
    device: str = "cuda",
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    label_counts = np.bincount(
        [train_dataset[i][1].item() for i in range(min(len(train_dataset), 10000))]
    )
    class_weights = 1.0 / (label_counts + 1)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
    )

    ckpt_path = str(checkpoint_dir / "best_model.pt")
    early_stopping = EarlyStopping(patience=config.get("patience", 15), mode="min")
    checkpoint_cb = ModelCheckpoint(ckpt_path, mode="max")
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0

    print(f"Training for up to {config['max_epochs']} epochs...")
    print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")
    print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

    for epoch in range(config["max_epochs"]):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 1.0))
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(metrics["accuracy"])
        history["val_f1"].append(metrics["f1_weighted"])

        lr_scheduler.step(metrics["f1_weighted"])
        checkpoint_cb.step(metrics["f1_weighted"], model)
        should_stop = early_stopping.step(-metrics["f1_weighted"])

        if (epoch + 1) % 5 == 0 or epoch == 0 or should_stop:
            print(
                f"Epoch {epoch+1:3d} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {metrics['accuracy']:.4f} | "
                f"val_f1: {metrics['f1_weighted']:.4f} | "
                f"lr: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if metrics["f1_weighted"] > best_val_f1:
            best_val_f1 = metrics["f1_weighted"]

        if should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        print(f"Loaded best model (val_f1={best_val_f1:.4f})")

    return {"model": model, "history": history, "best_val_f1": best_val_f1}


def evaluate_model(model: nn.Module, test_dataset, batch_size: int = 64, device: str = "cuda") -> dict:
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["predictions"] = np.array(all_preds)
    metrics["labels"] = np.array(all_labels)
    metrics["probabilities"] = np.array(all_probs)
    return metrics
