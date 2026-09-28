import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from crypto_transformer.utils.config import load_config
from crypto_transformer.data.collectors.binance import fetch_ohlcv
from crypto_transformer.data.features_v3 import compute_features_v3, NUM_FEATURES_V3
from crypto_transformer.data.dataset_v3 import prepare_multi_asset_dataset
from crypto_transformer.models.transformer_v3 import CryptoTransformerV3
from torch.utils.data import DataLoader


def main():
    config = load_config("v3")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = config["model"]
    tcfg = config["training"]
    dcfg = config["data"]

    print("Loading data...")
    btc = fetch_ohlcv("BTC/USDT", dcfg["timeframe"])
    all_dfs = {}
    for p in dcfg["pairs"]:
        all_dfs[p] = compute_features_v3(fetch_ohlcv(p, dcfg["timeframe"]), btc_df=btc)
        print(f"  {p}: {len(all_dfs[p])} rows")

    data = prepare_multi_asset_dataset(
        all_dfs, btc,
        seq_len=config["dataset"]["seq_len"],
        horizon=dcfg["prediction_horizon"],
        train_cutoff=dcfg["train_cutoff"],
        val_fraction=dcfg["val_fraction"],
    )
    print(f"\nData: train={len(data['train'])} val={len(data['val'])} test={len(data['test'])}")
    print(f"Stats: {data['stats']}")
    print(f"Pairs ({len(data['pair_names'])}): {data['pair_names']}")

    model = CryptoTransformerV3(
        num_features=NUM_FEATURES_V3,
        num_pairs=config["dataset"]["num_pairs"],
        d_model=mcfg["d_model"],
        nhead=mcfg["nhead"],
        num_layers=mcfg["num_layers"],
        dim_feedforward=mcfg["dim_feedforward"],
        dropout=mcfg["dropout"],
        pair_emb_dim=mcfg["pair_embedding_dim"],
    ).to(device)
    print(f"\nModel: {model.count_parameters():,} parameters")

    criterion = nn.HuberLoss(delta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])

    warmup = tcfg.get("warmup_epochs", 5)
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (tcfg["max_epochs"] - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = DataLoader(data["train"], batch_size=tcfg["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(data["val"], batch_size=tcfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(data["test"], batch_size=tcfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_dir_acc": []}

    print(f"\nTraining ({tcfg['max_epochs']} epochs)...")
    for epoch in range(tcfg["max_epochs"]):
        model.train()
        train_loss = 0
        for x, y, pid, _ in train_loader:
            x, y, pid = x.to(device), y.to(device), pid.to(device)
            optimizer.zero_grad()
            pred = model(x, pid)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg["gradient_clip"])
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y, pid, _ in val_loader:
                x, y, pid = x.to(device), y.to(device), pid.to(device)
                pred = model(x, pid)
                val_loss += criterion(pred, y).item()
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(y.cpu().numpy())
        val_loss /= len(val_loader)

        pred_arr = np.array(all_pred)
        true_arr = np.array(all_true)
        ic = np.corrcoef(pred_arr, true_arr)[0, 1] if len(pred_arr) > 2 else 0
        dir_acc = np.mean((pred_arr > 0) == (true_arr > 0))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ic"].append(float(ic))
        history["val_dir_acc"].append(float(dir_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d} | train={train_loss:.6f} val={val_loss:.6f} "
                f"IC={ic:.4f} dir_acc={dir_acc:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if patience_counter >= tcfg["patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.6f}")

    print("\nTest evaluation...")
    model.eval()
    all_pred, all_true, all_pid, all_close = [], [], [], []
    with torch.no_grad():
        for x, y, pid, cp in test_loader:
            x, pid = x.to(device), pid.to(device)
            pred = model(x, pid)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(y.numpy())
            all_pid.extend(pid.cpu().numpy())
            all_close.extend(cp.numpy())

    pred_arr = np.array(all_pred)
    true_arr = np.array(all_true)
    pid_arr = np.array(all_pid)
    close_arr = np.array(all_close)

    ic = np.corrcoef(pred_arr, true_arr)[0, 1]
    dir_acc = np.mean((pred_arr > 0) == (true_arr > 0))
    mae = np.mean(np.abs(pred_arr - true_arr))
    rmse = np.sqrt(np.mean((pred_arr - true_arr) ** 2))

    print(f"  IC (correlation): {ic:.4f}")
    print(f"  Directional acc:  {dir_acc:.4f}")
    print(f"  MAE:              {mae:.6f}")
    print(f"  RMSE:             {rmse:.6f}")

    pair_names = data["pair_names"]
    pair_to_id = data["pair_to_id"]
    id_to_pair = {v: k for k, v in pair_to_id.items()}
    print(f"\nPer-pair IC:")
    per_pair = {}
    for pid_val in sorted(np.unique(pid_arr)):
        mask = pid_arr == pid_val
        p_ic = np.corrcoef(pred_arr[mask], true_arr[mask])[0, 1]
        p_dir = np.mean((pred_arr[mask] > 0) == (true_arr[mask] > 0))
        p_count = mask.sum()
        pname = id_to_pair.get(pid_val, f"pair_{pid_val}")
        per_pair[pname] = {"ic": float(p_ic), "dir_acc": float(p_dir), "count": int(p_count)}
        print(f"  {pname:12s}  IC={p_ic:+.4f}  dir={p_dir:.3f}  n={p_count}")

    results = {
        "overall": {"ic": float(ic), "dir_acc": float(dir_acc), "mae": float(mae), "rmse": float(rmse)},
        "per_pair": per_pair,
        "history": history,
        "stats": data["stats"],
        "best_val_loss": float(best_val_loss),
    }

    Path("data").mkdir(exist_ok=True)
    with open("data/results_v3.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to data/results_v3.json")


if __name__ == "__main__":
    main()
