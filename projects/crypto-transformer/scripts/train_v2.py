import sys
import json
import numpy as np
import torch
import torch.nn as nn

from crypto_transformer.utils.config import load_config
from crypto_transformer.data.collectors.binance import fetch_ohlcv
from crypto_transformer.data.features_v2 import compute_features_v2, NUM_FEATURES_V2
from crypto_transformer.data.dataset_v2 import prepare_dataset_v2, LABEL_NAMES
from crypto_transformer.models.transformer_v2 import CryptoTransformerV2, FocalLoss
from crypto_transformer.models.baselines import XGBoostBaseline
from crypto_transformer.training.metrics import compute_metrics
from crypto_transformer.backtesting.engine import BacktestEngine
from torch.utils.data import DataLoader
from pathlib import Path


def train_one_epoch(model, loader, criterion_cls, criterion_reg, optimizer, device, cls_weight, reg_weight, clip):
    model.train()
    total_loss = 0
    for x, y_cls, y_reg in loader:
        x, y_cls, y_reg = x.to(device), y_cls.to(device), y_reg.to(device)
        optimizer.zero_grad()
        cls_logits, reg_pred = model(x)
        loss_cls = criterion_cls(cls_logits, y_cls)
        loss_reg = criterion_reg(reg_pred, y_reg)
        loss = cls_weight * loss_cls + reg_weight * loss_reg
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion_cls, criterion_reg, device, cls_weight, reg_weight):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs, all_reg_pred, all_reg_true = [], [], [], [], []
    for x, y_cls, y_reg in loader:
        x, y_cls, y_reg = x.to(device), y_cls.to(device), y_reg.to(device)
        cls_logits, reg_pred = model(x)
        loss_cls = criterion_cls(cls_logits, y_cls)
        loss_reg = criterion_reg(reg_pred, y_reg)
        loss = cls_weight * loss_cls + reg_weight * loss_reg
        total_loss += loss.item()
        probs = torch.softmax(cls_logits, dim=-1)
        all_preds.extend(cls_logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(y_cls.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_reg_pred.extend(reg_pred.cpu().numpy())
        all_reg_true.extend(y_reg.cpu().numpy())
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["probabilities"] = np.array(all_probs)
    metrics["predictions"] = np.array(all_preds)
    metrics["labels"] = np.array(all_labels)
    metrics["reg_pred"] = np.array(all_reg_pred)
    metrics["reg_true"] = np.array(all_reg_true)
    return total_loss / len(loader), metrics


def main():
    config = load_config("v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = config["model"]
    tcfg = config["training"]
    dcfg = config["data"]

    all_results = {}

    pairs = dcfg["binance_pairs"]
    dfs = {}
    for symbol in pairs:
        dfs[symbol] = fetch_ohlcv(symbol, dcfg["timeframe"])

    for symbol in pairs:
        print(f"\n{'='*70}")
        print(f" V2 Experiment: {symbol}")
        print(f"{'='*70}")

        cross_symbol = [s for s in pairs if s != symbol]
        cross_df = dfs[cross_symbol[0]] if cross_symbol else None

        df = compute_features_v2(dfs[symbol], cross_df=cross_df)

        data = prepare_dataset_v2(
            df,
            seq_len=config["dataset"]["seq_len"],
            horizon=dcfg["prediction_horizon"],
            label_threshold=dcfg["label_threshold"],
            train_cutoff=dcfg["train_cutoff"],
            val_fraction=dcfg["val_fraction"],
        )

        print(f"\n[1] Data: train={len(data['train'])} val={len(data['val'])} test={len(data['test'])}")
        print(f"    Labels (train): {dict(zip(LABEL_NAMES, data['label_counts']['train']))}")
        print(f"    Labels (test):  {dict(zip(LABEL_NAMES, data['label_counts']['test']))}")

        model = CryptoTransformerV2(
            num_features=NUM_FEATURES_V2,
            d_model=mcfg["d_model"],
            nhead=mcfg["nhead"],
            num_layers=mcfg["num_layers"],
            dim_feedforward=mcfg["dim_feedforward"],
            dropout=mcfg["dropout"],
            num_classes=mcfg["num_classes"],
        ).to(device)
        print(f"\n[2] Model: {model.count_parameters():,} parameters")

        label_counts = data["label_counts"]["train"]
        class_weights = 1.0 / (label_counts.astype(float) + 1)
        class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32).to(device)

        if mcfg.get("use_focal_loss", True):
            criterion_cls = FocalLoss(gamma=mcfg.get("focal_gamma", 2.0), weight=class_weights)
        else:
            criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
        criterion_reg = nn.HuberLoss()

        cls_w = mcfg.get("classification_weight", 0.7)
        reg_w = mcfg.get("regression_weight", 0.3)

        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        train_loader = DataLoader(data["train"], batch_size=tcfg["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(data["val"], batch_size=tcfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(data["test"], batch_size=tcfg["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

        best_val_f1 = 0
        patience_counter = 0
        best_state = None
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

        print(f"\n[3] Training (max {tcfg['max_epochs']} epochs, patience={tcfg['patience']})...")
        for epoch in range(tcfg["max_epochs"]):
            train_loss = train_one_epoch(
                model, train_loader, criterion_cls, criterion_reg,
                optimizer, device, cls_w, reg_w, tcfg["gradient_clip"],
            )
            scheduler.step()
            val_loss, val_metrics = evaluate(model, val_loader, criterion_cls, criterion_reg, device, cls_w, reg_w)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1_weighted"])

            if val_metrics["f1_weighted"] > best_val_f1:
                best_val_f1 = val_metrics["f1_weighted"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1:3d} | train={train_loss:.4f} val={val_loss:.4f} "
                    f"acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1_weighted']:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if patience_counter >= tcfg["patience"]:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)
        print(f"  Best val F1: {best_val_f1:.4f}")

        print(f"\n[4] Test evaluation...")
        _, test_metrics = evaluate(model, test_loader, criterion_cls, criterion_reg, device, cls_w, reg_w)
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  F1 (w):    {test_metrics['f1_weighted']:.4f}")
        print(f"  F1 (mac):  {test_metrics['f1_macro']:.4f}")
        for cls_name, cls_m in test_metrics["per_class"].items():
            print(f"    {cls_name}: P={cls_m['precision']:.3f} R={cls_m['recall']:.3f} F1={cls_m['f1']:.3f}")

        # Regression quality
        reg_pred = test_metrics["reg_pred"]
        reg_true = test_metrics["reg_true"]
        reg_corr = np.corrcoef(reg_pred, reg_true)[0, 1]
        reg_mae = np.mean(np.abs(reg_pred - reg_true))
        directional_acc = np.mean((reg_pred > 0) == (reg_true > 0))
        print(f"  Return corr: {reg_corr:.4f}")
        print(f"  Return MAE:  {reg_mae:.6f}")
        print(f"  Directional: {directional_acc:.4f}")

        # XGBoost baseline
        print(f"\n[5] XGBoost baseline...")
        train_data = data["train"]
        train_X = np.array([train_data[i][0].numpy() for i in range(len(train_data))])
        train_y = np.array([train_data[i][1].numpy() for i in range(len(train_data))])
        test_data = data["test"]
        test_X = np.array([test_data[i][0].numpy() for i in range(len(test_data))])
        test_y = np.array([test_data[i][1].numpy() for i in range(len(test_data))])

        xgb = XGBoostBaseline(seq_len=config["dataset"]["seq_len"])
        xgb.fit(train_X, train_y)
        xgb_preds = xgb.predict(test_X)
        xgb_metrics = compute_metrics(test_y, xgb_preds)
        print(f"  XGBoost: acc={xgb_metrics['accuracy']:.4f} f1={xgb_metrics['f1_weighted']:.4f}")

        # Backtest
        print(f"\n[6] Backtesting...")
        bt_cfg = config["backtesting"]
        engine = BacktestEngine(
            initial_capital=bt_cfg["initial_capital"],
            fee_rate=bt_cfg["fee_rate"],
            slippage=bt_cfg["slippage"],
            position_fraction=bt_cfg["position_fraction"],
        )

        test_close = df["close_raw"].values
        test_start = len(df) - len(test_data) - config["dataset"]["seq_len"]
        test_close_sub = test_close[test_start + config["dataset"]["seq_len"]:test_start + config["dataset"]["seq_len"] + len(test_data)]

        # Use regression predictions for trading: positive return -> buy, negative -> sell
        reg_signals = test_metrics["reg_pred"]
        min_ret = bt_cfg.get("min_pred_return", 0.002)
        bt_preds = np.where(reg_signals > min_ret, 0, np.where(reg_signals < -min_ret, 1, 2))
        bt_conf = np.abs(reg_signals) / (np.abs(reg_signals).max() + 1e-8)
        bt_conf = np.clip(bt_conf + 0.3, 0, 1)

        bt_result = engine.run(
            bt_preds, np.column_stack([
                (bt_preds == 0).astype(float) * bt_conf,
                (bt_preds == 1).astype(float) * bt_conf,
                (bt_preds == 2).astype(float) * bt_conf,
            ]),
            test_close_sub[:len(bt_preds)],
            confidence_threshold=bt_cfg["confidence_threshold"],
        )

        buy_hold = (test_close_sub[-1] / test_close_sub[0]) - 1

        print(f"  Strategy return: {bt_result.total_return*100:.2f}%")
        print(f"  Buy & Hold:      {buy_hold*100:.2f}%")
        print(f"  Sharpe:          {bt_result.sharpe_ratio:.2f}")
        print(f"  Max DD:          {bt_result.max_drawdown*100:.2f}%")
        print(f"  Trades:          {bt_result.num_trades}")
        print(f"  Win rate:        {bt_result.win_rate*100:.1f}%")

        all_results[symbol] = {
            "transformer": {
                "accuracy": test_metrics["accuracy"],
                "f1_weighted": test_metrics["f1_weighted"],
                "f1_macro": test_metrics["f1_macro"],
                "per_class": test_metrics["per_class"],
                "confusion_matrix": test_metrics["confusion_matrix"],
                "return_correlation": float(reg_corr),
                "directional_accuracy": float(directional_acc),
                "return_mae": float(reg_mae),
            },
            "xgboost": {"accuracy": xgb_metrics["accuracy"], "f1_weighted": xgb_metrics["f1_weighted"]},
            "backtest": {
                "strategy_return": bt_result.total_return,
                "buy_hold_return": float(buy_hold),
                "sharpe": bt_result.sharpe_ratio,
                "max_drawdown": bt_result.max_drawdown,
                "num_trades": bt_result.num_trades,
                "win_rate": bt_result.win_rate,
            },
            "history": history,
            "data": {
                "train": len(data["train"]),
                "val": len(data["val"]),
                "test": len(data["test"]),
                "labels": {k: v.tolist() for k, v in data["label_counts"].items()},
            },
        }

    Path("data").mkdir(exist_ok=True)
    with open("data/results_v2.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to data/results_v2.json")


if __name__ == "__main__":
    main()
