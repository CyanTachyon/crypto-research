import sys
import json
import numpy as np
import torch

from crypto_transformer.utils.config import load_config
from crypto_transformer.data.collectors.binance import fetch_ohlcv
from crypto_transformer.data.preprocessing import prepare_dataframe
from crypto_transformer.data.dataset import prepare_dataset
from crypto_transformer.models.transformer import CryptoTransformer
from crypto_transformer.models.baselines import RandomBaseline, MajorityBaseline, XGBoostBaseline, SMACrossoverBaseline
from crypto_transformer.training.trainer import train_model, evaluate_model
from crypto_transformer.backtesting.engine import BacktestEngine


def main():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for symbol in config["data"]["binance_pairs"]:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")

        print(f"\n[1] Loading data...")
        df = fetch_ohlcv(symbol, config["data"]["timeframe"])
        df = prepare_dataframe(df, config.get("features", {}).get("indicators"))

        dataset_config = {
            "seq_len": config["dataset"]["seq_len"],
            "label_threshold": config["dataset"]["label_threshold"],
            "train_cutoff": config["data"]["train_cutoff"],
            "val_fraction": config["data"]["val_fraction"],
        }
        data = prepare_dataset(df, **dataset_config)

        print(f"\n[2] Training Transformer...")
        model_config = config["model"]
        model = CryptoTransformer(
            num_features=model_config["num_features"],
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_layers=model_config["num_layers"],
            dim_feedforward=model_config["dim_feedforward"],
            dropout=model_config["dropout"],
            num_classes=model_config["num_classes"],
        )
        print(f"Model parameters: {model.count_parameters():,}")

        train_result = train_model(
            model, data["train"], data["val"],
            config=config["training"],
            device=device,
        )

        print(f"\n[3] Evaluating Transformer...")
        eval_result = evaluate_model(model, data["test"], config["training"]["batch_size"], device)
        print(f"Test accuracy: {eval_result['accuracy']:.4f}")
        print(f"Test F1 (weighted): {eval_result['f1_weighted']:.4f}")
        print(f"Test F1 (macro): {eval_result['f1_macro']:.4f}")
        for cls_name, cls_metrics in eval_result["per_class"].items():
            print(f"  {cls_name}: P={cls_metrics['precision']:.3f} R={cls_metrics['recall']:.3f} F1={cls_metrics['f1']:.3f}")

        print(f"\n[4] Training baselines...")
        train_data = data["train"]
        train_X = np.array([train_data[i][0].numpy() for i in range(len(train_data))])
        train_y = np.array([train_data[i][1].numpy() for i in range(len(train_data))])
        test_data = data["test"]
        test_X = np.array([test_data[i][0].numpy() for i in range(len(test_data))])
        test_y = np.array([test_data[i][1].numpy() for i in range(len(test_data))])

        baselines = {
            "random": RandomBaseline(),
            "majority": MajorityBaseline(),
            "xgboost": XGBoostBaseline(seq_len=dataset_config["seq_len"]),
            "sma_crossover": SMACrossoverBaseline(),
        }

        baseline_results = {}
        for name, baseline in baselines.items():
            print(f"  Training {name}...")
            baseline.fit(train_X, train_y)
            preds = baseline.predict(test_X)
            from crypto_transformer.training.metrics import compute_metrics
            metrics = compute_metrics(test_y, preds)
            baseline_results[name] = metrics
            print(f"    {name}: acc={metrics['accuracy']:.4f} f1={metrics['f1_weighted']:.4f}")

        print(f"\n[5] Backtesting...")
        bt_config = config["backtesting"]
        engine = BacktestEngine(
            initial_capital=bt_config["initial_capital"],
            fee_rate=bt_config["fee_rate"],
            slippage=bt_config["slippage"],
            position_fraction=bt_config["position_fraction"],
        )

        test_close = df["close_raw"].values
        test_start = len(df) - len(test_data) - dataset_config["seq_len"]
        test_close_subset = test_close[test_start + dataset_config["seq_len"]:test_start + dataset_config["seq_len"] + len(test_data)]

        bt_result = engine.run(
            eval_result["predictions"],
            eval_result["probabilities"],
            test_close_subset[:len(eval_result["predictions"])],
            confidence_threshold=bt_config["confidence_threshold"],
        )

        buy_hold_return = (test_close_subset[-1] / test_close_subset[0]) - 1

        print(f"\n  Backtest results:")
        print(f"    Total return: {bt_result.total_return*100:.2f}%")
        print(f"    Buy & Hold:   {buy_hold_return*100:.2f}%")
        print(f"    Sharpe ratio: {bt_result.sharpe_ratio:.2f}")
        print(f"    Max drawdown: {bt_result.max_drawdown*100:.2f}%")
        print(f"    Num trades:   {bt_result.num_trades}")
        print(f"    Win rate:     {bt_result.win_rate*100:.1f}%")

        results[symbol] = {
            "transformer": {
                "accuracy": eval_result["accuracy"],
                "f1_weighted": eval_result["f1_weighted"],
                "f1_macro": eval_result["f1_macro"],
                "per_class": eval_result["per_class"],
                "confusion_matrix": eval_result["confusion_matrix"],
            },
            "baselines": {k: {"accuracy": v["accuracy"], "f1_weighted": v["f1_weighted"]} for k, v in baseline_results.items()},
            "backtest": {
                "total_return": bt_result.total_return,
                "buy_hold_return": float(buy_hold_return),
                "sharpe_ratio": bt_result.sharpe_ratio,
                "max_drawdown": bt_result.max_drawdown,
                "num_trades": bt_result.num_trades,
                "win_rate": bt_result.win_rate,
            },
            "training_history": train_result["history"],
            "data_size": {
                "train": len(data["train"]),
                "val": len(data["val"]),
                "test": len(data["test"]),
                "label_distribution": {k: v.tolist() for k, v in data["label_counts"].items()},
            },
        }

    with open("data/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to data/results.json")


if __name__ == "__main__":
    main()
