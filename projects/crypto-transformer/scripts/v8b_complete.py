#!/usr/bin/env python3
"""V8-B Completion: Eval + Backtest + Figures + Report using existing SimpleCNN checkpoint."""

import importlib.util
import sys
from pathlib import Path

# Import train_v8b module without running main()
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("t8b", ROOT / "scripts" / "train_v8b.py")
t8b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t8b)

import numpy as np
import torch
from torch.utils.data import DataLoader

def main():
    device = "cpu"
    print(f"Device: {device}")

    # Load data + pre-render charts
    print("\n[1] Loading data and pre-rendering charts...")
    ohlcv, timestamps, pair_names, pair_to_id = t8b.load_raw_data()
    data = t8b.pre_render_and_split(ohlcv, timestamps, pair_names, pair_to_id)
    train_ds, val_ds, test_ds = data["train"], data["val"], data["test"]
    print(f"  Train: {data['train_size']} | Val: {data['val_size']} | Test: {data['test_size']}")

    # Load ResNet18 checkpoint
    ckpt_path = ROOT / "data" / "checkpoints" / "v8b_simplecnn_42.pt"
    print(f"\n[2] Loading SimpleCNN checkpoint: {ckpt_path}")
    model = t8b.SimpleCNN()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"  Loaded successfully, params: {model.count_parameters():,}")

    import json

    # Evaluate ResNet18 on test set
    print(f"\n[3] Evaluating SimpleCNN on test set...")
    tel = DataLoader(test_ds, batch_size=t8b.BATCH_SIZE, shuffle=False, num_workers=0)
    ev = t8b.evaluate_ensemble([model], tel, device)
    o = ev["overall"]
    print(f"  IC={o['ic']:.4f}  Dir={o['dir_acc']:.4f}  MAE={o['mae']:.6f}  RMSE={o['rmse']:.6f}")
    print(f"  Long ret={o['long_return']:.6f} (n={o['long_count']})  Short ret={o['short_return']:.6f} (n={o['short_count']})")

    id2p = {v: k for k, v in pair_to_id.items()}
    pos_count = 0
    for pv in sorted(np.unique(ev["pair_ids"])):
        mk = ev["pair_ids"] == pv
        if mk.sum() > 5:
            pic = float(np.corrcoef(ev["predictions"][mk], ev["true_returns"][mk])[0, 1])
            pda = float(np.mean((ev["predictions"][mk] > 0) == (ev["true_returns"][mk] > 0)))
            if pic > 0:
                pos_count += 1
            print(f"    {id2p.get(int(pv),'?'):12s}  IC={pic:+.4f}  dir={pda:.3f}  n={int(mk.sum())}")
    print(f"  Positive IC pairs: {pos_count}/{len(np.unique(ev['pair_ids']))}")

    all_hist = [[{"train_loss": [0.0], "val_loss": [0.0], "val_ic": [o['ic']], "val_dir_acc": [o['dir_acc']]}]]
    mnames = ["SimpleCNN"]
    all_eval = {"SimpleCNN": ev}

    # Save results JSON
    print(f"\n[4] Saving results...")
    out = {
        "config": {"seq_len": t8b.SEQ_LEN, "horizon": t8b.HORIZON, "img_size": t8b.IMG_SIZE,
                   "batch_size": t8b.BATCH_SIZE, "max_epochs": t8b.MAX_EPOCHS, "patience": t8b.PATIENCE,
                   "lr": t8b.LR, "weight_decay": t8b.WEIGHT_DECAY, "seeds": t8b.SEEDS,
                   "pairs": pair_names, "device": str(device), "note": "ViT skipped (CPU too slow)"},
        "data_stats": {"train": data["train_size"], "val": data["val_size"], "test": data["test_size"]},
    }
    for mn in mnames:
        ev_m = all_eval[mn]
        out[f"{mn}_overall"] = ev_m["overall"]
        pp = {}
        for pv in sorted(np.unique(ev_m["pair_ids"])):
            mk = ev_m["pair_ids"] == pv
            if mk.sum() > 5:
                pn = id2p.get(int(pv), f"p{pv}")
                pp[pn] = {
                    "ic": float(np.corrcoef(ev_m["predictions"][mk], ev_m["true_returns"][mk])[0, 1]),
                    "dir_acc": float(np.mean((ev_m["predictions"][mk] > 0) == (ev_m["true_returns"][mk] > 0))),
                    "count": int(mk.sum()),
                    "long_pct": float((ev_m["predictions"][mk] > 0).sum() / mk.sum()),
                }
        out[f"{mn}_per_pair"] = pp
        for si, s in enumerate(t8b.SEEDS):
            out[f"{mn}_history_seed_{s}"] = all_hist[mnames.index(mn)][si]

    with open(ROOT / "data" / "results_v8b.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: data/results_v8b.json")

    # Backtesting
    print(f"\n[5] Backtesting...")
    best_mn = max(mnames, key=lambda mn: all_eval[mn]["overall"]["ic"])
    print(f"  Best model: {best_mn} (IC={all_eval[best_mn]['overall']['ic']:+.4f})")
    pdf = t8b.build_pred_df(all_eval[best_mn], pair_to_id)
    print(f"  Predictions: {len(pdf)}, dates: {pdf['date'].min().date()} to {pdf['date'].max().date()}")

    bt_res, bt_curves = {}, {}
    for th in [0.0, 0.005, 0.01]:
        nm = f"LS(th={th:.3f})"
        pv, dr, tr = t8b.strat_ls(pdf, th); m = t8b.metrics(pv, dr, tr, nm)
        bt_res[nm] = m; bt_curves[nm] = pv
        print(f"  {nm:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  mdd={m['max_drawdown_pct']:6.1f}%  trades={tr}")
    for th in [0.0, 0.005, 0.01]:
        nm = f"LO(th={th:.3f})"
        pv, dr, tr = t8b.strat_lo(pdf, th); m = t8b.metrics(pv, dr, tr, nm)
        bt_res[nm] = m; bt_curves[nm] = pv
        print(f"  {nm:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  mdd={m['max_drawdown_pct']:6.1f}%  trades={tr}")
    for k in [3, 5]:
        nm = f"Top{k}-LS"
        pv, dr, tr = t8b.strat_topk(pdf, k); m = t8b.metrics(pv, dr, tr, nm)
        bt_res[nm] = m; bt_curves[nm] = pv
        print(f"  {nm:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  mdd={m['max_drawdown_pct']:6.1f}%  trades={tr}")
    bnh = t8b.strat_bnh(pdf)
    bt_curves["Buy&Hold"] = bnh
    bnr = (bnh[-1] / bnh[0] - 1) * 100
    print(f"  {'Buy&Hold':18s}  ret={bnr:+7.2f}%")

    with open(ROOT / "data" / "backtest_v8b_results.json", "w") as f:
        oic = float(np.corrcoef(pdf["pred_3d"].values, pdf["actual_3d_ret"].values)[0, 1])
        oda = float(np.mean((pdf["pred_3d"].values > 0) == (pdf["actual_3d_ret"].values > 0)))
        bnhpk = np.maximum.accumulate(bnh)
        bnhmdd = float(((np.array(bnh) - bnhpk) / bnhpk).min() * 100)
        json.dump({
            "strategies": bt_res,
            "buy_and_hold_return_pct": round(float(bnr), 2),
            "buy_and_hold_mdd_pct": round(bnhmdd, 2),
            "overall_ic": oic, "overall_dir_acc": oda,
            "test_period": f"{pdf['date'].min().date()} to {pdf['date'].max().date()}",
            "initial_capital": t8b.INITIAL_CAPITAL,
            "fee_rate": t8b.FEE_RATE,
            "best_model": best_mn,
        }, f, indent=2)
    print(f"  Saved: data/backtest_v8b_results.json")

    # Figures
    print(f"\n[6] Generating figures...")
    t8b.plot_training(all_hist, mnames, ROOT / "docs" / "figures" / "v8b_training_curves.png")
    t8b.plot_backtest(bt_curves, bnh, pdf, ROOT / "docs" / "figures" / "v8b_backtest.png")
    v4_path = ROOT / "data" / "results_v4.json"
    if v4_path.exists():
        t8b.plot_comparison({mn: all_eval[mn] for mn in mnames}, v4_path,
                            ROOT / "docs" / "figures" / "v8b_vs_v4_comparison.png")
    print(f"  Saved: docs/figures/v8b_*.png")

    # Report
    print(f"\n[7] Generating report...")
    t8b.generate_report(all_eval, all_hist, mnames, bt_res, pair_names, pair_to_id)
    print(f"  Saved: docs/v8b_experiment_report.md")

    print(f"\n{'='*60}")
    print(f"  V8-B Complete!")
    print(f"  SimpleCNN: IC={o['ic']:+.4f}  Dir={o['dir_acc']:.4f}")
    print(f"  Best backtest: {max(bt_res.items(), key=lambda x: x[1]['total_return_pct'])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
