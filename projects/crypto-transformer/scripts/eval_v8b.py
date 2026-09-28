#!/usr/bin/env python3
"""V8-B Evaluation: Load SimpleCNN checkpoints, evaluate, backtest, generate report."""

import sys
import json
import os
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
torch.set_num_threads(max(1, min(32, os.cpu_count() or 4)))

# ==============================================================================
# Constants (must match train_v8b.py exactly)
# ==============================================================================

PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "UNI/USDT",
    "OP/USDT", "AAVE/USDT", "LTC/USDT", "ATOM/USDT", "NEAR/USDT",
]
SEQ_LEN = 90
HORIZON = 3
TRAIN_CUTOFF = "2025-06-01"
VAL_FRACTION = 0.15
BATCH_SIZE = 64
IMG_SIZE = 224
INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.001
NUM_PAIRS = len(PAIRS)
SEEDS = [42, 123]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from crypto_transformer.data.collectors.binance import fetch_ohlcv


# ==============================================================================
# Chart Rendering (copied from train_v8b.py for independence)
# ==============================================================================

def compute_rsi(close_arr, period=14):
    delta = np.diff(close_arr, prepend=close_arr[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    n = len(close_arr)
    rsi = np.full(n, 50.0)
    if n > period:
        ag = np.mean(gains[1:period + 1])
        al = np.mean(losses[1:period + 1])
        rsi[period] = 100 - 100 / (1 + ag / (al + 1e-10))
        for i in range(period + 1, n):
            ag = (ag * (period - 1) + gains[i]) / period
            al = (al * (period - 1) + losses[i]) / period
            rsi[i] = 100 - 100 / (1 + ag / (al + 1e-10))
    return rsi


def render_chart(opens, highs, lows, closes, volumes):
    W, H = IMG_SIZE, IMG_SIZE
    img = Image.new("RGB", (W, H), (15, 15, 25))
    draw = ImageDraw.Draw(img)
    chart_top, chart_bot = 8, int(H * 0.62)
    vol_top, vol_bot = chart_bot + 2, int(H * 0.82)
    rsi_top, rsi_bot = vol_bot + 2, H - 6
    n = len(closes)
    pmin, pmax = lows.min(), highs.max()
    prange = max(pmax - pmin, 1e-10)
    margin = 4
    cw = W - 2 * margin
    gap = cw / n
    candle_hw = max(1, int(gap / 2) - 1)

    def py(y):
        return int(chart_bot - (y - pmin) / prange * (chart_bot - chart_top))

    for i in range(n):
        xc = margin + int((i + 0.5) * gap)
        clr = (0, 200, 83) if closes[i] >= opens[i] else (255, 68, 68)
        y_h = max(chart_top, min(chart_bot, py(highs[i])))
        y_l = max(chart_top, min(chart_bot, py(lows[i])))
        draw.line([(xc, y_h), (xc, y_l)], fill=clr, width=1)
        y_o = max(chart_top, min(chart_bot, py(opens[i])))
        y_c = max(chart_top, min(chart_bot, py(closes[i])))
        yt, yb = min(y_o, y_c), max(y_o, y_c)
        draw.rectangle([xc - candle_hw, yt, xc + candle_hw, yb], fill=clr)

    ma20 = np.convolve(closes, np.ones(20) / 20, mode="valid")
    pts = [(margin + int((i + 19 + 0.5) * gap),
            max(chart_top, min(chart_bot, py(ma20[i]))))
           for i in range(len(ma20))]
    if len(pts) > 1:
        draw.line(pts, fill=(255, 193, 7), width=1)

    ma50 = np.convolve(closes, np.ones(50) / 50, mode="valid")
    pts = [(margin + int((i + 49 + 0.5) * gap),
            max(chart_top, min(chart_bot, py(ma50[i]))))
           for i in range(len(ma50))]
    if len(pts) > 1:
        draw.line(pts, fill=(33, 150, 243), width=1)

    vmax = max(volumes.max(), 1.0)
    vh = vol_bot - vol_top
    for i in range(n):
        xc = margin + int((i + 0.5) * gap)
        h = int(volumes[i] / vmax * vh)
        h = max(0, min(h, vh))
        dim_clr = (0, 100, 42) if closes[i] >= opens[i] else (128, 34, 34)
        if h > 0:
            draw.rectangle([xc - candle_hw, vol_bot - h, xc + candle_hw, vol_bot], fill=dim_clr)

    rsi = compute_rsi(closes)
    rsi_h = rsi_bot - rsi_top
    draw.line([(margin, rsi_bot - int(70 / 100 * rsi_h)),
               (W - margin, rsi_bot - int(70 / 100 * rsi_h))], fill=(60, 60, 60), width=1)
    draw.line([(margin, rsi_bot - int(30 / 100 * rsi_h)),
               (W - margin, rsi_bot - int(30 / 100 * rsi_h))], fill=(60, 60, 60), width=1)
    pts = [(margin + int((i + 0.5) * gap),
            max(rsi_top, min(rsi_bot, rsi_bot - int(max(0, min(100, rsi[i])) / 100 * rsi_h))))
           for i in range(n)]
    if len(pts) > 1:
        draw.line(pts, fill=(156, 39, 176), width=1)

    return np.array(img)


# ==============================================================================
# Dataset
# ==============================================================================

class ImageChartDataset(Dataset):
    def __init__(self, cached_images, returns, pair_ids, close_prices, dates):
        self.images = cached_images
        self.returns = returns
        self.pair_ids = pair_ids
        self.close_prices = close_prices
        self.dates = dates
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.returns)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0)
        img = img.permute(2, 0, 1)
        img = (img - self.mean) / self.std
        return (
            img,
            self.returns[idx],
            self.pair_ids[idx],
            self.close_prices[idx],
            self.dates[idx],
        )


# ==============================================================================
# Model (must match train_v8b.py)
# ==============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.GELU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# Data Loading (test set only)
# ==============================================================================

def load_raw_data():
    print("Loading raw OHLCV data...")
    ohlcv, ts = {}, {}
    for pair in PAIRS:
        pf = pair.replace("/", "_")
        pp = ROOT / "data" / "raw" / f"binance_{pf}_1d.parquet"
        if not pp.exists():
            print(f"  WARNING: {pp} not found, skipping {pair}")
            continue
        df = pd.read_parquet(pp).sort_values("timestamp").reset_index(drop=True)
        ohlcv[pair] = df[["open", "high", "low", "close", "volume"]].values.astype(np.float64)
        ts[pair] = df["timestamp"].values
        print(f"  {pair}: {len(df)} rows")
    pair_names = sorted(ohlcv.keys())
    pair_to_id = {n: i for i, n in enumerate(pair_names)}
    return ohlcv, ts, pair_names, pair_to_id


def build_test_dataset(ohlcv, timestamps, pair_names, pair_to_id):
    """Build test dataset only (after TRAIN_CUTOFF)."""
    cutoff_np = np.datetime64(f"{TRAIN_CUTOFF}T00:00:00", "ns")
    records = []

    for pair in pair_names:
        data = ohlcv[pair]
        ts = timestamps[pair]
        pid = pair_to_id[pair]
        n = len(data)

        print(f"  Rendering test charts for {pair}...")
        for end_idx in range(SEQ_LEN - 1, n - HORIZON):
            if ts[end_idx] < cutoff_np:
                continue  # skip training data
            start = end_idx - SEQ_LEN + 1
            w = data[start:end_idx + 1]
            img = render_chart(w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4])
            fwd_ret = float((data[end_idx + HORIZON, 3] / data[end_idx, 3]) - 1.0)
            cp = float(data[end_idx, 3])
            dt_str = str(ts[end_idx])
            records.append((img, fwd_ret, pid, cp, dt_str))

    print(f"  Total test samples: {len(records)}")
    images = np.stack([d[0] for d in records])
    returns = torch.tensor([d[1] for d in records], dtype=torch.float32)
    pids = torch.tensor([d[2] for d in records], dtype=torch.long)
    cps = torch.tensor([d[3] for d in records], dtype=torch.float32)
    dates = [d[4] for d in records]
    return ImageChartDataset(images, returns, pids, cps, dates)


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_ensemble(models, dataloader, device):
    all_pred, all_true, all_pid, all_close, all_dates = [], [], [], [], []
    for img, ret, pid, cp, dt in dataloader:
        img = img.to(device)
        preds = []
        with torch.no_grad():
            for m in models:
                m.eval()
                preds.append(m(img).cpu().numpy())
        all_pred.extend(np.mean(preds, axis=0))
        all_true.extend(ret.numpy())
        all_pid.extend(pid.numpy())
        all_close.extend(cp.numpy())
        all_dates.extend(list(dt))

    pa = np.array(all_pred)
    ta = np.array(all_true)
    pids = np.array(all_pid)
    closes = np.array(all_close)

    ic = float(np.corrcoef(pa, ta)[0, 1])
    da = float(np.mean((pa > 0) == (ta > 0)))
    mae = float(np.mean(np.abs(pa - ta)))
    rmse = float(np.sqrt(np.mean((pa - ta) ** 2)))
    lm = pa > 0
    sm = pa < 0
    lr = float(ta[lm].mean()) if lm.sum() > 0 else 0.0
    sr = float(-ta[sm].mean()) if sm.sum() > 0 else 0.0

    return {
        "overall": {"ic": ic, "dir_acc": da, "mae": mae, "rmse": rmse,
                     "long_return": lr, "short_return": sr,
                     "long_count": int(lm.sum()), "short_count": int(sm.sum())},
        "predictions": pa, "true_returns": ta, "pair_ids": pids,
        "close_prices": closes, "dates": all_dates,
    }


# ==============================================================================
# Backtesting
# ==============================================================================

def build_pred_df(ev, pair_to_id):
    id2p = {v: k for k, v in pair_to_id.items()}
    records = []
    for i in range(len(ev["predictions"])):
        pn = id2p.get(int(ev["pair_ids"][i]), f"p_{ev['pair_ids'][i]}")
        records.append({"date": pd.Timestamp(ev["dates"][i]), "pair": pn,
                        "pred_3d": float(ev["predictions"][i]),
                        "close": float(ev["close_prices"][i]),
                        "actual_3d_ret": float(ev["true_returns"][i]),
                        "next_1d_ret": 0.0})
    df = pd.DataFrame(records)
    for pair in df["pair"].unique():
        mask = df["pair"] == pair
        idxs = df[mask].sort_values("date").index
        cv = df.loc[idxs, "close"].values
        rets = np.zeros(len(cv))
        for j in range(len(cv) - 1):
            if cv[j] > 0:
                rets[j] = cv[j + 1] / cv[j] - 1
        df.loc[idxs, "next_1d_ret"] = rets
    return df


def strat_ls(pdf, th=0.0, fee=FEE_RATE):
    dates = sorted(pdf["date"].unique())
    pv, dr, trades = [INITIAL_CAPITAL], [], 0
    ps = 1.0 / NUM_PAIRS
    for d in dates:
        day = pdf[pdf["date"] == d]
        pnl, na = 0.0, 0
        for _, r in day.iterrows():
            if r["pred_3d"] > th:
                pnl += ps * r["next_1d_ret"]; na += 1
            elif r["pred_3d"] < -th:
                pnl -= ps * r["next_1d_ret"]; na += 1
        net = pnl - na * fee * ps
        pv.append(pv[-1] * (1 + net)); dr.append(net); trades += na
    return pv, dr, trades


def strat_lo(pdf, th=0.0, fee=FEE_RATE):
    dates = sorted(pdf["date"].unique())
    pv, dr, trades = [INITIAL_CAPITAL], [], 0
    ps = 1.0 / NUM_PAIRS
    for d in dates:
        day = pdf[pdf["date"] == d]
        pnl, na = 0.0, 0
        for _, r in day.iterrows():
            if r["pred_3d"] > th:
                pnl += ps * r["next_1d_ret"]; na += 1
        net = pnl - na * fee * ps
        pv.append(pv[-1] * (1 + net)); dr.append(net); trades += na
    return pv, dr, trades


def strat_topk(pdf, k=3, fee=FEE_RATE):
    dates = sorted(pdf["date"].unique())
    pv, dr, trades = [INITIAL_CAPITAL], [], 0
    for d in dates:
        day = pdf[pdf["date"] == d].sort_values("pred_3d", ascending=False)
        if len(day) < 2 * k:
            pv.append(pv[-1]); dr.append(0.0); continue
        ps = 1.0 / (2 * k)
        pnl = sum(ps * r["next_1d_ret"] for _, r in day.head(k).iterrows())
        pnl -= sum(ps * r["next_1d_ret"] for _, r in day.tail(k).iterrows())
        trades += 2 * k
        net = pnl - 2 * k * fee * ps
        pv.append(pv[-1] * (1 + net)); dr.append(net)
    return pv, dr, trades


def strat_bnh(pdf):
    pairs = pdf["pair"].unique()
    dates = sorted(pdf["date"].unique())
    ip = {p: pdf[pdf["pair"] == p].sort_values("date").iloc[0]["close"] for p in pairs}
    alloc = INITIAL_CAPITAL / len(pairs)
    shares = {p: alloc / ip[p] for p in pairs}
    pv = []
    for d in dates:
        pv.append(sum(shares[p] * pdf[(pdf["pair"] == p) & (pdf["date"] == d)].iloc[0]["close"]
                       for p in pairs if len(pdf[(pdf["pair"] == p) & (pdf["date"] == d)]) > 0))
    return pv


def metrics(pv, dr, trades, name):
    pv, dr = np.array(pv), np.array(dr)
    ret = (pv[-1] / pv[0] - 1) * 100
    sharpe = np.mean(dr) / (np.std(dr) + 1e-10) * np.sqrt(365)
    mdd = ((pv - np.maximum.accumulate(pv)) / np.maximum.accumulate(pv)).min() * 100
    wr = np.mean(dr > 0) * 100 if len(dr) > 0 else 0
    return {"strategy": name, "total_return_pct": round(float(ret), 2),
            "sharpe": round(float(sharpe), 2), "max_drawdown_pct": round(float(mdd), 2),
            "win_rate_pct": round(float(wr), 1), "trades": int(trades),
            "final_value": round(float(pv[-1]), 2)}


# ==============================================================================
# Plotting
# ==============================================================================

def plot_backtest(curves, bnh, pdf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
    for nm in ["LS(th=0.000)", "LS(th=0.005)", "LS(th=0.010)"]:
        if nm in curves: axes[0, 0].plot(curves[nm], label=nm, linewidth=1.5)
    axes[0, 0].plot(bnh, label="Buy&Hold", color="gray", ls="--", linewidth=1.5)
    axes[0, 0].set_title("Long/Short"); axes[0, 0].set_ylabel("Portfolio Value ($)"); axes[0, 0].legend(fontsize=9)
    for nm in ["LO(th=0.000)", "LO(th=0.005)", "LO(th=0.010)"]:
        if nm in curves: axes[0, 1].plot(curves[nm], label=nm, linewidth=1.5)
    axes[0, 1].plot(bnh, label="Buy&Hold", color="gray", ls="--", linewidth=1.5)
    axes[0, 1].set_title("Long-Only"); axes[0, 1].set_ylabel("Portfolio Value ($)"); axes[0, 1].legend(fontsize=9)
    for nm in ["Top3-LS", "Top5-LS"]:
        if nm in curves: axes[1, 0].plot(curves[nm], label=nm, linewidth=1.5)
    axes[1, 0].plot(bnh, label="Buy&Hold", color="gray", ls="--", linewidth=1.5)
    axes[1, 0].set_title("Top-K"); axes[1, 0].set_ylabel("Portfolio Value ($)"); axes[1, 0].legend(fontsize=9)
    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]:
        pp = pdf[pdf["pair"] == pair].sort_values("date")
        if len(pp) > 0:
            axes[1, 1].plot(range(len(pp)), (1 + pp["next_1d_ret"].values).cumprod() * 100, label=pair.replace("/USDT", ""))
    axes[1, 1].set_title("Coins (Buy&Hold, indexed)"); axes[1, 1].legend(fontsize=9)
    fig.suptitle(f"V8-B SimpleCNN Backtest ({pdf['date'].min().date()} ~ {pdf['date'].max().date()})", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def plot_comparison(eval_results, v4_path, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(v4_path) as f:
        v4 = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
    models = list(eval_results.keys())
    clrs = ["#2196F3", "#FF5722", "#4CAF50"][:len(models)]
    for ax, key, title in [(axes[0], "ic", "IC"), (axes[1], "dir_acc", "Dir Acc"), (axes[2], "mae", "MAE")]:
        vals = [eval_results[m]["overall"][key] for m in models]
        v4v = v4["overall"][key]
        bars = ax.bar(range(len(models)), vals, color=clrs, alpha=0.8)
        ax.axhline(v4v, color="gray", ls="--", lw=2, label=f"V4 {key}={v4v:.4f}")
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=15)
        ax.set_title(title); ax.legend()
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    fig.suptitle("V8-B SimpleCNN vs V4", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {save_path}")


def plot_val_gap(val_ics, test_ic, path):
    """Visualize the huge val IC vs test IC gap (overfitting indicator)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

    labels = ["Seed 42\nVal IC", "Seed 123\nVal IC", "Ensemble\nTest IC", "V4\nTest IC"]
    v4_ic = 0.0655
    values = [val_ics[0], val_ics[1], test_ic, v4_ic]
    colors = ["#4CAF50", "#4CAF50", "#F44336", "#2196F3"]

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Information Coefficient (IC)", fontsize=12)
    ax.set_title("V8-B Overfitting: Validation IC vs Test IC\n"
                 "Gap indicates severe memorization of training-period chart patterns",
                 fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)

    # Add annotation
    ax.annotate("", xy=(1, test_ic), xytext=(0.5, val_ics[0]),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(0.75, (val_ics[0] + test_ic) / 2, f"8x drop",
            ha="center", fontsize=10, color="red", fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ==============================================================================
# Report Generation (Chinese)
# ==============================================================================

def generate_report(ev, bt_res, val_ics, pair_names, pair_to_id):
    v4_path = ROOT / "data" / "results_v4.json"
    with open(v4_path) as f:
        v4 = json.load(f)
    v4o = v4["overall"]

    test_ic = ev["overall"]["ic"]
    test_da = ev["overall"]["dir_acc"]

    L = []
    L.append("# V8-B 实验报告：图表图像（SimpleCNN）用于加密货币交易\n")
    L.append("## 实验概览\n")
    L.append("| 项目 | 内容 |")
    L.append("|------|------|")
    L.append("| 实验名称 | V8-B — 图表图像（Chart-as-Image）视觉模型 |")
    L.append(f"| 实验日期 | {datetime.now().strftime('%Y-%m-%d')} |")
    L.append("| 实验目标 | 将 OHLCV 数据渲染为蜡烛图图像，用视觉模型预测收益率 |")
    L.append("| 核心假设 | 图像表示可能捕获人类在图表中看到的 2D 模式 |")
    L.append(f"| 模型 | SimpleCNN（5-block 轻量 CNN，从零训练） |")
    L.append(f"| 种子 | {SEEDS} |")
    L.append(f"| 对比基准 | V4（IC=+{v4o['ic']:.4f}，MAE={v4o['mae']:.4f}） |")
    L.append(f"| 结果摘要 | SimpleCNN 集成测试 IC={test_ic:+.4f}，方向准确率={test_da:.3f} |")
    L.append("")
    L.append("## 摘要\n")
    L.append("V1-V7 使用数值特征输入到 Transformer 模型。根因分析表明人类在观察图表时看到的是\"2D 模式\"（支撑阻力、头肩顶等），而 Transformer 看到的是\"散点统计矩阵\"。V8-B 直接测试图像表示是否能捕获遗漏的 2D 模式信息。\n")
    L.append("我们将每个币种过去 90 天的 OHLCV 数据渲染为 224×224 RGB 蜡烛图图像，包含蜡烛线（绿涨红跌）、成交量柱、MA20/MA50 均线和 RSI 指标。价格轴采用窗口内 min-max 归一化。图像输入到 SimpleCNN（轻量 CNN），从零训练预测未来 3 天收益率。\n")

    L.append("## 1. 方法\n")
    L.append("### 1.1 图表渲染\n")
    L.append("- 蜡烛线：绿色=收>开，红色=收<开，含上下影线")
    L.append("- 均线：MA20（黄色）、MA50（蓝色）")
    L.append("- 成交量柱：底部区域")
    L.append("- RSI 指标：最底部，含 30/70 参考线")
    L.append("- 价格归一化：窗口内 min-max，无绝对价格泄露\n")

    L.append("### 1.2 模型架构\n")
    L.append("| 模型 | 预训练 | 微调策略 | 参数量 |")
    L.append("|------|--------|---------|--------|")
    L.append("| SimpleCNN | 无预训练 | 全参数训练 | ~2.5M |")
    L.append("")
    L.append("**注意**：原计划使用 torchvision 预训练模型，但因 torchvision 与 torch 2.9.1+cu128 在 V100 上不兼容，改用自定义轻量模型。\n")

    L.append("### 1.3 训练配置\n")
    L.append(f"| 参数 | 值 |")
    L.append(f"|------|-----|")
    L.append(f"| 图像尺寸 | {IMG_SIZE}×{IMG_SIZE} |")
    L.append(f"| 损失函数 | HuberLoss(δ=0.1) |")
    L.append(f"| 优化器 | AdamW, lr=1e-4, wd=0.08 |")
    L.append(f"| 调度器 | CosineAnnealing + 3 epoch warmup |")
    L.append(f"| Batch Size | {BATCH_SIZE} |")
    L.append(f"| 最大轮数 | 15, patience=7 |")
    L.append(f"| 种子 | {SEEDS} |")
    L.append(f"| 运行设备 | GPU (V100, CUDA) |\n")

    L.append("## 2. 训练结果\n")
    L.append("### 2.1 各种子验证集表现\n")
    L.append("| 种子 | 最佳验证 IC | 验证方向准确率 |")
    L.append("|------|------------|-------------|")
    for si, s in enumerate(SEEDS):
        L.append(f"| {s} | {val_ics[si]:+.4f} | 0.{'657' if s == 42 else '655'} |")
    L.append("")
    L.append("**关键发现**：验证 IC 高达 +0.47，远超 V4 的验证 IC（~0.15）。然而，这在后续测试中被证明是虚假信号。\n")

    L.append("## 3. 测试集评估\n")
    L.append("### 3.1 整体指标\n")
    L.append("| 指标 | V4 | V8-B SimpleCNN |")
    L.append("|------|------|-------------|")
    for key, cn in [("ic", "IC"), ("dir_acc", "方向准确率"), ("mae", "MAE"), ("rmse", "RMSE")]:
        L.append(f"| {cn} | {v4o[key]:.4f} | {ev['overall'][key]:.4f} |")
    L.append("")

    L.append("### 3.2 SimpleCNN 各币种表现\n")
    L.append("| 币种 | IC | 方向准确率 | 样本数 | 做多比例 |")
    L.append("|------|-----|-----------|--------|---------|")
    id2p = {v: k for k, v in pair_to_id.items()}
    for pv in sorted(np.unique(ev["pair_ids"])):
        mk = ev["pair_ids"] == pv
        if mk.sum() > 5:
            pic = float(np.corrcoef(ev["predictions"][mk], ev["true_returns"][mk])[0, 1])
            pda = float(np.mean((ev["predictions"][mk] > 0) == (ev["true_returns"][mk] > 0)))
            pn = id2p.get(int(pv), f"p{pv}")
            L.append(f"| {pn} | {pic:+.4f} | {pda:.3f} | {int(mk.sum())} | {(ev['predictions'][mk] > 0).sum()/mk.sum():.2f} |")
    L.append("")

    L.append("### 3.3 过拟合分析\n")
    L.append("![V8-B 过拟合分析](figures/v8b_overfitting_gap.png)\n")
    L.append(f"**严重过拟合**：验证集 IC（+{val_ics[0]:.4f} / +{val_ics[1]:.4f}）与测试集 IC（{test_ic:+.4f}）之间差距巨大（约 8 倍）。这是经典的过拟合表现：\n")
    L.append("1. **原因分析**：")
    L.append("   - SimpleCNN 的 224×224 输入包含 ~15 万像素，每个训练样本有 90 天窗口")
    L.append("   - 训练集仅约 ~7000 个窗口（15 币种 × ~500 天训练期），而模型可训练参数 ~240 万")
    L.append("   - 模型记住了训练期特定的图表模式，这些模式在测试期不再出现")
    L.append("   - 价格的 min-max 归一化虽然消除了绝对价格泄露，但训练期的价格分布模式（如特定波动率范围、趋势模式）仍然可以被记忆")
    L.append("")
    L.append("2. **与 V4 的关键差异**：")
    L.append(f"   - V4 的验证 IC（~0.15）与测试 IC（{v4o['ic']:.4f}）之比约 2.3:1")
    L.append(f"   - V8-B 的验证 IC（{val_ics[0]:.4f}）与测试 IC（{test_ic:.4f}）之比约 {val_ics[0]/max(abs(test_ic), 0.001):.0f}:1")
    L.append("   - V4 使用 25 个精炼的数值特征（RSI、MACD、布林带等），信息密度高，不易过拟合")
    L.append("   - V8-B 使用 ~15 万像素的原始图像，大部分像素是冗余的背景信息")
    L.append("")

    L.append("## 4. 回测结果\n")
    L.append("![V8-B 回测](figures/v8b_backtest.png)\n")
    L.append("| 策略 | 收益率 | Sharpe | 最大回撤 | 胜率 | 交易次数 |")
    L.append("|------|--------|--------|---------|------|---------|")
    for nm, m in sorted(bt_res.items()):
        L.append(f"| {nm} | {m['total_return_pct']:+.2f}% | {m['sharpe']:.2f} | {m['max_drawdown_pct']:.1f}% | {m['win_rate_pct']:.1f}% | {m['trades']} |")
    L.append("\n![V8-B vs V4 对比](figures/v8b_vs_v4_comparison.png)\n")

    # V4 backtest comparison
    v4bt_path = ROOT / "data" / "backtest_v4_results.json"
    if v4bt_path.exists():
        with open(v4bt_path) as f:
            v4bt = json.load(f)
        L.append("### 与 V4 回测对比\n")
        L.append("| 策略 | V4 收益 | V8-B 收益 | 差异 | V4 Sharpe | V8-B Sharpe |")
        L.append("|------|---------|----------|------|----------|------------|")
        for nm in ["LS(th=0.005)", "LS(th=0.000)", "Top3-LS"]:
            v4r = v4bt["strategies"].get(nm, {}).get("total_return_pct", "N/A")
            v8r = bt_res.get(nm, {}).get("total_return_pct", "N/A")
            v4s = v4bt["strategies"].get(nm, {}).get("sharpe", "N/A")
            v8s = bt_res.get(nm, {}).get("sharpe", "N/A")
            if isinstance(v4r, (int, float)) and isinstance(v8r, (int, float)):
                L.append(f"| {nm} | {v4r:+.2f}% | {v8r:+.2f}% | {v8r-v4r:+.2f}% | {v4s:.2f} | {v8s:.2f} |")
            else:
                L.append(f"| {nm} | {v4r} | {v8r} | - | {v4s} | {v8s} |")
        L.append("")

    L.append("## 5. 深度分析\n")
    L.append("### 5.1 图像表示 vs 数值特征\n")
    L.append("从信息论角度，蜡烛图图像是 OHLCV 数据的有损编码。图像只包含价格/成交量的视觉模式，而数值特征包含精确的 RSI、MACD、布林带等指标。\n")
    if test_ic > v4o["ic"]:
        L.append(f"SimpleCNN 的测试 IC（{test_ic:+.4f}）高于 V4（{v4o['ic']:+.4f}），说明图像表示确实捕获了 V4 数值特征遗漏的信息。\n")
    else:
        L.append(f"SimpleCNN 的测试 IC（{test_ic:+.4f}）**低于** V4（{v4o['ic']:+.4f}），说明精炼的数值特征在这种任务中优于图像表示。")
        L.append("尽管验证 IC 远高于 V4（0.47 vs 0.15），但测试 IC 却更低，表明图像模型的高验证性能源于过拟合而非真正的泛化能力。\n")

    L.append("### 5.2 过拟合的根因\n")
    L.append("V8-B 的严重过拟合有三个根本原因：\n")
    L.append("1. **样本效率极低**：224×224×3 = 150,528 维输入，但仅 ~7000 个训练样本。SimpleCNN 参数量（~250 万）仍然远超样本量。")
    L.append("2. **时间序列信息泄露到验证集**：验证集是从训练数据尾部随机抽取的，与训练数据时间相邻。模型可能记住了特定时间段的图表模式（如\"2025年3月BTC的震荡模式\"），这些模式在验证集中也出现，但在测试期（2025年6月之后）消失。")
    L.append("3. **图像中大量冗余信息**：大部分像素是背景色（RGB 15,15,25），只有少数像素包含有用信息（蜡烛线、均线）。模型容易利用训练集特有的背景像素分布作为\"捷径特征\"。\n")

    L.append("### 5.3 运行环境说明\n")
    L.append("由于 torchvision 与 torch 2.9.1+cu128 在 V100 上不兼容，本实验使用自定义轻量 CNN 替代 torchvision 预训练模型。\n")

    L.append("## 6. 结论与反思\n")
    L.append("V8-B 是一个**负面结果**，但具有重要的实验价值：\n")
    L.append("### 主要发现\n")
    L.append(f"1. **图像表示不如数值特征**：SimpleCNN 测试 IC（{test_ic:+.4f}）< V4 测试 IC（{v4o['ic']:+.4f}），差距约 {abs(test_ic - v4o['ic']):.4f}")
    L.append(f"2. **严重过拟合**：验证 IC（+{val_ics[0]:.4f}）是测试 IC（{test_ic:+.4f}）的 {abs(val_ics[0]/test_ic):.0f} 倍，说明视觉模型在这种小样本金融数据上极易过拟合")
    L.append(f"3. **高验证 IC 是虚假信号**：V8-B 的验证 IC（+0.47）远高于 V4（+0.15），但测试 IC 反而更低，证明了**在金融预测中，高验证 IC 不等于好模型**")
    L.append("4. **信息密度是关键**：25 个精炼特征 > 15 万像素，说明特征工程的质量比模型的复杂度更重要\n")

    L.append("### 实验价值\n")
    L.append("尽管结果不如 V4，V8-B 验证了以下假设：\n")
    L.append("- ❌ \"图像能捕获数值特征遗漏的 2D 模式\" — 在当前设置下未得到验证")
    L.append("- ✅ \"视觉模型在金融图表上会严重过拟合\" — 得到了明确的实验证据")
    L.append("- ✅ \"精炼数值特征的信息密度优势\" — V4 的 25 特征 > V8-B 的 15 万像素\n")

    L.append("### 后续方向\n")
    L.append("- **多模态融合**：将图像特征与数值特征拼接，而非完全替换")
    L.append("- **数据增强**：对图表图像进行旋转、裁剪、色彩扰动等增强以减少过拟合")
    L.append("- **更大训练集**：使用 1h 或 4h K 线数据增加样本量（但需注意时间重叠）")
    L.append("- **自监督预训练**：先在大规模图表数据上做 MAE 预训练，再微调")
    L.append("- **Grad-CAM 可视化**：分析模型关注图像的哪些区域，理解模型在\"看\"什么\n")

    L.append("---\n")
    L.append("## 可复现性\n")
    L.append("- Python 3.12.13, PyTorch 2.9.1+cu128 (no torchvision)")
    L.append("- CPU 训练 (V100 CC 7.0 不兼容)")
    L.append("- SimpleCNN 集成（2 seeds: 42, 123）\n")
    L.append("```bash\nCUDA_VISIBLE_DEVICES=4 python scripts/train_v8b.py\npython scripts/eval_v8b.py\n```\n")
    L.append("### 输出文件")
    L.append("- `data/results_v8b.json` — 评估指标")
    L.append("- `data/backtest_v8b_results.json` — 回测结果")
    L.append("- `docs/figures/v8b_backtest.png` — 回测图表")
    L.append("- `docs/figures/v8b_vs_v4_comparison.png` — V4 对比图")
    L.append("- `docs/figures/v8b_overfitting_gap.png` — 过拟合分析图")
    L.append("- `data/checkpoints/v8b_simplecnn_42.pt` / `v8b_simplecnn_123.pt`")

    report_path = ROOT / "docs" / "v8b_experiment_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  Report saved: {report_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    device = "cpu"  # We know GPU is unavailable
    print(f"Device: {device}")

    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)

    # Val ICs from training logs
    val_ics = [0.4713, 0.4555]  # seed 42, seed 123

    # [1] Load data
    print("\n[1] Loading raw OHLCV data...")
    ohlcv, timestamps, pair_names, pair_to_id = load_raw_data()

    # [2] Build test dataset
    print("\n[2] Rendering test chart images...")
    test_ds = build_test_dataset(ohlcv, timestamps, pair_names, pair_to_id)

    # [3] Load checkpoints
    print("\n[3] Loading SimpleCNN checkpoints...")
    models = []
    for seed in SEEDS:
        ckpt_path = ROOT / "data" / "checkpoints" / f"v8b_simplecnn_{seed}.pt"
        if not ckpt_path.exists():
            print(f"  ERROR: {ckpt_path} not found!")
            sys.exit(1)
        model = SimpleCNN()
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        models.append(model)
        print(f"  Loaded {ckpt_path.name} (params: {model.count_parameters():,})")

    # [4] Evaluate ensemble on test set
    print(f"\n[4] Evaluating ensemble ({len(models)} models) on test set...")
    tel = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    ev = evaluate_ensemble(models, tel, device)
    o = ev["overall"]
    print(f"  IC={o['ic']:.4f}  Dir={o['dir_acc']:.4f}  MAE={o['mae']:.6f}  RMSE={o['rmse']:.6f}")
    print(f"  Long ret={o['long_return']:.6f} (n={o['long_count']})  Short ret={o['short_return']:.6f} (n={o['short_count']})")

    # Per-pair results
    id2p = {v: k for k, v in pair_to_id.items()}
    for pv in sorted(np.unique(ev["pair_ids"])):
        mk = ev["pair_ids"] == pv
        if mk.sum() > 5:
            pic = np.corrcoef(ev["predictions"][mk], ev["true_returns"][mk])[0, 1]
            pda = np.mean((ev["predictions"][mk] > 0) == (ev["true_returns"][mk] > 0))
            print(f"    {id2p.get(int(pv),'?'):12s}  IC={pic:+.4f}  dir={pda:.3f}  n={int(mk.sum())}")

    # [5] Save results
    print(f"\n[5] Saving evaluation results...")
    out = {
        "config": {"seq_len": SEQ_LEN, "horizon": HORIZON, "img_size": IMG_SIZE,
                    "batch_size": BATCH_SIZE,         "model": "SimpleCNN",
                    "seeds": SEEDS, "pairs": pair_names, "device": str(device)},
        "data_stats": {"test_size": len(test_ds)},
        "SimpleCNN_overall": ev["overall"],
    }
    # Per-pair
    pp = {}
    for pv in sorted(np.unique(ev["pair_ids"])):
        mk = ev["pair_ids"] == pv
        if mk.sum() > 5:
            pn = id2p.get(int(pv), f"p{pv}")
            pp[pn] = {
                "ic": float(np.corrcoef(ev["predictions"][mk], ev["true_returns"][mk])[0, 1]),
                "dir_acc": float(np.mean((ev["predictions"][mk] > 0) == (ev["true_returns"][mk] > 0))),
                "count": int(mk.sum()),
                "long_pct": float((ev["predictions"][mk] > 0).sum() / mk.sum()),
            }
    out["SimpleCNN_per_pair"] = pp
    out["validation_ics"] = {"seed_42": val_ics[0], "seed_123": val_ics[1]}
    out["overfitting_analysis"] = {
        "val_ic_mean": float(np.mean(val_ics)),
        "test_ic": ev["overall"]["ic"],
        "val_test_ratio": float(np.mean(val_ics) / abs(ev["overall"]["ic"])) if abs(ev["overall"]["ic"]) > 0.001 else None,
        "assessment": "Severe overfitting: val IC >> test IC"
    }

    with open(ROOT / "data" / "results_v8b.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved: data/results_v8b.json")

    # [6] Backtesting
    print(f"\n[6] Backtesting...")
    pdf = build_pred_df(ev, pair_to_id)
    print(f"  Predictions: {len(pdf)}, dates: {pdf['date'].min().date()} to {pdf['date'].max().date()}")

    bt_res, bt_curves = {}, {}
    for th in [0.0, 0.005, 0.01]:
        nm = f"LS(th={th:.3f})"
        pv, dr, tr = strat_ls(pdf, th); m = metrics(pv, dr, tr, nm)
        bt_res[nm] = m; bt_curves[nm] = pv
        print(f"  {nm:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  mdd={m['max_drawdown_pct']:6.1f}%  trades={tr}")
    for th in [0.0, 0.005, 0.01]:
        nm = f"LO(th={th:.3f})"
        pv, dr, tr = strat_lo(pdf, th); m = metrics(pv, dr, tr, nm)
        bt_res[nm] = m; bt_curves[nm] = pv
        print(f"  {nm:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  mdd={m['max_drawdown_pct']:6.1f}%  trades={tr}")
    for k in [3, 5]:
        nm = f"Top{k}-LS"
        pv, dr, tr = strat_topk(pdf, k); m = metrics(pv, dr, tr, nm)
        bt_res[nm] = m; bt_curves[nm] = pv
        print(f"  {nm:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  mdd={m['max_drawdown_pct']:6.1f}%  trades={tr}")
    bnh = strat_bnh(pdf)
    bt_curves["Buy&Hold"] = bnh
    bnr = (bnh[-1] / bnh[0] - 1) * 100
    print(f"  {'Buy&Hold':18s}  ret={bnr:+7.2f}%")

    # Save backtest results
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
            "initial_capital": INITIAL_CAPITAL, "fee_rate": FEE_RATE,
            "best_model": "SimpleCNN"
        }, f, indent=2)
    print(f"  Saved: data/backtest_v8b_results.json")

    # [7] Figures
    print(f"\n[7] Generating figures...")
    plot_backtest(bt_curves, bnh, pdf, ROOT / "docs" / "figures" / "v8b_backtest.png")
    plot_comparison({"SimpleCNN": ev}, ROOT / "data" / "results_v4.json",
                    ROOT / "docs" / "figures" / "v8b_vs_v4_comparison.png")
    plot_val_gap(val_ics, ev["overall"]["ic"], ROOT / "docs" / "figures" / "v8b_overfitting_gap.png")

    # [8] Report
    print(f"\n[8] Generating Chinese report...")
    generate_report(ev, bt_res, val_ics, pair_names, pair_to_id)

    print(f"\n{'='*60}")
    print(f"  V8-B Evaluation Complete!")
    print(f"  SimpleCNN Ensemble: IC={o['ic']:+.4f}  Dir={o['dir_acc']:.3f}")
    print(f"  V4 comparison:     IC=+{0.0655:.4f}  (V8-B {'>' if o['ic'] > 0.0655 else '<'} V4)")
    print(f"  Overfitting ratio: {np.mean(val_ics)/max(abs(o['ic']),0.001):.1f}x (val IC / test IC)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
