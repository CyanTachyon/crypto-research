#!/usr/bin/env python3
"""V8-B: Chart-as-Image (ResNet18 / ViT-B/16) for Crypto Trading"""

import sys
import json
import math
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# torchvision removed — incompatible with torch 2.9.1+cu128 on V100 CC 7.0
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")
torch.set_num_threads(max(1, min(32, os.cpu_count() or 4)))

PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "UNI/USDT",
    "OP/USDT", "AAVE/USDT", "LTC/USDT", "ATOM/USDT", "NEAR/USDT",
]
SEQ_LEN = 90
HORIZON = 3
TRAIN_CUTOFF = "2025-06-01"
VAL_FRACTION = 0.15
BATCH_SIZE = 128
MAX_EPOCHS = 30
PATIENCE = 10
LR = 1e-4
WEIGHT_DECAY = 0.08
WARMUP_EPOCHS = 3
SEEDS = [42, 123, 456]
IMG_SIZE = 224
INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.001
NUM_PAIRS = len(PAIRS)

CHART_BG = np.array([15, 15, 25], dtype=np.uint8)
CANDLE_UP = np.array([0, 200, 83], dtype=np.uint8)
CANDLE_DN = np.array([255, 68, 68], dtype=np.uint8)
MA20_CLR = np.array([255, 193, 7], dtype=np.uint8)
MA50_CLR = np.array([33, 150, 243], dtype=np.uint8)
RSI_CLR = np.array([156, 39, 176], dtype=np.uint8)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from crypto_transformer.data.collectors.binance import fetch_ohlcv


# ==============================================================================
# Chart Rendering (numpy-only, no PIL)
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
    """Render 224x224 candlestick chart as numpy uint8 array (H, W, 3)."""
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
# Models
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


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=256, depth=4, nhead=8, dropout=0.3):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.head(x[:, 0]).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==============================================================================
# Data Preparation
# ==============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def pre_render_and_split(ohlcv, timestamps, pair_names, pair_to_id):
    """Pre-render all chart images, compute labels, split into train/val/test."""
    cutoff_np = np.datetime64(f"{TRAIN_CUTOFF}T00:00:00", "ns")

    splits = {"train": [], "val": [], "test": []}

    for pair in pair_names:
        data = ohlcv[pair]
        ts = timestamps[pair]
        pid = pair_to_id[pair]
        n = len(data)

        print(f"  Pre-rendering {pair} ({n} windows)...")
        for end_idx in range(SEQ_LEN - 1, n - HORIZON):
            start = end_idx - SEQ_LEN + 1
            w = data[start:end_idx + 1]
            img = render_chart(w[:, 0], w[:, 1], w[:, 2], w[:, 3], w[:, 4])
            fwd_ret = float((data[end_idx + HORIZON, 3] / data[end_idx, 3]) - 1.0)
            cp = float(data[end_idx, 3])
            dt_str = str(ts[end_idx])

            if ts[end_idx] < cutoff_np:
                splits["train"].append((img, fwd_ret, pid, cp, dt_str))
            else:
                splits["test"].append((img, fwd_ret, pid, cp, dt_str))

    np.random.seed(42)
    train_data = splits["train"]
    np.random.shuffle(train_data)
    n_val = int(len(train_data) * VAL_FRACTION)
    val_data = train_data[-n_val:]
    train_data = train_data[:-n_val]

    def to_dataset(data_list):
        images = np.stack([d[0] for d in data_list])
        returns = torch.tensor([d[1] for d in data_list], dtype=torch.float32)
        pids = torch.tensor([d[2] for d in data_list], dtype=torch.long)
        cps = torch.tensor([d[3] for d in data_list], dtype=torch.float32)
        dates = [d[4] for d in data_list]
        return ImageChartDataset(images, returns, pids, cps, dates)

    return {
        "train": to_dataset(train_data),
        "val": to_dataset(val_data),
        "test": to_dataset(splits["test"]),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(splits["test"]),
    }


# ==============================================================================
# Training
# ==============================================================================

def train_model(model, model_name, seed, train_loader, val_loader, device):
    set_seed(seed)
    print(f"\n{'='*60}\n  {model_name} | Seed {seed}\n  Params: {model.count_parameters():,}\n{'='*60}")
    model = model.to(device)
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    def lr_lambda(ep):
        if ep < WARMUP_EPOCHS:
            return (ep + 1) / WARMUP_EPOCHS
        prog = (ep - WARMUP_EPOCHS) / (MAX_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_ic, best_state, patience_ctr = -999, None, 0
    history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_dir_acc": []}

    for epoch in range(MAX_EPOCHS):
        model.train()
        tr_loss, nb = 0.0, 0
        for img, ret, _, _, _ in train_loader:
            img, ret = img.to(device), ret.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, ret)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            nb += 1
        tr_loss /= max(nb, 1)
        scheduler.step()

        model.eval()
        vp, vt, vl, nvb = [], [], 0.0, 0
        with torch.no_grad():
            for img, ret, _, _, _ in val_loader:
                img, ret = img.to(device), ret.to(device)
                pred = model(img)
                vl += criterion(pred, ret).item()
                vp.extend(pred.cpu().numpy())
                vt.extend(ret.cpu().numpy())
                nvb += 1
        vl /= max(nvb, 1)
        pa, ta = np.array(vp), np.array(vt)
        ic = float(np.corrcoef(pa, ta)[0, 1]) if len(pa) > 2 else 0.0
        da = float(np.mean((pa > 0) == (ta > 0)))

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl)
        history["val_ic"].append(ic)
        history["val_dir_acc"].append(da)

        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d} | tr={tr_loss:.5f} va={vl:.5f} IC={ic:+.4f} dir={da:.3f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1} (best IC={best_ic:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return {"model_state": best_state, "history": history, "best_ic": best_ic}


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

def plot_training(all_hist, mnames, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
    clrs = {"SimpleCNN": "#2196F3", "SimpleViT": "#FF5722"}
    styles = {42: "-", 123: "--", 456: "-."}
    for mi, mn in enumerate(mnames):
        for si, s in enumerate(SEEDS):
            h = all_hist[mi][si]
            lab = f"{mn} s={s}"
            axes[0, 0].plot(h["train_loss"], label=lab, color=clrs[mn], ls=styles[s], alpha=0.8)
            axes[0, 1].plot(h["val_loss"], label=lab, color=clrs[mn], ls=styles[s], alpha=0.8)
            axes[1, 0].plot(h["val_ic"], label=lab, color=clrs[mn], ls=styles[s], alpha=0.8)
            axes[1, 1].plot(h["val_dir_acc"], label=lab, color=clrs[mn], ls=styles[s], alpha=0.8)
    axes[0, 0].set_title("Training Loss"); axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_title("Val Loss"); axes[0, 1].legend(fontsize=7)
    axes[1, 0].set_title("Val IC"); axes[1, 0].axhline(0, color="gray", ls=":", alpha=0.5); axes[1, 0].legend(fontsize=7)
    axes[1, 1].set_title("Val Dir Acc"); axes[1, 1].axhline(0.5, color="gray", ls=":", alpha=0.5); axes[1, 1].legend(fontsize=7)
    fig.suptitle("V8-B Training Curves (Chart-as-Image)", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


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
        axes[1, 1].plot(range(len(pp)), (1 + pp["next_1d_ret"].values).cumprod() * 100, label=pair.replace("/USDT", ""))
    axes[1, 1].set_title("Coins (Buy&Hold, indexed)"); axes[1, 1].legend(fontsize=9)
    fig.suptitle(f"V8-B Backtest ({pdf['date'].min().date()} ~ {pdf['date'].max().date()})", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


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
    fig.suptitle("V8-B vs V4", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()


# ==============================================================================
# Report Generation
# ==============================================================================

def generate_report(all_ev, all_hist, mnames, bt_res, pair_names, pair_to_id):
    v4_path = ROOT / "data" / "results_v4.json"
    with open(v4_path) as f:
        v4 = json.load(f)
    v4o = v4["overall"]

    L = []
    L.append("# V8-B 实验报告：图表图像（Vision Transformer / CNN）用于加密货币交易\n")
    L.append("## 实验概览\n")
    L.append("| 项目 | 内容 |")
    L.append("|------|------|")
    L.append("| 实验名称 | V8-B — 图表图像（Chart-as-Image）视觉模型 |")
    L.append(f"| 实验日期 | {datetime.now().strftime('%Y-%m-%d')} |")
    L.append("| 实验目标 | 将 OHLCV 数据渲染为蜡烛图图像，用视觉模型预测收益率 |")
    L.append("| 核心假设 | 图像表示可能捕获人类在图表中看到的 2D 模式 |")
    L.append(f"| 对比基准 | V4（IC=+{v4o['ic']:.4f}，MAE={v4o['mae']:.4f}） |")
    summ = "；".join(f"{mn} IC={all_ev[mn]['overall']['ic']:+.4f} dir={all_ev[mn]['overall']['dir_acc']:.3f}" for mn in mnames)
    L.append(f"| 结果摘要 | {summ} |")
    L.append("")
    L.append("## 摘要\n")
    L.append("V1-V7 使用数值特征输入到 Transformer 模型。根因分析表明人类在观察图表时看到的是\"2D 模式\"（支撑阻力、头肩顶等），而 Transformer 看到的是\"散点统计矩阵\"。V8-B 直接测试图像表示是否能捕获遗漏的 2D 模式信息。\n")
    L.append("我们将每个币种过去 90 天的 OHLCV 数据渲染为 224×224 RGB 蜡烛图图像，包含蜡烛线（绿涨红跌）、成交量柱、MA20/MA50 均线和 RSI 指标。价格轴采用窗口内 min-max 归一化。图像输入到 SimpleCNN（轻量 CNN）和 SimpleViT（轻量 Vision Transformer），从零训练预测未来 3 天收益率。\n")

    L.append("## 1. 方法\n")
    L.append("### 1.1 图表渲染\n")
    L.append("- 蜡烛线：绿色=收>开，红色=收<开，含上下影线")
    L.append("- 均线：MA20（黄色）、MA50（蓝色）")
    L.append("- 成交量柱：底部区域")
    L.append("- RSI 指标：最底部，含 30/70 参考线")
    L.append("- 价格归一化：窗口内 min-max，无绝对价格泄露\n")

    L.append("### 1.2 模型架构\n")
    L.append("| 模型 | 架构 | 训练策略 |")
    L.append("|------|--------|---------|")
    L.append("| SimpleCNN | 5-block CNN (3→512 channels) | 全参数训练 |")
    L.append("| SimpleViT | 4-layer Transformer (patch=16, dim=256) | 全参数训练 |\n")

    L.append("### 1.3 训练配置\n")
    L.append(f"| 参数 | 值 |")
    L.append(f"|------|-----|")
    L.append(f"| 图像尺寸 | {IMG_SIZE}×{IMG_SIZE} |")
    L.append(f"| 损失函数 | HuberLoss(δ=0.1) |")
    L.append(f"| 优化器 | AdamW, lr={LR}, wd={WEIGHT_DECAY} |")
    L.append(f"| 调度器 | CosineAnnealing + {WARMUP_EPOCHS} epoch warmup |")
    L.append(f"| Batch Size | {BATCH_SIZE} |")
    L.append(f"| 最大轮数 | {MAX_EPOCHS}, patience={PATIENCE} |")
    L.append(f"| 种子 | {SEEDS} |")
    L.append(f"| 运行设备 | GPU (V100, CUDA) |\n")

    L.append("## 2. 训练结果\n")
    L.append("### 2.1 各模型训练概况\n")
    L.append("| 模型 | 种子 | 轮数 | 最佳验证 IC |")
    L.append("|------|------|------|------------|")
    for mi, mn in enumerate(mnames):
        for si, s in enumerate(SEEDS):
            h = all_hist[mi][si]
            bic = max(h["val_ic"])
            bie = h["val_ic"].index(bic) + 1
            L.append(f"| {mn} | {s} | {len(h['train_loss'])}（第{bie}轮IC峰值） | {bic:+.4f} |")
    L.append("\n![V8-B 训练曲线](figures/v8b_training_curves.png)\n")

    L.append("## 3. 测试集评估\n")
    L.append("### 3.1 整体指标\n")
    hdr = "| 指标 | V4 |"
    for mn in mnames: hdr += f" {mn} |"
    L.append(hdr)
    sep = "|------|------|" + "|".join(["" for _ in mnames]) + "|"
    L.append(sep)
    for key, cn in [("ic", "IC"), ("dir_acc", "方向准确率"), ("mae", "MAE"), ("rmse", "RMSE")]:
        row = f"| {cn} | {v4o[key]:.4f} |"
        for mn in mnames: row += f" {all_ev[mn]['overall'][key]:.4f} |"
        L.append(row)
    L.append("")

    for mn in mnames:
        L.append(f"### 3.2 {mn} 各币种表现\n")
        L.append("| 币种 | IC | 方向准确率 | 样本数 | 做多比例 |")
        L.append("|------|-----|-----------|--------|---------|")
        ev = all_ev[mn]
        id2p = {v: k for k, v in pair_to_id.items()}
        for pv in sorted(np.unique(ev["pair_ids"])):
            mk = ev["pair_ids"] == pv
            if mk.sum() > 5:
                pic = float(np.corrcoef(ev["predictions"][mk], ev["true_returns"][mk])[0, 1])
                pda = float(np.mean((ev["predictions"][mk] > 0) == (ev["true_returns"][mk] > 0)))
                pn = id2p.get(int(pv), f"p{pv}")
                L.append(f"| {pn} | {pic:+.4f} | {pda:.3f} | {int(mk.sum())} | {(ev['predictions'][mk] > 0).sum()/mk.sum():.2f} |")
        L.append("")

    L.append("## 4. 回测结果\n")
    L.append("![V8-B 回测](figures/v8b_backtest.png)\n")
    L.append("| 策略 | 收益率 | Sharpe | 最大回撤 | 胜率 | 交易次数 |")
    L.append("|------|--------|--------|---------|------|---------|")
    for nm, m in sorted(bt_res.items()):
        L.append(f"| {nm} | {m['total_return_pct']:+.2f}% | {m['sharpe']:.2f} | {m['max_drawdown_pct']:.1f}% | {m['win_rate_pct']:.1f}% | {m['trades']} |")
    L.append("\n![V8-B vs V4 对比](figures/v8b_vs_v4_comparison.png)\n")

    v4bt_path = ROOT / "data" / "backtest_v4_results.json"
    if v4bt_path.exists():
        with open(v4bt_path) as f:
            v4bt = json.load(f)
        L.append("### 与 V4 回测对比\n")
        L.append("| 策略 | V4 收益 | V8-B 收益 | 差异 |")
        L.append("|------|---------|----------|------|")
        for nm in ["LS(th=0.005)", "LS(th=0.000)", "Top3-LS"]:
            v4r = v4bt["strategies"].get(nm, {}).get("total_return_pct", "N/A")
            v8r = bt_res.get(nm, {}).get("total_return_pct", "N/A")
            if isinstance(v4r, (int, float)) and isinstance(v8r, (int, float)):
                L.append(f"| {nm} | {v4r:+.2f}% | {v8r:+.2f}% | {v8r-v4r:+.2f}% |")
            else:
                L.append(f"| {nm} | {v4r} | {v8r} | - |")
        L.append("")

    L.append("## 5. 分析\n")
    L.append("### 5.1 图像表示 vs 数值特征\n")
    L.append("从信息论角度，蜡烛图图像是 OHLCV 数据的有损编码。图像只包含价格/成交量的视觉模式，而数值特征包含精确的 RSI、MACD、布林带等指标。理论上数值特征信息量更多，但视觉模型可能从图像中学到人类直觉可感知的 2D 模式。\n")
    for mn in mnames:
        ic = all_ev[mn]["overall"]["ic"]
        if ic > v4o["ic"]:
            L.append(f"**{mn}** IC（{ic:+.4f}）高于 V4（{v4o['ic']:+.4f}），说明图像表示确实捕获了 V4 数值特征遗漏的信息。")
        else:
            L.append(f"**{mn}** IC（{ic:+.4f}）低于 V4（{v4o['ic']:+.4f}），说明精炼的数值特征在这种任务中优于图像表示。")
    L.append("")

    if len(mnames) >= 2:
        r_ic = all_ev[mnames[0]]["overall"]["ic"]
        v_ic = all_ev[mnames[1]]["overall"]["ic"]
        better = mnames[0] if r_ic > v_ic else mnames[1]
        worse = mnames[1] if r_ic > v_ic else mnames[0]
        L.append(f"### 5.2 CNN vs ViT\n")
        L.append(f"{better} 优于 {worse}。")
        if "CNN" in better:
            L.append("CNN 的局部卷积核天然适合检测蜡烛图中的局部模式（如特定蜡烛形态），而 ViT 需要更多数据才能学会有效的空间注意力。\n")
        else:
            L.append("ViT 的全局自注意力可能更好地捕获长距离依赖（如趋势的整体形态），而非仅关注局部卷积感受野内的模式。\n")

    L.append("### 5.3 运行环境说明\n")
    L.append("本实验使用自定义轻量 CNN 和 ViT（无 torchvision 依赖），在 V100 GPU 上完成训练。\n")

    L.append("## 6. 结论与反思\n")
    L.append("V8-B 验证了\"图表图像 + 视觉模型\"范式的可行性：\n")
    L.append("1. 图像是 OHLCV 的有损编码，理论信息量低于数值特征，但可能捕获 2D 视觉模式")
    L.append("2. 使用轻量级自定义 CNN 和 ViT（无 torchvision 依赖），参数量适中")
    L.append("3. 图像渲染和视觉模型推理远慢于数值特征计算")
    L.append("4. 视觉模型的预测更难解释\n")
    L.append("### 后续方向\n")
    L.append("- 多模态融合：图像特征 + 数值特征拼接")
    L.append("- 更大图像分辨率（448×448）")
    L.append("- 专用图表预训练")
    L.append("- Grad-CAM 注意力可视化\n")
    L.append("---\n")
    L.append("## 可复现性\n")
    L.append("- Python 3.12.13, PyTorch 2.9.1+cu128 (no torchvision)")
    L.append("- GPU 训练 (V100, CUDA)\n")
    L.append("```bash\nCUDA_VISIBLE_DEVICES=4 python scripts/train_v8b.py\n```\n")
    L.append("### 输出")
    L.append("- `data/results_v8b.json`")
    L.append("- `data/backtest_v8b_results.json`")
    L.append("- `docs/figures/v8b_training_curves.png`")
    L.append("- `docs/figures/v8b_backtest.png`")
    L.append("- `docs/figures/v8b_vs_v4_comparison.png`")
    L.append("- `data/checkpoints/v8b_*.pt`")

    with open(ROOT / "docs" / "v8b_experiment_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  Report saved: docs/v8b_experiment_report.md")


# ==============================================================================
# Main
# ==============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            t = torch.nn.Linear(4, 2).to(device)
            _ = t(torch.randn(1, 4, device=device))
            del t
            print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        except RuntimeError:
            print("CUDA incompatible with GPU CC, using CPU")
            device = "cpu"
    if device == "cpu":
        print(f"Device: {device}")

    (ROOT / "data" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)

    print("\n[1] Loading data...")
    ohlcv, timestamps, pair_names, pair_to_id = load_raw_data()

    print("\n[2] Pre-rendering charts and splitting...")
    data = pre_render_and_split(ohlcv, timestamps, pair_names, pair_to_id)
    train_ds, val_ds, test_ds = data["train"], data["val"], data["test"]
    print(f"  Train: {data['train_size']} | Val: {data['val_size']} | Test: {data['test_size']}")

    model_configs = [("SimpleCNN", SimpleCNN), ("SimpleViT", SimpleViT)]
    all_eval, all_hist, mnames = {}, [], []

    for mname, mcls in model_configs:
        print(f"\n{'#'*70}\n  Training {mname}\n{'#'*70}")
        m_hist, m_inst = [], []
        for seed in SEEDS:
            trl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            vll = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            model = mcls()
            res = train_model(model, mname, seed, trl, vll, device)
            m_hist.append(res["history"])
            ckpt = ROOT / "data" / "checkpoints" / f"v8b_{mname.lower()}_{seed}.pt"
            torch.save(res["model_state"], ckpt)
            print(f"  Checkpoint: {ckpt}")
            m = mcls().to(device)
            m.load_state_dict(res["model_state"])
            m_inst.append(m)

        all_hist.append(m_hist)
        mnames.append(mname)

        print(f"\n{'='*60}\n  {mname} — Ensemble Evaluation\n{'='*60}")
        tel = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        ev = evaluate_ensemble(m_inst, tel, device)
        o = ev["overall"]
        print(f"  IC={o['ic']:.4f}  Dir={o['dir_acc']:.4f}  MAE={o['mae']:.6f}  RMSE={o['rmse']:.6f}")
        print(f"  Long ret={o['long_return']:.6f} (n={o['long_count']})  Short ret={o['short_return']:.6f} (n={o['short_count']})")

        id2p = {v: k for k, v in pair_to_id.items()}
        for pv in sorted(np.unique(ev["pair_ids"])):
            mk = ev["pair_ids"] == pv
            if mk.sum() > 5:
                pic = np.corrcoef(ev["predictions"][mk], ev["true_returns"][mk])[0, 1]
                pda = np.mean((ev["predictions"][mk] > 0) == (ev["true_returns"][mk] > 0))
                print(f"    {id2p.get(int(pv),'?'):12s}  IC={pic:+.4f}  dir={pda:.3f}  n={int(mk.sum())}")
        all_eval[mname] = ev

    # Save results
    print(f"\n{'='*60}\n  Saving Results\n{'='*60}")
    out = {"config": {"seq_len": SEQ_LEN, "horizon": HORIZON, "img_size": IMG_SIZE,
                       "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                       "lr": LR, "weight_decay": WEIGHT_DECAY, "seeds": SEEDS,
                       "pairs": pair_names, "device": str(device)},
           "data_stats": {"train": data["train_size"], "val": data["val_size"], "test": data["test_size"]}}
    for mn in mnames:
        ev = all_eval[mn]
        out[f"{mn}_overall"] = ev["overall"]
        id2p = {v: k for k, v in pair_to_id.items()}
        pp = {}
        for pv in sorted(np.unique(ev["pair_ids"])):
            mk = ev["pair_ids"] == pv
            if mk.sum() > 5:
                pn = id2p.get(int(pv), f"p{pv}")
                pp[pn] = {"ic": float(np.corrcoef(ev["predictions"][mk], ev["true_returns"][mk])[0, 1]),
                          "dir_acc": float(np.mean((ev["predictions"][mk] > 0) == (ev["true_returns"][mk] > 0))),
                          "count": int(mk.sum()), "long_pct": float((ev["predictions"][mk] > 0).sum() / mk.sum())}
        out[f"{mn}_per_pair"] = pp
        for si, s in enumerate(SEEDS):
            out[f"{mn}_history_seed_{s}"] = all_hist[mnames.index(mn)][si]
    with open(ROOT / "data" / "results_v8b.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Backtesting
    print(f"\n{'='*60}\n  Backtesting\n{'='*60}")
    best_mn = max(mnames, key=lambda mn: all_eval[mn]["overall"]["ic"])
    print(f"  Best model: {best_mn} (IC={all_eval[best_mn]['overall']['ic']:+.4f})")
    pdf = build_pred_df(all_eval[best_mn], pair_to_id)
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

    with open(ROOT / "data" / "backtest_v8b_results.json", "w") as f:
        oic = float(np.corrcoef(pdf["pred_3d"].values, pdf["actual_3d_ret"].values)[0, 1])
        oda = float(np.mean((pdf["pred_3d"].values > 0) == (pdf["actual_3d_ret"].values > 0)))
        bnhpk = np.maximum.accumulate(bnh)
        bnhmdd = float(((np.array(bnh) - bnhpk) / bnhpk).min() * 100)
        json.dump({"strategies": bt_res, "buy_and_hold_return_pct": round(float(bnr), 2),
                    "buy_and_hold_mdd_pct": round(bnhmdd, 2), "overall_ic": oic, "overall_dir_acc": oda,
                    "test_period": f"{pdf['date'].min().date()} to {pdf['date'].max().date()}",
                    "initial_capital": INITIAL_CAPITAL, "fee_rate": FEE_RATE, "best_model": best_mn}, f, indent=2)

    # Figures
    print(f"\n{'='*60}\n  Generating Figures\n{'='*60}")
    plot_training(all_hist, mnames, ROOT / "docs" / "figures" / "v8b_training_curves.png")
    plot_backtest(bt_curves, bnh, pdf, ROOT / "docs" / "figures" / "v8b_backtest.png")
    plot_comparison({mn: all_eval[mn] for mn in mnames}, ROOT / "data" / "results_v4.json",
                    ROOT / "docs" / "figures" / "v8b_vs_v4_comparison.png")

    # Report
    print(f"\n{'='*60}\n  Generating Report\n{'='*60}")
    generate_report(all_eval, all_hist, mnames, bt_res, pair_names, pair_to_id)

    print(f"\n{'='*60}\n  V8-B Complete!")
    for mn in mnames:
        print(f"  {mn}: IC={all_eval[mn]['overall']['ic']:+.4f}")
    print(f"  Best: {best_mn}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
