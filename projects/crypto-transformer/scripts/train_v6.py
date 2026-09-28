#!/usr/bin/env python3
"""
V6: Walk-Forward Training + Market-Neutral Target
===================================================
Core innovations over V4:
1. Walk-forward (rolling window) training: train on [T-365, T], predict T+3.
   Retrain monthly. This directly addresses regime overfitting.
2. Market-neutral target: predict residual return after removing BTC beta.
   Focuses model on coin-specific alpha.
3. V4's 25 proven features (no external data).
4. Single-seed per window (fast), ensemble across last 3 windows.

Evaluation: aggregate all walk-forward predictions and compare with V4.
"""

import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

from crypto_transformer.data.collectors.binance import fetch_ohlcv
from crypto_transformer.data.features_v4 import compute_features_v4, compute_cross_sectional_features, FEATURE_COLUMNS_V4, NUM_FEATURES_V4
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ============================================================================
# Config
# ============================================================================
PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "UNI/USDT",
    "OP/USDT", "AAVE/USDT", "LTC/USDT", "ATOM/USDT", "NEAR/USDT",
]

SEQ_LEN = 90
HORIZON = 3
BATCH_SIZE = 128
MAX_EPOCHS = 80
PATIENCE = 15
LR = 1.5e-4
WEIGHT_DECAY = 0.10
GRADIENT_CLIP = 1.0
WARMUP_EPOCHS = 3
RANKING_MARGIN = 0.01
RANKING_WEIGHT = 0.3
HUBER_WEIGHT = 0.7
DROPOUT = 0.35

TRAIN_WINDOW_DAYS = 540
VAL_FRACTION = 0.12
STEP_DAYS = 30
MIN_TRAIN_DAYS = 365

BETA_LOOKBACK = 60

SEED = 42


# ============================================================================
# Model (same as V4)
# ============================================================================
class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin

    def forward(self, pred, target):
        half = len(pred) // 2
        if half < 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        p1, p2 = pred[:half], pred[half:2 * half]
        t1, t2 = target[:half], target[half:2 * half]
        loss = torch.relu(self.margin - (t1 - t2) * (p1 - p2))
        return loss.mean()


class RegressionTransformer(nn.Module):
    def __init__(self, num_features=25, num_pairs=15, d_model=128, nhead=4,
                 num_layers=3, dim_feedforward=512, dropout=0.35,
                 pair_emb_dim=16, max_len=256):
        super().__init__()
        self.pair_embedding = nn.Embedding(num_pairs, pair_emb_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(num_features + pair_emb_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 0.5),
        )
        self.pos_encoder = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pair_id):
        B, T, _ = x.shape
        pe = self.pair_embedding(pair_id).unsqueeze(1).expand(B, T, -1)
        x = self.input_proj(torch.cat([x, pe], dim=-1))
        x = x + self.pos_encoder[:, :T, :]
        mask = (x.abs().sum(dim=-1) == 0)
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.head(x[:, -1, :]).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Dataset
# ============================================================================
class CryptoDataset(Dataset):
    def __init__(self, features, returns, pair_ids, timestamps, close_prices, seq_len=90):
        self.features = features.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.pair_ids = pair_ids.astype(np.int64)
        self.timestamps = timestamps
        self.close_prices = close_prices.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len]
        y = self.returns[idx + self.seq_len - 1]
        pid = self.pair_ids[idx + self.seq_len - 1]
        ts = str(self.timestamps[idx + self.seq_len - 1])[:10]
        cp = self.close_prices[idx + self.seq_len - 1]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(pid, dtype=torch.long),
            ts,
            torch.tensor(cp, dtype=torch.float32),
        )


# ============================================================================
# Market-Neutral Return Computation
# ============================================================================
def compute_market_neutral_returns(all_dfs, btc_df, horizon=3, beta_lookback=60):
    """
    For each coin, compute residual return = raw_return - beta * btc_return.
    beta is computed from rolling regression vs BTC.
    """
    btc_close = btc_df.set_index("timestamp")["close"]
    neutral_dfs = {}

    for pair_name, df in all_dfs.items():
        df = df.copy()
        close = df["close_raw"].values

        # Raw future return
        raw_return = pd.Series(close).pct_change(horizon).shift(-horizon).values.copy()
        raw_return[np.isnan(raw_return)] = 0.0

        # BTC future return (aligned by timestamp)
        btc_aligned = df["timestamp"].map(
            btc_close.to_dict()
        ).values

        btc_future = pd.Series(btc_aligned).pct_change(horizon).shift(-horizon).values.copy()
        btc_future[np.isnan(btc_future)] = 0.0

        # Rolling beta
        raw_daily = pd.Series(close).pct_change().fillna(0).values
        btc_daily = pd.Series(btc_aligned).pct_change().fillna(0).values

        betas = np.zeros(len(df))
        for i in range(beta_lookback, len(df)):
            r = raw_daily[i - beta_lookback:i]
            b = btc_daily[i - beta_lookback:i]
            var_b = np.var(b)
            if var_b > 1e-12:
                betas[i] = np.cov(r, b)[0, 1] / var_b
            else:
                betas[i] = 1.0
        # Fill early values with 1.0
        betas[:beta_lookback] = 1.0

        # Residual return = raw - beta * btc_future
        residual_return = raw_return - betas * btc_future
        residual_return[np.isnan(residual_return)] = 0.0
        residual_return[-horizon:] = 0.0

        df["_raw_return"] = raw_return
        df["_residual_return"] = residual_return
        df["_beta"] = betas
        neutral_dfs[pair_name] = df

    return neutral_dfs


# ============================================================================
# Data Preparation for a Walk-Forward Window
# ============================================================================
def prepare_window_data(all_neutral_dfs, pair_to_id, train_start, train_end, test_end, use_residual=True):
    """Prepare train/val/test data for one walk-forward window."""

    train_f, train_r, train_p, train_t, train_c = [], [], [], [], []
    val_f, val_r, val_p, val_t, val_c = [], [], [], [], []
    test_f, test_r, test_p, test_t, test_c = [], [], [], [], []

    return_col = "_residual_return" if use_residual else "_raw_return"

    for pair_name, df in all_neutral_dfs.items():
        pair_id = pair_to_id[pair_name]
        features = df[FEATURE_COLUMNS_V4].values
        returns = df[return_col].values
        pids = np.full(len(df), pair_id)
        timestamps = df["timestamp"].values
        close = df["close_raw"].values

        # Time masks
        train_start_np = np.datetime64(train_start)
        train_end_np = np.datetime64(train_end)
        test_end_np = np.datetime64(test_end)

        train_mask = (timestamps >= train_start_np) & (timestamps < train_end_np)
        test_mask = (timestamps >= train_end_np) & (timestamps < test_end_np)

        # Skip pairs with insufficient data
        if train_mask.sum() < SEQ_LEN + 10 or test_mask.sum() < 1:
            continue

        tr_f = features[train_mask]
        tr_r = returns[train_mask]
        tr_p = pids[train_mask]
        tr_t = timestamps[train_mask]
        tr_c = close[train_mask]

        te_f = features[test_mask]
        te_r = returns[test_mask]
        te_p = pids[test_mask]
        te_t = timestamps[test_mask]
        te_c = close[test_mask]

        # Split train into train+val
        n_val = max(1, int(len(tr_f) * VAL_FRACTION))
        va_f, va_r, va_p, va_t, va_c = tr_f[-n_val:], tr_r[-n_val:], tr_p[-n_val:], tr_t[-n_val:], tr_c[-n_val:]
        tr_f, tr_r, tr_p, tr_t, tr_c = tr_f[:-n_val], tr_r[:-n_val], tr_p[:-n_val], tr_t[:-n_val], tr_c[:-n_val:]

        train_f.append(tr_f)
        train_r.append(tr_r)
        train_p.append(tr_p)
        train_t.append(tr_t)
        train_c.append(tr_c)
        val_f.append(va_f)
        val_r.append(va_r)
        val_p.append(va_p)
        val_t.append(va_t)
        val_c.append(va_c)
        test_f.append(te_f)
        test_r.append(te_r)
        test_p.append(te_p)
        test_t.append(te_t)
        test_c.append(te_c)

    if not train_f:
        return None

    # Concatenate
    tr_f = np.concatenate(train_f)
    tr_r = np.concatenate(train_r)
    tr_p = np.concatenate(train_p)
    tr_t = np.concatenate(train_t)
    tr_c = np.concatenate(train_c)
    va_f = np.concatenate(val_f)
    va_r = np.concatenate(val_r)
    va_p = np.concatenate(val_p)
    va_t = np.concatenate(val_t)
    va_c = np.concatenate(val_c)
    te_f = np.concatenate(test_f)
    te_r = np.concatenate(test_r)
    te_p = np.concatenate(test_p)
    te_t = np.concatenate(test_t)
    te_c = np.concatenate(test_c)

    # Scale features
    scaler = StandardScaler()
    tr_f = scaler.fit_transform(tr_f).astype(np.float32)
    va_f = scaler.transform(va_f).astype(np.float32)
    te_f = scaler.transform(te_f).astype(np.float32)

    # Handle NaN in features
    tr_f = np.nan_to_num(tr_f, nan=0.0)
    va_f = np.nan_to_num(va_f, nan=0.0)
    te_f = np.nan_to_num(te_f, nan=0.0)

    return {
        "train": CryptoDataset(tr_f, tr_r, tr_p, tr_t, tr_c, SEQ_LEN),
        "val": CryptoDataset(va_f, va_r, va_p, va_t, va_c, SEQ_LEN),
        "test": CryptoDataset(te_f, te_r, te_p, te_t, te_c, SEQ_LEN),
        "train_size": len(tr_f),
        "val_size": len(va_f),
        "test_size": len(te_f),
    }


# ============================================================================
# Training for a Single Window
# ============================================================================
def train_window(window_data, num_pairs, device):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = RegressionTransformer(
        num_features=NUM_FEATURES_V4, num_pairs=num_pairs,
        dropout=DROPOUT,
    ).to(device)

    huber = nn.HuberLoss(delta=0.1)
    ranking = PairwiseRankingLoss(margin=RANKING_MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (MAX_EPOCHS - WARMUP_EPOCHS)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = DataLoader(window_data["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(window_data["val"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    best_ic = -999
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
        for x, y, pid, _, _ in train_loader:
            x, y, pid = x.to(device), y.to(device), pid.to(device)
            optimizer.zero_grad()
            pred = model(x, pid)
            h_loss = huber(pred, y)
            idx = torch.randperm(len(pred))
            r_loss = ranking(pred[idx], y[idx])
            loss = HUBER_WEIGHT * h_loss + RANKING_WEIGHT * r_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y, pid, _, _ in val_loader:
                x, y, pid = x.to(device), y.to(device), pid.to(device)
                pred = model(x, pid)
                all_pred.extend(pred.cpu().numpy())
                all_true.extend(y.cpu().numpy())

        pred_arr = np.array(all_pred)
        true_arr = np.array(all_true)
        if len(pred_arr) > 2 and np.std(pred_arr) > 1e-8 and np.std(true_arr) > 1e-8:
            ic = np.corrcoef(pred_arr, true_arr)[0, 1]
        else:
            ic = 0.0

        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_ic


# ============================================================================
# Inference on Test Set
# ============================================================================
def predict_window(model, window_data, device):
    """Get predictions on the test portion of a window."""
    test_loader = DataLoader(window_data["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    all_pred, all_true, all_pid, all_ts, all_close = [], [], [], [], []

    with torch.no_grad():
        for x, y, pid, ts, cp in test_loader:
            x, pid = x.to(device), pid.to(device)
            pred = model(x, pid)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(y.numpy())
            all_pid.extend(pid.cpu().numpy())
            all_ts.extend(ts)
            all_close.extend(cp.numpy())

    return {
        "pred": np.array(all_pred),
        "true": np.array(all_true),
        "pair_id": np.array(all_pid),
        "timestamp": all_ts,
        "close": np.array(all_close),
    }


# ============================================================================
# Backtest Simulation
# ============================================================================
def run_backtest(all_predictions, pair_names, pair_to_id, initial_capital=10000, fee_rate=0.001):
    """Run trading simulation using aggregated walk-forward predictions."""
    id_to_pair = {v: k for k, v in pair_to_id.items()}

    # Organize predictions by date
    date_preds = defaultdict(list)
    for i in range(len(all_predictions["pred"])):
        ts = all_predictions["timestamp"][i]
        if isinstance(ts, np.datetime64):
            date_str = pd.Timestamp(ts).strftime("%Y-%m-%d")
        else:
            date_str = str(ts)[:10]
        date_preds[date_str].append({
            "pred": float(all_predictions["pred"][i]),
            "true": float(all_predictions["true"][i]),
            "pair_id": int(all_predictions["pair_id"][i]),
            "close": float(all_predictions["close"][i]),
        })

    dates = sorted(date_preds.keys())
    portfolio_value = initial_capital
    max_value = initial_capital
    dd_peak = initial_capital
    max_dd = 0
    equity_curve = [initial_capital]
    daily_returns = []
    all_trades = []

    for date in dates:
        day_items = date_preds[date]
        n_assets = len(day_items)
        if n_assets == 0:
            equity_curve.append(portfolio_value)
            continue

        position_size = portfolio_value / n_assets
        day_pnl = 0
        day_trades = 0
        day_wins = 0

        for item in day_items:
            pred = item["pred"]
            true_ret = item["true"]
            pair_name = id_to_pair.get(item["pair_id"], "UNKNOWN")

            # Long/Short strategy with threshold
            if abs(pred) > 0.005:
                direction = 1 if pred > 0 else -1
                trade_ret = direction * true_ret - fee_rate
                day_pnl += position_size * trade_ret
                day_trades += 1
                if trade_ret > 0:
                    day_wins += 1
                all_trades.append({
                    "date": date,
                    "pair": pair_name,
                    "direction": "long" if direction > 0 else "short",
                    "pred": pred,
                    "true_ret": true_ret,
                    "pnl": position_size * trade_ret,
                })

        portfolio_value += day_pnl
        dd_peak = max(dd_peak, portfolio_value)
        dd = (portfolio_value - dd_peak) / dd_peak
        max_dd = min(max_dd, dd)
        equity_curve.append(portfolio_value)

        daily_ret = day_pnl / (portfolio_value - day_pnl) if (portfolio_value - day_pnl) > 0 else 0
        daily_returns.append(daily_ret)

    # Buy and hold (equal weight across all pairs)
    # Use first and last close prices
    bh_start_close = {}
    bh_end_close = {}
    for item_list in date_preds.values():
        for item in item_list:
            pid = item["pair_id"]
            if pid not in bh_start_close:
                bh_start_close[pid] = item["close"]
            bh_end_close[pid] = item["close"]

    bh_returns = []
    for pid in bh_start_close:
        if bh_start_close[pid] > 0:
            bh_returns.append((bh_end_close[pid] / bh_start_close[pid]) - 1)
    bh_avg = np.mean(bh_returns) if bh_returns else 0

    # Stats
    total_return = (portfolio_value / initial_capital) - 1
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(365) if daily_returns else 0
    win_rate = sum(1 for t in all_trades if t["pnl"] > 0) / max(len(all_trades), 1)

    return {
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "win_rate_pct": win_rate * 100,
        "trades": len(all_trades),
        "final_value": portfolio_value,
        "buy_and_hold_return_pct": bh_avg * 100,
        "equity_curve": equity_curve,
        "daily_returns": daily_returns,
        "all_trades": all_trades[:10],  # just keep first 10 for inspection
    }


# ============================================================================
# Main
# ============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("V6 Walk-Forward + Market-Neutral Training")
    print("=" * 70)

    # [1] Load and compute features
    print("\n[1/5] Loading and computing features...")
    btc = fetch_ohlcv("BTC/USDT", "1d")
    raw = {p: fetch_ohlcv(p, "1d") for p in PAIRS}
    ranks = compute_cross_sectional_features(raw, lookback=10)
    all_dfs = {p: compute_features_v4(raw[p], btc_df=btc, cross_sectional_ranks=ranks.get(p)) for p in PAIRS}

    # [2] Compute market-neutral returns
    print("\n[2/5] Computing market-neutral returns...")
    all_neutral = compute_market_neutral_returns(all_dfs, btc, horizon=HORIZON, beta_lookback=BETA_LOOKBACK)

    pair_names = sorted(all_neutral.keys())
    pair_to_id = {name: i for i, name in enumerate(pair_names)}

    # [3] Generate walk-forward windows
    print("\n[3/5] Generating walk-forward windows...")

    # Find date range
    all_dates = []
    for df in all_neutral.values():
        all_dates.extend(df["timestamp"].values)
    all_dates = sorted(set(all_dates))
    min_date = pd.Timestamp(all_dates[0])
    max_date = pd.Timestamp(all_dates[-1])

    # First test date = min_date + MIN_TRAIN_DAYS
    # We want to predict from 2025-06-01 onward (same test period as V4)
    test_start = pd.Timestamp("2025-06-01")
    # But we need training data before that
    train_start = test_start - pd.Timedelta(days=TRAIN_WINDOW_DAYS)

    windows = []
    current_test_start = test_start
    while current_test_start < max_date:
        current_test_end = current_test_start + pd.Timedelta(days=STEP_DAYS)
        current_train_start = current_test_start - pd.Timedelta(days=TRAIN_WINDOW_DAYS)

        windows.append({
            "train_start": current_train_start.strftime("%Y-%m-%d"),
            "train_end": current_test_start.strftime("%Y-%m-%d"),
            "test_start": current_test_start.strftime("%Y-%m-%d"),
            "test_end": current_test_end.strftime("%Y-%m-%d"),
        })
        current_test_start = current_test_end

    print(f"  Total windows: {len(windows)}")
    print(f"  First window: {windows[0]['train_start']} -> {windows[0]['test_end']}")
    print(f"  Last window:  {windows[-1]['train_start']} -> {windows[-1]['test_end']}")

    # [4] Walk-forward training
    print(f"\n[4/5] Walk-forward training ({len(windows)} windows)...")

    all_predictions = {
        "pred": [], "true": [], "pair_id": [], "timestamp": [], "close": [],
    }
    window_results = []

    for wi, window in enumerate(windows):
        print(f"\n  Window {wi+1}/{len(windows)}: "
              f"train={window['train_start']}..{window['train_end']} "
              f"test={window['test_start']}..{window['test_end']}")

        wdata = prepare_window_data(
            all_neutral, pair_to_id,
            window["train_start"], window["train_end"], window["test_end"],
            use_residual=True,
        )

        if wdata is None or len(wdata["train"]) < 10 or len(wdata["test"]) < 1:
            print(f"    Skipped (insufficient data)")
            continue

        print(f"    Train: {wdata['train_size']} | Val: {wdata['val_size']} | Test: {wdata['test_size']}")

        model, best_ic = train_window(wdata, len(pair_names), device)
        print(f"    Best val IC: {best_ic:.4f}")

        # Predict on test
        preds = predict_window(model, wdata, device)
        if len(preds["pred"]) > 0:
            for key in all_predictions:
                if key == "timestamp":
                    all_predictions[key].extend(preds[key])
                else:
                    all_predictions[key].extend(preds[key].tolist() if hasattr(preds[key], 'tolist') else preds[key])

            # Per-window metrics
            pred_arr = np.array(preds["pred"])
            true_arr = np.array(preds["true"])
            if len(pred_arr) > 2 and np.std(pred_arr) > 1e-8:
                w_ic = np.corrcoef(pred_arr, true_arr)[0, 1]
                w_dir = np.mean((pred_arr > 0) == (true_arr > 0))
            else:
                w_ic = 0.0
                w_dir = 0.5

            window_results.append({
                "window": window,
                "best_val_ic": float(best_ic),
                "test_ic": float(w_ic),
                "test_dir": float(w_dir),
                "test_size": len(pred_arr),
            })
            print(f"    Test IC: {w_ic:+.4f} | Dir: {w_dir:.3f} | N: {len(pred_arr)}")

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Convert predictions
    for key in all_predictions:
        if key != "timestamp":
            all_predictions[key] = np.array(all_predictions[key])

    # [5] Aggregate evaluation
    print(f"\n{'='*60}")
    print("  V6 Aggregate Walk-Forward Results")
    print(f"{'='*60}")

    pred_arr = all_predictions["pred"]
    true_arr = all_predictions["true"]
    pid_arr = all_predictions["pair_id"]

    if len(pred_arr) < 2:
        print("  ERROR: Not enough predictions!")
        return

    overall_ic = np.corrcoef(pred_arr, true_arr)[0, 1] if np.std(pred_arr) > 1e-8 else 0
    overall_dir = np.mean((pred_arr > 0) == (true_arr > 0))
    mae = np.mean(np.abs(pred_arr - true_arr))
    rmse = np.sqrt(np.mean((pred_arr - true_arr) ** 2))

    long_mask = pred_arr > 0
    short_mask = pred_arr < 0
    long_return = true_arr[long_mask].mean() if long_mask.sum() > 0 else 0
    short_return = -true_arr[short_mask].mean() if short_mask.sum() > 0 else 0

    print(f"\n  IC:              {overall_ic:.4f}")
    print(f"  Directional Acc: {overall_dir:.4f}")
    print(f"  MAE:             {mae:.6f}")
    print(f"  RMSE:            {rmse:.6f}")
    print(f"  Long avg return: {long_return:.6f} (n={long_mask.sum()})")
    print(f"  Short avg return:{short_return:.6f} (n={short_mask.sum()})")

    # Per-pair
    id_to_pair = {v: k for k, v in pair_to_id.items()}
    per_pair = {}
    print(f"\n  Per-pair:")
    for pid_val in sorted(np.unique(pid_arr)):
        mask = pid_arr == pid_val
        if mask.sum() < 2:
            continue
        p_pred = pred_arr[mask]
        p_true = true_arr[mask]
        if np.std(p_pred) > 1e-8 and np.std(p_true) > 1e-8:
            p_ic = np.corrcoef(p_pred, p_true)[0, 1]
        else:
            p_ic = 0.0
        p_dir = np.mean((p_pred > 0) == (p_true > 0))
        p_name = id_to_pair.get(pid_val, f"pair_{pid_val}")
        per_pair[p_name] = {
            "ic": float(p_ic),
            "dir_acc": float(p_dir),
            "count": int(mask.sum()),
            "long_pct": float((p_pred > 0).sum() / mask.sum()),
            "avg_pred": float(p_pred.mean()),
            "avg_true": float(p_true.mean()),
        }
        print(f"    {p_name:12s}  IC={p_ic:+.4f}  dir={p_dir:.3f}  long%={per_pair[p_name]['long_pct']:.2f}  n={mask.sum()}")

    # Window-level IC distribution
    window_ics = [w["test_ic"] for w in window_results if not np.isnan(w["test_ic"])]
    if window_ics:
        print(f"\n  Window IC distribution:")
        print(f"    Mean: {np.mean(window_ics):.4f}")
        print(f"    Median: {np.median(window_ics):.4f}")
        print(f"    Positive: {sum(1 for ic in window_ics if ic > 0)}/{len(window_ics)}")
        print(f"    Best:  {max(window_ics):.4f}")
        print(f"    Worst: {min(window_ics):.4f}")

    # Backtest
    print(f"\n{'='*60}")
    print("  V6 Walk-Forward Backtest")
    print(f"{'='*60}")

    bt = run_backtest(all_predictions, pair_names, pair_to_id)
    print(f"\n  Total Return:  {bt['total_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio:  {bt['sharpe']:.2f}")
    print(f"  Max Drawdown:  {bt['max_drawdown_pct']:.1f}%")
    print(f"  Win Rate:      {bt['win_rate_pct']:.1f}%")
    print(f"  Trades:        {bt['trades']}")
    print(f"  Buy&Hold:      {bt['buy_and_hold_return_pct']:+.2f}%")
    print(f"  Final Value:   ${bt['final_value']:.2f}")

    # Save results
    print(f"\n[5/5] Saving results...")

    output = {
        "overall": {
            "ic": float(overall_ic),
            "dir_acc": float(overall_dir),
            "mae": float(mae),
            "rmse": float(rmse),
            "long_return": float(long_return),
            "short_return": float(short_return),
            "long_count": int(long_mask.sum()),
            "short_count": int(short_mask.sum()),
        },
        "per_pair": per_pair,
        "window_results": window_results,
        "backtest": {
            "total_return_pct": bt["total_return_pct"],
            "sharpe": bt["sharpe"],
            "max_drawdown_pct": bt["max_drawdown_pct"],
            "win_rate_pct": bt["win_rate_pct"],
            "trades": bt["trades"],
            "final_value": bt["final_value"],
            "buy_and_hold_return_pct": bt["buy_and_hold_return_pct"],
        },
        "config": {
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "d_model": 128,
            "num_layers": 3,
            "nhead": 4,
            "dim_feedforward": 512,
            "train_window_days": TRAIN_WINDOW_DAYS,
            "step_days": STEP_DAYS,
            "beta_lookback": BETA_LOOKBACK,
            "num_features": NUM_FEATURES_V4,
            "num_pairs": len(pair_names),
            "pairs": pair_names,
            "model_type": "walk_forward_market_neutral",
            "num_windows": len(windows),
            "num_windows_used": len(window_results),
        },
    }

    Path("data").mkdir(exist_ok=True)
    with open("data/results_v6.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to data/results_v6.json")

    # Save equity curve
    with open("data/v6_equity_curve.json", "w") as f:
        json.dump(bt["equity_curve"], f)
    print(f"  Equity curve saved to data/v6_equity_curve.json")

    print(f"\n{'='*70}")
    print("  V6 Walk-Forward Training Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
