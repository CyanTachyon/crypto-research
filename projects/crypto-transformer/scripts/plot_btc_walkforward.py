import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from crypto_transformer.data.collectors.binance import fetch_ohlcv
from crypto_transformer.data.features_v4 import (
    compute_features_v4, compute_cross_sectional_features,
    FEATURE_COLUMNS_V4, NUM_FEATURES_V4,
)

PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "UNI/USDT",
    "OP/USDT", "AAVE/USDT", "LTC/USDT", "ATOM/USDT", "NEAR/USDT",
]
SEQ_LEN = 90
HORIZON = 3
TRAIN_CUTOFF = "2025-06-01"
SEEDS = [42, 123, 456]
CKPT_DIR = Path("data/checkpoints")
THRESHOLD = 0.005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RegressionTransformer(nn.Module):
    def __init__(self, nf=25, np_=15, dm=128, nh=4, nl=3, df=512, dr=0.3, pe=16, ml=256):
        super().__init__()
        self.pair_emb = nn.Embedding(np_, pe)
        self.inp = nn.Sequential(nn.Linear(nf + pe, dm), nn.GELU(), nn.LayerNorm(dm), nn.Dropout(dr * 0.5))
        self.pos = nn.Parameter(torch.randn(1, ml, dm) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model=dm, nhead=nh, dim_feedforward=df, dropout=dr, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=nl)
        self.head = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 64), nn.GELU(), nn.Dropout(dr), nn.Linear(64, 1))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pid):
        B, T, _ = x.shape
        pe = self.pair_emb(pid).unsqueeze(1).expand(B, T, -1)
        x = self.inp(torch.cat([x, pe], dim=-1))
        x = x + self.pos[:, :T, :]
        x = self.enc(x, src_key_padding_mask=(x.abs().sum(-1) == 0))
        return self.head(x[:, -1, :]).squeeze(-1)


def main():
    print("Loading data...")
    raw = {p: fetch_ohlcv(p, "1d") for p in PAIRS}
    ranks = compute_cross_sectional_features(raw, lookback=10)
    all_dfs = {p: compute_features_v4(raw[p], btc_df=raw["BTC/USDT"], cross_sectional_ranks=ranks.get(p)) for p in PAIRS}

    pair_names = sorted(all_dfs.keys())
    pair_to_id = {n: i for i, n in enumerate(pair_names)}

    cutoff = np.datetime64(f"{TRAIN_CUTOFF}T00:00:00", "ns")
    train_concat = pd.concat([all_dfs[p][all_dfs[p]["timestamp"].values < cutoff] for p in pair_names], ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(train_concat[FEATURE_COLUMNS_V4].values)

    btc_df = all_dfs["BTC/USDT"].copy()
    btc_df[FEATURE_COLUMNS_V4] = scaler.transform(btc_df[FEATURE_COLUMNS_V4].values).astype(np.float32)

    test_btc = btc_df[btc_df["timestamp"].values >= cutoff].reset_index(drop=True)
    print(f"BTC test period: {test_btc['timestamp'].min().date()} ~ {test_btc['timestamp'].max().date()} ({len(test_btc)} days)")

    print("Loading models...")
    models = []
    for seed in SEEDS:
        m = RegressionTransformer(nf=NUM_FEATURES_V4, np_=len(pair_names)).to(DEVICE)
        m.load_state_dict(torch.load(CKPT_DIR / f"v4_seed_{seed}.pt", map_location=DEVICE, weights_only=True))
        m.eval()
        models.append(m)

    btc_pid = pair_to_id["BTC/USDT"]
    features = btc_df[FEATURE_COLUMNS_V4].values
    timestamps = btc_df["timestamp"].values
    closes = btc_df["close_raw"].values

    first_test_idx = None
    for i in range(len(btc_df)):
        if btc_df["timestamp"].values[i] >= cutoff:
            first_test_idx = i
            break

    n_pred = len(test_btc) - SEQ_LEN - HORIZON
    start_global = first_test_idx
    print(f"Inference: {n_pred} samples (global start idx={start_global})")

    records = []
    for i in range(n_pred):
        gi = start_global + i + SEQ_LEN
        if gi < SEQ_LEN:
            continue
        x = features[gi - SEQ_LEN:gi].reshape(1, SEQ_LEN, -1)
        x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        pid = torch.tensor([btc_pid], dtype=torch.long, device=DEVICE)

        preds = []
        with torch.no_grad():
            for m in models:
                preds.append(m(x, pid).cpu().item())
        avg_pred = np.mean(preds)

        actual_3d = closes[gi + HORIZON] / closes[gi] - 1 if gi + HORIZON < len(closes) else 0
        next_1d = closes[gi + 1] / closes[gi] - 1 if gi + 1 < len(closes) else 0

        records.append({
            "date": pd.Timestamp(timestamps[gi]),
            "close": float(closes[gi]),
            "pred_3d": float(avg_pred),
            "actual_3d": float(actual_3d),
            "next_1d": float(next_1d),
        })

    df = pd.DataFrame(records)
    print(f"Predictions: {len(df)} days, {df['date'].min().date()} ~ {df['date'].max().date()}")

    df["pred_close_3d"] = df["close"] * (1 + df["pred_3d"])

    df["pred_path"] = np.nan
    df.loc[df.index[0], "pred_path"] = df.loc[df.index[0], "close"]
    for i in range(1, len(df)):
        df.loc[df.index[i], "pred_path"] = df.loc[df.index[i - 1], "pred_path"] * (1 + df.loc[df.index[i], "actual_3d"] * np.sign(df.loc[df.index[i], "pred_3d"]))

    df["strategy_cumret"] = (1 + df["next_1d"] * 0).cumprod()
    cum = 1.0
    cum_arr = []
    for _, r in df.iterrows():
        if r["pred_3d"] > THRESHOLD:
            cum *= (1 + r["next_1d"] * 0.1)
        elif r["pred_3d"] < -THRESHOLD:
            cum *= (1 + (-r["next_1d"]) * 0.1)
        cum_arr.append(cum)
    df["strategy_cumret"] = cum_arr
    df["bnh_cumret"] = df["close"] / df["close"].iloc[0]

    buy_mask = df["pred_3d"] > THRESHOLD
    sell_mask = df["pred_3d"] < -THRESHOLD

    print(f"  BUY: {buy_mask.sum()}, SELL: {sell_mask.sum()}, HOLD: {(~buy_mask & ~sell_mask).sum()}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.collections import LineCollection

    BG = "#0d1117"
    TEXT = "#c9d1d9"
    GRID = "#21262d"
    BLUE = "#58a6ff"
    GREEN = "#3fb950"
    RED = "#f85149"
    ORANGE = "#d29922"
    PURPLE = "#bc8cff"

    fig = plt.figure(figsize=(24, 16), facecolor=BG)
    gs = fig.add_gridspec(4, 1, height_ratios=[4, 1.5, 1.5, 2], hspace=0.12)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(BG)
        ax.tick_params(colors="#8b949e", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["left"].set_color(GRID)
        ax.grid(True, alpha=0.15, color=GRID)

    dates = df["date"].values
    x_idx = np.arange(len(df))

    ax1.fill_between(x_idx, df["close"].min() * 0.97, df["close"], alpha=0.06, color=BLUE)
    ax1.plot(x_idx, df["close"], color=BLUE, linewidth=2.2, label="BTC Actual Price", zorder=3)

    buy_idx = df.index[buy_mask]
    sell_idx = df.index[sell_mask]
    ax1.scatter(buy_idx, df.loc[buy_mask, "close"], marker="^", s=120, color=GREEN,
                edgecolors="white", linewidths=0.8, zorder=10, label=f"BUY (pred > {THRESHOLD:.1%})")
    ax1.scatter(sell_idx, df.loc[sell_mask, "close"], marker="v", s=120, color=RED,
                edgecolors="white", linewidths=0.8, zorder=10, label=f"SELL (pred < -{THRESHOLD:.1%})")

    for i in buy_idx:
        ax1.axvspan(i - 0.4, i + 0.4, alpha=0.08, color=GREEN, zorder=1)
    for i in sell_idx:
        ax1.axvspan(i - 0.4, i + 0.4, alpha=0.08, color=RED, zorder=1)

    pred_dots = ax1.scatter(x_idx, df["pred_close_3d"], s=8, color=ORANGE, alpha=0.4, zorder=5, label="Predicted price (3d ahead)")

    ic = np.corrcoef(df["pred_3d"].values, df["actual_3d"].values)[0, 1]
    dir_acc = np.mean((df["pred_3d"].values > 0) == (df["actual_3d"].values > 0))
    buy_3d_avg = df.loc[buy_mask, "actual_3d"].mean() * 100 if buy_mask.sum() > 0 else 0
    sell_3d_avg = -df.loc[sell_mask, "actual_3d"].mean() * 100 if sell_mask.sum() > 0 else 0

    title_lines = (
        f"BTC/USDT Walk-Forward Prediction  |  "
        f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}  |  "
        f"IC={ic:.3f}  Dir={dir_acc:.1%}  |  "
        f"BUY avg 3d ret: {buy_3d_avg:+.2f}%  SELL avg short ret: {sell_3d_avg:+.2f}%"
    )
    ax1.set_title(title_lines, color=TEXT, fontsize=13, pad=15, fontfamily="monospace", fontweight="bold")
    ax1.set_ylabel("Price (USDT)", color=TEXT, fontsize=12)
    ax1.legend(loc="upper right", fontsize=10, facecolor="#161b22", edgecolor=GRID, labelcolor=TEXT, framealpha=0.95)

    for price_val, label, color in [
        (100000, "$100K", "#484f58"),
        (80000, "$80K", "#484f58"),
        (60000, "$60K", "#484f58"),
    ]:
        if df["close"].min() < price_val < df["close"].max():
            ax1.axhline(y=price_val, color=color, linestyle=":", alpha=0.4, linewidth=0.8)
            ax1.text(len(df) * 0.01, price_val, label, color=color, fontsize=9, va="bottom")

    pred_colors = np.where(df["pred_3d"].values > 0, GREEN, RED)
    ax2.bar(x_idx, df["pred_3d"].values * 100, color=pred_colors, alpha=0.7, width=0.8)
    ax2.axhline(y=THRESHOLD * 100, color=GREEN, linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=-THRESHOLD * 100, color=RED, linestyle="--", alpha=0.5, linewidth=1)
    ax2.axhline(y=0, color=GRID, linewidth=0.5)
    ax2.set_ylabel("Predicted\n3d Return (%)", color=TEXT, fontsize=10)

    actual_colors = np.where(df["actual_3d"].values > 0, GREEN, RED)
    ax3.bar(x_idx, df["actual_3d"].values * 100, color=actual_colors, alpha=0.7, width=0.8)
    ax3.axhline(y=0, color=GRID, linewidth=0.5)
    ax3.set_ylabel("Actual\n3d Return (%)", color=TEXT, fontsize=10)

    correct = (df["pred_3d"].values > 0) == (df["actual_3d"].values > 0)
    acc_pct = correct.mean() * 100
    ax3.text(len(df) * 0.01, ax3.get_ylim()[1] * 0.85 if ax3.get_ylim()[1] > 0 else 0.05,
             f"Direction match: {acc_pct:.1f}%", color=PURPLE, fontsize=10, fontweight="bold")

    ax4.plot(x_idx, df["bnh_cumret"].values * 100, color="#8b949e", linewidth=2, linestyle="--",
             label=f"Buy & Hold ({(df['bnh_cumret'].iloc[-1] - 1) * 100:+.1f}%)", zorder=3)
    ax4.plot(x_idx, df["strategy_cumret"].values * 100, color=PURPLE, linewidth=2.5,
             label=f"V4 Strategy 10% pos ({(df['strategy_cumret'].iloc[-1] - 1) * 100:+.1f}%)", zorder=4)
    ax4.axhline(y=100, color=GRID, linewidth=0.5)
    ax4.fill_between(x_idx, 100, df["strategy_cumret"].values * 100,
                     where=df["strategy_cumret"].values > 1, alpha=0.15, color=GREEN)
    ax4.fill_between(x_idx, 100, df["strategy_cumret"].values * 100,
                     where=df["strategy_cumret"].values <= 1, alpha=0.15, color=RED)
    ax4.set_ylabel("Portfolio Value\n(Indexed to 100)", color=TEXT, fontsize=11)
    ax4.set_xlabel("Date", color=TEXT, fontsize=12)
    ax4.legend(loc="upper left", fontsize=11, facecolor="#161b22", edgecolor=GRID, labelcolor=TEXT, framealpha=0.95)

    step = max(1, len(df) // 20)
    tick_pos = list(range(0, len(df), step))
    tick_labels = [pd.Timestamp(dates[i]).strftime("%Y-%m") for i in tick_pos]
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=9)

    fig.text(0.5, 0.01,
             "V4 Transformer  |  3-Seed Ensemble  |  Huber + Pairwise Ranking Loss  |  "
             f"Training: up to {TRAIN_CUTOFF}  |  Test: walk-forward, no lookahead",
             ha="center", color="#484f58", fontsize=10, fontfamily="monospace")

    plt.savefig("docs/figures/btc_walkforward_prediction.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\nSaved: docs/figures/btc_walkforward_prediction.png")
    print(f"\nSummary:")
    print(f"  Period:     {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"  BTC range:  ${df['close'].min():,.0f} ~ ${df['close'].max():,.0f}")
    print(f"  IC:         {ic:.4f}")
    print(f"  Dir acc:    {dir_acc:.2%}")
    print(f"  Buy&Hold:   {(df['bnh_cumret'].iloc[-1] - 1) * 100:+.1f}%")
    print(f"  V4 Strategy:{(df['strategy_cumret'].iloc[-1] - 1) * 100:+.1f}%")
    print(f"  BUY signals:  {buy_mask.sum()} (avg 3d ret after buy: {buy_3d_avg:+.2f}%)")
    print(f"  SELL signals: {sell_mask.sum()} (avg 3d ret after short: {sell_3d_avg:+.2f}%)")


if __name__ == "__main__":
    main()
