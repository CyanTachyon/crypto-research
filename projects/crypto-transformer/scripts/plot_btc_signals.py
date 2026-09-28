import numpy as np
import pandas as pd
import torch
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


class RegressionTransformer(torch.nn.Module):
    def __init__(self, nf=25, np_=15, dm=128, nh=4, nl=3, df=512, dr=0.3, pe=16, ml=256):
        super().__init__()
        self.pair_emb = torch.nn.Embedding(np_, pe)
        self.inp = torch.nn.Sequential(torch.nn.Linear(nf + pe, dm), torch.nn.GELU(), torch.nn.LayerNorm(dm), torch.nn.Dropout(dr * 0.5))
        self.pos = torch.nn.Parameter(torch.randn(1, ml, dm) * 0.02)
        enc = torch.nn.TransformerEncoderLayer(d_model=dm, nhead=nh, dim_feedforward=df, dropout=dr, activation="gelu", batch_first=True, norm_first=True)
        self.enc = torch.nn.TransformerEncoder(enc, num_layers=nl)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(dm), torch.nn.Linear(dm, 64), torch.nn.GELU(), torch.nn.Dropout(dr), torch.nn.Linear(64, 1))
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, pid):
        B, T, _ = x.shape
        pe = self.pair_emb(pid).unsqueeze(1).expand(B, T, -1)
        x = self.inp(torch.cat([x, pe], dim=-1))
        x = x + self.pos[:, :T, :]
        x = self.enc(x, src_key_padding_mask=(x.abs().sum(-1) == 0))
        return self.head(x[:, -1, :]).squeeze(-1)


def main():
    btc_raw = fetch_ohlcv("BTC/USDT", "1d")
    six_months_ago = pd.Timestamp("2025-12-01")
    print(f"BTC raw: {len(btc_raw)} days, {btc_raw['timestamp'].min().date()} to {btc_raw['timestamp'].max().date()}")

    raw = {p: fetch_ohlcv(p, "1d") for p in PAIRS}
    ranks = compute_cross_sectional_features(raw, lookback=10)
    all_dfs = {p: compute_features_v4(raw[p], btc_df=raw["BTC/USDT"], cross_sectional_ranks=ranks.get(p)) for p in PAIRS}

    pair_names = sorted(all_dfs.keys())
    cutoff = np.datetime64(f"{TRAIN_CUTOFF}T00:00:00", "ns")
    train_concat = pd.concat([all_dfs[p][all_dfs[p]["timestamp"].values < cutoff] for p in pair_names], ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(train_concat[FEATURE_COLUMNS_V4].values)

    btc_feat = all_dfs["BTC/USDT"].copy()
    btc_feat[FEATURE_COLUMNS_V4] = scaler.transform(btc_feat[FEATURE_COLUMNS_V4].values).astype(np.float32)

    btc_6m = btc_feat[btc_feat["timestamp"] >= "2025-06-01"].reset_index(drop=True)
    print(f"Feature rows in window (from 2025-06-01): {len(btc_6m)}")

    models = []
    for seed in SEEDS:
        ckpt = CKPT_DIR / f"v4_seed_{seed}.pt"
        m = RegressionTransformer(nf=NUM_FEATURES_V4, np_=len(pair_names)).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
        models.append(m)

    pair_to_id = {n: i for i, n in enumerate(pair_names)}
    btc_pid = pair_to_id["BTC/USDT"]

    features = btc_6m[FEATURE_COLUMNS_V4].values
    n = len(btc_6m) - SEQ_LEN - HORIZON
    print(f"Inference samples: {n}")

    all_preds = []
    batch_size = 256
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xs = np.array([features[i:i + SEQ_LEN] for i in range(start, end)])
        x = torch.tensor(xs, dtype=torch.float32).to(DEVICE)
        pid = torch.full((end - start,), btc_pid, dtype=torch.long, device=DEVICE)
        preds = []
        with torch.no_grad():
            for m in models:
                preds.append(m(x, pid).cpu().numpy())
        all_preds.extend(np.mean(preds, axis=0))

    dates = [btc_6m["timestamp"].values[i + SEQ_LEN] for i in range(n)]
    closes = [btc_6m["close_raw"].values[i + SEQ_LEN] for i in range(n)]
    actual_3d = []
    for i in range(n):
        ci = i + SEQ_LEN
        if ci + HORIZON < len(btc_6m):
            actual_3d.append(btc_6m["close_raw"].values[ci + HORIZON] / btc_6m["close_raw"].values[ci] - 1)
        else:
            actual_3d.append(0.0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import FancyBboxPatch

    pred_arr = np.array(all_preds)
    close_arr = np.array(closes)
    date_arr = pd.to_datetime(dates)
    actual_arr = np.array(actual_3d)

    buy_mask = pred_arr > THRESHOLD
    sell_mask = pred_arr < -THRESHOLD
    hold_mask = ~buy_mask & ~sell_mask

    fig, (ax_price, ax_pred) = plt.subplots(
        2, 1, figsize=(20, 12), height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.15},
    )

    ax_price.set_facecolor("#0d1117")
    ax_pred.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax_price.fill_between(range(len(close_arr)), close_arr.min() * 0.98, close_arr, alpha=0.08, color="#58a6ff")
    ax_price.plot(range(len(close_arr)), close_arr, color="#58a6ff", linewidth=2.0, label="BTC/USDT", zorder=2)

    for i in range(len(pred_arr)):
        if buy_mask[i]:
            ax_price.annotate(
                "", xy=(i, close_arr[i]), xytext=(i, close_arr[i] * 0.975),
                arrowprops=dict(arrowstyle="-|>", color="#3fb950", lw=2, mutation_scale=15),
                zorder=5,
            )
        elif sell_mask[i]:
            ax_price.annotate(
                "", xy=(i, close_arr[i]), xytext=(i, close_arr[i] * 1.025),
                arrowprops=dict(arrowstyle="-|>", color="#f85149", lw=2, mutation_scale=15),
                zorder=5,
            )

    buy_idx = np.where(buy_mask)[0]
    sell_idx = np.where(sell_mask)[0]
    ax_price.scatter(buy_idx, close_arr[buy_mask], marker="^", s=80, color="#3fb950",
                     edgecolors="white", linewidths=0.5, zorder=6, label=f"BUY (pred > {THRESHOLD:.1%})")
    ax_price.scatter(sell_idx, close_arr[sell_mask], marker="v", s=80, color="#f85149",
                     edgecolors="white", linewidths=0.5, zorder=6, label=f"SELL (pred < -{THRESHOLD:.1%})")

    n_buy = buy_mask.sum()
    n_sell = sell_mask.sum()
    n_hold = hold_mask.sum()
    buy_actual = actual_arr[buy_mask].mean() * 100 if n_buy > 0 else 0
    sell_actual = -actual_arr[sell_mask].mean() * 100 if n_sell > 0 else 0

    info_text = (
        f"Period: {date_arr[0].strftime('%Y-%m-%d')} ~ {date_arr[-1].strftime('%Y-%m-%d')}  |  "
        f"Days: {len(pred_arr)}  |  "
        f"BUY: {n_buy}  SELL: {n_sell}  HOLD: {n_hold}  |  "
        f"Avg 3d return after BUY: {buy_actual:+.2f}%  |  "
        f"Avg 3d return after SELL (short): {sell_actual:+.2f}%"
    )
    ax_price.set_title(info_text, color="#c9d1d9", fontsize=12, pad=15, fontfamily="monospace")

    ax_price.set_ylabel("Price (USDT)", color="#c9d1d9", fontsize=13)
    ax_price.legend(loc="upper left", fontsize=11, facecolor="#161b22", edgecolor="#30363d",
                    labelcolor="#c9d1d9", framealpha=0.9)
    ax_price.tick_params(colors="#8b949e")
    ax_price.spines["bottom"].set_color("#30363d")
    ax_price.spines["left"].set_color("#30363d")
    ax_price.spines["top"].set_visible(False)
    ax_price.spines["right"].set_visible(False)
    ax_price.grid(True, alpha=0.1, color="#30363d")

    step = max(1, len(date_arr) // 15)
    tick_positions = list(range(0, len(date_arr), step))
    tick_labels = [date_arr[i].strftime("%m-%d") for i in tick_positions]
    ax_price.set_xticks(tick_positions)
    ax_price.set_xticklabels(tick_labels, color="#8b949e", fontsize=9)

    colors_pred = np.where(pred_arr > 0, "#3fb950", "#f85149")
    ax_pred.bar(range(len(pred_arr)), pred_arr * 100, color=colors_pred, alpha=0.7, width=0.8)
    ax_pred.axhline(y=THRESHOLD * 100, color="#3fb950", linestyle="--", alpha=0.5, linewidth=1)
    ax_pred.axhline(y=-THRESHOLD * 100, color="#f85149", linestyle="--", alpha=0.5, linewidth=1)
    ax_pred.axhline(y=0, color="#484f58", linewidth=0.5)

    ax_pred.set_ylabel("Predicted 3d\nReturn (%)", color="#c9d1d9", fontsize=11)
    ax_pred.set_xlabel("Date", color="#c9d1d9", fontsize=11)
    ax_pred.tick_params(colors="#8b949e")
    ax_pred.spines["bottom"].set_color("#30363d")
    ax_pred.spines["left"].set_color("#30363d")
    ax_pred.spines["top"].set_visible(False)
    ax_pred.spines["right"].set_visible(False)
    ax_pred.grid(True, alpha=0.1, color="#30363d")
    ax_pred.set_xticks(tick_positions)
    ax_pred.set_xticklabels(tick_labels, color="#8b949e", fontsize=9, rotation=45)

    fig.text(0.5, 0.01, "V4 Transformer Model  |  3-Seed Ensemble  |  Prediction Horizon: 3 Days  |  Threshold: 0.5%",
             ha="center", color="#484f58", fontsize=10, fontfamily="monospace")

    plt.savefig("docs/figures/btc_trading_signals.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nChart saved: docs/figures/btc_trading_signals.png")
    print(f"\nSignal summary:")
    print(f"  BUY signals:  {n_buy:3d}  (avg actual 3d return: {buy_actual:+.2f}%)")
    print(f"  SELL signals: {n_sell:3d}  (avg actual 3d return after short: {sell_actual:+.2f}%)")
    print(f"  HOLD signals: {n_hold:3d}")
    print(f"  BTC price range: ${close_arr.min():,.0f} ~ ${close_arr.max():,.0f}")
    print(f"  Period: {date_arr[0].date()} ~ {date_arr[-1].date()}")


if __name__ == "__main__":
    main()
