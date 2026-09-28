import json
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
VAL_FRAC = 0.15
BATCH_SIZE = 256
SEEDS = [42, 123, 456]
CKPT_DIR = Path("data/checkpoints")
INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.001
NUM_PAIRS = len(PAIRS)


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin

    def forward(self, pred, target):
        half = len(pred) // 2
        if half < 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return torch.relu(self.margin - (target[:half] - target[half:2*half]) * (pred[:half] - pred[half:2*half])).mean()


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


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_test_index():
    btc = fetch_ohlcv("BTC/USDT", "1d")
    raw = {p: fetch_ohlcv(p, "1d") for p in PAIRS}
    ranks = compute_cross_sectional_features(raw, lookback=10)
    all_dfs = {p: compute_features_v4(raw[p], btc_df=btc, cross_sectional_ranks=ranks.get(p)) for p in PAIRS}

    pair_names = sorted(all_dfs.keys())
    pair_to_id = {n: i for i, n in enumerate(pair_names)}

    cutoff = np.datetime64(f"{TRAIN_CUTOFF}T00:00:00", "ns")
    train_concat = pd.concat([all_dfs[p][all_dfs[p]["timestamp"].values < cutoff] for p in pair_names], ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(train_concat[FEATURE_COLUMNS_V4].values)

    test_dfs = {}
    for p in pair_names:
        df = all_dfs[p][all_dfs[p]["timestamp"].values >= cutoff].copy().reset_index(drop=True)
        df[FEATURE_COLUMNS_V4] = scaler.transform(df[FEATURE_COLUMNS_V4].values).astype(np.float32)
        test_dfs[p] = df

    return test_dfs, pair_names, pair_to_id


def train_all_seeds(pair_names, pair_to_id, device):
    btc = fetch_ohlcv("BTC/USDT", "1d")
    raw = {p: fetch_ohlcv(p, "1d") for p in PAIRS}
    ranks = compute_cross_sectional_features(raw, lookback=10)
    all_dfs = {p: compute_features_v4(raw[p], btc_df=btc, cross_sectional_ranks=ranks.get(p)) for p in PAIRS}

    cutoff = np.datetime64(f"{TRAIN_CUTOFF}T00:00:00", "ns")
    train_parts = {}
    for p in pair_names:
        df = all_dfs[p]
        train_df = df[df["timestamp"].values < cutoff].copy().reset_index(drop=True)
        n_val = int(len(train_df) * VAL_FRAC)
        train_df = train_df.iloc[:-n_val].copy()
        train_parts[p] = train_df

    train_all = pd.concat([train_parts[p] for p in pair_names], ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(train_all[FEATURE_COLUMNS_V4].values)

    for p in pair_names:
        train_parts[p][FEATURE_COLUMNS_V4] = scaler.transform(train_parts[p][FEATURE_COLUMNS_V4].values).astype(np.float32)

    all_train = pd.concat([train_parts[p] for p in pair_names], ignore_index=True)
    feat_all = torch.tensor(all_train[FEATURE_COLUMNS_V4].values, dtype=torch.float32)
    close_all = all_train["close_raw"].values
    ret_all = pd.Series(close_all).pct_change(HORIZON).shift(-HORIZON).values.copy()
    ret_all[np.isnan(ret_all)] = 0.0
    ret_t = torch.tensor(ret_all, dtype=torch.float32)

    pid_map = {p: i for i, p in enumerate(pair_names)}

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    models = []

    for seed in SEEDS:
        set_seed(seed)
        ckpt = CKPT_DIR / f"v4_seed_{seed}.pt"
        model = RegressionTransformer(nf=NUM_FEATURES_V4, np_=len(pair_names)).to(device)

        if ckpt.exists():
            print(f"  Loading {ckpt.name}")
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        else:
            print(f"  Training seed {seed}...")
            huber = nn.HuberLoss(delta=0.1)
            ranking = PairwiseRankingLoss(margin=0.01)
            opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.08)

            n = len(feat_all)
            for epoch in range(100):
                model.train()
                idx = torch.randperm(n)
                eloss = 0
                nb = 0
                for s in range(0, n - SEQ_LEN - HORIZON, 256):
                    bi = idx[s:s+256]
                    bi = bi[(bi + SEQ_LEN < n) & (bi >= 0)]
                    if len(bi) == 0:
                        continue
                    xs = torch.stack([feat_all[b:b+SEQ_LEN] for b in bi]).to(device)
                    ys = torch.stack([ret_t[b+SEQ_LEN] for b in bi]).to(device)
                    pids = torch.zeros(len(bi), dtype=torch.long, device=device)
                    opt.zero_grad()
                    pred = model(xs, pids)
                    hl = huber(pred, ys)
                    pm = torch.randperm(len(pred))
                    rl = ranking(pred[pm], ys[pm])
                    loss = 0.7 * hl + 0.3 * rl
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    eloss += loss.item()
                    nb += 1
                if (epoch + 1) % 20 == 0:
                    print(f"    E{epoch+1:3d} loss={eloss/max(nb,1):.5f}")

            torch.save(model.state_dict(), ckpt)
            print(f"    Saved {ckpt.name}")

        models.append(model)

    return models


def run_inference(models, test_dfs, pair_names, pair_to_id, device):
    print("\nRunning inference...")
    records = []

    for pair_name in pair_names:
        df = test_dfs[pair_name]
        feat = df[FEATURE_COLUMNS_V4].values
        ts = df["timestamp"].values
        close = df["close_raw"].values
        pid = pair_to_id[pair_name]

        n = len(df) - SEQ_LEN - HORIZON
        if n <= 0:
            continue

        all_preds = []
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            xs = np.array([feat[i:i+SEQ_LEN] for i in range(start, end)])
            pids = np.full(end - start, pid)
            x = torch.tensor(xs, dtype=torch.float32).to(device)
            p = torch.tensor(pids, dtype=torch.long).to(device)

            preds = []
            with torch.no_grad():
                for m in models:
                    m.eval()
                    preds.append(m(x, p).cpu().numpy())
            all_preds.extend(np.mean(preds, axis=0))

        for i in range(n):
            di = i + SEQ_LEN
            rec = {
                "date": pd.Timestamp(ts[di]),
                "pair": pair_name,
                "pred_3d": float(all_preds[i]),
                "close": float(close[di]),
            }
            if di + 1 < len(close):
                rec["next_1d_ret"] = float(close[di+1] / close[di] - 1)
            else:
                rec["next_1d_ret"] = 0.0
            if di + HORIZON < len(close):
                rec["actual_3d_ret"] = float(close[di+HORIZON] / close[di] - 1)
            else:
                rec["actual_3d_ret"] = 0.0
            records.append(rec)

    df = pd.DataFrame(records)
    print(f"  Predictions: {len(df)}, dates: {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def strategy_long_short(pred_df, threshold=0.0, fee=FEE_RATE):
    dates = sorted(pred_df["date"].unique())
    pv = [INITIAL_CAPITAL]
    dr = []
    trades = 0
    pos_size = 1.0 / NUM_PAIRS

    for date in dates:
        day = pred_df[pred_df["date"] == date]
        pnl = 0.0
        n_act = 0
        for _, r in day.iterrows():
            p = r["pred_3d"]
            ret = r["next_1d_ret"]
            if p > threshold:
                pnl += pos_size * ret
                n_act += 1
            elif p < -threshold:
                pnl -= pos_size * ret
                n_act += 1
        fee_cost = n_act * fee * pos_size
        net = pnl - fee_cost
        pv.append(pv[-1] * (1 + net))
        dr.append(net)
        trades += n_act
    return pv, dr, trades


def strategy_long_only(pred_df, threshold=0.0, fee=FEE_RATE):
    dates = sorted(pred_df["date"].unique())
    pv = [INITIAL_CAPITAL]
    dr = []
    trades = 0
    pos_size = 1.0 / NUM_PAIRS

    for date in dates:
        day = pred_df[pred_df["date"] == date]
        pnl = 0.0
        n_act = 0
        for _, r in day.iterrows():
            if r["pred_3d"] > threshold:
                pnl += pos_size * r["next_1d_ret"]
                n_act += 1
        fee_cost = n_act * fee * pos_size
        net = pnl - fee_cost
        pv.append(pv[-1] * (1 + net))
        dr.append(net)
        trades += n_act
    return pv, dr, trades


def strategy_top_k(pred_df, k=3, fee=FEE_RATE):
    dates = sorted(pred_df["date"].unique())
    pv = [INITIAL_CAPITAL]
    dr = []
    trades = 0

    for date in dates:
        day = pred_df[pred_df["date"] == date].sort_values("pred_3d", ascending=False)
        n = len(day)
        if n < 2 * k:
            pv.append(pv[-1])
            dr.append(0.0)
            continue

        ps = 1.0 / (2 * k)
        pnl = 0.0
        for _, r in day.head(k).iterrows():
            pnl += ps * r["next_1d_ret"]
        for _, r in day.tail(k).iterrows():
            pnl -= ps * r["next_1d_ret"]
        trades += 2 * k
        fee_cost = 2 * k * fee * ps
        net = pnl - fee_cost
        pv.append(pv[-1] * (1 + net))
        dr.append(net)
    return pv, dr, trades


def strategy_buy_and_hold(pred_df):
    pairs = pred_df["pair"].unique()
    dates = sorted(pred_df["date"].unique())
    init_prices = {}
    for p in pairs:
        first = pred_df[pred_df["pair"] == p].sort_values("date").iloc[0]["close"]
        init_prices[p] = first

    alloc = INITIAL_CAPITAL / len(pairs)
    shares = {p: alloc / init_prices[p] for p in pairs}

    pv = []
    for date in dates:
        total = 0.0
        for p in pairs:
            row = pred_df[(pred_df["pair"] == p) & (pred_df["date"] == date)]
            if len(row) > 0:
                total += shares[p] * row.iloc[0]["close"]
        pv.append(total)
    return pv


def metrics(pv, dr, trades, name):
    pv, dr = np.array(pv), np.array(dr)
    ret = (pv[-1] / pv[0] - 1) * 100
    sharpe = np.mean(dr) / (np.std(dr) + 1e-10) * np.sqrt(365)
    peak = np.maximum.accumulate(pv)
    mdd = ((pv - peak) / peak).min() * 100
    wr = np.mean(dr > 0) * 100 if len(dr) > 0 else 0
    return {
        "strategy": name,
        "total_return_pct": round(ret, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(mdd, 2),
        "win_rate_pct": round(wr, 1),
        "trades": trades,
        "final_value": round(pv[-1], 2),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("V4 Backtest: Train + Inference + Trading Simulation")
    print("=" * 70)

    print("\n[1/4] Building test data index...")
    test_dfs, pair_names, pair_to_id = build_test_index()

    print("\n[2/4] Training/loading models...")
    models = train_all_seeds(pair_names, pair_to_id, device)

    print("\n[3/4] Running inference...")
    pred_df = run_inference(models, test_dfs, pair_names, pair_to_id, device)

    overall_ic = np.corrcoef(pred_df["pred_3d"].values, pred_df["actual_3d_ret"].values)[0, 1]
    overall_dir = np.mean((pred_df["pred_3d"].values > 0) == (pred_df["actual_3d_ret"].values > 0))
    print(f"  Overall IC: {overall_ic:.4f}, Dir acc: {overall_dir:.4f}")

    print("\n[4/4] Trading simulation...")
    print("-" * 70)

    all_results = {}
    all_curves = {}

    for th in [0.0, 0.005, 0.01]:
        name = f"LS(th={th:.3f})"
        pv, dr, tr = strategy_long_short(pred_df, threshold=th)
        m = metrics(pv, dr, tr, name)
        all_results[name] = m
        all_curves[name] = pv
        print(f"  {name:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  "
              f"mdd={m['max_drawdown_pct']:6.1f}%  win={m['win_rate_pct']:4.1f}%  trades={tr}")

    for th in [0.0, 0.005, 0.01]:
        name = f"LO(th={th:.3f})"
        pv, dr, tr = strategy_long_only(pred_df, threshold=th)
        m = metrics(pv, dr, tr, name)
        all_results[name] = m
        all_curves[name] = pv
        print(f"  {name:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  "
              f"mdd={m['max_drawdown_pct']:6.1f}%  win={m['win_rate_pct']:4.1f}%  trades={tr}")

    for k in [3, 5]:
        name = f"Top{k}-LS"
        pv, dr, tr = strategy_top_k(pred_df, k=k)
        m = metrics(pv, dr, tr, name)
        all_results[name] = m
        all_curves[name] = pv
        print(f"  {name:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  "
              f"mdd={m['max_drawdown_pct']:6.1f}%  win={m['win_rate_pct']:4.1f}%  trades={tr}")

    bnh = strategy_buy_and_hold(pred_df)
    bnh_ret = (bnh[-1] / bnh[0] - 1) * 100
    bnh_peak = np.maximum.accumulate(bnh)
    bnh_mdd = ((np.array(bnh) - bnh_peak) / bnh_peak).min() * 100
    all_curves["Buy&Hold"] = bnh
    print(f"  {'Buy&Hold':18s}  ret={bnh_ret:+7.2f}%  mdd={bnh_mdd:.1f}%")

    print(f"\nPer-pair breakdown (LS th=0.005, 10% position):")
    print("-" * 70)
    for pair in sorted(pred_df["pair"].unique()):
        pd_pair = pred_df[pred_df["pair"] == pair]
        ic = np.corrcoef(pd_pair["pred_3d"].values, pd_pair["actual_3d_ret"].values)[0, 1]
        dates = sorted(pd_pair["date"].unique())
        pv_p = [100.0]
        for d in dates:
            row = pd_pair[pd_pair["date"] == d].iloc[0]
            p = row["pred_3d"]
            ret = row["next_1d_ret"]
            pos = 0.0
            if p > 0.005:
                pos = ret
            elif p < -0.005:
                pos = -ret
            pv_p.append(pv_p[-1] * (1 + pos * 0.1))
        pair_ret = (pv_p[-1] / pv_p[0] - 1) * 100
        n_long = (pd_pair["pred_3d"] > 0.005).sum()
        n_short = (pd_pair["pred_3d"] < -0.005).sum()
        print(f"  {pair:12s}  IC={ic:+.4f}  ret={pair_ret:+6.2f}%  long={n_long:3d}  short={n_short:3d}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})

    ax = axes[0, 0]
    for name in ["LS(th=0.000)", "LS(th=0.005)", "LS(th=0.010)"]:
        if name in all_curves:
            ax.plot(all_curves[name], label=name, linewidth=1.5)
    ax.plot(all_curves["Buy&Hold"], label="Buy&Hold", color="gray", linestyle="--", linewidth=1.5)
    ax.set_title("Long/Short Strategies")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    for name in ["LO(th=0.000)", "LO(th=0.005)", "LO(th=0.010)"]:
        if name in all_curves:
            ax.plot(all_curves[name], label=name, linewidth=1.5)
    ax.plot(all_curves["Buy&Hold"], label="Buy&Hold", color="gray", linestyle="--", linewidth=1.5)
    ax.set_title("Long-Only Strategies")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    for name in ["Top3-LS", "Top5-LS"]:
        if name in all_curves:
            ax.plot(all_curves[name], label=name, linewidth=1.5)
    ax.plot(all_curves["Buy&Hold"], label="Buy&Hold", color="gray", linestyle="--", linewidth=1.5)
    ax.set_title("Top-K Strategies")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]:
        pd_p = pred_df[pred_df["pair"] == pair].sort_values("date")
        cumret = (1 + pd_p["next_1d_ret"].values).cumprod() * 100
        ax.plot(range(len(cumret)), cumret, label=pair.replace("/USDT", ""))
    ax.set_title("Individual Coins (Buy&Hold, Indexed to 100)")
    ax.set_ylabel("Indexed Value")
    ax.legend(fontsize=9)

    fig.suptitle(f"V4 Model Backtest ({pred_df['date'].min().date()} ~ {pred_df['date'].max().date()})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("docs/figures/v4_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()

    output = {
        "strategies": all_results,
        "buy_and_hold_return_pct": round(bnh_ret, 2),
        "buy_and_hold_mdd_pct": round(bnh_mdd, 2),
        "overall_ic": float(overall_ic),
        "overall_dir_acc": float(overall_dir),
        "test_period": f"{pred_df['date'].min().date()} to {pred_df['date'].max().date()}",
        "initial_capital": INITIAL_CAPITAL,
        "fee_rate": FEE_RATE,
    }
    with open("data/backtest_v4_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone! Chart: docs/figures/v4_backtest.png")
    print(f"Results: data/backtest_v4_results.json")


if __name__ == "__main__":
    main()
