import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from crypto_transformer.data.collectors.binance import fetch_ohlcv
from crypto_transformer.data.features_v5 import compute_features_v5, compute_cross_sectional_features, NUM_FEATURES_V5
from crypto_transformer.data.dataset_v3 import CryptoDatasetV3
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd


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
MAX_EPOCHS = 100
PATIENCE = 20
LR = 1.5e-4
WEIGHT_DECAY = 0.08
GRADIENT_CLIP = 1.0
WARMUP_EPOCHS = 5
NUM_SEEDS = 3
SEEDS = [42, 123, 456]
RANKING_MARGIN = 0.01
RANKING_WEIGHT = 0.3
HUBER_WEIGHT = 0.7
INITIAL_CAPITAL = 10000.0
FEE_RATE = 0.001
NUM_PAIRS = len(PAIRS)
CKPT_DIR = Path("data/checkpoints")
EXTERNAL_DIR = Path("data/external")


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin: float = 0.01):
        super().__init__()
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        half = len(pred) // 2
        if half < 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        p1, p2 = pred[:half], pred[half:2*half]
        t1, t2 = target[:half], target[half:2*half]
        diff_true = t1 - t2
        diff_pred = p1 - p2
        loss = torch.relu(self.margin - diff_true * diff_pred)
        return loss.mean()


class RegressionTransformer(nn.Module):
    def __init__(self, num_features=42, num_pairs=15, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=512, dropout=0.3, pair_emb_dim=16, max_len=256):
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


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_external_data():
    """Load all external data from parquet files."""
    external_data = {}

    fg_path = EXTERNAL_DIR / "fear_greed.parquet"
    if fg_path.exists():
        external_data["fear_greed"] = pd.read_parquet(fg_path)
        print(f"  Loaded fear_greed: {len(external_data['fear_greed'])} rows")
    else:
        external_data["fear_greed"] = pd.DataFrame()
        print(f"  fear_greed not found, using defaults")

    dom_path = EXTERNAL_DIR / "btc_dominance.parquet"
    if dom_path.exists():
        external_data["btc_dominance"] = pd.read_parquet(dom_path)
        print(f"  Loaded btc_dominance: {len(external_data['btc_dominance'])} rows")
    else:
        external_data["btc_dominance"] = pd.DataFrame()
        print(f"  btc_dominance not found, using defaults")

    oc_path = EXTERNAL_DIR / "blockchain_onchain.parquet"
    if oc_path.exists():
        external_data["onchain"] = pd.read_parquet(oc_path)
        print(f"  Loaded onchain: {len(external_data['onchain'])} rows")
    else:
        external_data["onchain"] = pd.DataFrame()
        print(f"  onchain not found, using defaults")

    tvl_path = EXTERNAL_DIR / "defi_tvl.parquet"
    if tvl_path.exists():
        external_data["defi_tvl"] = pd.read_parquet(tvl_path)
        print(f"  Loaded defi_tvl: {len(external_data['defi_tvl'])} rows")
    else:
        external_data["defi_tvl"] = pd.DataFrame()
        print(f"  defi_tvl not found, using defaults")

    funding = {}
    for pair in PAIRS:
        sym = pair.replace("/", "").upper()
        fr_path = EXTERNAL_DIR / f"funding_rate_{sym}.parquet"
        if fr_path.exists():
            funding[sym] = pd.read_parquet(fr_path)
            print(f"  Loaded funding {sym}: {len(funding[sym])} rows")
    external_data["funding"] = funding

    return external_data


def prepare_data(external_data):
    from crypto_transformer.data.features_v5 import FEATURE_COLUMNS_V5

    print("  Fetching BTC reference...")
    btc = fetch_ohlcv("BTC/USDT", "1d")

    print("  Fetching OHLCV for all pairs...")
    raw = {p: fetch_ohlcv(p, "1d") for p in PAIRS}

    print("  Computing cross-sectional features...")
    ranks = compute_cross_sectional_features(raw, lookback=10)

    print("  Computing V5 features with external data...")
    all_dfs = {}
    for p in PAIRS:
        df = raw[p].copy()
        df["pair"] = p
        all_dfs[p] = compute_features_v5(
            df,
            btc_df=btc,
            cross_sectional_ranks=ranks.get(p),
            external_data=external_data,
        )

    pair_names = sorted(all_dfs.keys())
    pair_to_id = {name: i for i, name in enumerate(pair_names)}

    splits = {"train": [[], [], [], []], "val": [[], [], [], []], "test": [[], [], [], []]}

    for pair_name in pair_names:
        df = all_dfs[pair_name]
        close = df["close_raw"].values
        future_return = pd.Series(close).pct_change(HORIZON).shift(-HORIZON).values.copy()
        future_return[np.isnan(future_return)] = 0.0

        features = df[FEATURE_COLUMNS_V5].values
        pair_id = pair_to_id[pair_name]
        pids = np.full(len(df), pair_id)
        timestamps = df["timestamp"].values

        valid = np.ones(len(df), dtype=bool)
        valid[-HORIZON:] = False

        cutoff_np = np.datetime64(f"{TRAIN_CUTOFF}T00:00:00", "ns")
        train_mask = (timestamps < cutoff_np) & valid
        test_mask = (timestamps >= cutoff_np) & valid

        tr_f, tr_r, tr_p, tr_c = features[train_mask], future_return[train_mask], pids[train_mask], close[train_mask]
        te_f, te_r, te_p, te_c = features[test_mask], future_return[test_mask], pids[test_mask], close[test_mask]

        n_val = int(len(tr_f) * VAL_FRACTION)
        va_f, va_r, va_p, va_c = tr_f[-n_val:], tr_r[-n_val:], tr_p[-n_val:], tr_c[-n_val:]
        tr_f, tr_r, tr_p, tr_c = tr_f[:-n_val], tr_r[:-n_val], tr_p[:-n_val], tr_c[:-n_val]

        for split, arrays in [
            ("train", [tr_f, tr_r, tr_p, tr_c]),
            ("val", [va_f, va_r, va_p, va_c]),
            ("test", [te_f, te_r, te_p, te_c]),
        ]:
            for i, arr in enumerate(arrays):
                splits[split][i].append(arr)

    train_arrays = [np.concatenate(splits["train"][i]) for i in range(4)]
    val_arrays = [np.concatenate(splits["val"][i]) for i in range(4)]
    test_arrays = [np.concatenate(splits["test"][i]) for i in range(4)]

    scaler = StandardScaler()
    train_arrays[0] = scaler.fit_transform(train_arrays[0]).astype(np.float32)
    val_arrays[0] = scaler.transform(val_arrays[0]).astype(np.float32)
    test_arrays[0] = scaler.transform(test_arrays[0]).astype(np.float32)

    print(f"  Feature dim: {train_arrays[0].shape[1]} (expected {NUM_FEATURES_V5})")

    return {
        "train": CryptoDatasetV3(*train_arrays, seq_len=SEQ_LEN),
        "val": CryptoDatasetV3(*val_arrays, seq_len=SEQ_LEN),
        "test": CryptoDatasetV3(*test_arrays, seq_len=SEQ_LEN),
        "scaler": scaler,
        "pair_names": pair_names,
        "pair_to_id": pair_to_id,
        "stats": {
            "train_size": len(train_arrays[0]),
            "val_size": len(val_arrays[0]),
            "test_size": len(test_arrays[0]),
            "train_return_mean": float(train_arrays[1].mean()),
            "train_return_std": float(train_arrays[1].std()),
            "test_return_mean": float(test_arrays[1].mean()),
            "test_return_std": float(test_arrays[1].std()),
        },
    }


def train_seed(seed, data, device):
    set_seed(seed)
    print(f"\n{'='*60}\n  Seed {seed}\n{'='*60}")

    model = RegressionTransformer(num_features=NUM_FEATURES_V5, num_pairs=len(data["pair_names"])).to(device)
    print(f"  Params: {model.count_parameters():,}")

    huber = nn.HuberLoss(delta=0.1)
    ranking = PairwiseRankingLoss(margin=RANKING_MARGIN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        return 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (MAX_EPOCHS - WARMUP_EPOCHS)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_loader = DataLoader(data["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(data["val"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    best_ic = -999
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_dir_acc": []}

    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0
        for x, y, pid, _ in train_loader:
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

        model.eval()
        all_pred, all_true = [], []
        val_loss = 0
        with torch.no_grad():
            for x, y, pid, _ in val_loader:
                x, y, pid = x.to(device), y.to(device), pid.to(device)
                pred = model(x, pid)
                val_loss += huber(pred, y).item()
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

        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  E{epoch+1:3d} | tr={train_loss:.5f} va={val_loss:.5f} "
                f"IC={ic:+.4f} dir={dir_acc:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if patience_counter >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1} (best IC={best_ic:.4f})")
            break

    if best_state:
        model.load_state_dict(best_state)
    return {"model_state": best_state, "history": history, "best_ic": best_ic}


def evaluate_ensemble(models, data, device):
    test_loader = DataLoader(data["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_pred, all_true, all_pid, all_close = [], [], [], []

    for x, y, pid, cp in test_loader:
        x, pid = x.to(device), pid.to(device)
        preds = []
        with torch.no_grad():
            for model in models:
                model.eval()
                preds.append(model(x, pid).cpu().numpy())
        avg_pred = np.mean(preds, axis=0)
        all_pred.extend(avg_pred)
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

    long_mask = pred_arr > 0
    short_mask = pred_arr < 0
    long_return = true_arr[long_mask].mean() if long_mask.sum() > 0 else 0
    short_return = -true_arr[short_mask].mean() if short_mask.sum() > 0 else 0

    id_to_pair = {v: k for k, v in data["pair_to_id"].items()}
    per_pair = {}
    for pid_val in sorted(np.unique(pid_arr)):
        mask = pid_arr == pid_val
        p_ic = np.corrcoef(pred_arr[mask], true_arr[mask])[0, 1]
        p_dir = np.mean((pred_arr[mask] > 0) == (true_arr[mask] > 0))
        p_long = (pred_arr[mask] > 0).sum()
        p_name = id_to_pair.get(pid_val, f"pair_{pid_val}")
        per_pair[p_name] = {
            "ic": float(p_ic),
            "dir_acc": float(p_dir),
            "count": int(mask.sum()),
            "long_pct": float(p_long / mask.sum()),
            "avg_pred": float(pred_arr[mask].mean()),
            "avg_true": float(true_arr[mask].mean()),
        }

    return {
        "overall": {
            "ic": float(ic), "dir_acc": float(dir_acc),
            "mae": float(mae), "rmse": float(rmse),
            "long_return": float(long_return),
            "short_return": float(short_return),
            "long_count": int(long_mask.sum()),
            "short_count": int(short_mask.sum()),
        },
        "per_pair": per_pair,
    }


# ─── Backtest Strategies ───────────────────────────────────────────────────

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


def backtest_metrics(pv, dr, trades, name):
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


def run_backtest(models, data, device):
    """Run full backtest: build test index, inference, trading simulation, chart."""
    from crypto_transformer.data.features_v5 import FEATURE_COLUMNS_V5

    print("\n" + "=" * 70)
    print("V5 Backtest: Inference + Trading Simulation")
    print("=" * 70)

    print("\n  Building test data index...")
    external_data = load_external_data()
    btc = fetch_ohlcv("BTC/USDT", "1d")
    raw = {p: fetch_ohlcv(p, "1d") for p in PAIRS}
    ranks = compute_cross_sectional_features(raw, lookback=10)
    all_dfs = {}
    for p in PAIRS:
        df = raw[p].copy()
        df["pair"] = p
        all_dfs[p] = compute_features_v5(df, btc_df=btc, cross_sectional_ranks=ranks.get(p), external_data=external_data)

    pair_names = sorted(all_dfs.keys())
    pair_to_id = {n: i for i, n in enumerate(pair_names)}

    cutoff = np.datetime64(f"{TRAIN_CUTOFF}T00:00:00", "ns")
    train_concat = pd.concat([all_dfs[p][all_dfs[p]["timestamp"].values < cutoff] for p in pair_names], ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(train_concat[FEATURE_COLUMNS_V5].values)

    test_dfs = {}
    for p in pair_names:
        df = all_dfs[p][all_dfs[p]["timestamp"].values >= cutoff].copy().reset_index(drop=True)
        df[FEATURE_COLUMNS_V5] = scaler.transform(df[FEATURE_COLUMNS_V5].values).astype(np.float32)
        test_dfs[p] = df

    print("\n  Running inference...")
    records = []
    batch_size = 256
    for pair_name in pair_names:
        df = test_dfs[pair_name]
        feat = df[FEATURE_COLUMNS_V5].values
        ts = df["timestamp"].values
        close = df["close_raw"].values
        pid = pair_to_id[pair_name]

        n = len(df) - SEQ_LEN - HORIZON
        if n <= 0:
            continue

        all_preds = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
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

    pred_df = pd.DataFrame(records)
    print(f"  Predictions: {len(pred_df)}, dates: {pred_df['date'].min().date()} to {pred_df['date'].max().date()}")

    overall_ic = np.corrcoef(pred_df["pred_3d"].values, pred_df["actual_3d_ret"].values)[0, 1]
    overall_dir = np.mean((pred_df["pred_3d"].values > 0) == (pred_df["actual_3d_ret"].values > 0))
    print(f"  Overall IC: {overall_ic:.4f}, Dir acc: {overall_dir:.4f}")

    print("\n  Trading simulation...")
    print("  " + "-" * 68)

    all_results = {}
    all_curves = {}

    for th in [0.0, 0.005, 0.01]:
        name = f"LS(th={th:.3f})"
        pv, dr, tr = strategy_long_short(pred_df, threshold=th)
        m = backtest_metrics(pv, dr, tr, name)
        all_results[name] = m
        all_curves[name] = pv
        print(f"  {name:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  "
              f"mdd={m['max_drawdown_pct']:6.1f}%  win={m['win_rate_pct']:4.1f}%  trades={tr}")

    for th in [0.0, 0.005, 0.01]:
        name = f"LO(th={th:.3f})"
        pv, dr, tr = strategy_long_only(pred_df, threshold=th)
        m = backtest_metrics(pv, dr, tr, name)
        all_results[name] = m
        all_curves[name] = pv
        print(f"  {name:18s}  ret={m['total_return_pct']:+7.2f}%  sharpe={m['sharpe']:5.2f}  "
              f"mdd={m['max_drawdown_pct']:6.1f}%  win={m['win_rate_pct']:4.1f}%  trades={tr}")

    for k in [3, 5]:
        name = f"Top{k}-LS"
        pv, dr, tr = strategy_top_k(pred_df, k=k)
        m = backtest_metrics(pv, dr, tr, name)
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

    print(f"\n  Per-pair breakdown (LS th=0.005, 10% position):")
    print("  " + "-" * 68)
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

    fig.suptitle(f"V5 Model Backtest ({pred_df['date'].min().date()} ~ {pred_df['date'].max().date()})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig("docs/figures/v5_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()

    backtest_output = {
        "strategies": all_results,
        "buy_and_hold_return_pct": round(bnh_ret, 2),
        "buy_and_hold_mdd_pct": round(bnh_mdd, 2),
        "overall_ic": float(overall_ic),
        "overall_dir_acc": float(overall_dir),
        "test_period": f"{pred_df['date'].min().date()} to {pred_df['date'].max().date()}",
        "initial_capital": INITIAL_CAPITAL,
        "fee_rate": FEE_RATE,
    }
    with open("data/backtest_v5_results.json", "w") as f:
        json.dump(backtest_output, f, indent=2)
    print(f"\n  Backtest chart: docs/figures/v5_backtest.png")
    print(f"  Backtest results: data/backtest_v5_results.json")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("V5 Training + Backtest (42 features, external data)")
    print("=" * 70)

    print("\n[1/5] Loading external data...")
    external_data = load_external_data()

    print("\n[2/5] Preparing data pipeline...")
    data = prepare_data(external_data)
    print(f"  Train: {len(data['train'])} | Val: {len(data['val'])} | Test: {len(data['test'])}")
    print(f"  Stats: {data['stats']}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[3/5] Training / loading models...")
    all_results = []
    all_models = []

    for seed in SEEDS[:NUM_SEEDS]:
        ckpt_path = CKPT_DIR / f"v5_seed_{seed}.pt"
        model = RegressionTransformer(num_features=NUM_FEATURES_V5, num_pairs=len(data["pair_names"])).to(device)

        if ckpt_path.exists():
            print(f"\n{'='*60}\n  Seed {seed} — loading checkpoint\n{'='*60}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            result = {"model_state": {k: v.clone() for k, v in model.state_dict().items()}, "history": {}, "best_ic": 0}
        else:
            result = train_seed(seed, data, device)
            torch.save(result["model_state"], ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        all_results.append(result)
        all_models.append(model)

    print(f"\n[4/5] Ensemble Evaluation...")
    print(f"{'='*60}\n  Ensemble Evaluation\n{'='*60}")
    eval_results = evaluate_ensemble(all_models, data, device)

    o = eval_results["overall"]
    print(f"\n  IC:              {o['ic']:.4f}")
    print(f"  Directional Acc: {o['dir_acc']:.4f}")
    print(f"  MAE:             {o['mae']:.6f}")
    print(f"  RMSE:            {o['rmse']:.6f}")
    print(f"  Long avg return: {o['long_return']:.6f} (n={o['long_count']})")
    print(f"  Short avg return:{o['short_return']:.6f} (n={o['short_count']})")

    print(f"\n  Per-pair:")
    for pname in sorted(eval_results["per_pair"].keys()):
        pp = eval_results["per_pair"][pname]
        print(f"  {pname:12s}  IC={pp['ic']:+.4f}  dir={pp['dir_acc']:.3f}  long%={pp['long_pct']:.2f}")

    output = {
        "overall": eval_results["overall"],
        "per_pair": eval_results["per_pair"],
        "histories": {f"seed_{s}": all_results[i]["history"] for i, s in enumerate(SEEDS[:NUM_SEEDS])},
        "stats": data["stats"],
        "config": {
            "seq_len": SEQ_LEN, "horizon": HORIZON, "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "dropout": 0.3, "d_model": 128,
            "num_layers": 3, "nhead": 4, "dim_feedforward": 512,
            "huber_weight": HUBER_WEIGHT, "ranking_weight": RANKING_WEIGHT,
            "huber_delta": 0.1, "ranking_margin": RANKING_MARGIN,
            "num_seeds": NUM_SEEDS, "num_features": NUM_FEATURES_V5,
            "pairs": PAIRS,
            "model_type": "v5_pure_regression_with_pairwise_ranking",
        },
    }

    Path("data").mkdir(exist_ok=True)
    with open("data/results_v5.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to data/results_v5.json")

    print("\n[5/5] Running backtest...")
    run_backtest(all_models, data, device)

    print("\n" + "=" * 70)
    print("V5 Training + Backtest complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
