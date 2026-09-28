import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from crypto_transformer.data.collectors.binance import fetch_ohlcv
from crypto_transformer.data.features_v4 import compute_features_v4, compute_cross_sectional_features, NUM_FEATURES_V4
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
    def __init__(self, num_features=25, num_pairs=15, d_model=128, nhead=4, num_layers=3,
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


def prepare_data():
    from crypto_transformer.data.features_v4 import FEATURE_COLUMNS_V4

    btc = fetch_ohlcv("BTC/USDT", "1d")
    raw = {p: fetch_ohlcv(p, "1d") for p in PAIRS}
    ranks = compute_cross_sectional_features(raw, lookback=10)
    all_dfs = {p: compute_features_v4(raw[p], btc_df=btc, cross_sectional_ranks=ranks.get(p)) for p in PAIRS}

    pair_names = sorted(all_dfs.keys())
    pair_to_id = {name: i for i, name in enumerate(pair_names)}

    splits = {"train": [[], [], [], []], "val": [[], [], [], []], "test": [[], [], [], []]}

    for pair_name in pair_names:
        df = all_dfs[pair_name]
        close = df["close_raw"].values
        future_return = pd.Series(close).pct_change(HORIZON).shift(-HORIZON).values.copy()
        future_return[np.isnan(future_return)] = 0.0

        features = df[FEATURE_COLUMNS_V4].values
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

    model = RegressionTransformer(num_features=NUM_FEATURES_V4, num_pairs=len(data["pair_names"])).to(device)
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    data = prepare_data()
    print(f"Train: {len(data['train'])} | Val: {len(data['val'])} | Test: {len(data['test'])}")
    print(f"Stats: {data['stats']}")

    all_results = []
    all_models = []

    for seed in SEEDS[:NUM_SEEDS]:
        result = train_seed(seed, data, device)
        all_results.append(result)
        model = RegressionTransformer(num_features=NUM_FEATURES_V4, num_pairs=len(data["pair_names"])).to(device)
        model.load_state_dict(result["model_state"])
        all_models.append(model)

    print(f"\n{'='*60}\n  Ensemble Evaluation\n{'='*60}")
    eval_results = evaluate_ensemble(all_models, data, device)

    o = eval_results["overall"]
    print(f"\n  IC:              {o['ic']:.4f}")
    print(f"  Directional Acc: {o['dir_acc']:.4f}")
    print(f"  MAE:             {o['mae']:.6f}")
    print(f"  RMSE:            {o['rmse']:.6f}")
    print(f"  Long avg return: {o['long_return']:.6f} (n={o['long_count']})")
    print(f"  Short avg return:{o['short_return']:.6f} (n={o['short_count']})")

    print(f"\nPer-pair:")
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
            "num_seeds": NUM_SEEDS, "pairs": PAIRS,
            "model_type": "pure_regression_with_pairwise_ranking",
        },
    }

    Path("data").mkdir(exist_ok=True)
    with open("data/results_v4.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to data/results_v4.json")


if __name__ == "__main__":
    main()
