import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from crypto_transformer.data.features_v3 import FEATURE_COLUMNS_V3


class CryptoDatasetV3(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        pair_ids: np.ndarray,
        close_prices: np.ndarray,
        seq_len: int = 60,
    ):
        self.features = features.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.pair_ids = pair_ids.astype(np.int64)
        self.close_prices = close_prices.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.features[idx: idx + self.seq_len]
        y = self.returns[idx + self.seq_len - 1]
        pid = self.pair_ids[idx + self.seq_len - 1]
        cp = self.close_prices[idx + self.seq_len - 1]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(pid, dtype=torch.long),
            torch.tensor(cp, dtype=torch.float32),
        )


def prepare_multi_asset_dataset(
    all_dfs: dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    seq_len: int = 60,
    horizon: int = 5,
    train_cutoff: str = "2025-06-01",
    val_fraction: float = 0.15,
) -> dict:
    from sklearn.preprocessing import StandardScaler

    pair_names = sorted(all_dfs.keys())
    pair_to_id = {name: i for i, name in enumerate(pair_names)}

    all_train_features, all_val_features, all_test_features = [], [], []
    all_train_returns, all_val_returns, all_test_returns = [], [], []
    all_train_pids, all_val_pids, all_test_pids = [], [], []
    all_train_close, all_val_close, all_test_close = [], [], []

    for pair_name in pair_names:
        df = all_dfs[pair_name]
        close = df["close_raw"].values
        future_return = pd.Series(close).pct_change(horizon).shift(-horizon).values.copy()
        future_return[np.isnan(future_return)] = 0.0

        features = df[FEATURE_COLUMNS_V3].values
        pair_id = pair_to_id[pair_name]
        pids = np.full(len(df), pair_id)
        timestamps = df["timestamp"].values

        valid = np.ones(len(df), dtype=bool)
        valid[-horizon:] = False

        cutoff_np = np.datetime64(f"{train_cutoff}T00:00:00", "ns")
        train_mask = (timestamps < cutoff_np) & valid
        test_mask = (timestamps >= cutoff_np) & valid

        tr_f, tr_r, tr_p, tr_c = features[train_mask], future_return[train_mask], pids[train_mask], close[train_mask]
        te_f, te_r, te_p, te_c = features[test_mask], future_return[test_mask], pids[test_mask], close[test_mask]

        n_val = int(len(tr_f) * val_fraction)
        va_f, va_r, va_p, va_c = tr_f[-n_val:], tr_r[-n_val:], tr_p[-n_val:], tr_c[-n_val:]
        tr_f, tr_r, tr_p, tr_c = tr_f[:-n_val], tr_r[:-n_val], tr_p[:-n_val], tr_c[:-n_val]

        all_train_features.append(tr_f)
        all_train_returns.append(tr_r)
        all_train_pids.append(tr_p)
        all_train_close.append(tr_c)
        all_val_features.append(va_f)
        all_val_returns.append(va_r)
        all_val_pids.append(va_p)
        all_val_close.append(va_c)
        all_test_features.append(te_f)
        all_test_returns.append(te_r)
        all_test_pids.append(te_p)
        all_test_close.append(te_c)

    train_features = np.concatenate(all_train_features)
    train_returns = np.concatenate(all_train_returns)
    train_pids = np.concatenate(all_train_pids)
    train_close = np.concatenate(all_train_close)
    val_features = np.concatenate(all_val_features)
    val_returns = np.concatenate(all_val_returns)
    val_pids = np.concatenate(all_val_pids)
    val_close = np.concatenate(all_val_close)
    test_features = np.concatenate(all_test_features)
    test_returns = np.concatenate(all_test_returns)
    test_pids = np.concatenate(all_test_pids)
    test_close = np.concatenate(all_test_close)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features).astype(np.float32)
    val_features = scaler.transform(val_features).astype(np.float32)
    test_features = scaler.transform(test_features).astype(np.float32)

    return {
        "train": CryptoDatasetV3(train_features, train_returns, train_pids, train_close, seq_len),
        "val": CryptoDatasetV3(val_features, val_returns, val_pids, val_close, seq_len),
        "test": CryptoDatasetV3(test_features, test_returns, test_pids, test_close, seq_len),
        "scaler": scaler,
        "pair_names": pair_names,
        "pair_to_id": pair_to_id,
        "stats": {
            "train_size": len(train_features),
            "val_size": len(val_features),
            "test_size": len(test_features),
            "train_return_mean": float(train_returns.mean()),
            "train_return_std": float(train_returns.std()),
            "test_return_mean": float(test_returns.mean()),
            "test_return_std": float(test_returns.std()),
        },
    }
