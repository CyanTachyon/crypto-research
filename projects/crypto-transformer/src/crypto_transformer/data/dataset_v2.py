import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from crypto_transformer.data.features_v2 import FEATURE_COLUMNS_V2


LABEL_UP = 0
LABEL_DOWN = 1
LABEL_FLAT = 2
LABEL_NAMES = ["up", "down", "flat"]


def compute_labels_multi_horizon(
    close_prices: pd.Series,
    horizon: int = 4,
    threshold: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    future_return = close_prices.pct_change(horizon).shift(-horizon)
    future_return_raw = future_return.values.copy()

    labels = np.full(len(close_prices), LABEL_FLAT, dtype=np.int64)
    valid = ~np.isnan(future_return_raw)
    labels[valid] = np.where(
        future_return_raw[valid] > threshold, LABEL_UP,
        np.where(future_return_raw[valid] < -threshold, LABEL_DOWN, LABEL_FLAT),
    )
    future_return_raw[np.isnan(future_return_raw)] = 0.0
    return labels, future_return_raw


class CryptoDatasetV2(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        returns: np.ndarray,
        seq_len: int = 128,
    ):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.returns = returns.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.features[idx: idx + self.seq_len]
        y_cls = self.labels[idx + self.seq_len - 1]
        y_reg = self.returns[idx + self.seq_len - 1]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_cls, dtype=torch.long),
            torch.tensor(y_reg, dtype=torch.float32),
        )


def prepare_dataset_v2(
    df: pd.DataFrame,
    seq_len: int = 128,
    horizon: int = 4,
    label_threshold: float = 0.01,
    train_cutoff: str = "2025-12-01",
    val_fraction: float = 0.2,
) -> dict:
    from sklearn.preprocessing import StandardScaler

    close = df["close_raw"] if "close_raw" in df.columns else df["close"]
    labels, returns = compute_labels_multi_horizon(close, horizon, label_threshold)

    feature_data = df[FEATURE_COLUMNS_V2].values

    timestamps = df["timestamp"].values
    train_cutoff_np = np.datetime64(f"{train_cutoff}T00:00:00", "ns")

    # invalidate labels for last `horizon` rows (no future data)
    valid_mask = np.ones(len(df), dtype=bool)
    valid_mask[-horizon:] = False

    train_mask = (timestamps < train_cutoff_np) & valid_mask
    test_mask = (timestamps >= train_cutoff_np) & valid_mask

    train_features = feature_data[train_mask]
    train_labels = labels[train_mask]
    train_returns = returns[train_mask]
    test_features = feature_data[test_mask]
    test_labels = labels[test_mask]
    test_returns = returns[test_mask]

    n_val = int(len(train_features) * val_fraction)
    val_features = train_features[-n_val:]
    val_labels = train_labels[-n_val:]
    val_returns = train_returns[-n_val:]
    train_features = train_features[:-n_val]
    train_labels = train_labels[:-n_val]
    train_returns = train_returns[:-n_val]

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features).astype(np.float32)
    val_features = scaler.transform(val_features).astype(np.float32)
    test_features = scaler.transform(test_features).astype(np.float32)

    return {
        "train": CryptoDatasetV2(train_features, train_labels, train_returns, seq_len),
        "val": CryptoDatasetV2(val_features, val_labels, val_returns, seq_len),
        "test": CryptoDatasetV2(test_features, test_labels, test_returns, seq_len),
        "scaler": scaler,
        "label_counts": {
            "train": np.bincount(train_labels, minlength=3),
            "val": np.bincount(val_labels, minlength=3),
            "test": np.bincount(test_labels, minlength=3),
        },
    }
