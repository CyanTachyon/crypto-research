from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from crypto_transformer.data.features import FEATURE_COLUMNS


LABEL_UP = 0
LABEL_DOWN = 1
LABEL_FLAT = 2
LABEL_NAMES = ["up", "down", "flat"]


def compute_labels(close_prices: pd.Series, threshold: float = 0.005) -> np.ndarray:
    returns = close_prices.pct_change().shift(-1)
    labels = np.full(len(close_prices), LABEL_FLAT, dtype=np.int64)
    valid = ~returns.isna()
    labels[valid.values] = np.where(
        returns[valid] > threshold, LABEL_UP,
        np.where(returns[valid] < -threshold, LABEL_DOWN, LABEL_FLAT)
    )
    return labels


class CryptoDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 48,
    ):
        self.features = features
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def prepare_dataset(
    df: pd.DataFrame,
    seq_len: int = 48,
    label_threshold: float = 0.005,
    train_cutoff: str = "2025-12-01",
    val_fraction: float = 0.2,
) -> dict:
    from sklearn.preprocessing import StandardScaler

    labels = compute_labels(df["close_raw"] if "close_raw" in df.columns else df["close"], label_threshold)

    feature_data = df[FEATURE_COLUMNS].values.astype(np.float32)

    timestamps = df["timestamp"].values
    train_cutoff_np = np.datetime64(f"{train_cutoff}T00:00:00", "ns")
    train_mask = timestamps < train_cutoff_np

    train_features = feature_data[train_mask]
    train_labels = labels[train_mask]
    test_features = feature_data[~train_mask]
    test_labels = labels[~train_mask]

    n_val = int(len(train_features) * val_fraction)
    val_features = train_features[-n_val:]
    val_labels = train_labels[-n_val:]
    train_features = train_features[:-n_val]
    train_labels = train_labels[:-n_val]

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features).astype(np.float32)
    val_features = scaler.transform(val_features).astype(np.float32)
    test_features = scaler.transform(test_features).astype(np.float32)

    return {
        "train": CryptoDataset(train_features, train_labels, seq_len),
        "val": CryptoDataset(val_features, val_labels, seq_len),
        "test": CryptoDataset(test_features, test_labels, seq_len),
        "scaler": scaler,
        "label_counts": {
            "train": np.bincount(train_labels, minlength=3),
            "val": np.bincount(val_labels, minlength=3),
            "test": np.bincount(test_labels, minlength=3),
        },
    }
