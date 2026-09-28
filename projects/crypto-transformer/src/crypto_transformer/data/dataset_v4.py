import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from crypto_transformer.data.features_v4 import FEATURE_COLUMNS_V4


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class CryptoDatasetV4(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        cls_labels: np.ndarray,
        reg_targets: np.ndarray,
        pair_ids: np.ndarray,
        close_prices: np.ndarray,
        seq_len: int = 90,
    ):
        self.features = features.astype(np.float32)
        self.cls_labels = cls_labels.astype(np.int64)
        self.reg_targets = reg_targets.astype(np.float32)
        self.pair_ids = pair_ids.astype(np.int64)
        self.close_prices = close_prices.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.features) - self.seq_len

    def __getitem__(self, idx: int):
        x = self.features[idx: idx + self.seq_len]
        cls = self.cls_labels[idx + self.seq_len - 1]
        reg = self.reg_targets[idx + self.seq_len - 1]
        pid = self.pair_ids[idx + self.seq_len - 1]
        cp = self.close_prices[idx + self.seq_len - 1]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(cls, dtype=torch.long),
            torch.tensor(reg, dtype=torch.float32),
            torch.tensor(pid, dtype=torch.long),
            torch.tensor(cp, dtype=torch.float32),
        )


def _make_labels(future_return: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    labels = np.ones(len(future_return), dtype=np.int64)
    labels[future_return > threshold] = 2
    labels[future_return < -threshold] = 0
    return labels


def prepare_multi_asset_dataset_v4(
    all_dfs: dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    seq_len: int = 90,
    horizon: int = 3,
    train_cutoff: str = "2025-06-01",
    val_fraction: float = 0.15,
    cls_threshold: float = 0.01,
) -> dict:
    from sklearn.preprocessing import StandardScaler

    pair_names = sorted(all_dfs.keys())
    pair_to_id = {name: i for i, name in enumerate(pair_names)}

    splits = {"train": [[], [], [], [], []], "val": [[], [], [], [], []], "test": [[], [], [], [], []]}
    keys = ["features", "cls", "reg", "pids", "close"]

    for pair_name in pair_names:
        df = all_dfs[pair_name]
        close = df["close_raw"].values
        future_return = pd.Series(close).pct_change(horizon).shift(-horizon).values.copy()
        future_return[np.isnan(future_return)] = 0.0

        features = df[FEATURE_COLUMNS_V4].values
        cls_labels = _make_labels(future_return, cls_threshold)
        pair_id = pair_to_id[pair_name]
        pids = np.full(len(df), pair_id)
        timestamps = df["timestamp"].values

        valid = np.ones(len(df), dtype=bool)
        valid[-horizon:] = False

        cutoff_np = np.datetime64(f"{train_cutoff}T00:00:00", "ns")
        train_mask = (timestamps < cutoff_np) & valid
        test_mask = (timestamps >= cutoff_np) & valid

        tr_f = features[train_mask]
        tr_c = cls_labels[train_mask]
        tr_r = future_return[train_mask]
        tr_p = pids[train_mask]
        tr_cp = close[train_mask]

        te_f = features[test_mask]
        te_c = cls_labels[test_mask]
        te_r = future_return[test_mask]
        te_p = pids[test_mask]
        te_cp = close[test_mask]

        n_val = int(len(tr_f) * val_fraction)
        va_f, va_c, va_r, va_p, va_cp = tr_f[-n_val:], tr_c[-n_val:], tr_r[-n_val:], tr_p[-n_val:], tr_cp[-n_val:]
        tr_f, tr_c, tr_r, tr_p, tr_cp = tr_f[:-n_val], tr_c[:-n_val], tr_r[:-n_val], tr_p[:-n_val], tr_cp[:-n_val]

        for split, arrays in [
            ("train", [tr_f, tr_c, tr_r, tr_p, tr_cp]),
            ("val", [va_f, va_c, va_r, va_p, va_cp]),
            ("test", [te_f, te_c, te_r, te_p, te_cp]),
        ]:
            for i, arr in enumerate(arrays):
                splits[split][i].append(arr)

    train_arrays = [np.concatenate(splits["train"][i]) for i in range(5)]
    val_arrays = [np.concatenate(splits["val"][i]) for i in range(5)]
    test_arrays = [np.concatenate(splits["test"][i]) for i in range(5)]

    scaler = StandardScaler()
    train_arrays[0] = scaler.fit_transform(train_arrays[0]).astype(np.float32)
    val_arrays[0] = scaler.transform(val_arrays[0]).astype(np.float32)
    test_arrays[0] = scaler.transform(test_arrays[0]).astype(np.float32)

    cls_weights = np.bincount(train_arrays[1], minlength=3).astype(np.float32)
    cls_weights = 1.0 / (cls_weights + 1)
    cls_weights = cls_weights / cls_weights.sum() * 3
    cls_weights = torch.tensor(cls_weights, dtype=torch.float32)

    train_ds = CryptoDatasetV4(*train_arrays, seq_len=seq_len)
    val_ds = CryptoDatasetV4(*val_arrays, seq_len=seq_len)
    test_ds = CryptoDatasetV4(*test_arrays, seq_len=seq_len)

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "scaler": scaler,
        "cls_weights": cls_weights,
        "pair_names": pair_names,
        "pair_to_id": pair_to_id,
        "label_counts": {
            "train": {i: int((train_arrays[1] == i).sum()) for i in range(3)},
            "val": {i: int((val_arrays[1] == i).sum()) for i in range(3)},
            "test": {i: int((test_arrays[1] == i).sum()) for i in range(3)},
        },
        "stats": {
            "train_size": len(train_arrays[0]),
            "val_size": len(val_arrays[0]),
            "test_size": len(test_arrays[0]),
            "train_return_mean": float(train_arrays[2].mean()),
            "train_return_std": float(train_arrays[2].std()),
            "test_return_mean": float(test_arrays[2].mean()),
            "test_return_std": float(test_arrays[2].std()),
        },
    }
