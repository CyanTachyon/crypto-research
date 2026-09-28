import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class RandomBaseline:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.num_classes = len(np.unique(y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.random.randint(0, self.num_classes, size=len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        probs = np.random.dirichlet(np.ones(self.num_classes), size=n)
        return probs


class MajorityBaseline:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        counts = np.bincount(y)
        self.majority_class = int(np.argmax(counts))
        self.num_classes = len(counts)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.majority_class)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        probs = np.zeros((n, self.num_classes))
        probs[:, self.majority_class] = 1.0
        return probs


class XGBoostBaseline:
    def __init__(self, seq_len: int = 48, num_features: int = 17, **kwargs):
        self.seq_len = seq_len
        self.num_features = num_features
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
            **kwargs,
        )

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_flat = self._flatten(X)
        self.model.fit(X_flat, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._flatten(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._flatten(X))


class SMACrossoverBaseline:
    def __init__(self, fast_period: int = 7, slow_period: int = 21):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._classes = 3

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.full(len(X), 2, dtype=np.int64)
        for i in range(len(X)):
            close_series = X[i, :, 3]
            if len(close_series) < self.slow_period:
                continue
            fast_ma = np.mean(close_series[-self.fast_period:])
            slow_ma = np.mean(close_series[-self.slow_period:])
            if fast_ma > slow_ma * 1.001:
                preds[i] = 0
            elif fast_ma < slow_ma * 0.999:
                preds[i] = 1
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        n = len(preds)
        probs = np.zeros((n, 3))
        for i in range(n):
            probs[i, preds[i]] = 1.0
        return probs
