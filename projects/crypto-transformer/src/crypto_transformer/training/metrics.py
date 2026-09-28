import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix


LABEL_NAMES = ["up", "down", "flat"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1_per, _, = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        per_class[name] = {
            "precision": float(precision[i]) if i < len(precision) else 0.0,
            "recall": float(recall[i]) if i < len(recall) else 0.0,
            "f1": float(f1_per[i]) if i < len(f1_per) else 0.0,
        }

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1_weighted),
        "f1_macro": float(f1_macro),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
