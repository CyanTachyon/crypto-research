import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score < self.best_score - self.min_delta
            if self.mode == "min"
            else score > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    def __init__(self, save_path: str, mode: str = "max"):
        self.save_path = save_path
        self.mode = mode
        self.best_score = float("-inf") if mode == "max" else float("inf")

    def step(self, score: float, model: torch.nn.Module) -> bool:
        improved = (
            score > self.best_score
            if self.mode == "max"
            else score < self.best_score
        )
        if improved:
            self.best_score = score
            torch.save(model.state_dict(), self.save_path)
            return True
        return False


class ReduceLROnPlateau:
    def __init__(self, optimizer: torch.optim.Optimizer, patience: int = 5, factor: float = 0.5, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_score = None
        self.counter = 0

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                for pg in self.optimizer.param_groups:
                    new_lr = max(pg["lr"] * self.factor, self.min_lr)
                    pg["lr"] = new_lr
                self.counter = 0
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False
