import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class Trade:
    entry_time: int
    entry_price: float
    direction: str
    size: float
    exit_time: int = 0
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_trade_return: float = 0.0
    num_trades: int = 0
    initial_capital: float = 0.0
    final_capital: float = 0.0


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_rate: float = 0.001,
        slippage: float = 0.0005,
        position_fraction: float = 0.25,
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.position_fraction = position_fraction

    def run(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        close_prices: np.ndarray,
        confidence_threshold: float = 0.6,
    ) -> BacktestResult:
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = 0
        trades = []
        equity = [capital]
        in_position = False

        for i in range(len(predictions)):
            price = close_prices[i]
            prob = probabilities[i]
            pred = predictions[i]
            max_prob = prob.max()

            if in_position:
                pnl_pct = (price - entry_price) / entry_price
                if position < 0:
                    pnl_pct = -pnl_pct

                if pred == 1 and max_prob >= confidence_threshold and position > 0:
                    exit_price = price * (1 - self.slippage)
                    fee = abs(position * exit_price) * self.fee_rate
                    pnl = position * (exit_price - entry_price) - fee
                    capital += position * exit_price + pnl - fee
                    trades.append(Trade(
                        entry_time=entry_time, entry_price=entry_price,
                        direction="long", size=abs(position),
                        exit_time=i, exit_price=exit_price,
                        pnl=pnl, pnl_pct=pnl_pct,
                    ))
                    position = 0.0
                    in_position = False

                elif pred == 0 and max_prob >= confidence_threshold and position < 0:
                    exit_price = price * (1 + self.slippage)
                    fee = abs(position * exit_price) * self.fee_rate
                    pnl = -position * (entry_price - exit_price) - fee
                    capital += abs(position) * entry_price + pnl
                    trades.append(Trade(
                        entry_time=entry_time, entry_price=entry_price,
                        direction="short", size=abs(position),
                        exit_time=i, exit_price=exit_price,
                        pnl=pnl, pnl_pct=pnl_pct,
                    ))
                    position = 0.0
                    in_position = False

            if not in_position and max_prob >= confidence_threshold:
                if pred == 0:
                    position_size = (capital * self.position_fraction) / (price * (1 + self.slippage))
                    fee = capital * self.position_fraction * self.fee_rate
                    capital -= capital * self.position_fraction + fee
                    position = position_size
                    entry_price = price * (1 + self.slippage)
                    entry_time = i
                    in_position = True

            if in_position:
                if position > 0:
                    equity.append(capital + position * price)
                else:
                    equity.append(capital + abs(position) * (2 * entry_price - price))
            else:
                equity.append(capital)

        if in_position:
            final_price = close_prices[-1]
            if position > 0:
                capital += position * final_price * (1 - self.slippage)
                capital -= position * final_price * self.fee_rate
            trades.append(Trade(
                entry_time=entry_time, entry_price=entry_price,
                direction="long" if position > 0 else "short",
                size=abs(position), exit_time=len(predictions) - 1,
                exit_price=final_price,
            ))

        equity = np.array(equity)
        result = BacktestResult(
            trades=trades,
            equity_curve=equity,
            initial_capital=self.initial_capital,
            final_capital=equity[-1],
        )
        return self._compute_stats(result)

    def _compute_stats(self, result: BacktestResult) -> BacktestResult:
        if len(result.equity_curve) < 2:
            return result

        returns = np.diff(result.equity_curve) / result.equity_curve[:-1]
        returns = returns[np.isfinite(returns)]

        result.total_return = (result.equity_curve[-1] / result.equity_curve[0]) - 1
        result.num_trades = len(result.trades)

        if len(returns) > 1:
            result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(8760) if returns.std() > 0 else 0.0

        peak = np.maximum.accumulate(result.equity_curve)
        drawdown = (result.equity_curve - peak) / peak
        result.max_drawdown = float(drawdown.min())

        winning_trades = [t for t in result.trades if t.pnl > 0]
        result.win_rate = len(winning_trades) / len(result.trades) if result.trades else 0.0
        result.avg_trade_return = np.mean([t.pnl_pct for t in result.trades]) if result.trades else 0.0

        return result
