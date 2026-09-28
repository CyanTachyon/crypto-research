# V18 趋势/突破策略验证报告

## 1. 方法
V18 测试 Donchian、EMA+Donchian、时间序列动量、横截面动量和 hybrid 趋势策略。参数只在 2024 selection 窗口选择，2025+ holdout 一次性评估。

## 2. 最佳 selection 候选
```json
{
  "family": "donchian",
  "lookback": 180,
  "slow": 120,
  "quantile": 0.1,
  "max_gross": 0.5,
  "pair_cap": 0.1,
  "rebalance_bars": 24,
  "long_only": true,
  "vol_window": 540,
  "target_vol": 0.2,
  "rebalance_threshold": 0.0
}
```

## 3. Holdout 排名
| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 换手 |
|---|---:|---:|---:|---:|---:|
| V12-LT30-reference | 19.4% | 1.46 | -5.71% | 0.3 | 0.0095 |
| V12-LT-reference | 12.68% | 1.46 | -3.83% | 0.2 | 0.0063 |
| AlwaysFlat | 0.0% | 0.0 | 0.0% | 0.0 | 0.0 |
| V18-TrendBreakout-SelectedOn2024 | -3.45% | 0.06 | -21.83% | 0.4763 | 0.0005 |

## 4. 结论
若 V18 未超过 V12-LT30-reference，则说明当前 4h 数据下最强候选仍是低换手 V12 冻结信号路线；趋势/突破可作为解释性基线而非最终冠军。