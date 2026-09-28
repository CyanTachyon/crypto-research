# V14 严格 Holdout 验证报告

## 1. 验证目的
本次 V14 不追求新策略，而是专门回答用户提出的训练/测试重合质疑：固定规则，只用 2020-2024 的历史训练和选参，然后在 2025-2026 holdout 上只跑一次。

## 2. 时间切分
- Fit 训练窗口：2020-11-08 04:00:00 to 2024-07-01 00:00:00
- Select 选参窗口：2024-07-23 00:00:00 to 2024-12-10 00:00:00
- Holdout 测试窗口：2025-01-01 00:00:00 to 2026-06-07 12:00:00
- Fit 与 Select、Select 与 Holdout 之间均留出约 132 根 4h bar 的 purge gap。

## 3. 候选策略
- `V14-V12LT-SelectedOn2024`：只用 2024 select 窗口选 V12 低换手参数，再应用到 2025+ holdout。
- `V14-V13-SelectedOn2024`：XGBoost 只在 fit 窗口训练，ensemble/router 参数只在 2024 select 窗口选择，再应用到 2025+ holdout。
- `V12-LT-original-reference`：历史 V12-LT 原始列在 holdout 的表现，只作参考，因为它的参数来自之前研究过程。

## 4. Holdout 结果
| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 平均换手 | 成本合计 |
|---|---:|---:|---:|---:|---:|---:|
| V14-V13-SelectedOn2024 | 11.63% | 1.87 | -2.19% | 0.1998 | 0.007 | 0.037558 |
| V12-LT-original-reference | 12.68% | 1.46 | -3.83% | 0.2 | 0.0063 | 0.033465 |
| V12-LT30-original-reference | 19.4% | 1.46 | -5.71% | 0.3 | 0.0095 | 0.050197 |
| V14-V12LT-SelectedOn2024 | 11.18% | 0.95 | -5.18% | 0.25 | 0.0068 | 0.035713 |
| AlwaysFlat | 0.0% | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 |

## 5. 方法学结论
V14 是比 V12/V13 更严格的验证口径。如果候选策略在这里失效，说明之前结果很可能依赖研究过程中的参数选择；如果仍然有效，才更接近可继续 paper trading 的候选。

## 6. 输出文件
- `data/results_v14_holdout.json`
- `data/backtest_v14_holdout_results.json`
- `data/v14_holdout_decisions.parquet`
- `docs/figures/v14_holdout_equity_curves.png`
