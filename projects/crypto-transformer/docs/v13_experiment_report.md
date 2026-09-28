# V13 4h 低换手 Ensemble + Regime Router 实验报告

## 1. 实验目标
V13 不再让 Transformer 直接满仓高频交易，而是把 V12 的 4h 冻结预测、XGBoost tabular 模型和规则因子合成弱信号，再用低换手、regime router 和风险控制转成仓位。

## 2. 数据与切分
输入为 `data/v12_trade_decisions.parquet`，周期为 4h，15 个主流币，覆盖 2020-11-08 至 2026-06-07。每个 fold 使用 fit -> purge -> select -> purge -> trade 的 walk-forward，参数只在过去 select 窗口选择，再应用到未来 trade 窗口。

## 3. 策略设计
- 信号层：XGBoost 回归预测 6-bar 风险调整收益，XGBoost 分类预测 short/flat/long 概率，融合 V12 Transformer 预测和动量/截面规则分数。
- 交易层：每 12 或 18 根 4h bar 再平衡一次，只交易置信度足够高的截面前/后 15%~20%。
- Regime router：上涨趋势偏多、下跌趋势偏空，高波动震荡/冲突 regime 降低总敞口。
- 风控层：单币仓位上限 4%，波动止损 2.0x vol，止盈 3.0x vol，组合回撤超过 12% 后冷却 36 根 4h bar。

## 4. 结果排名
| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 平均换手 | 成本合计 |
|---|---:|---:|---:|---:|---:|---:|
| V12-LowTurnover-same-period | 90.89% | 2.12 | -5.59% | 0.2 | 0.0054 | 0.093535 |
| V12-LowTurnover-Exploratory-same-period | 161.71% | 2.12 | -8.33% | 0.3 | 0.0081 | 0.140303 |
| V13-EnsembleRouterRisk | 26.71% | 1.21 | -3.71% | 0.1778 | 0.0066 | 0.114522 |
| V13-TargetNoKill | 26.71% | 1.21 | -3.71% | 0.1778 | 0.0066 | 0.114522 |
| AlwaysFlat | 0.0% | 0.0 | 0.0% | 0.0 | 0.0 | 0.0 |

## 5. Fold 参数选择
| Fold | Trade 起止 | weights | conf | q | gross | rebalance | select Sharpe |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | 2021-10-18 04:00:00 → 2022-06-15 00:00:00 | (0.55, 0.25, 0.2) | 0.25 | 0.2 | 0.2 | 12 | 6.08 |
| 2 | 2022-06-15 04:00:00 → 2023-02-10 00:00:00 | (1.0, 0.0, 0.0) | 0.25 | 0.2 | 0.15 | 12 | 4.27 |
| 3 | 2023-02-10 04:00:00 → 2023-10-08 00:00:00 | (0.55, 0.25, 0.2) | 0.35 | 0.2 | 0.2 | 12 | 4.78 |
| 4 | 2023-10-08 04:00:00 → 2024-06-04 00:00:00 | (1.0, 0.0, 0.0) | 0.25 | 0.2 | 0.2 | 12 | 2.39 |
| 5 | 2024-06-04 04:00:00 → 2025-01-30 00:00:00 | (0.55, 0.25, 0.2) | 0.25 | 0.2 | 0.15 | 12 | 2.6 |
| 6 | 2025-01-30 04:00:00 → 2025-09-27 00:00:00 | (0.55, 0.25, 0.2) | 0.35 | 0.2 | 0.15 | 12 | 4.48 |
| 7 | 2025-09-27 04:00:00 → 2026-05-25 00:00:00 | (0.55, 0.25, 0.2) | 0.35 | 0.2 | 0.2 | 12 | 3.47 |
| 8 | 2026-05-25 04:00:00 → 2026-06-07 12:00:00 | (1.0, 0.0, 0.0) | 0.35 | 0.2 | 0.2 | 12 | 1.02 |

## 6. 结论
本次 V13 表格中的最高 Sharpe 策略是 `V12-LowTurnover-same-period`，收益 90.89%，Sharpe 2.12，最大回撤 -5.59%。
V13 正式策略 `V13-EnsembleRouterRisk` 的收益为 26.71%、Sharpe 1.21、最大回撤 -3.71%。它的最大回撤低于 V12-LowTurnover，但收益和 Sharpe 明显落后，因此不能声称 V13 已经超过 V12。

需要强调：V13 是方法论更干净的研究框架，不是实盘承诺。它的改进点在于参数选择被放入 walk-forward select 窗口，并加入了低换手、regime router 和风险约束；但正式结果没有证明新 ensemble/router 信号优于 V12 冻结低换手基线。下一步必须把从诊断中提炼出的规则预注册成 V14，再用未参与选择的新窗口或前瞻 paper trading 验证。

## 7. 输出文件
- `data/results_v13.json`
- `data/backtest_v13_results.json`
- `data/v13_trade_decisions.parquet`
- `docs/figures/v13_equity_curves.png`
- `docs/figures/v13_strategy_comparison.png`

## 8. 第二轮诊断：模型/基线选择器

第一轮 V13 的新 ensemble/router 风控策略为正收益，但没有超过 V12 低换手基线。第二轮没有重新训练，而是做诊断：如果每个 fold 允许在 V13 与 V12 低换手之间选择，收益是否来自 V13 新信号，还是仍主要来自 V12 的低换手交易层。

| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 平均换手 |
|---|---:|---:|---:|---:|---:|
| V13-PostHocSelector | 153.43% | 2.94 | -4.3% | 0.2283 | 0.0065 |
| V13-ConservativeSelector-posthoc | 109.73% | 2.65 | -4.3% | 0.1999 | 0.0056 |
| V12-LowTurnover-same-period | 90.89% | 2.12 | -5.59% | 0.2 | 0.0054 |
| V12-LowTurnover-Exploratory-same-period | 161.71% | 2.12 | -8.33% | 0.3 | 0.0081 |
| V13-EnsembleRouterRisk | 26.71% | 1.21 | -3.71% | 0.1778 | 0.0066 |
| AlwaysFlat | 0.0% | 0.0 | 0.0% | 0.0 | 0.0 |

注意：`V13-PostHocSelector` 和 `V13-ConservativeSelector-posthoc` 使用了 trade fold 的事后表现来选择策略，因此不能进入正式排行榜，也不能视为可实盘策略。它们只能作为诊断：如果事先能够识别哪些 fold 应该用 V13、哪些 fold 应该回退到 V12-LT，理论上存在改进空间。但当前实验并没有提供这种事前识别能力。

Oracle 复核后的保守结论：V13 methodology promising, performance not superior to V12。正式可报告结果应以 `data/results_v13.json` 为准；`data/results_v13_improved.json` 只能标记为 post-hoc diagnostics。下一轮 V14 的最低验收标准应是，在预注册 selector 规则下至少超过 V12-LowTurnover 的 Sharpe 或收益/回撤组合，否则继续视为研究失败。
