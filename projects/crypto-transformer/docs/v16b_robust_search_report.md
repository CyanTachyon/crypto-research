# V16b 鲁棒搜索实验报告

## 1. 目标
在 4h 高频数据上系统搜索信号模型、交易层、regime 控制和敞口参数。本版修复 V16 第一轮与 V14 不一致的风控 clipping 口径，并惩罚高敞口/单边 beta。

## 2. 时间切分
- Fit: <= 2024-07-01 00:00:00
- Select: 2024-07-23 00:00:00 到 2024-12-10 00:00:00
- Holdout: >= 2025-01-01 00:00:00

## 3. 方法边界
V16b 同时搜索 strict_raw 与 augmented_v12 两类候选。strict_raw 只用行情衍生特征；augmented_v12 允许使用 V12 冻结预测列，因此只能作为增强候选而非纯 raw 结论。所有最终指标只在 holdout 上计算一次。

## 4. Holdout 排名
| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 换手 |
|---|---:|---:|---:|---:|---:|
| V12-LT30-reference | 19.4% | 1.46 | -5.71% | 0.3 | 0.0095 |
| V12-LT-reference | 12.68% | 1.46 | -3.83% | 0.2 | 0.0063 |
| V16-StrictRaw-SelectedOn2024 | 2.02% | 2.11 | -0.36% | 0.0225 | 0.0002 |
| V16-BestAny-SelectedOn2024 | 2.02% | 2.11 | -0.36% | 0.0225 | 0.0002 |
| AlwaysFlat | 0.0% | 0.0 | 0.0% | 0.0 | 0.0 |

## 5. 当前结论
本轮 holdout 第一名是 `V12-LT30-reference`：收益 19.4%，Sharpe 1.46，最大回撤 -5.71%。
如果第一名属于 augmented_v12，需要继续用未来 paper trading 或更严格重训来确认；如果 strict_raw 也表现良好，说明纯行情特征路线更有研究价值。

## 6. 输出文件
- `/home/cyan/default/crypto/data/results_v16b_robust_search.json`
- `/home/cyan/default/crypto/data/backtest_v16b_robust_search_results.json`
- `/home/cyan/default/crypto/data/v16b_robust_search_decisions.parquet`
- `/home/cyan/default/crypto/docs/figures/v16b_robust_search_equity.png`