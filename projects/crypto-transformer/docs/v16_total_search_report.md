# V16 总攻搜索实验报告

## 1. 目标
在 4h 高频数据上系统搜索信号模型、交易层、regime 控制和敞口参数，目标是在严格 2025+ holdout 上找出当前最快赚钱的候选方案。

## 2. 时间切分
- Fit: <= 2024-07-01 00:00:00
- Select: 2024-07-23 00:00:00 到 2024-12-10 00:00:00
- Holdout: >= 2025-01-01 00:00:00

## 3. 方法边界
V16 同时搜索 strict_raw 与 augmented_v12 两类候选。strict_raw 只用行情衍生特征；augmented_v12 允许使用 V12 冻结预测列，因此只能作为增强候选而非纯 raw 结论。所有最终指标只在 holdout 上计算一次。

## 4. Holdout 排名
| 策略 | 收益 | Sharpe | 最大回撤 | 平均总敞口 | 换手 |
|---|---:|---:|---:|---:|---:|
| AlwaysFlat | 0.0% | 0.0 | 0.0% | 0.0 | 0.0 |
| V12-LT-reference | -6.66% | -0.7 | -10.39% | 0.2 | 0.0063 |
| V12-LT30-reference | -10.03% | -0.7 | -15.26% | 0.3 | 0.0095 |
| V16-StrictRaw-SelectedOn2024 | -51.5% | -0.73 | -60.83% | 0.7538 | 0.0003 |
| V16-BestAny-SelectedOn2024 | -56.41% | -0.82 | -60.66% | 0.7712 | 0.0004 |

## 5. 当前结论
本轮 holdout 第一名是 `AlwaysFlat`：收益 0.0%，Sharpe 0.0，最大回撤 0.0%。
如果第一名属于 augmented_v12，需要继续用未来 paper trading 或更严格重训来确认；如果 strict_raw 也表现良好，说明纯行情特征路线更有研究价值。

## 6. 输出文件
- `/home/cyan/default/crypto/data/results_v16_total_search.json`
- `/home/cyan/default/crypto/data/backtest_v16_total_search_results.json`
- `/home/cyan/default/crypto/data/v16_total_search_decisions.parquet`
- `/home/cyan/default/crypto/docs/figures/v16_total_search_equity.png`