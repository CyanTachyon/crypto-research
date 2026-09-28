# BTC/ETH 永续研究：20 USDC

独立于旧交易机器人。只使用公开行情，不读取 `.env`、钱包密钥、交易账户，也没有 `/exchange` 下单实现。

第一轮结果：**8 个候选均未通过选择期，不能申请实盘；最后的保留区间未评估。** 完整说明见 `reports/experiment_v1_zh.md`，原始逐笔成交与净值见 `results/v1/results.json`。

## 本地复现

Python 3.9+，核心实验和测试只用标准库；静态图另需 matplotlib/numpy。当前已有数据不需要重新下载：

```bash
python3 -m unittest discover -s experiments/hl20 -p 'test_*.py' -v
python3 experiments/hl20/run_experiment.py --out experiments/hl20/results/reproduction
python3 experiments/hl20/render_report.py
```

输出目录必须不存在，防止覆盖旧实验。运行前保存策略、代码、数据和方案哈希；在读取选中候选的保留区间结果前保存选择决定。无人达标时不会运行任何候选的保留区间回测。

`data/hyperliquid_1h.json` 与 `.manifest.json` 必须一起保留。`load_verified()` 同时校验文件哈希和逐小时连续性。重新抓取会产生新的样本，不能冒充本轮冻结实验的重现。

## 文件

- `PROTOCOL.md`：预先固定的候选、时间切分、成本、风控与晋级标准。
- `data_source.py`：公开 API 采集、完整性校验、原始响应来源与哈希。
- `strategies.py`：8 个只使用已完成历史的策略信号。
- `engine.py`：一个持仓、双边费用、资金费、数量精度、真实清仓、日锁及回撤锁。
- `run_experiment.py`：冻结清单、开发/选择/保留区间纪律、基线与成本压力。
- `test_*.py`：54 项离线测试，含独立的交易引擎边界测试。
- `results/v1`：本轮冻结结果，不覆盖；`reports`：中文解释与静态图。

## 后续研究纪律

本轮不以增加杠杆、放宽亏损限制、删除亏损样本或改选保留区间赢家来制造成功。新的假设须单独编号、说明为何提出、哪些历史已被查看，并保留新的最终验证窗口。

即使未来历史指标通过，也必须另行验证订单拒绝、部分成交、超时后已成交、幂等、持久化和重启恢复，并积累冻结参数的未来公开行情模拟记录。当前研究引擎没有声称完成这些交易所执行能力。达到可靠性门槛前，不启动实盘、不请求实盘授权。
