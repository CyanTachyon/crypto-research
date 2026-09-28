# Crypto research

整合本机 Base 网格项目、服务器上的模型交易实验，以及 Hyperliquid BTC/ETH 永续合约研究。仓库创建于 2026-09-28，各项目保留原有代码和相对目录结构。

## 目录

| 路径 | 内容 |
| --- | --- |
| `projects/base-grid/` | 本机 TypeScript Base 网格交易项目及原始说明 |
| `projects/crypto-transformer/` | SSH U 上的 Python 模型、回测、历史交易脚本和研究报告 |
| `experiments/hl20/` | 20 USDC Hyperliquid 实验协议、数据、回测引擎、测试和冻结结果 |
| `docs/audits/` | 旧项目审计及证据 |
| `scripts/` | 只读远程审计工具 |
| `archive/` | 仅本机保留的代理笔记和服务器数据采集副本，不纳入 Git |

## 当前研究结论

Hyperliquid v1 的八个候选策略全部未通过选择阶段，尚无获准实盘的策略；最终保留集没有参与评估。详见 [实验报告](experiments/hl20/reports/experiment_v1_zh.md) 和 [实验协议](experiments/hl20/PROTOCOL.md)。

`projects/` 保存历史实现，其中包含实盘脚本和已知问题；这些脚本不是推荐的启动入口。整理仓库没有修改交易逻辑，也没有启动真实交易。密钥和 `.env` 未复制，依赖环境需要分别安装。

## 验证

Hyperliquid 研究测试只使用本地数据，不需要账户：

```sh
python3 -m unittest discover -s experiments/hl20 -p 'test_*.py'
```

Base 项目测试：

```sh
cd projects/base-grid
npm ci --ignore-scripts
npm test
npx tsc --noEmit
```

服务器 Python 项目的依赖见其 `requirements.txt` 和 `pyproject.toml`。历史训练数据和模型权重保留在本机 `projects/crypto-transformer/data/`，不随 Git 克隆分发。迁移映射、排除内容与验证记录见 [整理记录](docs/consolidation.md)。
