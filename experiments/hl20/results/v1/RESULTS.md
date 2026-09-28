# Frozen BTC/ETH 20-USDC experiment

Selected: `NONE`

Status: NOT_QUALIFIED

Real trading remains disabled. No real-trading approval requested.

## Selection, base costs

| Candidate | Net USDC | Return | MDD | Trades | Fees | Funding paid |
|---|---:|---:|---:|---:|---:|---:|
| breakout_72 | -0.0904 | -0.45% | 1.64% | 3 | 0.0349 | -0.0002 |
| breakout_168 | -0.2372 | -1.19% | 1.44% | 2 | 0.0184 | +0.0054 |
| breakout_336 | -0.2372 | -1.19% | 1.44% | 2 | 0.0184 | +0.0054 |
| momentum_72 | -1.9767 | -9.88% | 10.04% | 27 | 0.2908 | -0.0027 |
| momentum_168 | -0.2596 | -1.30% | 5.17% | 25 | 0.2690 | +0.0235 |
| momentum_336 | -0.8978 | -4.49% | 5.81% | 21 | 0.2224 | +0.0079 |
| ema_24_120 | -1.9778 | -9.89% | 10.04% | 19 | 0.1994 | +0.0130 |
| range_48 | -0.8289 | -4.14% | 6.35% | 14 | 0.1530 | -0.0032 |

## Selection, moderate cost stress

| Candidate | Net USDC | Return | MDD | Trades | Fees | Funding paid |
|---|---:|---:|---:|---:|---:|---:|
| breakout_72 | -0.1189 | -0.59% | 1.70% | 3 | 0.0456 | -0.0000 |
| breakout_168 | -0.1743 | -0.87% | 1.04% | 1 | 0.0120 | +0.0011 |
| breakout_336 | -0.1743 | -0.87% | 1.04% | 1 | 0.0120 | +0.0011 |
| momentum_72 | -1.9865 | -9.93% | 10.06% | 18 | 0.2526 | -0.0025 |
| momentum_168 | -0.7996 | -4.00% | 6.67% | 22 | 0.3088 | +0.0200 |
| momentum_336 | -0.8565 | -4.28% | 5.52% | 20 | 0.2797 | +0.0063 |
| ema_24_120 | -1.9901 | -9.95% | 10.07% | 19 | 0.2663 | +0.0072 |
| range_48 | -1.0418 | -5.21% | 6.35% | 12 | 0.1764 | -0.0026 |

## Selection benchmarks (75% initial notional, no stop)

| Candidate | Net USDC | Return | MDD | Trades | Fees | Funding paid |
|---|---:|---:|---:|---:|---:|---:|
| BTC | +0.2153 | +1.08% | 8.73% | 1 | 0.0132 | +0.1479 |
| ETH | +1.5609 | +7.80% | 11.23% | 1 | 0.0142 | +0.1490 |

USDC cash benchmark: 0%.

No candidate passed predeclared selection criteria. Holdout strategy returns remain unevaluated.

## Gate details

```json
{
  "selection_passed": false,
  "historical_passed": false,
  "forward_paper_passed": false,
  "exchange_fault_tests_passed": false,
  "eligible_to_ask_for_real_trading": false
}
```

Full assumptions, windows and file hashes: manifest.json. Funding valuations use a trade-price proxy; hourly OHLC cannot validate exchange fills.
