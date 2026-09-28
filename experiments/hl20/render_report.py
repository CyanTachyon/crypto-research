"""Render static experiment figures from frozen outputs, without running strategies."""
from pathlib import Path
from datetime import datetime, timezone
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

root = Path(__file__).resolve().parent
summary = json.loads((root / "results/v1/summary.json").read_text())
results = json.loads((root / "results/v1/results.json").read_text())
manifest = json.loads((root / "results/v1/manifest.json").read_text())
names = manifest["candidates"]
labels = ["BO 72h", "BO 168h", "BO 336h", "MOM 72h", "MOM 168h", "MOM 336h", "EMA 24/120", "RANGE 48h"]
out = root / "reports"
out.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1.2]})
x = np.arange(len(names))
for shift, scenario, color, label in [(-.18, "base", "#2865ac", "Base: 4.5 bp fee + 2 bp slippage / side"),
                                      (.18, "moderate", "#de8b37", "Stress: 6 bp fee + 5 bp slippage / side")]:
    values = [summary["selection"][scenario][n]["return_pct"] for n in names]
    bars = axes[0].bar(x + shift, values, .35, label=label, color=color)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, value - .22, f"{value:.2f}", ha="center", va="top", fontsize=8)
axes[0].axhline(0, color="#333333", lw=1)
axes[0].set_xticks(x, labels)
axes[0].set_ylim(-12, 1.8)
axes[0].set_ylabel("Net return (%)")
axes[0].set_title("All eight candidates failed the predeclared selection gate", loc="left", pad=28)
axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.12), ncol=2, fontsize=9)
axes[0].grid(axis="y", alpha=.15)
colors = {"breakout_72": "#4f9c85", "momentum_168": "#2865ac", "ema_24_120": "#b04c60", "range_48": "#ac7c24"}
for name, color in colors.items():
    curve = results["selection"]["base"][name]["equity_curve"]
    dates = [datetime.fromtimestamp(p["t"]/1000, timezone.utc) for p in curve]
    axes[1].plot(dates, [p["equity"] for p in curve], label=name, color=color, lw=1.5)
for coin, color in (("BTC", "#777777"), ("ETH", "#b4a1b9")):
    curve = results["benchmarks"]["selection"][coin]["equity_curve"]
    dates = [datetime.fromtimestamp(p["t"]/1000, timezone.utc) for p in curve]
    axes[1].plot(dates, [p["equity"] for p in curve], label=f"{coin} passive 75% notional", color=color, ls="--", lw=1)
axes[1].axhline(20, color="#202020", lw=1, label="USDC cash")
axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axes[1].set_ylabel("Equity (USDC)")
axes[1].set_title("Selection period: Jun 19 – Aug 6, 2026 (UTC)", loc="left")
axes[1].legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
axes[1].grid(alpha=.15)
fig.suptitle("BTC/ETH perpetual research • 20 USDC • No real orders", fontsize=16, x=.07, ha="left")
fig.text(.07, .025, "Historical OHLC fill approximation; real hourly funding rates use trade-price proxy. Final holdout remains unevaluated.", fontsize=9, color="#555555")
fig.tight_layout(rect=(.03, .05, 1, .95), h_pad=2)
fig.savefig(out / "selection_comparison.png", dpi=170, facecolor="white")
fig.savefig(out / "selection_comparison.svg", facecolor="white")
plt.close(fig)
print(out / "selection_comparison.png")
