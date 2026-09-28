import { describe, it, expect } from "vitest";
import {
  createGridLevels,
  checkTriggers,
  markFilled,
  shouldRecenter,
  getGridStats,
} from "./gridManager.js";
import type { GridConfig, GridTier, GridLevel } from "./gridManager.js";
import { calcGeometricSpacing, toUsdcValue, fromUsdcValue, pctDifference } from "../../utils/math.js";

function assertApprox(actual: bigint, expected: bigint, maxBps: number) {
  const diff = actual > expected ? actual - expected : expected - actual;
  const bps = Number((diff * 10000n) / expected);
  expect(bps, `${actual} vs ${expected}: ${bps} bps > ${maxBps} bps tolerance`).toBeLessThan(maxBps);
}

describe("gridManager", () => {
  const config: GridConfig = {
    pair: "ETH/USDC",
    centerPrice: 2_500_000_000n,
    gridLevels: 5,
    rangePct: 3,
    capitalUsd: 1000,
    reservePct: 10,
  };

  describe("createGridLevels", () => {
    it("creates 5 geometric-spaced levels within ±3% range", () => {
      const levels = createGridLevels(config);
      expect(levels).toHaveLength(5);

      assertApprox(levels[0].price, 2_425_000_000n, 50);
      assertApprox(levels[1].price, 2_462_000_000n, 50);
      assertApprox(levels[2].price, 2_500_000_000n, 200);
      assertApprox(levels[3].price, 2_538_000_000n, 50);
      assertApprox(levels[4].price, 2_575_000_000n, 50);
    });

    it("assigns sides: buy, buy, none, sell, sell", () => {
      const levels = createGridLevels(config);
      expect(levels.map((l) => l.side)).toEqual(["buy", "buy", "none", "sell", "sell"]);
    });

    it("all levels start pending with equal allocation", () => {
      const levels = createGridLevels(config);
      const expectedPerLevel = BigInt(Math.floor((1000 * 90) / 100) / 5) * 1_000_000n;

      for (const level of levels) {
        expect(level.status).toBe("pending");
        expect(level.amount).toBe(expectedPerLevel);
      }
    });

    it("has ~1.5% geometric spacing between adjacent levels", () => {
      const levels = createGridLevels(config);
      for (let i = 1; i < levels.length; i++) {
        const spacingBps = Number(
          ((levels[i].price - levels[i - 1].price) * 10000n) / levels[i - 1].price,
        );
        expect(spacingBps).toBeGreaterThan(100);
        expect(spacingBps).toBeLessThan(200);
      }
    });

    it("assigns correct indices", () => {
      const levels = createGridLevels(config);
      expect(levels.map((l) => l.index)).toEqual([0, 1, 2, 3, 4]);
    });
  });

  describe("createGridLevels — multi-tier", () => {
    const multiTierConfig: GridConfig = {
      pair: "ETH/USDC",
      centerPrice: 2_500_000_000n,
      gridLevels: 4,
      tiers: [
        { rangePct: 0.8, count: 1 },
        { rangePct: 2.5, count: 1 },
      ],
      capitalUsd: 25,
      reservePct: 10,
    };

    it("creates 4 levels with two tiers (1+1 each side)", () => {
      const levels = createGridLevels(multiTierConfig);
      expect(levels).toHaveLength(4);
    });

    it("assigns 2 buy and 2 sell sides", () => {
      const levels = createGridLevels(multiTierConfig);
      const buys = levels.filter((l) => l.side === "buy");
      const sells = levels.filter((l) => l.side === "sell");
      expect(buys).toHaveLength(2);
      expect(sells).toHaveLength(2);
    });

    it("places inner tier at 0.8% from center", () => {
      const levels = createGridLevels(multiTierConfig);
      const innerBuy = levels.filter((l) => l.side === "buy").sort((a, b) => Number(b.price - a.price))[0];
      const innerSell = levels.filter((l) => l.side === "sell").sort((a, b) => Number(a.price - b.price))[0];

      const buyPct = Number(((2_500_000_000n - innerBuy.price) * 10000n) / 2_500_000_000n);
      const sellPct = Number(((innerSell.price - 2_500_000_000n) * 10000n) / 2_500_000_000n);
      expect(buyPct).toBe(80);
      expect(sellPct).toBe(80);
    });

    it("places outer tier at 2.5% from center", () => {
      const levels = createGridLevels(multiTierConfig);
      const outerBuy = levels.filter((l) => l.side === "buy").sort((a, b) => Number(a.price - b.price))[0];
      const outerSell = levels.filter((l) => l.side === "sell").sort((a, b) => Number(b.price - a.price))[0];

      const buyPct = Number(((2_500_000_000n - outerBuy.price) * 10000n) / 2_500_000_000n);
      const sellPct = Number(((outerSell.price - 2_500_000_000n) * 10000n) / 2_500_000_000n);
      expect(buyPct).toBe(250);
      expect(sellPct).toBe(250);
    });

    it("sorts levels by price ascending", () => {
      const levels = createGridLevels(multiTierConfig);
      for (let i = 1; i < levels.length; i++) {
        expect(levels[i].price).toBeGreaterThanOrEqual(levels[i - 1].price);
      }
    });

    it("all levels start pending with equal allocation ($5.625 each)", () => {
      const levels = createGridLevels(multiTierConfig);
      const totalLevels = 4;
      const investableUsd = (25 * 90) / 100;
      const perLevelUsd = Math.floor(investableUsd / totalLevels);
      const expectedAmount = BigInt(perLevelUsd) * 1_000_000n;

      for (const level of levels) {
        expect(level.status).toBe("pending");
        expect(level.amount).toBe(expectedAmount);
      }
    });

    it("assigns sequential indices", () => {
      const levels = createGridLevels(multiTierConfig);
      expect(levels.map((l) => l.index)).toEqual([0, 1, 2, 3]);
    });
  });

  describe("checkTriggers", () => {
    it("detects price drop triggering 2 buy levels", () => {
      const levels = createGridLevels(config);
      const triggered = checkTriggers(levels, 2_490_000_000n, 2_420_000_000n);

      expect(triggered).toHaveLength(2);
      expect(triggered.every((l) => l.side === "buy")).toBe(true);
    });

    it("detects price rise triggering 2 sell levels", () => {
      const levels = createGridLevels(config);
      const triggered = checkTriggers(levels, 2_510_000_000n, 2_580_000_000n);

      expect(triggered).toHaveLength(2);
      expect(triggered.every((l) => l.side === "sell")).toBe(true);
    });

    it("detects flash crash triggering all buy levels", () => {
      const levels = createGridLevels(config);
      const triggered = checkTriggers(levels, 2_490_000_000n, 2_400_000_000n);

      const buyLevels = levels.filter((l) => l.side === "buy");
      expect(triggered).toHaveLength(buyLevels.length);
      expect(triggered.every((l) => l.side === "buy")).toBe(true);
    });

    it("returns empty when no levels are crossed", () => {
      const levels = createGridLevels(config);
      const triggered = checkTriggers(levels, 2_500_000_000n, 2_505_000_000n);
      expect(triggered).toHaveLength(0);
    });

    it("skips filled levels", () => {
      let levels = createGridLevels(config);
      const buyLevels = levels.filter((l) => l.side === "buy");
      levels = markFilled(levels, buyLevels[1].index, buyLevels[1].price);

      const triggered = checkTriggers(levels, 2_490_000_000n, 2_400_000_000n);
      expect(triggered).toHaveLength(1);
    });
  });

  describe("markFilled", () => {
    it("marks buy level filled and generates sell at adjacent level", () => {
      const levels = createGridLevels(config);
      const result = markFilled(levels, 1, 2_462_000_000n);

      expect(result[1].status).toBe("filled");
      expect(result[1].fillPrice).toBe(2_462_000_000n);

      expect(result[2].side).toBe("sell");
      expect(result[2].status).toBe("pending");
    });

    it("marks sell level filled and generates buy at adjacent level", () => {
      const levels = createGridLevels(config);
      const result = markFilled(levels, 3, 2_538_000_000n);

      expect(result[3].status).toBe("filled");
      expect(result[3].fillPrice).toBe(2_538_000_000n);

      expect(result[2].side).toBe("buy");
      expect(result[2].status).toBe("pending");
    });

    it("returns immutable copy — original unchanged", () => {
      const levels = createGridLevels(config);
      const result = markFilled(levels, 0, levels[0].price);

      expect(levels[0].status).toBe("pending");
      expect(result[0].status).toBe("filled");
    });

    it("handles filling at top boundary (no level above)", () => {
      const levels = createGridLevels(config);
      const result = markFilled(levels, 4, levels[4].price);

      expect(result[4].status).toBe("filled");
      expect(result).toHaveLength(levels.length);
    });

    it("handles filling at bottom boundary (no level below)", () => {
      const levels = createGridLevels(config);
      const result = markFilled(levels, 0, levels[0].price);

      expect(result[0].status).toBe("filled");
      expect(result).toHaveLength(levels.length);
    });
  });

  describe("shouldRecenter", () => {
    it("returns true when price exceeds +5% of center", () => {
      const levels = createGridLevels(config);
      expect(shouldRecenter(levels, 2_640_000_000n, config.centerPrice)).toBe(true);
    });

    it("returns true when price drops below -5% of center", () => {
      const levels = createGridLevels(config);
      expect(shouldRecenter(levels, 2_370_000_000n, config.centerPrice)).toBe(true);
    });

    it("returns false when price within range", () => {
      const levels = createGridLevels(config);
      expect(shouldRecenter(levels, 2_500_000_000n, config.centerPrice)).toBe(false);
    });

    it("returns false at boundary exactly at threshold", () => {
      const levels = createGridLevels(config);
      const upperBound = (config.centerPrice * 105n) / 100n;
      expect(shouldRecenter(levels, upperBound, config.centerPrice)).toBe(false);

      const lowerBound = (config.centerPrice * 95n) / 100n;
      expect(shouldRecenter(levels, lowerBound, config.centerPrice)).toBe(false);
    });
  });

  describe("getGridStats", () => {
    it("returns correct stats for fresh grid", () => {
      const levels = createGridLevels(config);
      const stats = getGridStats(levels, config.centerPrice);

      expect(stats.filledCount).toBe(0);
      expect(stats.pendingCount).toBe(5);
      expect(stats.totalBuyAmount).toBeGreaterThan(0n);
      expect(stats.totalSellAmount).toBeGreaterThan(0n);
    });

    it("updates counts after fills", () => {
      let levels = createGridLevels(config);
      levels = markFilled(levels, 1, levels[1].price);
      const stats = getGridStats(levels, config.centerPrice);

      expect(stats.filledCount).toBe(1);
      expect(stats.pendingCount).toBe(4);
    });
  });

  describe("calcGeometricSpacing", () => {
    it("returns single level for N=1", () => {
      const prices = calcGeometricSpacing(1000n, 2000n, 1);
      expect(prices).toEqual([1000n]);
    });

    it("returns empty for N=0", () => {
      const prices = calcGeometricSpacing(1000n, 2000n, 0);
      expect(prices).toEqual([]);
    });

    it("returns identical values when lower === upper", () => {
      const prices = calcGeometricSpacing(1000n, 1000n, 3);
      expect(prices).toEqual([1000n, 1000n, 1000n]);
    });
  });

  describe("toUsdcValue / fromUsdcValue", () => {
    it("round-trips 18-decimal ETH amount", () => {
      const ethAmount = 1_000_000_000_000_000_000n;
      const usdcValue = toUsdcValue(ethAmount, 18);
      const back = fromUsdcValue(usdcValue, 18);
      expect(usdcValue).toBe(1_000_000n);
      expect(back).toBe(ethAmount);
    });

    it("handles same decimals (6)", () => {
      expect(toUsdcValue(123n, 6)).toBe(123n);
      expect(fromUsdcValue(123n, 6)).toBe(123n);
    });
  });

  describe("pctDifference", () => {
    it("returns 0 for equal values", () => {
      expect(pctDifference(100n, 100n)).toBe(0n);
    });

    it("returns ~100 bps for 1% difference", () => {
      const bps = pctDifference(100n, 101n);
      expect(Number(bps)).toBeGreaterThan(90);
      expect(Number(bps)).toBeLessThan(110);
    });
  });
});
