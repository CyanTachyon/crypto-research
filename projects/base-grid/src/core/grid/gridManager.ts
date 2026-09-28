import { calcGeometricSpacing } from "../../utils/math.js";

export interface GridLevel {
  index: number;
  price: bigint;
  side: "buy" | "sell" | "none";
  status: "pending" | "filled";
  amount: bigint;
  fillPrice?: bigint;
}

export interface GridTier {
  rangePct: number;
  count: number;
}

export interface GridConfig {
  pair: string;
  centerPrice: bigint;
  gridLevels: number;
  rangePct?: number;
  tiers?: GridTier[];
  capitalUsd: number;
  reservePct: number;
}

export interface GridStats {
  totalBuyAmount: bigint;
  totalSellAmount: bigint;
  filledCount: number;
  pendingCount: number;
}

export function createGridLevels(config: GridConfig): GridLevel[] {
  if (config.tiers) {
    return createMultiTierGrid(config);
  }
  return createUniformGrid(config);
}

function createUniformGrid(config: GridConfig): GridLevel[] {
  const rangePct = config.rangePct ?? 3;
  const { centerPrice, gridLevels, capitalUsd, reservePct } = config;

  const lower = (centerPrice * BigInt(100 - rangePct)) / 100n;
  const upper = (centerPrice * BigInt(100 + rangePct)) / 100n;

  const prices = calcGeometricSpacing(lower, upper, gridLevels);

  const investableUsd = (capitalUsd * (100 - reservePct)) / 100;
  const perLevelUsd = Math.floor(investableUsd / gridLevels);
  const amount = BigInt(perLevelUsd) * 1_000_000n;

  const middleIndex = Math.floor(gridLevels / 2);

  return prices.map((price, i) => ({
    index: i,
    price,
    side: i < middleIndex ? "buy" as const : i > middleIndex ? "sell" as const : "none" as const,
    status: "pending" as const,
    amount,
  }));
}

function createMultiTierGrid(config: GridConfig): GridLevel[] {
  const { centerPrice, tiers, capitalUsd, reservePct } = config;

  const totalLevels = tiers!.reduce((sum, t) => sum + t.count * 2, 0);
  const investableUsd = (capitalUsd * (100 - reservePct)) / 100;
  const perLevelUsd = Math.floor(investableUsd / totalLevels);
  const amount = BigInt(perLevelUsd) * 1_000_000n;

  const prices: bigint[] = [];

  for (const tier of tiers!) {
    const rangeBps = BigInt(Math.round(tier.rangePct * 100));
    const buyPrice = (centerPrice * (10000n - rangeBps)) / 10000n;
    const sellPrice = (centerPrice * (10000n + rangeBps)) / 10000n;

    for (let i = 0; i < tier.count; i++) {
      prices.push(buyPrice);
      prices.push(sellPrice);
    }
  }

  prices.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

  return prices.map((price, i) => {
    let side: "buy" | "sell" | "none";
    if (price < centerPrice) {
      side = "buy";
    } else if (price > centerPrice) {
      side = "sell";
    } else {
      side = "none";
    }
    return {
      index: i,
      price,
      side,
      status: "pending" as const,
      amount,
    };
  });
}

export function checkTriggers(
  levels: GridLevel[],
  previousPrice: bigint,
  currentPrice: bigint,
): GridLevel[] {
  const triggered: GridLevel[] = [];

  for (const level of levels) {
    if (level.status !== "pending") continue;

    if (level.side === "buy" && previousPrice > level.price && level.price >= currentPrice) {
      triggered.push(level);
    } else if (level.side === "sell" && previousPrice < level.price && level.price <= currentPrice) {
      triggered.push(level);
    }
  }

  return triggered;
}

export function markFilled(
  levels: GridLevel[],
  levelIndex: number,
  fillPrice: bigint,
): GridLevel[] {
  const updated = levels.map((l) => ({ ...l }));

  const filled = updated[levelIndex];
  updated[levelIndex] = { ...filled, status: "filled", fillPrice };

  if (filled.side === "buy" && levelIndex + 1 < updated.length) {
    const above = updated[levelIndex + 1];
    updated[levelIndex + 1] = { ...above, side: "sell", status: "pending" };
  }

  if (filled.side === "sell" && levelIndex - 1 >= 0) {
    const below = updated[levelIndex - 1];
    updated[levelIndex - 1] = { ...below, side: "buy", status: "pending" };
  }

  return updated;
}

export function shouldRecenter(
  levels: GridLevel[],
  currentPrice: bigint,
  centerPrice: bigint,
  thresholdPct: number = 5,
): boolean {
  const upperBound = (centerPrice * BigInt(100 + thresholdPct)) / 100n;
  const lowerBound = (centerPrice * BigInt(100 - thresholdPct)) / 100n;
  return currentPrice > upperBound || currentPrice < lowerBound;
}

export function getGridStats(
  levels: GridLevel[],
  _centerPrice: bigint,
): GridStats {
  let totalBuyAmount = 0n;
  let totalSellAmount = 0n;
  let filledCount = 0;
  let pendingCount = 0;

  for (const level of levels) {
    if (level.status === "filled") {
      filledCount++;
    } else {
      pendingCount++;
    }
    if (level.side === "buy") {
      totalBuyAmount += level.amount;
    } else if (level.side === "sell") {
      totalSellAmount += level.amount;
    }
  }

  return { totalBuyAmount, totalSellAmount, filledCount, pendingCount };
}
