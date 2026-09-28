import { describe, it, expect } from "vitest";
import { GridBot, DEFAULT_BOT_CONFIG, type BotConfig } from "./bot.js";
import type { PriceFeed } from "./core/price/priceFeed.js";
import { EmergencyStop } from "./risk/emergencyStop.js";
import type { GridLevel } from "./core/grid/gridManager.js";
import pino from "pino";

function makeConfig(overrides: Partial<BotConfig> = {}): BotConfig {
  return {
    ...DEFAULT_BOT_CONFIG,
    mode: "sim",
    pair: "CBBTC/USDC",
    stateFilePath: `/tmp/test-grid-state-${Date.now()}.json`,
    capitalUsd: 100,
    ...overrides,
  };
}

function makeLogger() {
  return pino({ level: "silent" });
}

function makeGridState(levels: GridLevel[] = []): import("./types.js").GridState {
  return {
    pair: "CBBTC/USDC",
    centerPrice: "0x" + (100_000_000n).toString(16),
    levels: levels.map(l => ({
      price: "0x" + l.price.toString(16),
      buyAmount: "0x" + l.amount.toString(16),
      sellAmount: "0x" + l.amount.toString(16),
      filled: l.status === "filled",
      side: l.side,
    })),
    config: { gridLevels: levels.length || 2, rangePct: 1, capitalUsd: 100 },
    fillHistory: [],
  };
}

function makeBotState(levels: GridLevel[] = []): import("./types.js").BotState {
  return {
    version: 1,
    lastBlockNumber: "0x0",
    lastUpdated: Date.now(),
    grids: { "CBBTC/USDC": makeGridState(levels) },
    pendingTx: null,
  };
}

class MockPriceFeed implements PriceFeed {
  private price: bigint;
  constructor(price: bigint) { this.price = price; }
  async getPrice(_pair: string): Promise<bigint> { return this.price; }
}

type SimPortfolio = {
  startingUsd: bigint; usdHolding: bigint; tokenHolding: bigint;
  tokenDecimals: number; tokenSymbol: string;
  totalFeesUsd: bigint; totalGasUsd: bigint; totalSlippageUsd: bigint; tradeCount: number;
};

interface BotAny {
  levels: GridLevel[];
  centerPrice: bigint | null;
  previousPrice: bigint | null;
  tickCount: number;
  config: BotConfig;
  simPortfolio: SimPortfolio | null;
  walletBalances: { usdc: bigint; token: bigint; eth: bigint };
  state: import("./types.js").BotState;
  processTrigger(level: GridLevel, currentPrice: bigint): Promise<void>;
  calculatePerLevelAmount(): bigint;
  getPortfolioValueUsd(): bigint;
  getPortfolioBias(): "buy" | "sell" | "neutral";
  checkWalletBalances(): Promise<void>;
}

function asAny(bot: GridBot): BotAny {
  return bot as unknown as BotAny;
}

function setupSim(b: BotAny, centerPrice: bigint, capitalUsd = 100) {
  const startingUsd = BigInt(capitalUsd) * 1_000_000n;
  b.simPortfolio = {
    startingUsd,
    usdHolding: startingUsd / 2n,
    tokenHolding: (startingUsd / 2n) * 10n ** 8n / centerPrice,
    tokenDecimals: 8, tokenSymbol: "CBBTC",
    totalFeesUsd: 0n, totalGasUsd: 0n, totalSlippageUsd: 0n, tradeCount: 0,
  };
}

describe("GridBot — Ping-Pong Feature", () => {
  const centerPrice = 100_000_000n; // $100 in 6-decimal

  it("creates a new sell level after buy fills", async () => {
    const config = makeConfig({ tiers: [{ rangePct: 1, count: 1 }], pingPongSpreadBps: 60 });
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    bot.centerPrice = centerPrice;
    bot.previousPrice = centerPrice;
    bot.state = makeBotState();
    setupSim(bot, centerPrice);

    const buyPrice = (centerPrice * 9900n) / 10000n; // 1% below center
    bot.levels = [
      { index: 0, price: buyPrice, side: "buy", status: "pending", amount: 9_000_000n },
      { index: 1, price: (centerPrice * 10100n) / 10000n, side: "sell", status: "pending", amount: 9_000_000n },
    ];
    bot.state = makeBotState(bot.levels);

    await bot.processTrigger(bot.levels[0], centerPrice);

    expect(bot.levels.find(l => l.price === buyPrice)?.status).toBe("filled");

    const pendingSells = bot.levels.filter(l => l.side === "sell" && l.status === "pending");
    expect(pendingSells.length).toBeGreaterThanOrEqual(1);

    const pingPongSell = pendingSells.find(l => l.price > buyPrice);
    expect(pingPongSell).toBeDefined();
    expect(pingPongSell!.price).toBe((buyPrice * 10060n) / 10000n);
  });

  it("creates a new buy level after sell fills", async () => {
    const config = makeConfig({ tiers: [{ rangePct: 1, count: 1 }], pingPongSpreadBps: 60 });
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    bot.centerPrice = centerPrice;
    bot.previousPrice = centerPrice;
    bot.state = makeBotState();
    setupSim(bot, centerPrice);

    const sellPrice = (centerPrice * 10100n) / 10000n;
    bot.levels = [
      { index: 0, price: (centerPrice * 9900n) / 10000n, side: "buy", status: "pending", amount: 9_000_000n },
      { index: 1, price: sellPrice, side: "sell", status: "pending", amount: 9_000_000n },
    ];
    bot.state = makeBotState(bot.levels);

    await bot.processTrigger(bot.levels[1], centerPrice);

    expect(bot.levels.find(l => l.price === sellPrice)?.status).toBe("filled");

    const newBuy = bot.levels.find(l => l.side === "buy" && l.status === "pending" && l.price === (sellPrice * 9940n) / 10000n);
    expect(newBuy).toBeDefined();
  });

  it("respects maxLevels cap", async () => {
    const config = makeConfig({ tiers: [{ rangePct: 1, count: 1 }], pingPongSpreadBps: 60, maxLevels: 4 });
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    bot.centerPrice = centerPrice;
    bot.previousPrice = centerPrice;
    bot.state = makeBotState();
    setupSim(bot, centerPrice);

    bot.levels = [
      { index: 0, price: 97_000_000n, side: "buy", status: "pending", amount: 9_000_000n },
      { index: 1, price: 99_000_000n, side: "buy", status: "filled", fillPrice: 99_000_000n, amount: 9_000_000n },
      { index: 2, price: 101_000_000n, side: "sell", status: "pending", amount: 9_000_000n },
      { index: 3, price: 103_000_000n, side: "sell", status: "pending", amount: 9_000_000n },
    ];
    bot.state = makeBotState(bot.levels);

    await bot.processTrigger(
      { index: 0, price: 97_000_000n, side: "buy", status: "pending", amount: 9_000_000n },
      centerPrice,
    );

    expect(bot.levels.length).toBeLessThanOrEqual(4);
  });

  it("maintains sorted levels with sequential indices after ping-pong", async () => {
    const config = makeConfig({ tiers: [{ rangePct: 1, count: 1 }], pingPongSpreadBps: 60 });
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    bot.centerPrice = centerPrice;
    bot.previousPrice = centerPrice;
    bot.state = makeBotState();
    setupSim(bot, centerPrice);

    bot.levels = [
      { index: 0, price: 99_000_000n, side: "buy", status: "pending", amount: 9_000_000n },
      { index: 1, price: 101_000_000n, side: "sell", status: "pending", amount: 9_000_000n },
    ];
    bot.state = makeBotState(bot.levels);

    await bot.processTrigger(bot.levels[0], centerPrice);

    for (let i = 1; i < bot.levels.length; i++) {
      expect(bot.levels[i].price).toBeGreaterThanOrEqual(bot.levels[i - 1].price);
    }
    for (let i = 0; i < bot.levels.length; i++) {
      expect(bot.levels[i].index).toBe(i);
    }
  });
});

describe("GridBot — Dynamic Per-Level Amount", () => {
  const centerPrice = 100_000_000n;

  it("calculates amount based on portfolio value", () => {
    const config = makeConfig({ tiers: [{ rangePct: 1, count: 1 }], pingPongSpreadBps: 60 });
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    bot.centerPrice = centerPrice;
    bot.previousPrice = centerPrice;
    setupSim(bot, centerPrice);
    bot.levels = [
      { index: 0, price: 99_000_000n, side: "buy", status: "pending", amount: 9_000_000n },
      { index: 1, price: 101_000_000n, side: "sell", status: "pending", amount: 9_000_000n },
    ];

    const portfolioValue = bot.getPortfolioValueUsd();
    expect(portfolioValue).toBeGreaterThan(0n);

    const perLevel = bot.calculatePerLevelAmount();
    expect(perLevel).toBeGreaterThan(0n);

    const investable = (portfolioValue * 90n) / 100n; // 90% of portfolio
    const slots = Math.max(4, 4); // 2 active + 2 for room
    expect(perLevel).toBe(investable / BigInt(slots));
  });

  it("returns 0 when no active levels", () => {
    const config = makeConfig();
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    bot.centerPrice = centerPrice;
    bot.previousPrice = centerPrice;
    setupSim(bot, centerPrice);
    bot.levels = [
      { index: 0, price: 99_000_000n, side: "buy", status: "filled", fillPrice: 99_000_000n, amount: 9_000_000n },
    ];

    expect(bot.calculatePerLevelAmount()).toBe(0n);
  });
});

describe("GridBot — Portfolio Balance Bias", () => {
  const centerPrice = 100_000_000n;

  it("returns buy bias when USDC-heavy", () => {
    const config = makeConfig();
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    bot.centerPrice = centerPrice;
    bot.previousPrice = centerPrice;
    bot.simPortfolio = {
      startingUsd: 100_000_000n, usdHolding: 80_000_000n,
      tokenHolding: (20_000_000n * 10n ** 8n) / centerPrice,
      tokenDecimals: 8, tokenSymbol: "CBBTC",
      totalFeesUsd: 0n, totalGasUsd: 0n, totalSlippageUsd: 0n, tradeCount: 0,
    };

    expect(bot.getPortfolioBias()).toBe("buy");
  });

  it("returns sell bias when token-heavy", () => {
    const config = makeConfig();
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    bot.centerPrice = centerPrice;
    bot.previousPrice = centerPrice;
    bot.simPortfolio = {
      startingUsd: 100_000_000n, usdHolding: 20_000_000n,
      tokenHolding: (80_000_000n * 10n ** 8n) / centerPrice,
      tokenDecimals: 8, tokenSymbol: "CBBTC",
      totalFeesUsd: 0n, totalGasUsd: 0n, totalSlippageUsd: 0n, tradeCount: 0,
    };

    expect(bot.getPortfolioBias()).toBe("sell");
  });

  it("returns neutral when balanced", () => {
    const config = makeConfig();
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    bot.centerPrice = centerPrice;
    bot.previousPrice = centerPrice;
    setupSim(bot, centerPrice); // 50/50 split

    expect(bot.getPortfolioBias()).toBe("neutral");
  });

  it("bias overrides ping-pong side direction", async () => {
    const config = makeConfig({ tiers: [{ rangePct: 1, count: 1 }], pingPongSpreadBps: 60 });
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    bot.centerPrice = centerPrice;
    bot.previousPrice = centerPrice;

    // USDC-heavy (90 USDC, 10 token) → buy bias
    bot.simPortfolio = {
      startingUsd: 100_000_000n, usdHolding: 90_000_000n,
      tokenHolding: (10_000_000n * 10n ** 8n) / centerPrice,
      tokenDecimals: 8, tokenSymbol: "CBBTC",
      totalFeesUsd: 0n, totalGasUsd: 0n, totalSlippageUsd: 0n, tradeCount: 0,
    };

    const sellPrice = (centerPrice * 10100n) / 10000n;
    bot.levels = [
      { index: 0, price: 99_000_000n, side: "buy", status: "pending", amount: 9_000_000n },
      { index: 1, price: sellPrice, side: "sell", status: "pending", amount: 9_000_000n },
    ];
    bot.state = makeBotState(bot.levels);

    await bot.processTrigger(bot.levels[1], centerPrice);

    const newBuys = bot.levels.filter(
      l => l.side === "buy" && l.status === "pending" && l.price !== 99_000_000n,
    );
    expect(newBuys.length).toBeGreaterThanOrEqual(1);
  });
});

describe("GridBot — Live Balance Detection", () => {
  const centerPrice = 100_000_000n;

  it("skips balance check in sim mode", async () => {
    const config = makeConfig({ mode: "sim" });
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    await bot.checkWalletBalances();
    expect(bot.walletBalances.usdc).toBe(0n);
    expect(bot.walletBalances.token).toBe(0n);
  });

  it("skips balance check without swapExecutor", async () => {
    const config = makeConfig({ mode: "live" });
    const bot = asAny(new GridBot(config, new MockPriceFeed(centerPrice), new EmergencyStop(), null, makeLogger()));

    await bot.checkWalletBalances();
    expect(bot.walletBalances.usdc).toBe(0n);
  });
});

describe("GridBot — Config Defaults", () => {
  it("DEFAULT_BOT_CONFIG includes pingPongSpreadBps and maxLevels", () => {
    expect(DEFAULT_BOT_CONFIG.pingPongSpreadBps).toBe(60);
    expect(DEFAULT_BOT_CONFIG.maxLevels).toBe(20);
  });

  it("config accepts custom pingPongSpreadBps", () => {
    const config = makeConfig({ pingPongSpreadBps: 100 });
    expect(config.pingPongSpreadBps).toBe(100);
  });

  it("config accepts custom maxLevels", () => {
    const config = makeConfig({ maxLevels: 30 });
    expect(config.maxLevels).toBe(30);
  });
});
