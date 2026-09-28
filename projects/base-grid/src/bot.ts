import type { Address } from "viem";
import type pino from "pino";
import type { PriceFeed } from "./core/price/priceFeed.js";
import type { SwapExecutor, SwapOrder } from "./core/swap/swapExecutor.js";
import type { BotState, GridLevelState } from "./types.js";
import type { RiskConfig, TradeContext } from "./risk/riskManager.js";
import type { Notifier } from "./utils/notify.js";
import { EmergencyStop } from "./risk/emergencyStop.js";
import {
  createGridLevels,
  checkTriggers,
  markFilled,
  shouldRecenter,
  getGridStats,
} from "./core/grid/gridManager.js";
import type { GridLevel, GridConfig, GridStats } from "./core/grid/gridManager.js";
import {
  loadState,
  saveState,
  createDefaultState,
  recordFill,
  clearPendingTx,
} from "./storage/orderStore.js";
import { DEFAULT_RISK_CONFIG } from "./risk/riskManager.js";
import { TOKENS } from "./config/chains.js";
import { erc20Abi } from "./abis/erc20.js";
import { VolatilityTracker } from "./utils/volatility.js";

type Logger = pino.Logger;

export interface SimCosts {
  dexFeeBps: number;
  gasUsd: number;
  slippageBps: number;
}

export const DEFAULT_SIM_COSTS: SimCosts = {
  dexFeeBps: 1,
  gasUsd: 0.01,
  slippageBps: 1,
};

export interface GridTier {
  rangePct: number;
  count: number;
}

export interface BotConfig {
  mode: "sim" | "live";
  pair: string;
  gridLevels: number;
  rangePct?: number;
  tiers?: GridTier[];
  capitalUsd: number;
  reservePct: number;
  tickIntervalMs: number;
  stateFilePath: string;
  slippageBps: number;
  simCosts?: SimCosts;
  pingPongSpreadBps?: number;
  maxLevels?: number;
}

export const DEFAULT_BOT_CONFIG: Omit<BotConfig, "mode" | "pair" | "stateFilePath"> = {
  gridLevels: 10,
  tiers: [
    { rangePct: 0.3, count: 1 },
    { rangePct: 1.0, count: 1 },
    { rangePct: 1.8, count: 1 },
    { rangePct: 3.0, count: 1 },
    { rangePct: 5.0, count: 1 },
  ],
  capitalUsd: 100,
  reservePct: 10,
  tickIntervalMs: 10_000,
  slippageBps: 50,
  pingPongSpreadBps: 60,
  maxLevels: 20,
};

interface SimPortfolio {
  startingUsd: bigint;
  usdHolding: bigint;
  tokenHolding: bigint;
  tokenDecimals: number;
  tokenSymbol: string;
  totalFeesUsd: bigint;
  totalGasUsd: bigint;
  totalSlippageUsd: bigint;
  tradeCount: number;
}

function formatPrice(price: bigint): string {
  return (Number(price) / 1_000_000).toFixed(2);
}

function formatToken(amount: bigint, decimals: number): string {
  return (Number(amount) / 10 ** decimals).toFixed(6);
}

function bigintToHex(value: bigint): string {
  return "0x" + value.toString(16);
}

function gridLevelToState(level: GridLevel): GridLevelState {
  return {
    price: bigintToHex(level.price),
    buyAmount: bigintToHex(level.amount),
    sellAmount: bigintToHex(level.amount),
    filled: level.status === "filled",
    side: level.side,
  };
}

function stateToGridLevel(state: GridLevelState, index: number): GridLevel {
  const amount =
    state.side === "sell"
      ? BigInt(state.sellAmount)
      : BigInt(state.buyAmount);
  return {
    index,
    price: BigInt(state.price),
    side: state.side,
    status: state.filled ? "filled" : "pending",
    amount,
  };
}

export class GridBot {
  private running = false;
  private previousPrice: bigint | null = null;
  private levels: GridLevel[] = [];
  private centerPrice: bigint | null = null;
  private state: BotState;
  private timeoutId: ReturnType<typeof setTimeout> | null = null;
  private simPortfolio: SimPortfolio | null = null;
  private volTracker = new VolatilityTracker(360);
  private tickCount = 0;
  private lastVolAdjustTick = 0;
  private currentTiers: GridTier[];
  private lastBalanceCheck = 0;
  private walletBalances = { usdc: 0n, token: 0n, eth: 0n };

  constructor(
    private config: BotConfig,
    private priceFeed: PriceFeed,
    private emergencyStop: EmergencyStop,
    private swapExecutor: SwapExecutor | null,
    private logger: Logger,
    private riskConfig: RiskConfig = DEFAULT_RISK_CONFIG,
    private notifier: Notifier | null = null,
  ) {
    this.state = createDefaultState();
    this.currentTiers = this.config.tiers ?? [
      { rangePct: 0.3, count: 1 },
      { rangePct: 1.0, count: 1 },
      { rangePct: 1.8, count: 1 },
      { rangePct: 3.0, count: 1 },
      { rangePct: 5.0, count: 1 },
    ];
  }

  async start(): Promise<void> {
    this.state = await loadState(this.config.stateFilePath);

    const initialPrice = await this.priceFeed.getPrice(this.config.pair);
    this.logger.info({ price: formatPrice(initialPrice), pair: this.config.pair }, "Initial price fetched");

    if (this.config.mode === "sim") {
      const startingUsd = BigInt(this.config.capitalUsd) * 1_000_000n;
      const halfUsd = startingUsd / 2n;
      const { base: baseDecimals } = this.getTokenDecimals();
      const tokenSymbol = this.config.pair.split("/")[0];
      const halfToken = (halfUsd * 10n ** BigInt(baseDecimals)) / initialPrice;
      this.simPortfolio = {
        startingUsd,
        usdHolding: halfUsd,
        tokenHolding: halfToken,
        tokenDecimals: baseDecimals,
        tokenSymbol,
        totalFeesUsd: 0n,
        totalGasUsd: 0n,
        totalSlippageUsd: 0n,
        tradeCount: 0,
      };
    }

    const existingGrid = this.state.grids[this.config.pair];
    if (existingGrid) {
      this.levels = existingGrid.levels.map((l, i) => stateToGridLevel(l, i));
      this.centerPrice = BigInt(existingGrid.centerPrice);
      this.logger.info(
        { levels: this.levels.length, filled: this.levels.filter((l) => l.status === "filled").length },
        "Recovered grid state",
      );

      if (this.state.pendingTx) {
        if (this.config.mode === "sim") {
          this.logger.warn({ txHash: this.state.pendingTx.txHash }, "Pending tx found — clearing in sim mode");
          this.state = clearPendingTx(this.state);
        } else if (this.swapExecutor) {
          this.logger.warn({ txHash: this.state.pendingTx.txHash }, "Recovering pending transaction");
          try {
            const recovered = await this.swapExecutor.recoverPendingTx(this.state, this.config.stateFilePath);
            if (recovered) {
              this.logger.info("Pending tx confirmed — state updated");
              this.state = recovered;
            } else {
              this.logger.warn("Pending tx not found or failed — clearing");
              this.state = clearPendingTx(this.state);
            }
          } catch (err) {
            this.logger.error({ err }, "Failed to recover pending tx — clearing");
            this.state = clearPendingTx(this.state);
          }
        } else {
          this.logger.warn({ txHash: this.state.pendingTx.txHash }, "Pending tx found — no swapExecutor, clearing");
          this.state = clearPendingTx(this.state);
        }
      }

      const prevHex = (existingGrid as unknown as Record<string, unknown>).previousPrice as string | undefined;
      if (prevHex) {
        this.previousPrice = BigInt(prevHex);
        this.logger.info({ previousPrice: formatPrice(this.previousPrice) }, "Recovered previous price");
      }
    } else {
      this.centerPrice = initialPrice;
      this.levels = createGridLevels({
        pair: this.config.pair,
        centerPrice: initialPrice,
        gridLevels: this.config.gridLevels,
        rangePct: this.config.rangePct,
        tiers: this.currentTiers,
        capitalUsd: this.config.capitalUsd,
        reservePct: this.config.reservePct,
      });
      this.persistGridState();
      this.logger.info({ centerPrice: formatPrice(initialPrice), levels: this.levels.length }, "Created new grid");
    }

    this.logGridLevels();

    this.running = true;
    this.logger.info({ mode: this.config.mode, pair: this.config.pair, intervalMs: this.config.tickIntervalMs }, "Bot started");
    if (this.notifier) {
      await this.notifier.notify({
        type: "info",
        pair: this.config.pair,
        message: `Bot started in ${this.config.mode} mode, capital: $${this.config.capitalUsd.toFixed(2)}`,
      });
    }
    await this.tickLoop();
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.timeoutId !== null) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
    this.persistGridState();
    await saveState(this.config.stateFilePath, this.state);
    console.log("Graceful shutdown — state saved");
  }

  private async tickLoop(): Promise<void> {
    if (!this.running) return;

    const tickStart = Date.now();
    try {
      await this.tick();
    } catch (err) {
      this.logger.error({ err }, "Tick error");
    }

    const elapsed = Date.now() - tickStart;
    const remaining = Math.max(0, this.config.tickIntervalMs - elapsed);

    if (this.running) {
      this.timeoutId = setTimeout(() => this.tickLoop(), remaining);
    }
  }

  private async tick(): Promise<void> {
    // 1. Emergency stop check
    this.emergencyStop.checkFileTrigger(process.cwd());
    this.emergencyStop.checkEnvTrigger();
    if (this.emergencyStop.isActive()) {
      this.logger.warn({ reason: this.emergencyStop.getReason() }, "Emergency stop active — pausing ticks");
      if (this.notifier) {
        await this.notifier.notify({
          type: "emergency",
          pair: this.config.pair,
          message: this.emergencyStop.getReason() ?? "unknown",
        });
      }
      return;
    }

    // 2. Get current price
    const currentPrice = await this.priceFeed.getPrice(this.config.pair);

    // 2b. Track volatility and adjust grid tiers
    this.volTracker.addPrice(currentPrice);
    this.tickCount++;

    await this.checkWalletBalances();

    if (this.tickCount >= 360 && this.tickCount - this.lastVolAdjustTick >= 100) {
      const vol = this.volTracker.getVolatility();
      if (vol !== null) {
        const newTiers = this.volToTiers(vol);
        if (this.tiersChangedSignificantly(this.currentTiers, newTiers)) {
          this.currentTiers = newTiers;
          this.logger.info(
            { vol: vol.toFixed(3), tiers: newTiers.map(t => `${t.rangePct}%`) },
            "Volatility changed — adjusting grid tiers",
          );
          if (this.centerPrice) {
            this.levels = createGridLevels({
              pair: this.config.pair,
              centerPrice: this.centerPrice,
              gridLevels: this.config.gridLevels,
              tiers: this.currentTiers,
              capitalUsd: this.config.capitalUsd,
              reservePct: this.config.reservePct,
            });
            this.logGridLevels();
          }
        }
      }
      this.lastVolAdjustTick = this.tickCount;
    }

    // 3. First tick — initialize previous price
    if (this.previousPrice === null) {
      this.previousPrice = currentPrice;
      this.logger.info({ price: formatPrice(currentPrice) }, "Previous price initialized");
      this.persistPreviousPrice();
      return;
    }

    // 4. Check grid triggers
    const triggered = checkTriggers(this.levels, this.previousPrice, currentPrice);

    // 5. Process each triggered level
    for (const level of triggered) {
      await this.processTrigger(level, currentPrice);
    }

    // 6. Recenter check
    if (this.centerPrice && shouldRecenter(this.levels, currentPrice, this.centerPrice)) {
      this.logger.warn(
        { currentPrice: formatPrice(currentPrice), centerPrice: formatPrice(this.centerPrice) },
        "Price outside grid range — recentering grid",
      );
      this.centerPrice = currentPrice;
      this.levels = createGridLevels({
        pair: this.config.pair,
        centerPrice: currentPrice,
        gridLevels: this.config.gridLevels,
        rangePct: this.config.rangePct,
        tiers: this.currentTiers,
        capitalUsd: this.config.capitalUsd,
        reservePct: this.config.reservePct,
      });
      this.persistGridState();
      this.logGridLevels();
      if (this.notifier) {
        await this.notifier.notify({
          type: "recenter",
          pair: this.config.pair,
          message: `$${formatPrice(currentPrice)}`,
        });
      }
    }

    // 7. Update previous price
    this.previousPrice = currentPrice;
    this.persistPreviousPrice();

    if (triggered.length > 0) {
      const stats = getGridStats(this.levels, this.centerPrice ?? 0n);
      this.logTickStatus(currentPrice, stats);
    }
  }

  private async processTrigger(level: GridLevel, currentPrice: bigint): Promise<void> {
    const side = level.side as "buy" | "sell";
    const amountUsd = Number(level.amount) / 1_000_000;

    if (this.config.mode === "sim") {
      const simCosts = this.config.simCosts ?? DEFAULT_SIM_COSTS;
      const dexFeeBps = BigInt(simCosts.dexFeeBps);
      const gasCostMicro = BigInt(Math.round(simCosts.gasUsd * 1_000_000));
      const slippageBps = BigInt(simCosts.slippageBps);
      const BPS = 10_000n;

      this.logger.info(
        `WOULD ${side === "buy" ? "BUY" : "SELL"} ${this.config.pair} at $${formatPrice(level.price)}, amount: $${amountUsd.toFixed(2)} USDC`,
      );

      if (this.simPortfolio) {
        if (side === "buy") {
          const feeAmount = (level.amount * dexFeeBps) / BPS;
          const afterFee = level.amount - feeAmount;
          const grossToken = (afterFee * 10n ** BigInt(this.simPortfolio.tokenDecimals)) / level.price;
          const slippageCost = (afterFee * slippageBps) / BPS;
          const tokenReceived = (grossToken * (BPS - slippageBps)) / BPS;
          const totalDeducted = level.amount + gasCostMicro;

          this.simPortfolio.usdHolding -= totalDeducted;
          this.simPortfolio.tokenHolding += tokenReceived;
          this.simPortfolio.totalFeesUsd += feeAmount;
          this.simPortfolio.totalGasUsd += gasCostMicro;
          this.simPortfolio.totalSlippageUsd += slippageCost;
          this.simPortfolio.tradeCount++;

          const feeUsd = Number(feeAmount) / 1e6;
          const slippageUsd = Number(slippageCost) / 1e6;
          const gasUsd = Number(gasCostMicro) / 1e6;
          const netCost = Number(totalDeducted) / 1e6;
          this.logger.info(
            `  ├ DEX fee:  -$${feeUsd.toFixed(4)} (${simCosts.dexFeeBps / 100}%)`,
          );
          this.logger.info(
            `  ├ Gas:      -$${gasUsd.toFixed(4)}`,
          );
          this.logger.info(
            `  ├ Slippage: -$${slippageUsd.toFixed(4)} (${simCosts.slippageBps / 100}%)`,
          );
          this.logger.info(
            `  └ Net cost:  $${netCost.toFixed(4)} → receive ${formatToken(tokenReceived, this.simPortfolio.tokenDecimals)} ${this.simPortfolio.tokenSymbol}`,
          );
        } else {
          const tokenDelta = (level.amount * 10n ** BigInt(this.simPortfolio.tokenDecimals)) / level.price;
          const grossUsdc = level.amount;
          const feeAmount = (grossUsdc * dexFeeBps) / BPS;
          const slippageCost = (grossUsdc * slippageBps) / BPS;
          const usdcReceived = grossUsdc - feeAmount - slippageCost;

          this.simPortfolio.tokenHolding -= tokenDelta;
          this.simPortfolio.usdHolding += usdcReceived - gasCostMicro;
          this.simPortfolio.totalFeesUsd += feeAmount;
          this.simPortfolio.totalGasUsd += gasCostMicro;
          this.simPortfolio.totalSlippageUsd += slippageCost;
          this.simPortfolio.tradeCount++;

          const feeUsd = Number(feeAmount) / 1e6;
          const slippageUsd = Number(slippageCost) / 1e6;
          const gasUsd = Number(gasCostMicro) / 1e6;
          const netReceived = Number(usdcReceived - gasCostMicro) / 1e6;
          this.logger.info(
            `  ├ DEX fee:  -$${feeUsd.toFixed(4)} (${simCosts.dexFeeBps / 100}%)`,
          );
          this.logger.info(
            `  ├ Gas:      -$${gasUsd.toFixed(4)}`,
          );
          this.logger.info(
            `  ├ Slippage: -$${slippageUsd.toFixed(4)} (${simCosts.slippageBps / 100}%)`,
          );
          this.logger.info(
            `  └ Net recv:  $${netReceived.toFixed(4)} USDC (spent ${formatToken(tokenDelta, this.simPortfolio.tokenDecimals)} ${this.simPortfolio.tokenSymbol})`,
          );
        }
      }
    } else {
      if (!this.swapExecutor) {
        this.logger.error("SwapExecutor not available in live mode");
        return;
      }

      let order: SwapOrder;
      const pairTokens = this.getPairTokens();
      if (side === "buy") {
        order = {
          pair: this.config.pair,
          side: "buy",
          tokenIn: TOKENS.USDC.address,
          tokenOut: pairTokens.base,
          amountIn: level.amount,
          slippageBps: this.config.slippageBps,
        };
      } else {
        const { base: baseDecimals } = this.getTokenDecimals();
        const tokenAmount = (level.amount * 10n ** BigInt(baseDecimals)) / level.price;
        order = {
          pair: this.config.pair,
          side: "sell",
          tokenIn: pairTokens.base,
          tokenOut: TOKENS.USDC.address,
          amountIn: tokenAmount,
          slippageBps: this.config.slippageBps,
        };
      }

      this.logger.info(
        { side, pair: this.config.pair, price: formatPrice(level.price), amountUsd: amountUsd.toFixed(2) },
        "Executing live swap",
      );

      const context: TradeContext = {
        currentPrice,
        centerPrice: this.centerPrice ?? currentPrice,
        initialValueUsd: this.config.capitalUsd,
        currentValueUsd: this.config.capitalUsd,
        triggeredLevels: 1,
        ethBalance: 0n,
        spentTodayUsd: 0,
        estimatedGasCostUsd: 0.001,
      };

      const result = await this.swapExecutor.executeSwap(order, context, this.riskConfig);

      if (result.success) {
        this.logger.info(
          { txHash: result.txHash, amountOut: result.amountOut.toString(), gasUsed: result.gasUsed.toString() },
          "Swap executed successfully",
        );
      } else {
        this.logger.error({ error: result.error }, "Swap execution failed");
        return;
      }
    }

    const filledIdx = this.levels.findIndex((l) => l.index === level.index);
    if (filledIdx >= 0) {
      this.levels[filledIdx] = { ...this.levels[filledIdx], status: "filled", fillPrice: currentPrice };
    }

    // Ping-pong: buy filled → sell above, sell filled → buy below
    const maxLevels = this.config.maxLevels ?? 20;
    const filledPrice = level.price;
    const minSpreadBps = this.getDynamicSpreadBps();
    let newPrice: bigint;
    let newSide: "buy" | "sell";

    if (side === "buy") {
      newPrice = (filledPrice * (10000n + minSpreadBps)) / 10000n;
      newSide = "sell";
    } else {
      newPrice = (filledPrice * (10000n - minSpreadBps)) / 10000n;
      newSide = "buy";
    }

    // Portfolio bias overrides ping-pong side when unbalanced
    const bias = this.getPortfolioBias();
    if (bias === "buy" && newSide === "sell") {
      newSide = "buy";
      newPrice = (filledPrice * (10000n - minSpreadBps)) / 10000n;
    } else if (bias === "sell" && newSide === "buy") {
      newSide = "sell";
      newPrice = (filledPrice * (10000n + minSpreadBps)) / 10000n;
    }

    const newAmount = this.calculatePerLevelAmount();

    if (newAmount > 0n) {
      // Evict farthest filled level when at capacity
      if (this.levels.length >= maxLevels) {
        const filledLevels = this.levels.filter((l) => l.status === "filled");
        if (filledLevels.length > 0) {
          filledLevels.sort((a, b) => {
            const distA = a.price > currentPrice ? a.price - currentPrice : currentPrice - a.price;
            const distB = b.price > currentPrice ? b.price - currentPrice : currentPrice - b.price;
            return distB > distA ? 1 : distB < distA ? -1 : 0;
          });
          const toRemove = filledLevels[filledLevels.length - 1];
          this.levels = this.levels.filter((l) => l !== toRemove);
        }
      }

      if (this.levels.length < maxLevels) {
        const newLevel: GridLevel = {
          index: 0,
          price: newPrice,
          side: newSide,
          status: "pending",
          amount: newAmount,
        };
        this.levels.push(newLevel);
        this.levels.sort((a, b) => (a.price < b.price ? -1 : a.price > b.price ? 1 : 0));
        this.levels = this.levels.map((l, i) => ({ ...l, index: i }));

        this.logger.info(
          { side: newSide, price: formatPrice(newPrice), amount: `$${(Number(newAmount) / 1e6).toFixed(2)}` },
          `Ping-pong: new ${newSide} level created`,
        );
      }
    }

    if (this.config.mode === "sim" && this.simPortfolio) {
      const usdVal = Number(this.simPortfolio.usdHolding) / 1e6;
      const tokVal =
        Number((this.simPortfolio.tokenHolding * currentPrice) / 10n ** BigInt(this.simPortfolio.tokenDecimals)) /
        1e6;
      this.logger.info(
        `  └ Portfolio: $${usdVal.toFixed(2)} USDC + $${tokVal.toFixed(2)} ${this.simPortfolio.tokenSymbol} = TOTAL $${(usdVal + tokVal).toFixed(2)} | bias: ${this.getPortfolioBias()}`,
      );
    }

    this.state = recordFill(this.state, this.config.pair, level.index, {
      side,
      price: bigintToHex(level.price),
      amount: bigintToHex(level.amount),
    });

    if (this.notifier) {
      let portfolioMsg = "";
      if (this.config.mode === "sim" && this.simPortfolio) {
        const usdVal = Number(this.simPortfolio.usdHolding) / 1e6;
        const tokVal = Number(
          (this.simPortfolio.tokenHolding * currentPrice) / 10n ** BigInt(this.simPortfolio.tokenDecimals),
        ) / 1e6;
        const total = usdVal + tokVal;
        const pnl = total - Number(this.simPortfolio.startingUsd) / 1e6;
        portfolioMsg = `\n💰 Total: $${total.toFixed(2)} (USDC $${usdVal.toFixed(2)} + ${this.simPortfolio.tokenSymbol} $${tokVal.toFixed(2)})\n📈 P&L: $${pnl >= 0 ? "+" : ""}${pnl.toFixed(4)}`;
      }
      await this.notifier.notify({
        type: side === "buy" ? "grid_trigger" : "grid_trigger",
        pair: this.config.pair,
        message: `${side.toUpperCase()} $${amountUsd.toFixed(2)} at $${formatPrice(level.price)}${portfolioMsg}`,
      });
    }

    this.persistGridState();
    await saveState(this.config.stateFilePath, this.state);
  }

  private getTokenDecimals(): { base: number; quote: number } {
    if (this.config.pair === "ETH/USDC") return { base: TOKENS.WETH.decimals, quote: TOKENS.USDC.decimals };
    if (this.config.pair === "CBBTC/USDC") return { base: TOKENS.CBBTC.decimals, quote: TOKENS.USDC.decimals };
    throw new Error(`Unknown pair: ${this.config.pair}`);
  }

  private getPairTokens(): { base: Address; quote: Address } {
    if (this.config.pair === "ETH/USDC") return { base: TOKENS.WETH.address, quote: TOKENS.USDC.address };
    if (this.config.pair === "CBBTC/USDC") return { base: TOKENS.CBBTC.address, quote: TOKENS.USDC.address };
    throw new Error(`Unknown pair: ${this.config.pair}`);
  }

  private calculatePerLevelAmount(): bigint {
    const totalValueUsd = this.getPortfolioValueUsd();
    if (totalValueUsd === 0n) return 0n;

    const activeLevels = this.levels.filter((l) => l.status === "pending").length;
    if (activeLevels === 0) return 0n;

    const investable = (totalValueUsd * 90n) / 100n;
    const slotsForCalc = Math.max(activeLevels + 2, 4);
    return investable / BigInt(slotsForCalc);
  }

  private getPortfolioValueUsd(): bigint {
    if (this.config.mode === "sim" && this.simPortfolio) {
      const tokenValue =
        (this.simPortfolio.tokenHolding * (this.previousPrice ?? 0n)) /
        10n ** BigInt(this.simPortfolio.tokenDecimals);
      return this.simPortfolio.usdHolding + tokenValue;
    }
    if (this.walletBalances.usdc > 0n || this.walletBalances.token > 0n) {
      const { base: baseDecimals } = this.getTokenDecimals();
      const tokenValue =
        (this.walletBalances.token * (this.previousPrice ?? 0n)) / 10n ** BigInt(baseDecimals);
      return this.walletBalances.usdc + tokenValue;
    }
    return BigInt(this.config.capitalUsd) * 1_000_000n;
  }

  private getPortfolioBias(): "buy" | "sell" | "neutral" {
    if (this.config.mode === "sim" && this.simPortfolio) {
      const usdValue = Number(this.simPortfolio.usdHolding);
      const tokenValue = Number(
        (this.simPortfolio.tokenHolding * (this.previousPrice ?? 0n)) /
          10n ** BigInt(this.simPortfolio.tokenDecimals),
      );
      const total = usdValue + tokenValue;
      if (total === 0) return "neutral";

      const usdPct = usdValue / total;
      if (usdPct > 0.65) return "buy";
      if (usdPct < 0.35) return "sell";
      return "neutral";
    }
    if (this.walletBalances.usdc > 0n || this.walletBalances.token > 0n) {
      const usdValue = Number(this.walletBalances.usdc);
      const { base: baseDecimals } = this.getTokenDecimals();
      const tokenValue = Number(
        (this.walletBalances.token * (this.previousPrice ?? 0n)) / 10n ** BigInt(baseDecimals),
      );
      const total = usdValue + tokenValue;
      if (total === 0) return "neutral";

      const usdPct = usdValue / total;
      if (usdPct > 0.65) return "buy";
      if (usdPct < 0.35) return "sell";
    }
    return "neutral";
  }

  private async checkWalletBalances(): Promise<void> {
    if (this.config.mode !== "live" || !this.swapExecutor) return;

    if (this.tickCount - this.lastBalanceCheck < 60) return;
    this.lastBalanceCheck = this.tickCount;

    try {
      const { publicClient: pubClient, walletClient: wClient } = await import("./clients/rpc.js");
      if (!wClient) return;

      const address = wClient.account.address;

      const ethBalance = await pubClient.getBalance({ address });
      const usdcBalance = (await pubClient.readContract({
        address: TOKENS.USDC.address,
        abi: erc20Abi,
        functionName: "balanceOf",
        args: [address],
      })) as bigint;

      const pairTokens = this.getPairTokens();
      const tokenBalance = (await pubClient.readContract({
        address: pairTokens.base,
        abi: erc20Abi,
        functionName: "balanceOf",
        args: [address],
      })) as bigint;

      const prevUsdc = this.walletBalances.usdc;
      const prevToken = this.walletBalances.token;

      this.walletBalances = { usdc: usdcBalance, token: tokenBalance, eth: ethBalance };

      if (prevUsdc > 0n || prevToken > 0n) {
        const usdcDelta = usdcBalance > prevUsdc ? usdcBalance - prevUsdc : prevUsdc - usdcBalance;
        const tokenDelta =
          tokenBalance > prevToken ? tokenBalance - prevToken : prevToken - tokenBalance;

        if (usdcDelta > prevUsdc / 10n || tokenDelta > prevToken / 10n) {
          this.logger.info(
            {
              usdcBefore: this.formatUsdc(prevUsdc),
              usdcAfter: this.formatUsdc(usdcBalance),
              tokenDelta: tokenDelta > 0n ? "significant" : "none",
            },
            "External balance change detected (manual deposit/withdrawal)",
          );
          await this.rebalanceGrid();
        }
      }
    } catch (err) {
      this.logger.warn({ error: (err as Error).message }, "Balance check failed");
    }
  }

  private formatUsdc(val: bigint): string {
    return `$${(Number(val) / 1e6).toFixed(2)}`;
  }

  private async rebalanceGrid(): Promise<void> {
    const bias = this.getPortfolioBias();
    if (bias === "neutral") return;

    const minSpreadBps = this.getDynamicSpreadBps();

    this.logger.info({ bias }, "Rebalancing grid bias");

    const buyPending = this.levels.filter((l) => l.side === "buy" && l.status === "pending").length;
    const sellPending = this.levels.filter((l) => l.side === "sell" && l.status === "pending").length;

    if (bias === "buy" && sellPending > buyPending + 1) {
      const outerSell = this.levels.find((l) => l.side === "sell" && l.status === "pending");
      if (outerSell) {
        outerSell.side = "buy";
        outerSell.price = (outerSell.price * (10000n - minSpreadBps * 2n)) / 10000n;
        this.logger.info({ price: formatPrice(outerSell.price) }, "Converted sell → buy for rebalance");
      }
    } else if (bias === "sell" && buyPending > sellPending + 1) {
      const outerBuy = [...this.levels]
        .reverse()
        .find((l) => l.side === "buy" && l.status === "pending");
      if (outerBuy) {
        outerBuy.side = "sell";
        outerBuy.price = (outerBuy.price * (10000n + minSpreadBps * 2n)) / 10000n;
        this.logger.info({ price: formatPrice(outerBuy.price) }, "Converted buy → sell for rebalance");
      }
    }

    this.levels.sort((a, b) => (a.price < b.price ? -1 : a.price > b.price ? 1 : 0));
    this.levels = this.levels.map((l, i) => ({ ...l, index: i }));
  }

  private getDynamicSpreadBps(): bigint {
    const amount = this.calculatePerLevelAmount();
    const amountUsd = Number(amount) / 1e6;
    if (amountUsd <= 0) return 200n;
    const gasRatioPct = (0.01 / amountUsd) * 100;
    const breakevenSpreadPct = (gasRatioPct + 0.02) * 2;
    const breakevenBps = Math.ceil(breakevenSpreadPct * 100);
    const safeBps = Math.max(breakevenBps * 2, this.config.pingPongSpreadBps ?? 60);
    return BigInt(Math.max(safeBps, 40));
  }

  private volToTiers(hourlyVolPct: number): GridTier[] {
    const dynamicBps = Number(this.getDynamicSpreadBps());
    const minBase = Math.max(dynamicBps / 200, 0.3);
    const base = Math.max(minBase, Math.min(1.2, hourlyVolPct * 2));

    return [
      { rangePct: parseFloat(base.toFixed(2)), count: 1 },
      { rangePct: parseFloat((base * 3.3).toFixed(2)), count: 1 },
      { rangePct: parseFloat((base * 6).toFixed(2)), count: 1 },
      { rangePct: parseFloat((base * 10).toFixed(2)), count: 1 },
      { rangePct: parseFloat((base * 16).toFixed(2)), count: 1 },
    ];
  }

  private tiersChangedSignificantly(oldTiers: GridTier[], newTiers: GridTier[]): boolean {
    if (oldTiers.length !== newTiers.length) return true;
    for (let i = 0; i < oldTiers.length; i++) {
      const pctChange = Math.abs(oldTiers[i].rangePct - newTiers[i].rangePct) / oldTiers[i].rangePct;
      if (pctChange > 0.20) return true;
    }
    return false;
  }

  private persistGridState(): void {
    if (!this.centerPrice) return;

    const existingGrid = this.state.grids[this.config.pair];
    const gridState: Record<string, unknown> = {
      pair: this.config.pair,
      centerPrice: bigintToHex(this.centerPrice),
      levels: this.levels.map(gridLevelToState),
      config: {
        gridLevels: this.config.gridLevels,
        rangePct: this.config.rangePct ?? 0,
        capitalUsd: this.config.capitalUsd,
      },
      fillHistory: existingGrid?.fillHistory ?? [],
    };
    if (this.previousPrice) {
      gridState.previousPrice = bigintToHex(this.previousPrice);
    }

    this.state = {
      ...this.state,
      lastUpdated: Date.now(),
      grids: {
        ...this.state.grids,
        [this.config.pair]: gridState as unknown as import("./types.js").GridState,
      },
    };
  }

  private persistPreviousPrice(): void {
    this.persistGridState();
  }

  private logGridLevels(): void {
    this.logger.info("=== Grid Levels ===");
    for (const level of this.levels) {
      const tag =
        level.side === "none"
          ? "----"
          : level.status === "filled"
            ? "DONE"
            : level.side.toUpperCase();
      this.logger.info(`  [${tag}] $${formatPrice(level.price)} | amount: $${(Number(level.amount) / 1_000_000).toFixed(2)}`);
    }
    this.logger.info("===================");
  }

  private logTickStatus(currentPrice: bigint, stats: GridStats): void {
    const totalLevels = stats.filledCount + stats.pendingCount;
    const vol = this.volTracker.getVolatility();
    const volStr = vol !== null ? ` | vol: ${vol.toFixed(2)}%` : "";
    let simInfo = "";
    if (this.config.mode === "sim" && this.simPortfolio) {
      const tokenValueUsd = (this.simPortfolio.tokenHolding * currentPrice) / 10n ** BigInt(this.simPortfolio.tokenDecimals);
      const totalValue = this.simPortfolio.usdHolding + tokenValueUsd;
      const pnl = totalValue - this.simPortfolio.startingUsd;
      const feesUsd = Number(this.simPortfolio.totalFeesUsd) / 1e6;
      const gasUsd = Number(this.simPortfolio.totalGasUsd) / 1e6;
      const slipUsd = Number(this.simPortfolio.totalSlippageUsd) / 1e6;
      simInfo = ` | sim: $${(Number(this.simPortfolio.usdHolding) / 1e6).toFixed(2)} USDC + ${formatToken(this.simPortfolio.tokenHolding, this.simPortfolio.tokenDecimals)} ${this.simPortfolio.tokenSymbol} | P&L: $${(Number(pnl) / 1e6).toFixed(4)} (fees: $${feesUsd.toFixed(4)}, gas: $${gasUsd.toFixed(4)}, slippage: $${slipUsd.toFixed(4)}, trades: ${this.simPortfolio.tradeCount})`;
    }
    this.logger.info(`[${this.config.pair}] Price: $${formatPrice(currentPrice)} | Filled: ${stats.filledCount}/${totalLevels}${volStr}${simInfo}`);
  }
}
