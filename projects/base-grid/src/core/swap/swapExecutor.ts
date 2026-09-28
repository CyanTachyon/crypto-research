import type { Address, PublicClient, WalletClient } from "viem";
import type { SwapResult, BotState } from "../../types.js";
import type {
  canTrade as canTradeType,
  RiskConfig,
  RiskCheckResult,
  TradeContext,
} from "../../risk/riskManager.js";
import { getQuote, approveToken, executeSwap } from "./aerodrome.js";
import { CONTRACTS } from "../../config/chains.js";

export interface SwapOrder {
  pair: string;
  tokenIn: Address;
  tokenOut: Address;
  amountIn: bigint;
  slippageBps: number;
  side: "buy" | "sell";
}

export interface SwapExecutionResult {
  success: boolean;
  order: SwapOrder;
  amountOut: bigint;
  txHash: string;
  gasUsed: bigint;
  error?: string;
}

export interface StateUpdate {
  type: "swap_pending" | "swap_filled" | "swap_failed" | "recovered";
  pair: string;
  txHash?: string;
  amountOut?: bigint;
  error?: string;
}

const DEADLINE_MS = 120_000;
const MAX_RETRIES = 2;
const RETRY_DELAY_MS = 5_000;
const GAS_THRESHOLD_PCT = 0.02;

/** Assumes base token 18 decimals, price normalized to 6-decimal quote. */
function estimateTradeValueUsd(
  amountIn: bigint,
  currentPrice: bigint,
): number {
  const SCALE_18 = 10n ** 18n;
  const tradeValueMicroUsd = (amountIn * currentPrice) / SCALE_18;
  return Number(tradeValueMicroUsd) / 1_000_000;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export class SwapExecutor {
  private pendingSwaps: Map<string, boolean>;

  constructor(
    private publicClient: PublicClient,
    private walletClient: WalletClient,
    private getQuoteFn: typeof getQuote,
    private approveFn: typeof approveToken,
    private executeSwapFn: typeof executeSwap,
    private canTradeFn: typeof canTradeType,
    private onStateUpdate: (update: StateUpdate) => void,
  ) {
    this.pendingSwaps = new Map();
  }

  isPairLocked(pair: string): boolean {
    return this.pendingSwaps.get(pair) === true;
  }

  async executeSwap(
    order: SwapOrder,
    context: TradeContext,
    riskConfig: RiskConfig,
  ): Promise<SwapExecutionResult> {
    const fail = (
      error: string,
      amountOut = 0n,
      txHash = "",
    ): SwapExecutionResult => ({
      success: false, order, amountOut, txHash, gasUsed: 0n, error,
    });

    // 1. Serial lock
    if (this.isPairLocked(order.pair)) {
      return fail(`Swap already pending for pair: ${order.pair}`);
    }

    // 2. Risk check
    const tradeAmountUsd = estimateTradeValueUsd(order.amountIn, context.currentPrice);
    const riskResult: RiskCheckResult = this.canTradeFn(tradeAmountUsd, context, riskConfig);
    if (!riskResult.allowed) {
      return fail(`Risk check failed: ${riskResult.reason ?? "unknown"}`);
    }

    // 3. Get quote
    let quote: bigint;
    try {
      quote = await this.getQuoteFn(this.publicClient, order.tokenIn, order.tokenOut, order.amountIn);
    } catch (err) {
      return fail(`Quote error: ${err instanceof Error ? err.message : "Quote failed"}`);
    }

    // 4. Gas cost check
    if (tradeAmountUsd > 0 && context.estimatedGasCostUsd > tradeAmountUsd * GAS_THRESHOLD_PCT) {
      return fail(`Gas cost $${context.estimatedGasCostUsd.toFixed(4)} exceeds 2% of trade value $${tradeAmountUsd.toFixed(4)}`);
    }

    // 5. Allowance check & approve
    try {
      await this.approveFn(this.publicClient, this.walletClient, order.tokenIn, CONTRACTS.AERODROME_ROUTER, order.amountIn);
    } catch (err) {
      return fail(`Approval error: ${err instanceof Error ? err.message : "Approval failed"}`);
    }

    // 6. simulateContract — handled internally by executeSwapFn

    // 7. Lock pair, emit pending state
    this.pendingSwaps.set(order.pair, true);
    this.onStateUpdate({ type: "swap_pending", pair: order.pair });

    // 8-9. Execute with retries and 120s deadline
    const deadline = Date.now() + DEADLINE_MS;
    let lastError: string | undefined;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      if (Date.now() > deadline) {
        this.clearLock(order.pair, "swap_failed", "Swap deadline exceeded");
        return fail("Swap deadline exceeded (120s)");
      }

      try {
        const swapResult = await this.executeWithDeadline(order, deadline - Date.now());

        // 10. Verify
        if (swapResult.success) {
          let gasUsed = 0n;
          if (swapResult.txHash) {
            try {
              const receipt = await this.publicClient.getTransactionReceipt({ hash: swapResult.txHash as `0x${string}` });
              gasUsed = receipt.gasUsed;
            } catch { /* gasUsed fetch is non-critical */ }
          }
          // 11. Clear lock
          this.clearLock(order.pair, "swap_filled", undefined, swapResult.txHash);
          return { success: true, order, amountOut: swapResult.amountOut, txHash: swapResult.txHash ?? "", gasUsed };
        }

        // Revert is permanent — clear lock, no retry
        this.clearLock(order.pair, "swap_failed", swapResult.error, swapResult.txHash);
        return fail(swapResult.error ?? "Swap reverted", swapResult.amountOut, swapResult.txHash ?? "");
      } catch (err) {
        lastError = err instanceof Error ? err.message : "Network error";
        if (attempt < MAX_RETRIES) {
          await sleep(RETRY_DELAY_MS);
        }
      }
    }

    this.clearLock(order.pair, "swap_failed", lastError);
    return fail(lastError ?? "All retries exhausted");
  }

  /** Wraps executeSwapFn with a timeout via Promise.race. */
  private async executeWithDeadline(order: SwapOrder, timeoutMs: number): Promise<SwapResult> {
    const swapPromise = this.executeSwapFn(
      this.publicClient, this.walletClient, order.tokenIn, order.tokenOut, order.amountIn, order.slippageBps,
    );
    const timeoutPromise = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error("Swap execution timeout")), Math.max(timeoutMs, 0)),
    );
    return Promise.race([swapPromise, timeoutPromise]);
  }

  async recoverPendingTx(state: BotState, _filePath: string): Promise<BotState | null> {
    if (!state.pendingTx) return null;
    const { txHash } = state.pendingTx;

    try {
      const receipt = await this.publicClient.getTransactionReceipt({ hash: txHash as `0x${string}` });
      if (receipt.status === "success") {
        this.onStateUpdate({ type: "recovered", pair: "", txHash });
        return { ...state, pendingTx: null };
      }
      this.onStateUpdate({ type: "swap_failed", pair: "", txHash, error: "Recovered transaction reverted on-chain" });
      return { ...state, pendingTx: null };
    } catch {
      this.onStateUpdate({ type: "swap_failed", pair: "", txHash, error: `Could not retrieve receipt for pending tx: ${txHash}` });
      return { ...state, pendingTx: null };
    }
  }

  private clearLock(pair: string, type: StateUpdate["type"], error?: string, txHash?: string): void {
    this.pendingSwaps.delete(pair);
    this.onStateUpdate({ type, pair, error, txHash });
  }
}
