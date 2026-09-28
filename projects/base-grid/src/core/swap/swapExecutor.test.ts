import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Address, PublicClient, WalletClient } from "viem";
import type { SwapResult, BotState } from "../../types.js";
import type { RiskConfig, RiskCheckResult, TradeContext } from "../../risk/riskManager.js";
import { SwapExecutor } from "./swapExecutor.js";
import type { SwapOrder, SwapExecutionResult, StateUpdate } from "./swapExecutor.js";

const WETH = "0x4200000000000000000000000000000000000006" as Address;
const USDC = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913" as Address;
const ROUTER = "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43" as Address;

function makeOrder(overrides: Partial<SwapOrder> = {}): SwapOrder {
  return {
    pair: "ETH/USDC",
    tokenIn: WETH,
    tokenOut: USDC,
    amountIn: 1000000000000000000n,
    slippageBps: 50,
    side: "sell",
    ...overrides,
  };
}

function makeContext(overrides: Partial<TradeContext> = {}): TradeContext {
  return {
    currentPrice: 2500000000n,
    centerPrice: 2500000000n,
    initialValueUsd: 10000,
    currentValueUsd: 10000,
    triggeredLevels: 0,
    ethBalance: 1000000000000000000n,
    spentTodayUsd: 0,
    estimatedGasCostUsd: 0.01,
    ...overrides,
  };
}

function makeRiskConfig(overrides: Partial<RiskConfig> = {}): RiskConfig {
  return {
    stopLossPct: 8,
    maxDrawdownPct: 25,
    circuitBreakerThreshold: 2,
    circuitBreakerPauseMs: 60_000,
    maxGasCostUsd: 0.1,
    maxDailySpendUsd: 10,
    minEthReserve: 5000000000000000n,
    ...overrides,
  };
}

function createMocks() {
  const stateUpdates: StateUpdate[] = [];
  const onStateUpdate = vi.fn((update: StateUpdate) => {
    stateUpdates.push(update);
  });

  const getQuoteFn = vi.fn().mockResolvedValue(2500000000n);
  const approveFn = vi.fn().mockResolvedValue("0xapprovehash");
  const executeSwapFn = vi.fn().mockResolvedValue({
    success: true,
    amountIn: 1000000000000000000n,
    amountOut: 2500000000n,
    txHash: "0xswaphash",
  } satisfies SwapResult);
  const canTradeFn = vi.fn().mockReturnValue({ allowed: true } as RiskCheckResult);

  const getTransactionReceipt = vi.fn().mockResolvedValue({
    status: "success",
    gasUsed: 21000n,
  });

  const publicClient = {
    getTransactionReceipt,
  } as unknown as PublicClient;

  const walletClient = {
    account: { address: "0x1234567890123456789012345678901234567890" as Address },
    chain: { id: 8453 },
  } as unknown as WalletClient;

  const executor = new SwapExecutor(
    publicClient,
    walletClient,
    getQuoteFn,
    approveFn,
    executeSwapFn,
    canTradeFn,
    onStateUpdate,
  );

  return {
    executor,
    stateUpdates,
    onStateUpdate,
    getQuoteFn,
    approveFn,
    executeSwapFn,
    canTradeFn,
    getTransactionReceipt,
    publicClient,
    walletClient,
  };
}

describe("SwapExecutor", () => {
  describe("executeSwap — full 11-step flow", () => {
    it("executes the full flow and returns success", async () => {
      const {
        executor, getQuoteFn, approveFn, executeSwapFn, canTradeFn,
        onStateUpdate, getTransactionReceipt, stateUpdates,
      } = createMocks();
      const order = makeOrder();
      const context = makeContext();
      const config = makeRiskConfig();

      const result = await executor.executeSwap(order, context, config);

      expect(result.success).toBe(true);
      expect(result.amountOut).toBe(2500000000n);
      expect(result.txHash).toBe("0xswaphash");
      expect(result.gasUsed).toBe(21000n);
      expect(result.order).toBe(order);

      expect(canTradeFn).toHaveBeenCalledTimes(1);
      expect(getQuoteFn).toHaveBeenCalledTimes(1);
      expect(approveFn).toHaveBeenCalledTimes(1);
      expect(executeSwapFn).toHaveBeenCalledTimes(1);
      expect(getTransactionReceipt).toHaveBeenCalledWith({ hash: "0xswaphash" });

      expect(onStateUpdate).toHaveBeenCalledTimes(2);
      expect(stateUpdates[0]).toEqual({ type: "swap_pending", pair: "ETH/USDC" });
      expect(stateUpdates[1]).toEqual({ type: "swap_filled", pair: "ETH/USDC", txHash: "0xswaphash" });
    });

    it("calls dependencies in correct order: risk → quote → approve → execute", async () => {
      const { executor, canTradeFn, getQuoteFn, approveFn, executeSwapFn } = createMocks();

      await executor.executeSwap(makeOrder(), makeContext(), makeRiskConfig());

      const riskOrder = vi.mocked(canTradeFn).mock.invocationCallOrder[0];
      const quoteOrder = vi.mocked(getQuoteFn).mock.invocationCallOrder[0];
      const approveOrder = vi.mocked(approveFn).mock.invocationCallOrder[0];
      const swapOrder = vi.mocked(executeSwapFn).mock.invocationCallOrder[0];

      expect(riskOrder).toBeLessThan(quoteOrder);
      expect(quoteOrder).toBeLessThan(approveOrder);
      expect(approveOrder).toBeLessThan(swapOrder);
    });
  });

  describe("serial lock", () => {
    it("rejects second swap on same pair while first is pending", async () => {
      const { executor, executeSwapFn } = createMocks();

      const order = makeOrder();
      const context = makeContext();

      executeSwapFn.mockReturnValueOnce(new Promise<SwapResult>(() => {}));

      executor.executeSwap(order, context, makeRiskConfig());

      // Let microtasks drain so first call progresses past step 7 (lock acquired)
      await new Promise((r) => setTimeout(r, 10));

      expect(executor.isPairLocked("ETH/USDC")).toBe(true);

      const secondResult = await executor.executeSwap(order, context, makeRiskConfig());
      expect(secondResult.success).toBe(false);
      expect(secondResult.error).toContain("already pending");
    });

    it("allows swaps on different pairs concurrently", async () => {
      const { executor, executeSwapFn } = createMocks();
      executeSwapFn.mockResolvedValue({
        success: true, amountIn: 1000n, amountOut: 2000n, txHash: "0xhash1",
      });

      const order1 = makeOrder({ pair: "ETH/USDC" });
      const order2 = makeOrder({ pair: "CBBTC/USDC", tokenIn: "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf" as Address });
      const context = makeContext();

      const result1 = await executor.executeSwap(order1, context, makeRiskConfig());
      const result2 = await executor.executeSwap(order2, context, makeRiskConfig());

      expect(result1.success).toBe(true);
      expect(result2.success).toBe(true);
    });

    it("unlocks pair after successful swap", async () => {
      const { executor } = createMocks();
      expect(executor.isPairLocked("ETH/USDC")).toBe(false);

      await executor.executeSwap(makeOrder(), makeContext(), makeRiskConfig());

      expect(executor.isPairLocked("ETH/USDC")).toBe(false);
    });

    it("unlocks pair after failed swap", async () => {
      const { executor, executeSwapFn } = createMocks();
      executeSwapFn.mockResolvedValueOnce({
        success: false, amountIn: 1000n, amountOut: 0n, error: "reverted",
      });

      await executor.executeSwap(makeOrder(), makeContext(), makeRiskConfig());

      expect(executor.isPairLocked("ETH/USDC")).toBe(false);
    });
  });

  describe("risk check failure", () => {
    it("returns failure and does not execute swap when risk check fails", async () => {
      const { executor, canTradeFn, getQuoteFn, approveFn, executeSwapFn } = createMocks();
      canTradeFn.mockReturnValueOnce({
        allowed: false,
        reason: "Stop loss triggered",
        action: "STOP_LOSS",
      } as RiskCheckResult);

      const result = await executor.executeSwap(makeOrder(), makeContext(), makeRiskConfig());

      expect(result.success).toBe(false);
      expect(result.error).toContain("Risk check failed");
      expect(result.error).toContain("Stop loss triggered");

      expect(getQuoteFn).not.toHaveBeenCalled();
      expect(approveFn).not.toHaveBeenCalled();
      expect(executeSwapFn).not.toHaveBeenCalled();
    });
  });

  describe("swap revert", () => {
    it("clears lock and returns failure without updating grid state", async () => {
      const { executor, executeSwapFn, onStateUpdate, stateUpdates } = createMocks();
      executeSwapFn.mockResolvedValueOnce({
        success: false,
        amountIn: 1000000000000000000n,
        amountOut: 0n,
        error: "Transaction reverted on-chain",
      });

      const result = await executor.executeSwap(makeOrder(), makeContext(), makeRiskConfig());

      expect(result.success).toBe(false);
      expect(result.error).toContain("reverted");
      expect(executor.isPairLocked("ETH/USDC")).toBe(false);

      expect(stateUpdates).toEqual([
        { type: "swap_pending", pair: "ETH/USDC" },
        { type: "swap_failed", pair: "ETH/USDC", error: "Transaction reverted on-chain", txHash: undefined },
      ]);
    });
  });

  describe("network error retry", () => {
    it("retries 2 times then fails on network errors", async () => {
      const { executor, executeSwapFn, onStateUpdate } = createMocks();
      executeSwapFn.mockRejectedValue(new Error("Network timeout"));

      const result = await executor.executeSwap(makeOrder(), makeContext(), makeRiskConfig());

      expect(result.success).toBe(false);
      expect(result.error).toContain("Network timeout");
      expect(executeSwapFn).toHaveBeenCalledTimes(3); // 1 initial + 2 retries
      expect(executor.isPairLocked("ETH/USDC")).toBe(false);

      const failedUpdate = onStateUpdate.mock.calls.find(
        (c: [StateUpdate]) => c[0].type === "swap_failed",
      );
      expect(failedUpdate).toBeDefined();
    }, 20_000);

    it("succeeds on retry after first network error", async () => {
      const { executor, executeSwapFn } = createMocks();
      executeSwapFn
        .mockRejectedValueOnce(new Error("Network error"))
        .mockResolvedValueOnce({
          success: true,
          amountIn: 1000000000000000000n,
          amountOut: 2500000000n,
          txHash: "0xretriedhash",
        });

      const result = await executor.executeSwap(makeOrder(), makeContext(), makeRiskConfig());

      expect(result.success).toBe(true);
      expect(result.txHash).toBe("0xretriedhash");
      expect(executeSwapFn).toHaveBeenCalledTimes(2);
    }, 15_000);
  });

  describe("gas cost check", () => {
    it("rejects swap when gas exceeds 2% of trade value", async () => {
      const { executor, executeSwapFn } = createMocks();
      const context = makeContext({ estimatedGasCostUsd: 100 });

      const result = await executor.executeSwap(makeOrder(), context, makeRiskConfig());

      expect(result.success).toBe(false);
      expect(result.error).toContain("Gas cost");
      expect(executeSwapFn).not.toHaveBeenCalled();
    });
  });

  describe("recoverPendingTx", () => {
    it("returns null when no pending tx", async () => {
      const { executor } = createMocks();
      const state: BotState = {
        version: 1,
        lastBlockNumber: "0x0",
        lastUpdated: Date.now(),
        grids: {},
        pendingTx: null,
      };

      const result = await executor.recoverPendingTx(state, "/tmp/state.json");

      expect(result).toBeNull();
    });

    it("clears pendingTx and returns updated state on successful receipt", async () => {
      const { executor, onStateUpdate, getTransactionReceipt } = createMocks();
      const state: BotState = {
        version: 1,
        lastBlockNumber: "0x0",
        lastUpdated: Date.now(),
        grids: {},
        pendingTx: { txHash: "0xpending", submittedAt: Date.now() },
      };

      getTransactionReceipt.mockResolvedValueOnce({ status: "success", gasUsed: 21000n });

      const result = await executor.recoverPendingTx(state, "/tmp/state.json");

      expect(result).not.toBeNull();
      expect(result!.pendingTx).toBeNull();
      expect(onStateUpdate).toHaveBeenCalledWith({
        type: "recovered",
        pair: "",
        txHash: "0xpending",
      });
    });

    it("clears pendingTx when receipt shows failure", async () => {
      const { executor, onStateUpdate, getTransactionReceipt } = createMocks();
      const state: BotState = {
        version: 1,
        lastBlockNumber: "0x0",
        lastUpdated: Date.now(),
        grids: {},
        pendingTx: { txHash: "0xfailed", submittedAt: Date.now() },
      };

      getTransactionReceipt.mockResolvedValueOnce({ status: "reverted", gasUsed: 21000n });

      const result = await executor.recoverPendingTx(state, "/tmp/state.json");

      expect(result).not.toBeNull();
      expect(result!.pendingTx).toBeNull();
      expect(onStateUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "swap_failed",
          txHash: "0xfailed",
        }),
      );
    });

    it("clears pendingTx when receipt cannot be found", async () => {
      const { executor, onStateUpdate, getTransactionReceipt } = createMocks();
      const state: BotState = {
        version: 1,
        lastBlockNumber: "0x0",
        lastUpdated: Date.now(),
        grids: {},
        pendingTx: { txHash: "0xmissing", submittedAt: Date.now() },
      };

      getTransactionReceipt.mockRejectedValueOnce(new Error("Not found"));

      const result = await executor.recoverPendingTx(state, "/tmp/state.json");

      expect(result).not.toBeNull();
      expect(result!.pendingTx).toBeNull();
      expect(onStateUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          type: "swap_failed",
          txHash: "0xmissing",
        }),
      );
    });
  });
});
