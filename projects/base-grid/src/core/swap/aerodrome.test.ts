import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Address, PublicClient, WalletClient } from "viem";
import { TOKENS, CONTRACTS } from "../../config/chains.js";
import {
  createRoute,
  getQuote,
  approveToken,
  executeSwap,
} from "./aerodrome.js";

const WETH = TOKENS.WETH.address;
const USDC = TOKENS.USDC.address;
const ROUTER = CONTRACTS.AERODROME_ROUTER;



function createMockPublicClient(
  overrides: Record<string, unknown> = {},
): PublicClient {
  return {
    readContract: vi.fn().mockResolvedValue(undefined),
    simulateContract: vi.fn().mockResolvedValue({}),
    waitForTransactionReceipt: vi.fn().mockResolvedValue({
      status: "success",
      transactionHash: "0xabc123",
    }),
    ...overrides,
  } as unknown as PublicClient;
}

function createMockWalletClient(
  accountAddress = "0x1234567890123456789012345678901234567890" as Address,
): WalletClient {
  return {
    account: { address: accountAddress },
    chain: { id: 8453 },
    writeContract: vi.fn().mockResolvedValue("0xdeadbeef"),
  } as unknown as WalletClient;
}

describe("createRoute", () => {
  it("creates route with default stable=false and factory=zero address", () => {
    const route = createRoute(WETH, USDC);
    expect(route).toEqual({
      from: WETH,
      to: USDC,
      stable: false,
      factory: "0x0000000000000000000000000000000000000000",
    });
  });

  it("creates route with explicit stable and factory", () => {
    const factory = "0x420DD381b31aEf6683db6B902084cB0FFECe40Da" as Address;
    const route = createRoute(WETH, USDC, true, factory);
    expect(route).toEqual({
      from: WETH,
      to: USDC,
      stable: true,
      factory,
    });
  });
});

describe("getQuote", () => {
  it("calls getAmountsOut with correct route struct and returns last amount", async () => {
    const amounts = [1000000000000000000n, 2500000000n];
    const mockPublicClient = createMockPublicClient({
      readContract: vi.fn().mockResolvedValue(amounts),
    });

    const result = await getQuote(mockPublicClient, WETH, USDC, 1000000000000000000n);

    expect(mockPublicClient.readContract).toHaveBeenCalledWith(
      expect.objectContaining({
        address: ROUTER,
        functionName: "getAmountsOut",
        args: [
          1000000000000000000n,
          [{ from: WETH, to: USDC, stable: false, factory: "0x0000000000000000000000000000000000000000" }],
        ],
      }),
    );
    expect(result).toBe(2500000000n);
  });

  it("passes stable=true when specified", async () => {
    const mockPublicClient = createMockPublicClient({
      readContract: vi.fn().mockResolvedValue([100n, 200n]),
    });

    await getQuote(mockPublicClient, WETH, USDC, 100n, true);

    const callArgs = (mockPublicClient.readContract as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(callArgs.args[1][0].stable).toBe(true);
  });
});

describe("approveToken", () => {
  const owner = "0x1234567890123456789012345678901234567890" as Address;
  const amount = 1000000000000000000n;

  it("skips approve when allowance is already sufficient", async () => {
    const mockPublicClient = createMockPublicClient({
      readContract: vi.fn().mockResolvedValue(amount),
    });
    const mockWalletClient = createMockWalletClient(owner);

    const hash = await approveToken(
      mockPublicClient,
      mockWalletClient,
      WETH,
      ROUTER,
      amount,
    );

    expect(hash).toBe("0x0000000000000000000000000000000000000000000000000000000000000000");
    expect(mockWalletClient.writeContract).not.toHaveBeenCalled();
  });

  it("approves when allowance is insufficient", async () => {
    const mockPublicClient = createMockPublicClient({
      readContract: vi.fn().mockResolvedValue(0n),
    });
    const mockWalletClient = createMockWalletClient(owner);

    const hash = await approveToken(
      mockPublicClient,
      mockWalletClient,
      WETH,
      ROUTER,
      amount,
    );

    expect(hash).toBe("0xdeadbeef");
    expect(mockWalletClient.writeContract).toHaveBeenCalledWith(
      expect.objectContaining({
        address: WETH,
        functionName: "approve",
        args: [ROUTER, amount],
      }),
    );
  });

  it("checks allowance for owner and spender", async () => {
    const mockPublicClient = createMockPublicClient({
      readContract: vi.fn().mockResolvedValue(999n),
    });
    const mockWalletClient = createMockWalletClient(owner);

    await approveToken(mockPublicClient, mockWalletClient, WETH, ROUTER, amount);

    expect(mockPublicClient.readContract).toHaveBeenCalledWith(
      expect.objectContaining({
        address: WETH,
        functionName: "allowance",
        args: [owner, ROUTER],
      }),
    );
  });
});

describe("executeSwap", () => {
  const owner = "0x1234567890123456789012345678901234567890" as Address;
  const amountIn = 1000000000000000000n; // 1 WETH
  const expectedOut = 2500000000n; // 2500 USDC

  function createSwapMocks() {
    let readCallCount = 0;
    const mockPublicClient = createMockPublicClient({
      readContract: vi.fn().mockImplementation(() => {
        readCallCount++;
        return [amountIn, expectedOut];
      }),
      simulateContract: vi.fn().mockResolvedValue({ result: [amountIn, expectedOut] }),
      waitForTransactionReceipt: vi.fn().mockResolvedValue({
        status: "success",
        transactionHash: "0xswaphash",
      }),
    });
    const mockWalletClient = createMockWalletClient(owner);
    return { mockPublicClient, mockWalletClient };
  }

  it("performs full swap flow and returns success result", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    const result = await executeSwap(
      mockPublicClient,
      mockWalletClient,
      WETH,
      USDC,
      amountIn,
    );

    expect(result.success).toBe(true);
    expect(result.amountIn).toBe(amountIn);
    expect(result.amountOut).toBe(expectedOut);
    expect(result.txHash).toBe("0xdeadbeef");
  });

  it("calls simulateContract before writeContract", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(mockPublicClient, mockWalletClient, WETH, USDC, amountIn);

    const simCallOrder = vi.mocked(mockPublicClient.simulateContract).mock.invocationCallOrder[0];
    const writeCallOrder = vi.mocked(mockWalletClient.writeContract).mock.invocationCallOrder[0];
    expect(simCallOrder).toBeLessThan(writeCallOrder);
  });

  it("calculates slippage correctly with default 50 bps", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(mockPublicClient, mockWalletClient, WETH, USDC, amountIn);

    const simArgs = (mockPublicClient.simulateContract as ReturnType<typeof vi.fn>).mock.calls[0][0];
    const amountOutMin = simArgs.args[1];
    const expectedMin = (expectedOut * 9950n) / 10000n;
    expect(amountOutMin).toBe(expectedMin);
  });

  it("calculates slippage correctly with custom bps", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(
      mockPublicClient,
      mockWalletClient,
      WETH,
      USDC,
      amountIn,
      100, // 1% slippage
    );

    const simArgs = (mockPublicClient.simulateContract as ReturnType<typeof vi.fn>).mock.calls[0][0];
    const amountOutMin = simArgs.args[1];
    const expectedMin = (expectedOut * 9900n) / 10000n;
    expect(amountOutMin).toBe(expectedMin);
  });

  it("waits for 2 confirmations", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(mockPublicClient, mockWalletClient, WETH, USDC, amountIn);

    expect(mockPublicClient.waitForTransactionReceipt).toHaveBeenCalledWith(
      expect.objectContaining({ confirmations: 2 }),
    );
  });

  it("returns failure when receipt status is not success", async () => {
    const mockPublicClient = createMockPublicClient({
      readContract: vi.fn().mockResolvedValue([amountIn, expectedOut]),
      simulateContract: vi.fn().mockResolvedValue({}),
      waitForTransactionReceipt: vi.fn().mockResolvedValue({
        status: "reverted",
        transactionHash: "0xreverted",
      }),
    });
    const mockWalletClient = createMockWalletClient(owner);

    const result = await executeSwap(
      mockPublicClient,
      mockWalletClient,
      WETH,
      USDC,
      amountIn,
    );

    expect(result.success).toBe(false);
    expect(result.error).toBe("Transaction reverted on-chain");
  });

  it("returns failure when simulation throws", async () => {
    const mockPublicClient = createMockPublicClient({
      readContract: vi.fn().mockResolvedValue([amountIn, expectedOut]),
      simulateContract: vi.fn().mockRejectedValue(new Error("Simulation failed: insufficient balance")),
    });
    const mockWalletClient = createMockWalletClient(owner);

    const result = await executeSwap(
      mockPublicClient,
      mockWalletClient,
      WETH,
      USDC,
      amountIn,
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Simulation failed");
  });

  it("passes correct route struct to swap", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(
      mockPublicClient,
      mockWalletClient,
      WETH,
      USDC,
      amountIn,
      50,
      true, // stable
    );

    const simArgs = (mockPublicClient.simulateContract as ReturnType<typeof vi.fn>).mock.calls[0][0];
    const route = simArgs.args[2][0];
    expect(route).toEqual({
      from: WETH,
      to: USDC,
      stable: true,
      factory: "0x0000000000000000000000000000000000000000",
    });
  });

  it("uses correct recipient from wallet account", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(mockPublicClient, mockWalletClient, WETH, USDC, amountIn);

    const simArgs = (mockPublicClient.simulateContract as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(simArgs.args[3]).toBe(owner);
  });
});
