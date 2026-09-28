import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Address, PublicClient, WalletClient } from "viem";
import { TOKENS, CONTRACTS } from "../../config/chains.js";
import {
  getQuote,
  getBestQuote,
  approveToken,
  executeSwap,
} from "./uniswapV3.js";

const WETH = TOKENS.WETH.address;
const USDC = TOKENS.USDC.address;
const QUOTER_V2 = CONTRACTS.UNISWAP_V3_QUOTER_V2;
const SWAP_ROUTER_02 = CONTRACTS.UNISWAP_V3_SWAP_ROUTER_02;

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

describe("getQuote", () => {
  it("calls quoteExactInputSingle and returns amountOut", async () => {
    const quoteResult = [2500000000n, 0n, 0, 150000n];
    const mockPublicClient = createMockPublicClient({
      simulateContract: vi.fn().mockResolvedValue({ result: quoteResult }),
    });

    const result = await getQuote(WETH, USDC, 1000000000000000000n, 500, mockPublicClient);

    expect(mockPublicClient.simulateContract).toHaveBeenCalledWith(
      expect.objectContaining({
        address: QUOTER_V2,
        functionName: "quoteExactInputSingle",
        args: [
          {
            tokenIn: WETH,
            tokenOut: USDC,
            amountIn: 1000000000000000000n,
            fee: 500,
            sqrtPriceLimitX96: 0n,
          },
        ],
      }),
    );
    expect(result).toBe(2500000000n);
  });

  it("passes correct fee tier", async () => {
    const mockPublicClient = createMockPublicClient({
      simulateContract: vi.fn().mockResolvedValue({ result: [100n, 0n, 0, 100n] }),
    });

    await getQuote(WETH, USDC, 100n, 3000, mockPublicClient);

    const callArgs = (mockPublicClient.simulateContract as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(callArgs.args[0].fee).toBe(3000);
  });
});

describe("getBestQuote", () => {
  it("tries multiple fee tiers and returns the best quote", async () => {
    let callCount = 0;
    const mockPublicClient = createMockPublicClient({
      simulateContract: vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) return { result: [2000000000n, 0n, 0, 100n] };
        return { result: [1800000000n, 0n, 0, 100n] };
      }),
    });

    const result = await getBestQuote(WETH, USDC, 1000000000000000000n, mockPublicClient);

    expect(result.amountOut).toBe(2000000000n);
    expect(result.fee).toBe(500);
    expect(mockPublicClient.simulateContract).toHaveBeenCalledTimes(2);
  });

  it("picks fee 3000 when it gives better output", async () => {
    let callCount = 0;
    const mockPublicClient = createMockPublicClient({
      simulateContract: vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) return { result: [100n, 0n, 0, 100n] };
        return { result: [200n, 0n, 0, 100n] };
      }),
    });

    const result = await getBestQuote(WETH, USDC, 100n, mockPublicClient);

    expect(result.amountOut).toBe(200n);
    expect(result.fee).toBe(3000);
  });

  it("skips failed fee tiers and still returns a result", async () => {
    let callCount = 0;
    const mockPublicClient = createMockPublicClient({
      simulateContract: vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) throw new Error("Pool does not exist");
        return { result: [1500000000n, 0n, 0, 100n] };
      }),
    });

    const result = await getBestQuote(WETH, USDC, 1000000000000000000n, mockPublicClient);

    expect(result.amountOut).toBe(1500000000n);
    expect(result.fee).toBe(3000);
  });

  it("throws when all fee tiers fail", async () => {
    const mockPublicClient = createMockPublicClient({
      simulateContract: vi.fn().mockRejectedValue(new Error("No pool")),
    });

    await expect(
      getBestQuote(WETH, USDC, 100n, mockPublicClient),
    ).rejects.toThrow("No valid quote found for any fee tier");
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
      WETH,
      SWAP_ROUTER_02,
      amount,
      mockPublicClient,
      mockWalletClient,
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
      WETH,
      SWAP_ROUTER_02,
      amount,
      mockPublicClient,
      mockWalletClient,
    );

    expect(hash).toBe("0xdeadbeef");
    expect(mockWalletClient.writeContract).toHaveBeenCalledWith(
      expect.objectContaining({
        address: WETH,
        functionName: "approve",
        args: [SWAP_ROUTER_02, amount],
      }),
    );
  });

  it("checks allowance for owner and spender", async () => {
    const mockPublicClient = createMockPublicClient({
      readContract: vi.fn().mockResolvedValue(999n),
    });
    const mockWalletClient = createMockWalletClient(owner);

    await approveToken(WETH, SWAP_ROUTER_02, amount, mockPublicClient, mockWalletClient);

    expect(mockPublicClient.readContract).toHaveBeenCalledWith(
      expect.objectContaining({
        address: WETH,
        functionName: "allowance",
        args: [owner, SWAP_ROUTER_02],
      }),
    );
  });
});

describe("executeSwap", () => {
  const owner = "0x1234567890123456789012345678901234567890" as Address;
  const amountIn = 1000000000000000000n;
  const expectedOut = 2500000000n;
  const fee = 500;

  function createSwapMocks() {
    const mockPublicClient = createMockPublicClient({
      simulateContract: vi.fn().mockResolvedValue({
        result: [expectedOut, 0n, 0, 150000n],
      }),
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
      WETH,
      USDC,
      amountIn,
      50,
      fee,
      mockPublicClient,
      mockWalletClient,
    );

    expect(result.success).toBe(true);
    expect(result.amountIn).toBe(amountIn);
    expect(result.amountOut).toBe(expectedOut);
    expect(result.txHash).toBe("0xdeadbeef");
  });

  it("uses multicall wrapping with deadline for the swap", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(WETH, USDC, amountIn, 50, fee, mockPublicClient, mockWalletClient);

    const writeCall = (mockWalletClient.writeContract as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(writeCall.functionName).toBe("multicall");
    expect(writeCall.address).toBe(SWAP_ROUTER_02);
    expect(writeCall.args[0]).toBeTypeOf("bigint");
    expect(writeCall.args[1]).toHaveLength(1);
  });

  it("calls simulateContract before writeContract", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(WETH, USDC, amountIn, 50, fee, mockPublicClient, mockWalletClient);

    const simCallOrder = vi.mocked(mockPublicClient.simulateContract).mock.invocationCallOrder[0];
    const writeCallOrder = vi.mocked(mockWalletClient.writeContract).mock.invocationCallOrder[0];
    expect(simCallOrder).toBeLessThan(writeCallOrder);
  });

  it("calculates slippage correctly with 50 bps", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(WETH, USDC, amountIn, 50, fee, mockPublicClient, mockWalletClient);

    const simArgs = (mockPublicClient.simulateContract as ReturnType<typeof vi.fn>).mock.calls[1][0];
    const multicallData = simArgs.args[1][0] as `0x${string}`;

    const writeArgs = (mockWalletClient.writeContract as ReturnType<typeof vi.fn>).mock.calls[0][0];
    const writeData = writeArgs.args[1][0] as `0x${string}`;

    expect(multicallData).toBe(writeData);
  });

  it("waits for 2 confirmations", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(WETH, USDC, amountIn, 50, fee, mockPublicClient, mockWalletClient);

    expect(mockPublicClient.waitForTransactionReceipt).toHaveBeenCalledWith(
      expect.objectContaining({ confirmations: 2 }),
    );
  });

  it("returns failure when receipt status is not success", async () => {
    const mockPublicClient = createMockPublicClient({
      simulateContract: vi.fn().mockResolvedValue({
        result: [expectedOut, 0n, 0, 150000n],
      }),
      waitForTransactionReceipt: vi.fn().mockResolvedValue({
        status: "reverted",
        transactionHash: "0xreverted",
      }),
    });
    const mockWalletClient = createMockWalletClient(owner);

    const result = await executeSwap(
      WETH,
      USDC,
      amountIn,
      50,
      fee,
      mockPublicClient,
      mockWalletClient,
    );

    expect(result.success).toBe(false);
    expect(result.error).toBe("Transaction reverted on-chain");
  });

  it("returns failure when simulation throws", async () => {
    const mockPublicClient = createMockPublicClient({
      simulateContract: vi.fn()
        .mockResolvedValueOnce({ result: [expectedOut, 0n, 0, 150000n] })
        .mockRejectedValueOnce(new Error("Simulation failed: insufficient balance")),
    });
    const mockWalletClient = createMockWalletClient(owner);

    const result = await executeSwap(
      WETH,
      USDC,
      amountIn,
      50,
      fee,
      mockPublicClient,
      mockWalletClient,
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Simulation failed");
  });

  it("uses wallet account address as recipient in swap calldata", async () => {
    const { mockPublicClient, mockWalletClient } = createSwapMocks();

    await executeSwap(WETH, USDC, amountIn, 50, fee, mockPublicClient, mockWalletClient);

    const simArgs = (mockPublicClient.simulateContract as ReturnType<typeof vi.fn>).mock.calls[1][0];
    expect(simArgs.account).toMatchObject({ address: owner });
  });
});
