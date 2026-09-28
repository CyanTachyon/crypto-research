import type { Address, PublicClient, WalletClient } from "viem";
import { aerodromeRouterAbi } from "../../abis/aerodromeRouter.js";
import { erc20Abi } from "../../abis/erc20.js";
import { CONTRACTS } from "../../config/chains.js";
import type { SwapResult } from "../../types.js";

const AERODROME_ROUTER = CONTRACTS.AERODROME_ROUTER;

export interface Route {
  from: Address;
  to: Address;
  stable: boolean;
  factory: Address;
}

/** Create an Aerodrome Route struct */
export function createRoute(
  from: Address,
  to: Address,
  stable = false,
  factory: Address = "0x0000000000000000000000000000000000000000",
): Route {
  return { from, to, stable, factory };
}

/** Get a quote from Aerodrome router for a token swap */
export async function getQuote(
  publicClient: PublicClient,
  tokenIn: Address,
  tokenOut: Address,
  amountIn: bigint,
  stable = false,
): Promise<bigint> {
  const route = createRoute(tokenIn, tokenOut, stable);

  const amounts = await publicClient.readContract({
    address: AERODROME_ROUTER,
    abi: aerodromeRouterAbi,
    functionName: "getAmountsOut",
    args: [amountIn, [route]],
  });

  return amounts[amounts.length - 1];
}

/** Approve token spending — only calls approve if current allowance is insufficient */
export async function approveToken(
  publicClient: PublicClient,
  walletClient: WalletClient,
  tokenAddress: Address,
  spender: Address,
  amount: bigint,
): Promise<`0x${string}`> {
  const owner = walletClient.account!.address;

  const currentAllowance = await publicClient.readContract({
    address: tokenAddress,
    abi: erc20Abi,
    functionName: "allowance",
    args: [owner, spender],
  });

  if (currentAllowance >= amount) {
    return "0x0000000000000000000000000000000000000000000000000000000000000000";
  }

  const hash = await walletClient.writeContract({
    address: tokenAddress,
    abi: erc20Abi,
    functionName: "approve",
    args: [spender, amount],
    account: walletClient.account!,
    chain: walletClient.chain!,
  });

  return hash;
}

/** Execute a swap on Aerodrome with slippage protection and pre-flight simulation */
export async function executeSwap(
  publicClient: PublicClient,
  walletClient: WalletClient,
  tokenIn: Address,
  tokenOut: Address,
  amountIn: bigint,
  slippageBps = 50,
  stable = false,
): Promise<SwapResult> {
  try {
    // Step 1: Get quote
    const expectedOutput = await getQuote(
      publicClient,
      tokenIn,
      tokenOut,
      amountIn,
      stable,
    );

    // Step 2: Calculate minimum output with slippage
    const amountOutMin =
      (expectedOutput * BigInt(10000 - slippageBps)) / 10000n;

    const route = createRoute(tokenIn, tokenOut, stable);
    const FIVE_MINUTES = 300;
  const deadline = BigInt(Math.floor(Date.now() / 1000) + FIVE_MINUTES);
    const recipient = walletClient.account!.address;

    // Step 3: Simulate contract (pre-flight check)
    await publicClient.simulateContract({
      address: AERODROME_ROUTER,
      abi: aerodromeRouterAbi,
      functionName: "swapExactTokensForTokens",
      args: [amountIn, amountOutMin, [route], recipient, deadline],
      account: walletClient.account!,
    });

    // Step 4: Execute swap
    const hash = await walletClient.writeContract({
      address: AERODROME_ROUTER,
      abi: aerodromeRouterAbi,
      functionName: "swapExactTokensForTokens",
      args: [amountIn, amountOutMin, [route], recipient, deadline],
      account: walletClient.account!,
      chain: walletClient.chain!,
    });

    // Step 5: Wait for receipt
    const receipt = await publicClient.waitForTransactionReceipt({
      hash,
      confirmations: 2,
    });

    // Step 6: Verify success
    if (receipt.status !== "success") {
      return {
        success: false,
        amountIn,
        amountOut: 0n,
        txHash: hash,
        error: "Transaction reverted on-chain",
      };
    }

    return {
      success: true,
      amountIn,
      amountOut: expectedOutput,
      txHash: hash,
    };
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Unknown error during swap";
    return {
      success: false,
      amountIn,
      amountOut: 0n,
      error: message,
    };
  }
}
