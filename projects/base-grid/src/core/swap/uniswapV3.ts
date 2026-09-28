import type { Address, PublicClient, WalletClient } from "viem";
import { encodeFunctionData } from "viem";
import { quoterV2Abi } from "../../abis/quoterV2.js";
import { erc20Abi } from "../../abis/erc20.js";
import { swapRouter02Abi } from "../../abis/swapRouter02.js";
import { CONTRACTS } from "../../config/chains.js";
import type { SwapResult } from "../../types.js";

const QUOTER_V2 = CONTRACTS.UNISWAP_V3_QUOTER_V2;
const SWAP_ROUTER_02 = CONTRACTS.UNISWAP_V3_SWAP_ROUTER_02;

const FEE_TIERS = [500, 3000] as const;
const DEADLINE_SECONDS = 120;

export async function getQuote(
  tokenIn: Address,
  tokenOut: Address,
  amountIn: bigint,
  fee: number,
  publicClient: PublicClient,
): Promise<bigint> {
  const { result } = await publicClient.simulateContract({
    address: QUOTER_V2,
    abi: quoterV2Abi,
    functionName: "quoteExactInputSingle",
    args: [
      {
        tokenIn,
        tokenOut,
        amountIn,
        fee,
        sqrtPriceLimitX96: 0n,
      },
    ],
    account: "0x0000000000000000000000000000000000000001",
  });

  const [amountOut] = result;
  return amountOut;
}

export async function getBestQuote(
  tokenIn: Address,
  tokenOut: Address,
  amountIn: bigint,
  publicClient: PublicClient,
): Promise<{ amountOut: bigint; fee: number }> {
  let bestAmountOut = 0n;
  let bestFee: number = FEE_TIERS[0];

  for (const fee of FEE_TIERS) {
    try {
      const amountOut = await getQuote(
        tokenIn,
        tokenOut,
        amountIn,
        fee,
        publicClient,
      );
      if (amountOut > bestAmountOut) {
        bestAmountOut = amountOut;
        bestFee = fee;
      }
    } catch {
      // Pool may not exist for this fee tier — skip
    }
  }

  if (bestAmountOut === 0n) {
    throw new Error("No valid quote found for any fee tier");
  }

  return { amountOut: bestAmountOut, fee: bestFee };
}

export async function approveToken(
  tokenAddress: Address,
  spender: Address,
  amount: bigint,
  publicClient: PublicClient,
  walletClient: WalletClient,
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

export async function executeSwap(
  tokenIn: Address,
  tokenOut: Address,
  amountIn: bigint,
  slippageBps: number,
  fee: number,
  publicClient: PublicClient,
  walletClient: WalletClient,
): Promise<SwapResult> {
  try {
    const expectedOutput = await getQuote(
      tokenIn,
      tokenOut,
      amountIn,
      fee,
      publicClient,
    );

    const amountOutMinimum =
      (expectedOutput * BigInt(10000 - slippageBps)) / 10000n;

    const deadline = BigInt(
      Math.floor(Date.now() / 1000) + DEADLINE_SECONDS,
    );
    const recipient = walletClient.account!.address;

    const swapCalldata = encodeFunctionData({
      abi: swapRouter02Abi,
      functionName: "exactInputSingle",
      args: [
        {
          tokenIn,
          tokenOut,
          fee,
          recipient,
          amountIn,
          amountOutMinimum,
          sqrtPriceLimitX96: 0n,
        },
      ],
    });

    await publicClient.simulateContract({
      address: SWAP_ROUTER_02,
      abi: swapRouter02Abi,
      functionName: "multicall",
      args: [deadline, [swapCalldata]],
      account: walletClient.account!,
    });

    const hash = await walletClient.writeContract({
      address: SWAP_ROUTER_02,
      abi: swapRouter02Abi,
      functionName: "multicall",
      args: [deadline, [swapCalldata]],
      account: walletClient.account!,
      chain: walletClient.chain!,
    });

    const receipt = await publicClient.waitForTransactionReceipt({
      hash,
      confirmations: 2,
    });

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
