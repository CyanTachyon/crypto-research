import type { Address } from "viem";
import { publicClient } from "../../clients/rpc.js";
import { quoterV2Abi } from "../../abis/quoterV2.js";
import { TOKENS, CONTRACTS } from "../../config/chains.js";

export interface PriceFeed {
  getPrice(pair: string): Promise<bigint>;
}

const PAIR_CONFIG: Record<
  string,
  { tokenIn: Address; tokenOut: Address; fee: number; invert: boolean }
> = {
  "ETH/USDC": {
    tokenIn: TOKENS.USDC.address,
    tokenOut: TOKENS.WETH.address,
    fee: 100,
    invert: true,
  },
  "CBBTC/USDC": {
    tokenIn: TOKENS.USDC.address,
    tokenOut: TOKENS.CBBTC.address,
    fee: 100,
    invert: true,
  },
};

export class OnChainPriceFeed implements PriceFeed {
  async getPrice(pair: string): Promise<bigint> {
    const cfg = PAIR_CONFIG[pair];
    if (!cfg) throw new Error(`Unknown pair: ${pair}`);

    const { result } = await publicClient.simulateContract({
      address: CONTRACTS.UNISWAP_V3_QUOTER_V2,
      abi: quoterV2Abi,
      functionName: "quoteExactInputSingle",
      args: [
        {
          tokenIn: cfg.tokenIn,
          tokenOut: cfg.tokenOut,
          amountIn: 1_000_000n,
          fee: cfg.fee,
          sqrtPriceLimitX96: 0n,
        },
      ],
    });

    const [amountOut] = result as [bigint, bigint, number, bigint];

    if (cfg.invert) {
      const tokenInfo =
        pair === "ETH/USDC"
          ? TOKENS.WETH
          : pair === "CBBTC/USDC"
            ? TOKENS.CBBTC
            : null;
      if (!tokenInfo) throw new Error(`Unknown pair: ${pair}`);
      const decimalShift = BigInt(tokenInfo.decimals - 6);
      const scale = 10n ** decimalShift;
      return (1_000_000n * scale * 1_000_000n) / amountOut;
    }

    return amountOut;
  }
}

export class DefiLlamaPriceFeed implements PriceFeed {
  private tokenMap: Record<string, string> = {
    "ETH/USDC": TOKENS.WETH.address,
    "CBBTC/USDC": TOKENS.CBBTC.address,
  };

  async getPrice(pair: string): Promise<bigint> {
    const address = this.tokenMap[pair];
    if (!address) throw new Error(`Unknown pair: ${pair}`);

    const url = `https://api.llama.fi/prices/current/base:${address}`;
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`DefiLlama API error: ${res.status} ${res.statusText}`);
    }

    const json = (await res.json()) as {
      coins: Record<string, { price: number }>;
    };
    const coinKey = `base:${address}`;
    const priceData = json.coins[coinKey];
    if (!priceData || priceData.price === undefined) {
      throw new Error(`No price data from DefiLlama for ${pair}`);
    }

    return BigInt(Math.floor(priceData.price * 1_000_000));
  }
}
