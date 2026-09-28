import { createPublicClient, http } from "viem";
import type { Address } from "viem";
import { base } from "viem/chains";
import { TOKENS, CONTRACTS } from "../src/config/chains.js";
import { quoterV2Abi } from "../src/abis/quoterV2.js";

const AERODROME_POOL_FACTORY_ABI = [
  {
    inputs: [
      { name: "", type: "address" },
      { name: "", type: "address" },
      { name: "", type: "bool" },
    ],
    name: "getPool",
    outputs: [{ name: "", type: "address" }],
    stateMutability: "view",
    type: "function",
  },
] as const;

const UNISWAP_V3_FACTORY_ABI = [
  {
    inputs: [
      { name: "", type: "address" },
      { name: "", type: "address" },
      { name: "", type: "uint24" },
    ],
    name: "getPool",
    outputs: [{ name: "", type: "address" }],
    stateMutability: "view",
    type: "function",
  },
] as const;

const ZERO_ADDRESS = "0x0000000000000000000000000000000000000000" as Address;

const { WETH, USDC, CBBTC } = TOKENS;

interface PoolResult {
  name: string;
  dex: string;
  pair: string;
  address: Address | null;
  fee?: number;
  hasLiquidity?: boolean;
  quoteOut?: bigint;
}

async function main() {
  const client = createPublicClient({
    chain: base,
    transport: http("https://mainnet.base.org"),
  });

  console.log("=== Pool Verification on Base Mainnet ===\n");

  const results: PoolResult[] = [];

  console.log("--- Aerodrome (Volatile Pools) ---");

  const aeroPairs: Array<{ tokenA: Address; tokenB: Address; label: string }> =
    [
      { tokenA: WETH.address, tokenB: USDC.address, label: "ETH/USDC" },
      { tokenA: CBBTC.address, tokenB: USDC.address, label: "CBBTC/USDC" },
    ];

  for (const pair of aeroPairs) {
    try {
      const poolAddress = await client.readContract({
        address: CONTRACTS.AERODROME_POOL_FACTORY,
        abi: AERODROME_POOL_FACTORY_ABI,
        functionName: "getPool",
        args: [pair.tokenA, pair.tokenB, false], // false = volatile
      });

      const exists =
        typeof poolAddress === "string" && poolAddress !== ZERO_ADDRESS;

      console.log(
        `  ${pair.label}: ${exists ? poolAddress : "NOT FOUND (address(0))"}`,
      );

      results.push({
        name: `Aerodrome ${pair.label}`,
        dex: "aerodrome",
        pair: pair.label,
        address: exists ? (poolAddress as Address) : null,
      });
    } catch (err) {
      console.log(`  ${pair.label}: ERROR - ${(err as Error).message}`);
      results.push({
        name: `Aerodrome ${pair.label}`,
        dex: "aerodrome",
        pair: pair.label,
        address: null,
      });
    }
  }

  console.log("\n--- Uniswap V3 ---");

  const uniPairs: Array<{
    tokenA: Address;
    tokenB: Address;
    label: string;
    fee: number;
    amountIn: bigint;
  }> = [
    {
      tokenA: WETH.address,
      tokenB: USDC.address,
      label: "ETH/USDC",
      fee: 500,
      amountIn: 100000000000000000n, // 0.1 WETH
    },
    {
      tokenA: WETH.address,
      tokenB: USDC.address,
      label: "ETH/USDC",
      fee: 3000,
      amountIn: 100000000000000000n, // 0.1 WETH
    },
    {
      tokenA: CBBTC.address,
      tokenB: USDC.address,
      label: "CBBTC/USDC",
      fee: 500,
      amountIn: 10000000n, // 0.1 CBBTC (8 decimals)
    },
    {
      tokenA: CBBTC.address,
      tokenB: USDC.address,
      label: "CBBTC/USDC",
      fee: 3000,
      amountIn: 10000000n, // 0.1 CBBTC (8 decimals)
    },
  ];

  for (const pair of uniPairs) {
    try {
      const poolAddress = await client.readContract({
        address: CONTRACTS.UNISWAP_V3_FACTORY,
        abi: UNISWAP_V3_FACTORY_ABI,
        functionName: "getPool",
        args: [pair.tokenA, pair.tokenB, pair.fee],
      });

      const exists =
        typeof poolAddress === "string" && poolAddress !== ZERO_ADDRESS;

      console.log(
        `  ${pair.label} (fee ${pair.fee}): ${exists ? poolAddress : "NOT FOUND (address(0))"}`,
      );

      let hasLiquidity = false;
      let quoteOut: bigint | undefined;

      if (exists) {
        try {
          const quoteResult = await client.simulateContract({
            address: CONTRACTS.UNISWAP_V3_QUOTER_V2,
            abi: quoterV2Abi,
            functionName: "quoteExactInputSingle",
            args: [
              {
                tokenIn: pair.tokenA,
                tokenOut: pair.tokenB,
                amountIn: pair.amountIn,
                fee: pair.fee,
                sqrtPriceLimitX96: 0n,
              },
            ],
          });

          quoteOut = quoteResult.result[0];
          hasLiquidity = quoteOut > 0n;
          console.log(
            `    Quote: ${pair.amountIn.toString()} -> ${quoteOut.toString()} (has liquidity: ${hasLiquidity})`,
          );
        } catch (quoteErr) {
          console.log(
            `    Quote failed (no liquidity or error): ${(quoteErr as Error).message}`,
          );
        }
      }

      results.push({
        name: `UniswapV3 ${pair.label} (${pair.fee})`,
        dex: "uniswapV3",
        pair: pair.label,
        address: exists ? (poolAddress as Address) : null,
        fee: pair.fee,
        hasLiquidity,
        quoteOut,
      });
    } catch (err) {
      console.log(
        `  ${pair.label} (fee ${pair.fee}): ERROR - ${(err as Error).message}`,
      );
      results.push({
        name: `UniswapV3 ${pair.label} (${pair.fee})`,
        dex: "uniswapV3",
        pair: pair.label,
        address: null,
        fee: pair.fee,
      });
    }
  }

  console.log("\n=== Summary ===");
  for (const r of results) {
    const status = r.address
      ? r.hasLiquidity === false
        ? "FOUND (no liquidity)"
        : "FOUND"
      : "NOT FOUND";
    console.log(
      `  ${r.name}: ${status} ${r.address ? r.address : ""}`,
    );
  }

  const aerodromePools = results.filter((r) => r.dex === "aerodrome");
  const uniswapV3Pools = results.filter((r) => r.dex === "uniswapV3");

  const poolsTs = generatePoolsTs(aerodromePools, uniswapV3Pools);

  const outputPath = new URL("../src/config/pools.ts", import.meta.url);
  const { writeFileSync } = await import("node:fs");
  writeFileSync(outputPath, poolsTs);
  console.log(`\nGenerated src/config/pools.ts`);
}

function generatePoolsTs(
  aerodromePools: PoolResult[],
  uniswapV3Pools: PoolResult[],
): string {
  const aeroEthUsdc = aerodromePools.find((p) => p.pair === "ETH/USDC");
  const aeroCbbtcUsdc = aerodromePools.find((p) => p.pair === "CBBTC/USDC");

  const uniEthUsdc500 = uniswapV3Pools.find(
    (p) => p.pair === "ETH/USDC" && p.fee === 500,
  );
  const uniEthUsdc3000 = uniswapV3Pools.find(
    (p) => p.pair === "ETH/USDC" && p.fee === 3000,
  );
  const uniCbbtcUsdc500 = uniswapV3Pools.find(
    (p) => p.pair === "CBBTC/USDC" && p.fee === 500,
  );
  const uniCbbtcUsdc3000 = uniswapV3Pools.find(
    (p) => p.pair === "CBBTC/USDC" && p.fee === 3000,
  );

  const fmt = (addr: Address | null) =>
    addr ? `"${addr}" as Address` : `null as unknown as Address /* NOT FOUND */`;

  return `import type { Address } from "viem";

/**
 * Verified pool addresses on Base mainnet.
 * Generated by scripts/verify-pools.ts — re-run to refresh.
 *
 * Pool addresses:
 *   Aerodrome ETH/USDC:      ${aeroEthUsdc?.address ?? "NOT FOUND"}
 *   Aerodrome CBBTC/USDC:    ${aeroCbbtcUsdc?.address ?? "NOT FOUND"}
 *   UniswapV3 ETH/USDC 500:  ${uniEthUsdc500?.address ?? "NOT FOUND"}
 *   UniswapV3 ETH/USDC 3000: ${uniEthUsdc3000?.address ?? "NOT FOUND"}
 *   UniswapV3 CBBTC/USDC 500:  ${uniCbbtcUsdc500?.address ?? "NOT FOUND"}
 *   UniswapV3 CBBTC/USDC 3000: ${uniCbbtcUsdc3000?.address ?? "NOT FOUND"}
 */

export const POOLS = {
  aerodrome: {
    ETH_USDC: ${fmt(aeroEthUsdc?.address ?? null)},
    CBBTC_USDC: ${fmt(aeroCbbtcUsdc?.address ?? null)},
  },
  uniswapV3: {
    ETH_USDC_500: ${fmt(uniEthUsdc500?.address ?? null)},
    ETH_USDC_3000: ${fmt(uniEthUsdc3000?.address ?? null)},
    CBBTC_USDC_500: ${fmt(uniCbbtcUsdc500?.address ?? null)},
    CBBTC_USDC_3000: ${fmt(uniCbbtcUsdc3000?.address ?? null)},
  },
} as const;
`;
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
