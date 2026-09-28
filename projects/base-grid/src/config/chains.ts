import { base } from "viem/chains";
import type { Address } from "viem";

export const BASE_CHAIN = base;

export const TOKENS = {
  WETH: {
    address: "0x4200000000000000000000000000000000000006" as Address,
    decimals: 18,
    symbol: "WETH",
  },
  USDC: {
    address: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913" as Address,
    decimals: 6,
    symbol: "USDC",
  },
  CBBTC: {
    address: "0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf" as Address,
    decimals: 8,
    symbol: "CBBTC",
  },
} as const;

export const CONTRACTS = {
  AERODROME_ROUTER: "0xcF77a3Ba9A5CA399B7c97c74d54e5b1Beb874E43" as Address,
  AERODROME_POOL_FACTORY:
    "0x420DD381b31aEf6683db6B902084cB0FFECe40Da" as Address,
  UNISWAP_V3_SWAP_ROUTER_02:
    "0x2626664c2603336E57B271c5C0b26F421741e481" as Address,
  UNISWAP_V3_QUOTER_V2:
    "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a" as Address,
  UNISWAP_V3_FACTORY:
    "0x33128a8fC17869897dcE68Ed026d694621f6FDfD" as Address,
} as const;

/** All internal calculations use USDC (6 decimals) as base unit */
export const BASE_DECIMALS = 6;
