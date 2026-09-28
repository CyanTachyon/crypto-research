import type { Address } from "viem";

export interface TokenInfo {
  address: Address;
  decimals: number;
  symbol: string;
}

export interface GridLevel {
  price: bigint;
  buyAmount: bigint;
  sellAmount: bigint;
  filled: boolean;
}

export interface GridConfig {
  baseToken: TokenInfo;
  quoteToken: TokenInfo;
  lowerPrice: bigint;
  upperPrice: bigint;
  gridLevels: number;
  investmentAmount: bigint;
}

export interface SwapResult {
  success: boolean;
  amountIn: bigint;
  amountOut: bigint;
  txHash?: string;
  error?: string;
}

export type BotMode = "sim" | "live";

// --- State Persistence Types ---

export interface BotState {
  version: number;
  lastBlockNumber: string; // hex string for BigInt
  lastUpdated: number; // unix timestamp ms
  grids: Record<string, GridState>; // key = "ETH/USDC" etc
  pendingTx: { txHash: string; submittedAt: number } | null;
}

export interface GridState {
  pair: string;
  centerPrice: string; // hex string for BigInt
  levels: GridLevelState[];
  config: {
    gridLevels: number;
    rangePct: number;
    capitalUsd: number;
  };
  fillHistory: FillRecord[];
}

export interface GridLevelState {
  price: string; // hex
  buyAmount: string; // hex
  sellAmount: string; // hex
  filled: boolean;
  side: "buy" | "sell" | "none";
}

export interface FillRecord {
  timestamp: number;
  levelIndex: number;
  side: "buy" | "sell";
  price: string; // hex
  amount: string; // hex
  txHash?: string;
}
