import { parseEther } from "viem";

// --- Types ---

export interface RiskCheckResult {
  allowed: boolean;
  reason?: string;
  action?: "STOP_LOSS" | "EMERGENCY_EXIT" | "PAUSE" | "SKIP";
  pauseDurationMs?: number;
}

export interface TradeContext {
  currentPrice: bigint;
  centerPrice: bigint;
  initialValueUsd: number;
  currentValueUsd: number;
  triggeredLevels: number;
  ethBalance: bigint;
  spentTodayUsd: number;
  estimatedGasCostUsd: number;
}

export interface RiskConfig {
  stopLossPct: number;
  maxDrawdownPct: number;
  circuitBreakerThreshold: number;
  circuitBreakerPauseMs: number;
  maxGasCostUsd: number;
  maxDailySpendUsd: number;
  minEthReserve: bigint;
}

export const DEFAULT_RISK_CONFIG: RiskConfig = {
  stopLossPct: 8,
  maxDrawdownPct: 25,
  circuitBreakerThreshold: 2,
  circuitBreakerPauseMs: 60_000,
  maxGasCostUsd: 0.1,
  maxDailySpendUsd: 10,
  minEthReserve: parseEther("0.005"),
};

// --- Individual Risk Rules ---

export function checkStopLoss(
  currentPrice: bigint,
  centerPrice: bigint,
  stopLossPct: number = 8,
): RiskCheckResult {
  // Pure bigint: price < center * (100 - pct) / 100
  const threshold = (centerPrice * BigInt(100 - stopLossPct)) / 100n;
  if (currentPrice < threshold) {
    return {
      allowed: false,
      reason: `Stop loss: price ${currentPrice} below threshold ${threshold} (${stopLossPct}% below center ${centerPrice})`,
      action: "STOP_LOSS",
    };
  }
  return { allowed: true };
}

export function checkMaxDrawdown(
  initialValue: number,
  currentValue: number,
  maxDrawdownPct: number = 25,
): RiskCheckResult {
  if (initialValue <= 0) return { allowed: true };
  const drawdownPct = ((initialValue - currentValue) / initialValue) * 100;
  if (drawdownPct > maxDrawdownPct) {
    return {
      allowed: false,
      reason: `Max drawdown exceeded: ${drawdownPct.toFixed(2)}% > ${maxDrawdownPct}%`,
      action: "EMERGENCY_EXIT",
    };
  }
  return { allowed: true };
}

export function checkCircuitBreaker(
  triggeredLevels: number,
  maxConsecutive: number = 2,
): RiskCheckResult {
  if (triggeredLevels > maxConsecutive) {
    return {
      allowed: false,
      reason: `Circuit breaker: ${triggeredLevels} levels triggered > ${maxConsecutive} max`,
      action: "PAUSE",
      pauseDurationMs: 60_000,
    };
  }
  return { allowed: true };
}

export function checkGasPrice(
  estimatedGasCostUsd: number,
  maxGasUsd: number = 0.1,
): RiskCheckResult {
  if (estimatedGasCostUsd > maxGasUsd) {
    return {
      allowed: false,
      reason: `Gas too high: $${estimatedGasCostUsd.toFixed(4)} > $${maxGasUsd} max`,
      action: "SKIP",
    };
  }
  return { allowed: true };
}

export function checkDailyLimit(
  spentTodayUsd: number,
  maxDailyUsd: number = 10,
): RiskCheckResult {
  if (spentTodayUsd > maxDailyUsd) {
    return {
      allowed: false,
      reason: `Daily limit exceeded: $${spentTodayUsd.toFixed(2)} > $${maxDailyUsd} max`,
      action: "SKIP",
    };
  }
  return { allowed: true };
}

export function checkBalanceReserve(
  ethBalance: bigint,
  minReserve: bigint,
): RiskCheckResult {
  if (ethBalance < minReserve) {
    return {
      allowed: false,
      reason: `ETH balance ${ethBalance} below minimum reserve ${minReserve}`,
      action: "EMERGENCY_EXIT",
    };
  }
  return { allowed: true };
}

// --- Composite Check ---

export function canTrade(
  tradeAmountUsd: number,
  context: TradeContext,
  config: RiskConfig = DEFAULT_RISK_CONFIG,
): RiskCheckResult {
  const checks: RiskCheckResult[] = [
    checkStopLoss(context.currentPrice, context.centerPrice, config.stopLossPct),
    checkMaxDrawdown(context.initialValueUsd, context.currentValueUsd, config.maxDrawdownPct),
    checkCircuitBreaker(context.triggeredLevels, config.circuitBreakerThreshold),
    checkGasPrice(context.estimatedGasCostUsd, config.maxGasCostUsd),
    checkDailyLimit(context.spentTodayUsd + tradeAmountUsd, config.maxDailySpendUsd),
    checkBalanceReserve(context.ethBalance, config.minEthReserve),
  ];

  for (const result of checks) {
    if (!result.allowed) return result;
  }

  return { allowed: true };
}
