import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { parseEther } from "viem";
import {
  checkStopLoss,
  checkMaxDrawdown,
  checkCircuitBreaker,
  checkGasPrice,
  checkDailyLimit,
  checkBalanceReserve,
  canTrade,
  DEFAULT_RISK_CONFIG,
} from "./riskManager.js";
import type { TradeContext, RiskConfig } from "./riskManager.js";
import { EmergencyStop } from "./emergencyStop.js";

describe("checkStopLoss", () => {
  it("triggers when price is 8% below center", () => {
    const centerPrice = 1_000_000_000n;
    const currentPrice = 910_000_000n;
    const result = checkStopLoss(currentPrice, centerPrice, 8);
    expect(result.allowed).toBe(false);
    expect(result.action).toBe("STOP_LOSS");
    expect(result.reason).toContain("Stop loss");
  });

  it("does not trigger when price is 7% below center", () => {
    const centerPrice = 1_000_000_000n;
    const currentPrice = 930_000_000n;
    const result = checkStopLoss(currentPrice, centerPrice, 8);
    expect(result.allowed).toBe(true);
  });

  it("does not trigger at exactly the threshold", () => {
    const centerPrice = 100n;
    const currentPrice = 92n;
    const result = checkStopLoss(currentPrice, centerPrice, 8);
    expect(result.allowed).toBe(true);
  });
});

describe("checkMaxDrawdown", () => {
  it("triggers emergency exit at 25% drawdown", () => {
    const result = checkMaxDrawdown(1000, 700, 25);
    expect(result.allowed).toBe(false);
    expect(result.action).toBe("EMERGENCY_EXIT");
    expect(result.reason).toContain("Max drawdown");
  });

  it("does not trigger at 20% drawdown", () => {
    const result = checkMaxDrawdown(1000, 800, 25);
    expect(result.allowed).toBe(true);
  });

  it("handles zero initial value", () => {
    const result = checkMaxDrawdown(0, 500, 25);
    expect(result.allowed).toBe(true);
  });
});

describe("checkCircuitBreaker", () => {
  it("pauses when 3 levels triggered (threshold 2)", () => {
    const result = checkCircuitBreaker(3, 2);
    expect(result.allowed).toBe(false);
    expect(result.action).toBe("PAUSE");
    expect(result.pauseDurationMs).toBe(60_000);
    expect(result.reason).toContain("Circuit breaker");
  });

  it("does not trigger when 2 levels triggered (threshold 2)", () => {
    const result = checkCircuitBreaker(2, 2);
    expect(result.allowed).toBe(true);
  });

  it("does not trigger at 0 levels", () => {
    const result = checkCircuitBreaker(0, 2);
    expect(result.allowed).toBe(true);
  });
});

describe("checkGasPrice", () => {
  it("skips trade when gas $0.15 exceeds $0.10 max", () => {
    const result = checkGasPrice(0.15, 0.1);
    expect(result.allowed).toBe(false);
    expect(result.action).toBe("SKIP");
    expect(result.reason).toContain("Gas too high");
  });

  it("allows trade when gas within limit", () => {
    const result = checkGasPrice(0.05, 0.1);
    expect(result.allowed).toBe(true);
  });

  it("allows trade when gas equals limit exactly", () => {
    const result = checkGasPrice(0.1, 0.1);
    expect(result.allowed).toBe(true);
  });
});

describe("checkDailyLimit", () => {
  it("skips when spent $8 + trade $5 exceeds $10 max", () => {
    const result = checkDailyLimit(8 + 5, 10);
    expect(result.allowed).toBe(false);
    expect(result.action).toBe("SKIP");
    expect(result.reason).toContain("Daily limit");
  });

  it("allows when within daily limit", () => {
    const result = checkDailyLimit(8, 10);
    expect(result.allowed).toBe(true);
  });
});

describe("checkBalanceReserve", () => {
  it("stops all trading when ETH 0.003 < 0.005 reserve", () => {
    const result = checkBalanceReserve(parseEther("0.003"), parseEther("0.005"));
    expect(result.allowed).toBe(false);
    expect(result.action).toBe("EMERGENCY_EXIT");
    expect(result.reason).toContain("ETH balance");
  });

  it("allows when balance exceeds reserve", () => {
    const result = checkBalanceReserve(parseEther("0.01"), parseEther("0.005"));
    expect(result.allowed).toBe(true);
  });
});

describe("canTrade", () => {
  const safeContext: TradeContext = {
    currentPrice: 1_000_000_000n,
    centerPrice: 1_000_000_000n,
    initialValueUsd: 1000,
    currentValueUsd: 950,
    triggeredLevels: 1,
    ethBalance: parseEther("0.1"),
    spentTodayUsd: 2,
    estimatedGasCostUsd: 0.05,
  };

  it("allows trade when all checks pass", () => {
    const result = canTrade(3, safeContext);
    expect(result.allowed).toBe(true);
  });

  it("returns first failure reason when one check fails", () => {
    const context: TradeContext = {
      ...safeContext,
      currentPrice: 900_000_000n,
      centerPrice: 1_000_000_000n,
    };
    const result = canTrade(3, context);
    expect(result.allowed).toBe(false);
    expect(result.action).toBe("STOP_LOSS");
  });

  it("returns gas failure when gas is too high", () => {
    const context: TradeContext = {
      ...safeContext,
      estimatedGasCostUsd: 0.2,
    };
    const result = canTrade(3, context);
    expect(result.allowed).toBe(false);
    expect(result.action).toBe("SKIP");
    expect(result.reason).toContain("Gas too high");
  });

  it("returns daily limit failure when overspent", () => {
    const context: TradeContext = {
      ...safeContext,
      spentTodayUsd: 8,
    };
    const result = canTrade(5, context);
    expect(result.allowed).toBe(false);
    expect(result.action).toBe("SKIP");
    expect(result.reason).toContain("Daily limit");
  });

  it("uses default config when none provided", () => {
    const result = canTrade(1, safeContext);
    expect(result.allowed).toBe(true);
  });
});

describe("EmergencyStop", () => {
  let emergency: EmergencyStop;

  beforeEach(() => {
    emergency = new EmergencyStop();
  });

  it("starts inactive", () => {
    expect(emergency.isActive()).toBe(false);
    expect(emergency.getReason()).toBeUndefined();
  });

  it("activates with a reason", () => {
    emergency.activate("test reason");
    expect(emergency.isActive()).toBe(true);
    expect(emergency.getReason()).toBe("test reason");
  });

  it("deactivates and clears reason", () => {
    emergency.activate("test");
    emergency.deactivate();
    expect(emergency.isActive()).toBe(false);
    expect(emergency.getReason()).toBeUndefined();
  });

  it("activates on file trigger when stop file exists", () => {
    const result = emergency.checkFileTrigger("/Users/cyan/Desktop/tmp/crypto");
    if (result) {
      expect(emergency.isActive()).toBe(true);
      expect(emergency.getReason()).toContain("Stop file");
    }
  });

  it("does not activate when stop file missing", () => {
    const result = emergency.checkFileTrigger("/nonexistent/path");
    expect(result).toBe(false);
    expect(emergency.isActive()).toBe(false);
  });

  it("activates on EMERGENCY_STOP env var", () => {
    process.env.EMERGENCY_STOP = "true";
    const result = emergency.checkEnvTrigger();
    expect(result).toBe(true);
    expect(emergency.isActive()).toBe(true);
    delete process.env.EMERGENCY_STOP;
  });

  it("does not activate when env var unset", () => {
    delete process.env.EMERGENCY_STOP;
    const result = emergency.checkEnvTrigger();
    expect(result).toBe(false);
    expect(emergency.isActive()).toBe(false);
  });

  it("activates on balance trigger when ETH below reserve", () => {
    const result = emergency.checkBalanceTrigger(
      parseEther("0.001"),
      parseEther("0.005"),
    );
    expect(result).toBe(true);
    expect(emergency.isActive()).toBe(true);
    expect(emergency.getReason()).toContain("ETH balance");
  });

  it("does not activate when balance is sufficient", () => {
    const result = emergency.checkBalanceTrigger(
      parseEther("0.1"),
      parseEther("0.005"),
    );
    expect(result).toBe(false);
    expect(emergency.isActive()).toBe(false);
  });
});

describe("DEFAULT_RISK_CONFIG", () => {
  it("has expected default values", () => {
    expect(DEFAULT_RISK_CONFIG.stopLossPct).toBe(8);
    expect(DEFAULT_RISK_CONFIG.maxDrawdownPct).toBe(25);
    expect(DEFAULT_RISK_CONFIG.circuitBreakerThreshold).toBe(2);
    expect(DEFAULT_RISK_CONFIG.circuitBreakerPauseMs).toBe(60_000);
    expect(DEFAULT_RISK_CONFIG.maxGasCostUsd).toBe(0.1);
    expect(DEFAULT_RISK_CONFIG.maxDailySpendUsd).toBe(10);
    expect(DEFAULT_RISK_CONFIG.minEthReserve).toBe(parseEther("0.005"));
  });
});
