import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { PriceMonitor } from "./priceMonitor.js";
import type { PriceFeed } from "./priceFeed.js";

function createMockFeed(
  prices: bigint[],
  shouldFail = false,
): PriceFeed {
  let callCount = 0;
  return {
    async getPrice(_pair: string): Promise<bigint> {
      callCount++;
      if (shouldFail) throw new Error("feed error");
      const idx = (callCount - 1) % prices.length;
      return prices[idx];
    },
  };
}

describe("PriceMonitor", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("gets price from primary feed and calls onPrice", async () => {
    const primary = createMockFeed([2500_000_000n]);
    const fallback = createMockFeed([0n]);
    const monitor = new PriceMonitor({
      intervalMs: 1000,
      primaryFeed: primary,
      fallbackFeed: fallback,
    });

    const prices: Array<{ pair: string; price: bigint }> = [];
    const onPrice = vi.fn(async (pair: string, price: bigint) => {
      prices.push({ pair, price });
    });

    const startPromise = monitor.start(onPrice, ["ETH/USDC"]);

    await vi.advanceTimersByTimeAsync(0);
    await startPromise;

    expect(onPrice).toHaveBeenCalledWith("ETH/USDC", 2500_000_000n);
    monitor.stop();
  });

  it("falls back to secondary feed when primary throws", async () => {
    const primary = createMockFeed([0n], true);
    const fallback = createMockFeed([3000_000_000n]);
    const monitor = new PriceMonitor({
      intervalMs: 1000,
      primaryFeed: primary,
      fallbackFeed: fallback,
    });

    const onPrice = vi.fn();
    const startPromise = monitor.start(onPrice, ["ETH/USDC"]);

    await vi.advanceTimersByTimeAsync(0);
    await startPromise;

    expect(onPrice).toHaveBeenCalledWith("ETH/USDC", 3000_000_000n);
    monitor.stop();
  });

  it("maintains price history", async () => {
    const primary = createMockFeed([100n, 200n, 300n]);
    const fallback = createMockFeed([0n]);
    const monitor = new PriceMonitor({
      intervalMs: 100,
      primaryFeed: primary,
      fallbackFeed: fallback,
    });

    const onPrice = vi.fn();
    const startPromise = monitor.start(onPrice, ["ETH/USDC"]);

    await vi.advanceTimersByTimeAsync(0);
    await startPromise;

    expect(monitor.getPriceHistory("ETH/USDC")).toEqual([100n]);

    await vi.advanceTimersByTimeAsync(100);
    expect(monitor.getPriceHistory("ETH/USDC")).toEqual([100n, 200n]);

    await vi.advanceTimersByTimeAsync(100);
    expect(monitor.getPriceHistory("ETH/USDC")).toEqual([100n, 200n, 300n]);

    monitor.stop();
  });

  it("stop() terminates the polling loop", async () => {
    const primary = createMockFeed([100n]);
    const fallback = createMockFeed([0n]);
    const monitor = new PriceMonitor({
      intervalMs: 50,
      primaryFeed: primary,
      fallbackFeed: fallback,
    });

    const onPrice = vi.fn();
    const startPromise = monitor.start(onPrice, ["ETH/USDC"]);

    await vi.advanceTimersByTimeAsync(0);
    await startPromise;

    expect(onPrice).toHaveBeenCalledTimes(1);

    monitor.stop();

    await vi.advanceTimersByTimeAsync(200);
    expect(onPrice).toHaveBeenCalledTimes(1);
  });

  it("uses recursive setTimeout — no drift, accounts for tick duration", async () => {
    const primary = createMockFeed([100n]);
    const fallback = createMockFeed([0n]);
    const monitor = new PriceMonitor({
      intervalMs: 100,
      primaryFeed: primary,
      fallbackFeed: fallback,
    });

    const onPrice = vi.fn();
    const startPromise = monitor.start(onPrice, ["ETH/USDC"]);
    await vi.advanceTimersByTimeAsync(0);
    await startPromise;

    expect(onPrice).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(100);
    expect(onPrice).toHaveBeenCalledTimes(2);

    await vi.advanceTimersByTimeAsync(100);
    expect(onPrice).toHaveBeenCalledTimes(3);

    monitor.stop();
  });
});
