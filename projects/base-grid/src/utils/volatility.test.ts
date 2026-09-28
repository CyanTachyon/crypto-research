import { describe, it, expect } from "vitest";
import { VolatilityTracker } from "./volatility.js";

describe("VolatilityTracker", () => {
  it("returns null with fewer than 30 prices", () => {
    const tracker = new VolatilityTracker();
    for (let i = 0; i < 29; i++) {
      tracker.addPrice(1_000_000n);
    }
    expect(tracker.getVolatility()).toBeNull();
  });

  it("returns 0 for constant prices", () => {
    const tracker = new VolatilityTracker();
    for (let i = 0; i < 100; i++) {
      tracker.addPrice(1_000_000n);
    }
    expect(tracker.getVolatility()).toBe(0);
  });

  it("returns correct volatility for known alternating series", () => {
    const tracker = new VolatilityTracker(360);
    const base = 1_000_000n;
    for (let i = 0; i < 40; i++) {
      tracker.addPrice(i % 2 === 0 ? base : (base * 101n) / 100n);
    }
    const vol = tracker.getVolatility();
    expect(vol).not.toBeNull();
    expect(vol!).toBeCloseTo(1.0, 1);
  });

  it("rolling window drops oldest prices", () => {
    const tracker = new VolatilityTracker(35);
    const base = 1_000_000n;
    for (let i = 0; i < 35; i++) {
      tracker.addPrice(base);
    }
    expect(tracker.getVolatility()).toBe(0);

    for (let i = 0; i < 35; i++) {
      tracker.addPrice((base * 110n) / 100n);
    }
    const vol = tracker.getVolatility();
    expect(vol).not.toBeNull();
    expect(vol!).toBe(0);
  });

  it("addPrice and reset work correctly", () => {
    const tracker = new VolatilityTracker();
    for (let i = 0; i < 50; i++) {
      tracker.addPrice(1_000_000n + BigInt(i) * 1000n);
    }
    expect(tracker.getVolatility()).not.toBeNull();

    tracker.reset();
    expect(tracker.getVolatility()).toBeNull();

    for (let i = 0; i < 29; i++) {
      tracker.addPrice(1_000_000n);
    }
    expect(tracker.getVolatility()).toBeNull();
  });
});
