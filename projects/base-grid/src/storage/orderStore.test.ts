import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import {
  serialize,
  deserialize,
  createDefaultState,
  loadState,
  saveState,
  updateGridLevel,
  recordFill,
  setPendingTx,
  clearPendingTx,
  recoverPendingTx,
} from "./orderStore.js";
import type { BotState, GridState } from "../types.js";

let testDir: string;
let statePath: string;

function makeGridState(pair: string): GridState {
  return {
    pair,
    centerPrice: "0xde0b6b3a7640000", // 1e18
    levels: [
      { price: "0xde0b6b3a7640000", buyAmount: "0x8ac7230489e80000", sellAmount: "0x0", filled: false, side: "none" },
      { price: "0x1bc16d674ec80000", buyAmount: "0x0", sellAmount: "0x8ac7230489e80000", filled: false, side: "none" },
    ],
    config: { gridLevels: 2, rangePct: 10, capitalUsd: 1000 },
    fillHistory: [],
  };
}

function makeStateWithGrid(): BotState {
  const state = createDefaultState();
  state.grids["ETH/USDC"] = makeGridState("ETH/USDC");
  state.lastBlockNumber = "0x100";
  return state;
}

beforeEach(async () => {
  testDir = await mkdtemp(join(tmpdir(), "orderstore-test-"));
  statePath = join(testDir, "state.json");
});

afterEach(async () => {
  await rm(testDir, { recursive: true, force: true });
});

describe("BigInt serialization", () => {
  it("serializes bigint to hex string", () => {
    const state = createDefaultState();
    const json = serialize(state);
    expect(json).toContain('"0x0"');
    const parsed = JSON.parse(json);
    expect(parsed.lastBlockNumber).toBe("0x0");
  });

  it("deserializes preserving hex strings", () => {
    const original = createDefaultState();
    original.lastBlockNumber = "0xff";
    const json = serialize(original);
    const restored = deserialize(json);
    expect(restored.lastBlockNumber).toBe("0xff");
  });

  it("roundtrips 0x0 correctly", () => {
    const state = createDefaultState();
    const json = serialize(state);
    const restored = deserialize(json);
    expect(restored.lastBlockNumber).toBe("0x0");
  });

  it("roundtrips very large numbers", () => {
    const large = 115792089237316195423570985008687907853269984665640564039457584007913129639935n;
    const hexLarge = `0x${large.toString(16)}`;
    const state = createDefaultState();
    state.lastBlockNumber = hexLarge;
    const json = serialize(state);
    const restored = deserialize(json);
    expect(restored.lastBlockNumber).toBe(hexLarge);
  });

  it("does not mangle non-hex strings starting with 0x that have non-hex chars", () => {
    const json = '{"val": "0xgghi"}';
    const restored = deserialize(json);
    expect((restored as unknown as Record<string, unknown>).val).toBe("0xgghi");
  });

  it("handles single hex digit values", () => {
    const json = '{"val": "0xa"}';
    const restored = deserialize(json);
    expect((restored as unknown as Record<string, unknown>).val).toBe("0xa");
  });
});

describe("createDefaultState", () => {
  it("returns a valid default state", () => {
    const state = createDefaultState();
    expect(state.version).toBe(1);
    expect(state.grids).toEqual({});
    expect(state.pendingTx).toBeNull();
    expect(state.lastBlockNumber).toBe("0x0");
    expect(state.lastUpdated).toBeGreaterThan(0);
  });
});

describe("saveState + loadState", () => {
  it("roundtrips state preserving all data", async () => {
    const original = makeStateWithGrid();
    await saveState(statePath, original);
    const loaded = await loadState(statePath);

    expect(loaded.version).toBe(original.version);
    expect(loaded.lastBlockNumber).toBe(original.lastBlockNumber);
    expect(loaded.grids["ETH/USDC"].pair).toBe("ETH/USDC");
    expect(loaded.grids["ETH/USDC"].levels).toHaveLength(2);
    expect(loaded.grids["ETH/USDC"].levels[0].price).toBe("0xde0b6b3a7640000");
    expect(loaded.pendingTx).toBeNull();
  });

  it("creates tmp file during write then renames", async () => {
    const state = createDefaultState();
    await saveState(statePath, state);

    const tmpPath = `${statePath}.tmp`;
    const { existsSync } = await import("node:fs");
    expect(existsSync(tmpPath)).toBe(false);
    expect(existsSync(statePath)).toBe(true);
  });

  it("multiple consecutive saves do not corrupt state", async () => {
    const state1 = makeStateWithGrid();
    await saveState(statePath, state1);

    const state2 = { ...state1, lastBlockNumber: "0x200" };
    await saveState(statePath, state2);

    const state3 = { ...state2, lastBlockNumber: "0x300" };
    await saveState(statePath, state3);

    const loaded = await loadState(statePath);
    expect(loaded.lastBlockNumber).toBe("0x300");
    expect(loaded.grids["ETH/USDC"].levels).toHaveLength(2);
  });

  it("loadState returns default when file does not exist", async () => {
    const loaded = await loadState(join(testDir, "nonexistent.json"));
    expect(loaded.version).toBe(1);
    expect(loaded.grids).toEqual({});
  });

  it("creates parent directories if missing", async () => {
    const deepPath = join(testDir, "a", "b", "c", "state.json");
    const state = createDefaultState();
    await saveState(deepPath, state);
    const loaded = await loadState(deepPath);
    expect(loaded.version).toBe(state.version);
  });
});

describe("updateGridLevel", () => {
  it("updates a specific level immutably", () => {
    const state = makeStateWithGrid();
    const updated = updateGridLevel(state, "ETH/USDC", 0, { filled: true, side: "buy" });

    expect(updated.grids["ETH/USDC"].levels[0].filled).toBe(true);
    expect(updated.grids["ETH/USDC"].levels[0].side).toBe("buy");
    expect(state.grids["ETH/USDC"].levels[0].filled).toBe(false);
  });

  it("throws for unknown pair", () => {
    const state = makeStateWithGrid();
    expect(() => updateGridLevel(state, "BTC/USDC", 0, { filled: true })).toThrow(
      "Grid not found for pair: BTC/USDC",
    );
  });

  it("throws for out-of-range level index", () => {
    const state = makeStateWithGrid();
    expect(() => updateGridLevel(state, "ETH/USDC", 99, { filled: true })).toThrow(
      "Level index 99 out of range",
    );
  });
});

describe("recordFill", () => {
  it("adds fill to history and marks level as filled", () => {
    const state = makeStateWithGrid();
    const updated = recordFill(state, "ETH/USDC", 0, {
      side: "buy",
      price: "0xde0b6b3a7640000",
      amount: "0x8ac7230489e80000",
      txHash: "0xabc123",
    });

    expect(updated.grids["ETH/USDC"].fillHistory).toHaveLength(1);
    expect(updated.grids["ETH/USDC"].fillHistory[0].side).toBe("buy");
    expect(updated.grids["ETH/USDC"].fillHistory[0].levelIndex).toBe(0);
    expect(updated.grids["ETH/USDC"].fillHistory[0].txHash).toBe("0xabc123");
    expect(updated.grids["ETH/USDC"].levels[0].filled).toBe(true);
    expect(updated.grids["ETH/USDC"].levels[0].side).toBe("buy");
  });

  it("accumulates multiple fills", () => {
    let state = makeStateWithGrid();
    state = recordFill(state, "ETH/USDC", 0, {
      side: "buy",
      price: "0xde0b6b3a7640000",
      amount: "0x8ac7230489e80000",
    });
    state = recordFill(state, "ETH/USDC", 1, {
      side: "sell",
      price: "0x1bc16d674ec80000",
      amount: "0x8ac7230489e80000",
    });

    expect(state.grids["ETH/USDC"].fillHistory).toHaveLength(2);
    expect(state.grids["ETH/USDC"].levels[0].filled).toBe(true);
    expect(state.grids["ETH/USDC"].levels[1].filled).toBe(true);
  });
});

describe("pendingTx", () => {
  it("setPendingTx sets the pending transaction", () => {
    const state = createDefaultState();
    const updated = setPendingTx(state, "0xdeadbeef");
    expect(updated.pendingTx).not.toBeNull();
    expect(updated.pendingTx!.txHash).toBe("0xdeadbeef");
    expect(updated.pendingTx!.submittedAt).toBeGreaterThan(0);
  });

  it("clearPendingTx clears the pending transaction", () => {
    const state = setPendingTx(createDefaultState(), "0xdeadbeef");
    expect(state.pendingTx).not.toBeNull();
    const cleared = clearPendingTx(state);
    expect(cleared.pendingTx).toBeNull();
  });

  it("set/clear cycle preserves other state", () => {
    const state = makeStateWithGrid();
    const withPending = setPendingTx(state, "0xabc");
    const cleared = clearPendingTx(withPending);
    expect(cleared.grids["ETH/USDC"]).toBeDefined();
    expect(cleared.lastBlockNumber).toBe(state.lastBlockNumber);
  });

  it("pendingTx persists through save/load", async () => {
    const state = setPendingTx(makeStateWithGrid(), "0xtxhash123");
    await saveState(statePath, state);
    const loaded = await loadState(statePath);
    expect(loaded.pendingTx).not.toBeNull();
    expect(loaded.pendingTx!.txHash).toBe("0xtxhash123");
  });
});

describe("recoverPendingTx", () => {
  it("returns null when no state file exists", async () => {
    const result = await recoverPendingTx(join(testDir, "none.json"), async () => {});
    expect(result).toBeNull();
  });

  it("returns null when no pending tx", async () => {
    await saveState(statePath, createDefaultState());
    const result = await recoverPendingTx(statePath, async () => {});
    expect(result).toBeNull();
  });

  it("calls recovery callback with tx details and returns state", async () => {
    const state = setPendingTx(makeStateWithGrid(), "0xrecoverme");
    await saveState(statePath, state);

    let capturedHash: string | undefined;
    let capturedAt: number | undefined;
    const result = await recoverPendingTx(statePath, async (hash, submittedAt) => {
      capturedHash = hash;
      capturedAt = submittedAt;
    });

    expect(result).not.toBeNull();
    expect(capturedHash).toBe("0xrecoverme");
    expect(capturedAt).toBeGreaterThan(0);
  });
});
