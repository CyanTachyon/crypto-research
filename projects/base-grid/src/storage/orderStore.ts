import { randomUUID } from "node:crypto";
import { readFile, writeFile, rename, unlink, readdir, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import type {
  BotState,
  GridLevelState,
  FillRecord,
} from "../types.js";

export type { BotState, GridState, GridLevelState, FillRecord } from "../types.js";

const STATE_VERSION = 1;
const HEX_PREFIX = "0x";

function bigintToHex(value: bigint): string {
  return `${HEX_PREFIX}${value.toString(16)}`;
}

function hexToBigInt(value: string): bigint {
  return BigInt(value);
}

function replacer(_key: string, value: unknown): unknown {
  if (typeof value === "bigint") {
    return bigintToHex(value);
  }
  return value;
}

export function serialize(state: BotState): string {
  return JSON.stringify(state, replacer, 2);
}

export function deserialize(json: string): BotState {
  return JSON.parse(json) as BotState;
}

export function hexToBigIntSafe(hex: string): bigint {
  return BigInt(hex);
}

export function createDefaultState(): BotState {
  return {
    version: STATE_VERSION,
    lastBlockNumber: bigintToHex(0n),
    lastUpdated: Date.now(),
    grids: {},
    pendingTx: null,
  };
}

export async function loadState(filePath: string): Promise<BotState> {
  if (!existsSync(filePath)) {
    return createDefaultState();
  }
  try {
    const raw = await readFile(filePath, "utf-8");
    return deserialize(raw);
  } catch {
    return createDefaultState();
  }
}

export async function saveState(filePath: string, state: BotState): Promise<void> {
  const dir = dirname(filePath);
  if (!existsSync(dir)) {
    await mkdir(dir, { recursive: true });
  }

  const tmpPath = `${filePath}.${randomUUID()}.tmp`;
  const data = serialize(state);
  await writeFile(tmpPath, data, "utf-8");
  await rename(tmpPath, filePath);

  try {
    const dirFiles = await readdir(dir);
    for (const f of dirFiles) {
      if (f.endsWith(".tmp")) {
        await unlink(join(dir, f)).catch(() => {});
      }
    }
  } catch {
    // best-effort cleanup
  }
}

export function updateGridLevel(
  state: BotState,
  pair: string,
  levelIndex: number,
  updates: Partial<GridLevelState>,
): BotState {
  const grid = state.grids[pair];
  if (!grid) {
    throw new Error(`Grid not found for pair: ${pair}`);
  }
  if (levelIndex < 0 || levelIndex >= grid.levels.length) {
    throw new Error(`Level index ${levelIndex} out of range for pair: ${pair}`);
  }

  const updatedLevels = grid.levels.map((level, i) =>
    i === levelIndex ? { ...level, ...updates } : level,
  );

  return {
    ...state,
    grids: {
      ...state.grids,
      [pair]: {
        ...grid,
        levels: updatedLevels,
      },
    },
  };
}

export function recordFill(
  state: BotState,
  pair: string,
  levelIndex: number,
  fill: Omit<FillRecord, "timestamp" | "levelIndex">,
): BotState {
  const grid = state.grids[pair];
  if (!grid) {
    throw new Error(`Grid not found for pair: ${pair}`);
  }

  const record: FillRecord = {
    ...fill,
    timestamp: Date.now(),
    levelIndex,
  };

  const updatedGrid = {
    ...grid,
    fillHistory: [...grid.fillHistory, record],
  };

  let result: BotState = {
    ...state,
    grids: {
      ...state.grids,
      [pair]: updatedGrid,
    },
  };

  result = updateGridLevel(result, pair, levelIndex, {
    filled: true,
    side: fill.side,
  });

  return result;
}

export function setPendingTx(state: BotState, txHash: string): BotState {
  return {
    ...state,
    pendingTx: {
      txHash,
      submittedAt: Date.now(),
    },
  };
}

export function clearPendingTx(state: BotState): BotState {
  return {
    ...state,
    pendingTx: null,
  };
}

export async function recoverPendingTx(
  filePath: string,
  onRecovery: (txHash: string, submittedAt: number) => Promise<void>,
): Promise<BotState | null> {
  if (!existsSync(filePath)) {
    return null;
  }

  const state = await loadState(filePath);
  if (!state.pendingTx) {
    return null;
  }

  await onRecovery(state.pendingTx.txHash, state.pendingTx.submittedAt);
  return state;
}
