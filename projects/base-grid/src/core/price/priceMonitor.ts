import type { PriceFeed } from "./priceFeed.js";

const MAX_HISTORY = 100;

export class PriceMonitor {
  private intervalMs: number;
  private primaryFeed: PriceFeed;
  private fallbackFeed: PriceFeed;
  private running = false;
  private timeoutId: ReturnType<typeof setTimeout> | null = null;
  private history = new Map<string, bigint[]>();

  constructor(config: {
    intervalMs: number;
    primaryFeed: PriceFeed;
    fallbackFeed: PriceFeed;
  }) {
    this.intervalMs = config.intervalMs;
    this.primaryFeed = config.primaryFeed;
    this.fallbackFeed = config.fallbackFeed;
  }

  async start(
    onPrice: (pair: string, price: bigint) => Promise<void>,
    pairs: string[] = ["ETH/USDC"],
  ): Promise<void> {
    this.running = true;
    const loop = async () => {
      if (!this.running) return;

      const tickStart = Date.now();

      for (const pair of pairs) {
        try {
          const price = await this.fetchPrice(pair);
          this.pushHistory(pair, price);
          await onPrice(pair, price);
        } catch {
          // swallow — will retry next tick
        }
      }

      const elapsed = Date.now() - tickStart;
      const remaining = Math.max(0, this.intervalMs - elapsed);

      if (this.running) {
        this.timeoutId = setTimeout(loop, remaining);
      }
    };

    await loop();
  }

  stop(): void {
    this.running = false;
    if (this.timeoutId !== null) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }

  getPriceHistory(pair: string): bigint[] {
    return this.history.get(pair)?.slice() ?? [];
  }

  private async fetchPrice(pair: string): Promise<bigint> {
    try {
      return await this.primaryFeed.getPrice(pair);
    } catch {
      return await this.fallbackFeed.getPrice(pair);
    }
  }

  private pushHistory(pair: string, price: bigint): void {
    const arr = this.history.get(pair) ?? [];
    arr.push(price);
    if (arr.length > MAX_HISTORY) arr.shift();
    this.history.set(pair, arr);
  }
}
