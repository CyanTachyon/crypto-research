export class VolatilityTracker {
  private prices: bigint[] = [];
  private readonly maxSamples: number;

  constructor(maxSamples = 360) {
    this.maxSamples = maxSamples;
  }

  addPrice(price: bigint): void {
    this.prices.push(price);
    if (this.prices.length > this.maxSamples) {
      this.prices.shift();
    }
  }

  /** Returns mean absolute % change (e.g. 0.15 = 0.15%). null if < 30 samples. */
  getVolatility(): number | null {
    if (this.prices.length < 30) return null;

    let sumPctChange = 0;
    for (let i = 1; i < this.prices.length; i++) {
      const diff =
        this.prices[i] > this.prices[i - 1]
          ? this.prices[i] - this.prices[i - 1]
          : this.prices[i - 1] - this.prices[i];
      const pctChange = Number((diff * 10000n) / this.prices[i - 1]) / 100;
      sumPctChange += pctChange;
    }
    return sumPctChange / (this.prices.length - 1);
  }

  reset(): void {
    this.prices = [];
  }
}
