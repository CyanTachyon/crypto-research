/**
 * BigInt-only price math for grid trading. All prices USDC 6-decimal.
 * No floating point — uses scaled integer arithmetic (1e18).
 */

const SCALE = 10n ** 18n;
const USDC_DECIMALS = 6;

export function toUsdcValue(amount: bigint, tokenDecimals: number): bigint {
  if (tokenDecimals > USDC_DECIMALS) {
    return amount / 10n ** BigInt(tokenDecimals - USDC_DECIMALS);
  }
  if (tokenDecimals < USDC_DECIMALS) {
    return amount * 10n ** BigInt(USDC_DECIMALS - tokenDecimals);
  }
  return amount;
}

export function fromUsdcValue(usdcAmount: bigint, tokenDecimals: number): bigint {
  if (tokenDecimals > USDC_DECIMALS) {
    return usdcAmount * 10n ** BigInt(tokenDecimals - USDC_DECIMALS);
  }
  if (tokenDecimals < USDC_DECIMALS) {
    return usdcAmount / 10n ** BigInt(USDC_DECIMALS - tokenDecimals);
  }
  return usdcAmount;
}

function bigintPow(base: bigint, exp: number): bigint {
  if (exp === 0) return 1n;
  let result = 1n;
  for (let i = 0; i < exp; i++) {
    result *= base;
  }
  return result;
}

/**
 * BigInt integer nth-root via Newton's method: x' = ((n-1)*x + A/x^(n-1)) / n
 */
function bigintNthRoot(value: bigint, n: number): bigint {
  if (n <= 0) throw new Error("n must be positive");
  if (n === 1) return value;
  if (value <= 1n) return value;

  const bitLen = BigInt(value.toString(2).length);
  let x = 1n << (bitLen / BigInt(n) + 1n);

  const bn = BigInt(n);
  const bn1 = BigInt(n - 1);

  for (;;) {
    const xk1 = (bn1 * x + value / bigintPow(x, n - 1)) / bn;
    if (xk1 >= x) break;
    x = xk1;
  }
  return x;
}

/**
 * price_i = lower * (upper/lower)^(i/(N-1)) via scaled arithmetic.
 * r_scaled = nthRoot(upper * SCALE^n / lower, n); price_i = prev * r_scaled / SCALE
 */
export function calcGeometricSpacing(
  lowerPrice: bigint,
  upperPrice: bigint,
  levels: number,
): bigint[] {
  if (levels <= 0) return [];
  if (levels === 1) return [lowerPrice];
  if (lowerPrice === upperPrice) return Array(levels).fill(lowerPrice);

  const n = levels - 1;
  const inner = (upperPrice * bigintPow(SCALE, n)) / lowerPrice;
  const rScaled = bigintNthRoot(inner, n);

  const result: bigint[] = [lowerPrice];
  for (let i = 1; i < levels; i++) {
    result.push((result[i - 1] * rScaled) / SCALE);
  }

  result[levels - 1] = upperPrice;
  return result;
}

/**
 * Symmetric pct difference in basis points (100 = 1%).
 * |a-b| * 20000 / (a+b) — avoids midpoint precision loss.
 */
export function pctDifference(a: bigint, b: bigint): bigint {
  if (a === b) return 0n;
  const diff = a > b ? a - b : b - a;
  const sum = a + b;
  if (sum === 0n) return 0n;
  return (diff * 20000n) / sum;
}
