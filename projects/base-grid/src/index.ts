// PM2 Process Manager (optional, for production):
//   pm2 start "npx tsx src/index.ts -- --mode=live" --name grid-bot
//   pm2 logs grid-bot
//   pm2 stop grid-bot
//   pm2 restart grid-bot
//
// Environment via .env file or PM2 ecosystem:
//   PRIVATE_KEY=0x...
//   MODE=live
//   TELEGRAM_BOT_TOKEN=123456:ABC (optional)
//   TELEGRAM_CHAT_ID=123456789 (optional)

import { createPublicClient, http, formatEther, formatUnits } from "viem";
import { base } from "viem/chains";
import type { Address, PublicClient, WalletClient } from "viem";
import type { PriceFeed } from "./core/price/priceFeed.js";
import type { SwapExecutor } from "./core/swap/swapExecutor.js";
import { config } from "./config/index.js";
import { TOKENS, CONTRACTS } from "./config/chains.js";
import { erc20Abi } from "./abis/erc20.js";
import { quoterV2Abi } from "./abis/quoterV2.js";
import { EmergencyStop } from "./risk/emergencyStop.js";
import { GridBot, DEFAULT_BOT_CONFIG, type BotConfig } from "./bot.js";
import { Notifier, type NotifyConfig } from "./utils/notify.js";
import pino from "pino";
import { createWriteStream } from "node:fs";
import { mkdirSync } from "node:fs";

// ---------------------------------------------------------------------------
// Sim-mode price feeds (no PRIVATE_KEY dependency)
// ---------------------------------------------------------------------------

const PAIR_TOKEN_MAP: Record<string, { base: Address; quote: Address; fee: number }> = {
  "ETH/USDC": {
    base: TOKENS.WETH.address,
    quote: TOKENS.USDC.address,
    fee: 100,
  },
  "CBBTC/USDC": {
    base: TOKENS.CBBTC.address,
    quote: TOKENS.USDC.address,
    fee: 100,
  },
};

class SimOnChainPriceFeed implements PriceFeed {
  private pairConfig = PAIR_TOKEN_MAP;
  private client = createPublicClient({
    chain: base,
    transport: http("https://mainnet.base.org"),
  });

  async getPrice(pair: string): Promise<bigint> {
    const cfg = this.pairConfig[pair];
    if (!cfg) throw new Error(`Unknown pair: ${pair}`);

    const { result } = await this.client.simulateContract({
      address: CONTRACTS.UNISWAP_V3_QUOTER_V2,
      abi: quoterV2Abi,
      functionName: "quoteExactInputSingle",
      args: [
        {
          tokenIn: cfg.quote,
          tokenOut: cfg.base,
          amountIn: 1_000_000n,
          fee: cfg.fee,
          sqrtPriceLimitX96: 0n,
        },
      ],
    });

    const [amountOut] = result as [bigint, bigint, number, bigint];

    // Invert: price = 1 USDC * scale * 1e6 / amountOut
    const tokenInfo =
      pair === "ETH/USDC" ? TOKENS.WETH : pair === "CBBTC/USDC" ? TOKENS.CBBTC : null;
    if (!tokenInfo) throw new Error(`Unknown pair: ${pair}`);
    const decimalShift = BigInt(tokenInfo.decimals - 6);
    return (1_000_000n * 10n ** decimalShift * 1_000_000n) / amountOut;
  }
}

class DefiLlamaFallbackFeed implements PriceFeed {
  private tokenMap: Record<string, string> = {
    "ETH/USDC": TOKENS.WETH.address,
    "CBBTC/USDC": TOKENS.CBBTC.address,
  };

  async getPrice(pair: string): Promise<bigint> {
    const address = this.tokenMap[pair];
    if (!address) throw new Error(`Unknown pair: ${pair}`);

    const url = `https://api.llama.fi/prices/current/base:${address}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`DefiLlama API error: ${res.status}`);

    const json = (await res.json()) as { coins: Record<string, { price: number }> };
    const priceData = json.coins[`base:${address}`];
    if (!priceData || priceData.price === undefined) {
      throw new Error(`No price from DefiLlama for ${pair}`);
    }
    return BigInt(Math.floor(priceData.price * 1_000_000));
  }
}

/** Composite feed: tries on-chain first, falls back to DefiLlama */
class CompositePriceFeed implements PriceFeed {
  constructor(
    private primary: PriceFeed,
    private fallback: PriceFeed,
  ) {}

  async getPrice(pair: string): Promise<bigint> {
    try {
      return await this.primary.getPrice(pair);
    } catch {
      return await this.fallback.getPrice(pair);
    }
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const mode = config.MODE;

  // Ensure data directory exists
  mkdirSync("data", { recursive: true });

  // Multi-destination logger: console (pretty) + file (JSON for analysis)
  const logFile = `data/bot-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}.log`;
  const fileStream = createWriteStream(logFile, { flags: "a" });

  const logger = pino(
    { level: "info" },
    pino.multistream([
      { level: "info", stream: pino.transport({ target: "pino-pretty" }) },
      { level: "info", stream: fileStream },
    ]),
  );

  logger.info({ logFile }, "Log file created");
  logger.info({ mode }, "Grid trading bot starting");

  const emergencyStop = new EmergencyStop();

  let priceFeed: PriceFeed;
  let swapExecutor: SwapExecutor | null = null;
  let notifier: Notifier | null = null;

  const telegramToken = process.env.TELEGRAM_BOT_TOKEN;
  const telegramChatId = process.env.TELEGRAM_CHAT_ID;
  if (telegramToken && telegramChatId) {
    const notifyConfig: NotifyConfig = {
      telegramBotToken: telegramToken,
      telegramChatId: telegramChatId,
      throttleMs: 300_000,
    };
    notifier = new Notifier(notifyConfig);
    logger.info("Telegram notifications enabled");
  }

  if (mode === "sim") {
    const onChainFeed = new SimOnChainPriceFeed();
    const fallbackFeed = new DefiLlamaFallbackFeed();
    priceFeed = new CompositePriceFeed(onChainFeed, fallbackFeed);
  } else {
    if (!config.PRIVATE_KEY) {
      logger.error("PRIVATE_KEY is required for live mode. Set it in .env");
      process.exit(1);
    }

    const { publicClient: pubClient, walletClient } = await import("./clients/rpc.js");
    if (!walletClient) {
      logger.error("walletClient not initialized — check PRIVATE_KEY");
      process.exit(1);
    }
    const { OnChainPriceFeed, DefiLlamaPriceFeed } = await import("./core/price/priceFeed.js");
    const SwapModule = await import("./core/swap/swapExecutor.js");
    const { getQuote, approveToken, executeSwap } = await import("./core/swap/aerodrome.js");
    const { canTrade } = await import("./risk/riskManager.js");

    priceFeed = new CompositePriceFeed(new OnChainPriceFeed(), new DefiLlamaPriceFeed());

    // base-chain PublicClient has deposit tx types absent from generic PublicClient
    swapExecutor = new SwapModule.SwapExecutor(
      pubClient as unknown as PublicClient,
      walletClient as unknown as WalletClient,
      getQuote,
      approveToken,
      executeSwap,
      canTrade,
      (update) => logger.info({ update }, "Swap state update"),
    );

    const walletAddress = walletClient.account.address;

    logger.info({ wallet: walletAddress }, "Running pre-flight safety checks");

    const ethBalance = await pubClient.getBalance({ address: walletAddress });
    logger.info({ ethBalance: formatEther(ethBalance) }, "ETH balance");
    if (ethBalance < toWei("0.005")) {
      logger.error({ ethBalance: formatEther(ethBalance) }, "ETH balance below 0.005 ETH gas reserve — aborting");
      process.exit(1);
    }

    const usdcBalance = (await pubClient.readContract({
      address: TOKENS.USDC.address,
      abi: erc20Abi,
      functionName: "balanceOf",
      args: [walletAddress],
    })) as bigint;
    logger.info({ usdcBalance: formatUnits(usdcBalance, TOKENS.USDC.decimals) }, "USDC balance");
    if (usdcBalance === 0n) {
      logger.error("USDC balance is 0 — no trading capital. Aborting.");
      process.exit(1);
    }

    const usdcAllowance = (await pubClient.readContract({
      address: TOKENS.USDC.address,
      abi: erc20Abi,
      functionName: "allowance",
      args: [walletAddress, CONTRACTS.AERODROME_ROUTER],
    })) as bigint;
    logger.info({ usdcAllowance: usdcAllowance.toString() }, "USDC allowance for Aerodrome Router");
    if (usdcAllowance === 0n) {
      logger.error("USDC not approved for Aerodrome Router — run setup-wallet.ts first. Aborting.");
      process.exit(1);
    }

    const wethAllowance = (await pubClient.readContract({
      address: TOKENS.WETH.address,
      abi: erc20Abi,
      functionName: "allowance",
      args: [walletAddress, CONTRACTS.AERODROME_ROUTER],
    })) as bigint;
    logger.info({ wethAllowance: wethAllowance.toString() }, "WETH allowance for Aerodrome Router");
    if (wethAllowance === 0n) {
      logger.error("WETH not approved for Aerodrome Router — run setup-wallet.ts first. Aborting.");
      process.exit(1);
    }

    logger.info("All pre-flight checks passed");
  }

  const PAIRS = ["CBBTC/USDC"] as const;

  const bots = PAIRS.map((pair) => {
    const safeFileName = pair.replace("/", "-");
    const pairConfig: BotConfig = {
      ...DEFAULT_BOT_CONFIG,
      mode,
      pair,
      stateFilePath: `data/grid-state-${safeFileName}.json`,
    };
    return new GridBot(pairConfig, priceFeed, emergencyStop, swapExecutor, logger.child({ pair }), undefined, notifier);
  });

  let shuttingDown = false;
  const shutdown = async (signal: string) => {
    if (shuttingDown) return;
    shuttingDown = true;
    await Promise.all(bots.map((b) => b.stop()));
    process.exit(0);
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));

  await Promise.all(bots.map((b) => b.start()));
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});

function toWei(eth: string): bigint {
  const [intPart = "0", fracPart = ""] = eth.split(".");
  return BigInt(intPart + fracPart.padEnd(18, "0").slice(0, 18));
}
