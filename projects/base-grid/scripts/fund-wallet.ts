/**
 * Fund Wallet Script — displays transfer instructions and verifies balances.
 * Usage: npx tsx scripts/fund-wallet.ts
 */
import { createPublicClient, http, formatEther, formatUnits } from "viem";
import { base } from "viem/chains";
import { existsSync, readFileSync } from "node:fs";
import { dirname } from "node:path";
import { createInterface } from "node:readline";
import { privateKeyToAccount } from "viem/accounts";
import { TOKENS } from "../src/config/chains.js";
import { erc20Abi } from "../src/abis/erc20.js";

const rl = createInterface({ input: process.stdin, output: process.stdout });
const question = (prompt: string) => new Promise<string>((res) => rl.question(prompt, res));

function getProjectRoot(): string {
  return dirname(new URL(import.meta.url).pathname);
}

async function main(): Promise<void> {
  const envPath = `${getProjectRoot()}/.env`;

  if (!existsSync(envPath)) {
    console.error("No .env file found. Run setup-wallet.ts first.");
    process.exit(1);
  }

  const envContent = readFileSync(envPath, "utf-8");
  const match = envContent.match(/PRIVATE_KEY=(0x[0-9a-fA-F]{64})/);
  if (!match) {
    console.error("PRIVATE_KEY not found in .env. Run setup-wallet.ts first.");
    process.exit(1);
  }

  const account = privateKeyToAccount(match[1] as `0x${string}`);

  console.log("=== Fund Your Bot Wallet ===\n");
  console.log("Please transfer from your main wallet:");
  console.log("  - ~$50 USDC (or equivalent in ETH/CBBTC)");
  console.log("  - 0.01 ETH (for gas fees)");
  console.log(`  To: ${account.address}`);
  console.log("");

  await question("Press ENTER once you have completed the transfer...");

  const publicClient = createPublicClient({
    chain: base,
    transport: http("https://mainnet.base.org"),
  });

  console.log("\nChecking balances...");

  const ethBalance = await publicClient.getBalance({ address: account.address });
  console.log(`  ETH:  ${formatEther(ethBalance)} ETH`);

  const usdcBalance = (await publicClient.readContract({
    address: TOKENS.USDC.address,
    abi: erc20Abi,
    functionName: "balanceOf",
    args: [account.address],
  })) as bigint;
  console.log(`  USDC: ${formatUnits(usdcBalance, TOKENS.USDC.decimals)} USDC`);

  if (ethBalance === 0n && usdcBalance === 0n) {
    console.log("\n  ⚠ No funds detected yet. Confirm the transfer completed and re-run.");
  } else {
    console.log("\n  ✓ Funds detected — wallet is ready.");
  }

  rl.close();
}

main().catch((err) => {
  console.error("Fund check failed:", err);
  process.exit(1);
});
