/**
 * Wallet Setup Script
 *
 * Generates a new wallet, displays address + private key,
 * offers to write PRIVATE_KEY to .env, waits for funding,
 * then verifies balances and sets token approvals.
 *
 * Usage: npx tsx scripts/setup-wallet.ts
 */
import { generatePrivateKey, privateKeyToAccount } from "viem/accounts";
import { createPublicClient, createWalletClient, http, formatEther, formatUnits, maxUint256 } from "viem";
import { base } from "viem/chains";
import { readFileSync, writeFileSync, existsSync, appendFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { createInterface } from "node:readline";
import { TOKENS, CONTRACTS } from "../src/config/chains.js";
import { erc20Abi } from "../src/abis/erc20.js";

const rl = createInterface({ input: process.stdin, output: process.stdout });

function question(prompt: string): Promise<string> {
  return new Promise((res) => rl.question(prompt, res));
}

function getProjectRoot(): string {
  return dirname(new URL(import.meta.url).pathname);
}

const ETH_RESERVE_MIN = toWei("0.005");

function toWei(eth: string): bigint {
  const [intPart = "0", fracPart = ""] = eth.split(".");
  return BigInt(intPart + fracPart.padEnd(18, "0").slice(0, 18));
}

async function main(): Promise<void> {
  console.log("=== Grid Bot Wallet Setup ===\n");

  const rawPrivateKey = generatePrivateKey();
  const account = privateKeyToAccount(rawPrivateKey);
  const prefixedKey: `0x${string}` = rawPrivateKey.startsWith("0x")
    ? (rawPrivateKey as `0x${string}`)
    : `0x${rawPrivateKey}`;

  console.log("Generated new wallet:");
  console.log(`  Address:     ${account.address}`);
  console.log("");
  console.log("  ⚠️  PRIVATE KEY (SAVE THIS SECURELY — it will NOT be shown again):");
  console.log(`  ${prefixedKey}`);
  console.log("");

  const envPath = resolve(getProjectRoot(), ".env");

  if (existsSync(envPath)) {
    const envContent = readFileSync(envPath, "utf-8");
    if (envContent.includes("PRIVATE_KEY=")) {
      console.log(".env already contains PRIVATE_KEY — skipping write.");
    } else {
      const answer = await question("Write PRIVATE_KEY to .env? [y/N]: ");
      if (answer.toLowerCase() === "y" || answer.toLowerCase() === "yes") {
        appendFileSync(envPath, `\nPRIVATE_KEY=${prefixedKey}\n`);
        console.log("PRIVATE_KEY appended to .env");
      }
    }
  } else {
    const answer = await question("Create .env with PRIVATE_KEY? [y/N]: ");
    if (answer.toLowerCase() === "y" || answer.toLowerCase() === "yes") {
      writeFileSync(envPath, `PRIVATE_KEY=${prefixedKey}\nBASE_RPC_URL=https://mainnet.base.org\n`);
      console.log(".env created with PRIVATE_KEY");
    }
  }

  const transport = http("https://mainnet.base.org");
  const publicClient = createPublicClient({ chain: base, transport });
  const walletClient = createWalletClient({ account, chain: base, transport });

  console.log("");
  console.log("Please send funds to your wallet:");
  console.log("  - At least 0.01 ETH (for gas fees)");
  console.log("  - USDC (trading capital, e.g. ~$50)");
  console.log(`  → To: ${account.address}`);
  console.log("");

  await question("Press ENTER once you have transferred funds...");

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

  let hasErrors = false;

  if (ethBalance < ETH_RESERVE_MIN) {
    console.error("  ✗ ETH balance too low. Need >0.005 ETH for gas.");
    hasErrors = true;
  } else {
    console.log("  ✓ ETH balance sufficient for gas");
  }

  if (usdcBalance === 0n) {
    console.error("  ✗ USDC balance is 0. Need USDC for trading.");
    hasErrors = true;
  } else {
    console.log("  ✓ USDC balance > 0");
  }

  if (hasErrors) {
    console.error("\nPlease fund your wallet and re-run this script.");
    rl.close();
    process.exit(1);
  }

  console.log("\nSetting token approvals...");

  const router = CONTRACTS.AERODROME_ROUTER;
  const approvalAmount = maxUint256;

  console.log("  Approving USDC for Aerodrome Router...");
  const usdcAllowanceBefore = (await publicClient.readContract({
    address: TOKENS.USDC.address,
    abi: erc20Abi,
    functionName: "allowance",
    args: [account.address, router],
  })) as bigint;

  if (usdcAllowanceBefore > 0n) {
    console.log("  ✓ USDC already approved (allowance > 0)");
  } else {
    const usdcTxHash = await walletClient.writeContract({
      address: TOKENS.USDC.address,
      abi: erc20Abi,
      functionName: "approve",
      args: [router, approvalAmount],
      account,
      chain: base,
    });
    await publicClient.waitForTransactionReceipt({ hash: usdcTxHash, confirmations: 1 });
    console.log(`  ✓ USDC approved (tx: ${usdcTxHash})`);
  }

  console.log("  Approving WETH for Aerodrome Router...");
  const wethAllowanceBefore = (await publicClient.readContract({
    address: TOKENS.WETH.address,
    abi: erc20Abi,
    functionName: "allowance",
    args: [account.address, router],
  })) as bigint;

  if (wethAllowanceBefore > 0n) {
    console.log("  ✓ WETH already approved (allowance > 0)");
  } else {
    const wethTxHash = await walletClient.writeContract({
      address: TOKENS.WETH.address,
      abi: erc20Abi,
      functionName: "approve",
      args: [router, approvalAmount],
      account,
      chain: base,
    });
    await publicClient.waitForTransactionReceipt({ hash: wethTxHash, confirmations: 1 });
    console.log(`  ✓ WETH approved (tx: ${wethTxHash})`);
  }

  const usdcAllowanceAfter = (await publicClient.readContract({
    address: TOKENS.USDC.address,
    abi: erc20Abi,
    functionName: "allowance",
    args: [account.address, router],
  })) as bigint;

  const wethAllowanceAfter = (await publicClient.readContract({
    address: TOKENS.WETH.address,
    abi: erc20Abi,
    functionName: "allowance",
    args: [account.address, router],
  })) as bigint;

  console.log("\n=== Wallet Setup Complete ===");
  console.log(`  Address:        ${account.address}`);
  console.log(`  ETH:            ${formatEther(ethBalance)} ETH`);
  console.log(`  USDC:           ${formatUnits(usdcBalance, TOKENS.USDC.decimals)} USDC`);
  console.log(`  USDC Allowance: ${usdcAllowanceAfter > 0n ? "✓ Approved" : "✗ NOT approved"}`);
  console.log(`  WETH Allowance: ${wethAllowanceAfter > 0n ? "✓ Approved" : "✗ NOT approved"}`);
  console.log("");

  rl.close();
}

main().catch((err) => {
  console.error("Setup failed:", err);
  process.exit(1);
});
