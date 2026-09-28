import { createPublicClient, createWalletClient, http } from "viem";
import { base } from "viem/chains";
import { privateKeyToAccount } from "viem/accounts";
import { config } from "../config/index.js";

const transport = http(config.BASE_RPC_URL);

export const publicClient = createPublicClient({
  chain: base,
  transport,
});

const account = config.PRIVATE_KEY
  ? privateKeyToAccount(config.PRIVATE_KEY as `0x${string}`)
  : undefined;

export const walletClient = account
  ? createWalletClient({
      account,
      chain: base,
      transport,
    })
  : undefined;

export const readOnlyClient = createPublicClient({
  chain: base,
  transport: http("https://mainnet.base.org"),
});
