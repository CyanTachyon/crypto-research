import { z } from "zod";
import "dotenv/config";

const envSchema = z.object({
  PRIVATE_KEY: z.string().startsWith("0x").optional(),
  BASE_RPC_URL: z.string().url().default("https://mainnet.base.org"),
  MODE: z.enum(["sim", "live"]).default("sim"),
  TELEGRAM_BOT_TOKEN: z.string().optional(),
  TELEGRAM_CHAT_ID: z.string().optional(),
});

export type EnvConfig = z.infer<typeof envSchema>;

function loadConfig(): EnvConfig {
  const parsed = envSchema.safeParse(process.env);
  if (!parsed.success) {
    const errors = parsed.error.issues
      .map((i) => `${i.path.join(".")}: ${i.message}`)
      .join("\n");
    throw new Error(`Invalid environment config:\n${errors}`);
  }
  return parsed.data;
}

export const config = loadConfig();
