import { createHash } from "node:crypto";

export interface NotifyConfig {
  telegramBotToken?: string; // from env TELEGRAM_BOT_TOKEN
  telegramChatId?: string; // from env TELEGRAM_CHAT_ID
  throttleMs: number; // default 300_000 (5 min)
}

export interface NotifyEvent {
  type: "grid_trigger" | "fill" | "risk" | "recenter" | "emergency" | "info";
  pair: string;
  message: string;
  data?: Record<string, unknown>;
}

export class Notifier {
  private lastSent: Map<string, number> = new Map();

  constructor(private config: NotifyConfig) {}

  async notify(event: NotifyEvent): Promise<void> {
    const dedupKey = this.buildDedupKey(event);
    if (this.shouldThrottle(dedupKey)) {
      return;
    }

    const formatted = this.formatMessage(event);

    if (this.config.telegramBotToken && this.config.telegramChatId) {
      await this.sendTelegram(formatted);
    }

    this.lastSent.set(dedupKey, Date.now());
  }

  private buildDedupKey(event: NotifyEvent): string {
    const raw = `${event.type}:${event.pair}:${event.message}`;
    return createHash("sha256").update(raw).digest("hex").slice(0, 16);
  }

  private shouldThrottle(key: string): boolean {
    const last = this.lastSent.get(key);
    if (last === undefined) return false;
    return Date.now() - last < this.config.throttleMs;
  }

  private formatMessage(event: NotifyEvent): string {
    switch (event.type) {
      case "grid_trigger":
        return `\u{1F4CA} ${event.pair}: ${event.message}`;
      case "fill":
        return `\u2705 ${event.pair}: ${event.message}`;
      case "risk":
        return `\u26A0\uFE0F STOP LOSS triggered \u2014 ${event.message}`;
      case "recenter":
        return `\u{1F504} Grid recentered to ${event.message}`;
      case "emergency":
        return `\u{1F6D1} EMERGENCY STOP \u2014 reason: ${event.message}`;
      case "info":
        return `\u2139\uFE0F ${event.message}`;
    }
  }

  private async sendTelegram(message: string): Promise<void> {
    const url = `https://api.telegram.org/bot${this.config.telegramBotToken}/sendMessage`;
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          chat_id: this.config.telegramChatId,
          text: message,
          parse_mode: "HTML",
        }),
      });
      if (!res.ok) {
        const body = await res.text().catch(() => "");
        console.error(`Telegram API error ${res.status}: ${body}`);
      }
    } catch (err) {
      console.error("Telegram send failed:", err);
    }
  }
}
