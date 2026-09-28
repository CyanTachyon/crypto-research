import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { Notifier } from "./notify.js";
import type { NotifyConfig, NotifyEvent } from "./notify.js";

describe("Notifier", () => {
  let fetchSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve("{}"),
    });
    vi.stubGlobal("fetch", fetchSpy);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  function makeConfig(overrides: Partial<NotifyConfig> = {}): NotifyConfig {
    return {
      telegramBotToken: "123456:ABC-DEF",
      telegramChatId: "987654321",
      throttleMs: 300_000,
      ...overrides,
    };
  }

  function makeEvent(overrides: Partial<NotifyEvent> = {}): NotifyEvent {
    return {
      type: "grid_trigger",
      pair: "ETH/USDC",
      message: "BUY triggered at $2,425.00",
      ...overrides,
    };
  }

  it("sends notification via Telegram when configured", async () => {
    const notifier = new Notifier(makeConfig());
    await notifier.notify(makeEvent());

    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, options] = fetchSpy.mock.calls[0];
    expect(url).toBe("https://api.telegram.org/bot123456:ABC-DEF/sendMessage");
    const body = JSON.parse((options as RequestInit).body as string);
    expect(body.chat_id).toBe("987654321");
    expect(body.parse_mode).toBe("HTML");
    expect(body.text).toContain("ETH/USDC");
    expect(body.text).toContain("BUY triggered at $2,425.00");
  });

  it("skips Telegram call when token not configured", async () => {
    const notifier = new Notifier(
      makeConfig({ telegramBotToken: undefined, telegramChatId: undefined }),
    );
    await notifier.notify(makeEvent());

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("skips Telegram call when chatId not configured", async () => {
    const notifier = new Notifier(makeConfig({ telegramChatId: undefined }));
    await notifier.notify(makeEvent());

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("throttles duplicate events within throttle window", async () => {
    const notifier = new Notifier(makeConfig({ throttleMs: 300_000 }));
    const event = makeEvent();

    await notifier.notify(event);
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    await notifier.notify(event);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("allows same event after throttle window expires", async () => {
    vi.useFakeTimers();
    const notifier = new Notifier(makeConfig({ throttleMs: 100 }));
    const event = makeEvent();

    await notifier.notify(event);
    expect(fetchSpy).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(150);

    await notifier.notify(event);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    vi.useRealTimers();
  });

  it("uses different dedup keys for different events", async () => {
    const notifier = new Notifier(makeConfig({ throttleMs: 300_000 }));

    await notifier.notify(makeEvent({ type: "grid_trigger", message: "BUY at $100" }));
    await notifier.notify(makeEvent({ type: "grid_trigger", message: "SELL at $110" }));

    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  it("formats grid_trigger message with chart emoji", async () => {
    const notifier = new Notifier(makeConfig());
    await notifier.notify(makeEvent({ type: "grid_trigger" }));

    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string);
    expect(body.text).toContain("\u{1F4CA}");
  });

  it("formats fill message with checkmark emoji", async () => {
    const notifier = new Notifier(makeConfig());
    await notifier.notify(makeEvent({ type: "fill", message: "BUY filled" }));

    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string);
    expect(body.text).toContain("\u2705");
  });

  it("formats emergency message with stop emoji", async () => {
    const notifier = new Notifier(makeConfig());
    await notifier.notify(makeEvent({ type: "emergency", message: "max drawdown exceeded" }));

    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string);
    expect(body.text).toContain("\u{1F6D1}");
    expect(body.text).toContain("EMERGENCY STOP");
  });

  it("does not throw when Telegram API returns error", async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: false,
      status: 429,
      text: () => Promise.resolve("Too Many Requests"),
    });

    const notifier = new Notifier(makeConfig());
    await expect(notifier.notify(makeEvent())).resolves.toBeUndefined();
  });

  it("does not throw when fetch itself fails", async () => {
    fetchSpy.mockRejectedValueOnce(new Error("network error"));

    const notifier = new Notifier(makeConfig());
    await expect(notifier.notify(makeEvent())).resolves.toBeUndefined();
  });
});
