#!/usr/bin/env python3
"""LLM AI trading agent — DeepSeek V4-Pro + tool calling + web search.

Model: deepseek-v4-pro (with thinking mode + reasoning_effort)
Data:  Real Hyperliquid candles/price
Mode:  SIMULATED trading (no real money)

Tools given to LLM:
  - open_position(side, take_profit_points, stop_loss_points, reason)
  - close_position(reason)
  - search_web(query)  — DuckDuckGo search for crypto news
  - hold(reason)

Setup:
    export DEEPSEEK_API_KEY="sk-your-key"
    python scripts/llm_scalper.py
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import requests as req

# ─── Load .env file ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = val

LOG_DIR = Path("data/llm_logs")
STATE_FILE = Path("data/llm_scalper_state.json")
REPORT_FILE = Path("docs/llm_scalper_report.md")

COIN = os.environ.get("LLM_COIN", "BTC")
LOOKBACK_CANDLES = int(os.environ.get("LLM_LOOKBACK_CANDLES", "48"))
CANDLE_TIMEFRAME = os.environ.get("LLM_CANDLE_TIMEFRAME", "1h")
LLM_MODEL = "deepseek-v4-pro"
LLM_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "high")
ENABLE_THINKING = os.environ.get("LLM_ENABLE_THINKING", "1") == "1"
ENABLE_SEARCH = os.environ.get("LLM_ENABLE_SEARCH", "1") == "1"
DECISION_TIMEOUT = 600
MAX_WAIT = int(os.environ.get("LLM_MAX_WAIT", "1200"))
MIN_WAIT = 30  # 防止 AI 设过短间隔狂打 API

LEVERAGE = int(os.environ.get("LLM_LEVERAGE", "10"))
MAX_MARGIN_USD = float(os.environ.get("LLM_MAX_MARGIN_USD", "30"))
FEE_RATE = 0.00035
SLIPPAGE = 0.0001
INITIAL_BALANCE = float(os.environ.get("LLM_INITIAL_BALANCE", "100"))
MAX_DAILY_LOSS = float(os.environ.get("LLM_MAX_DAILY_LOSS", "10"))

# Telegram 通知
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")
TG_ENABLED = bool(TG_BOT_TOKEN and TG_CHAT_ID)


def retry(fn, attempts=3, delay=5, label=""):
    """通用重试包装器。网络抖动时自动重试，全失败才抛异常。"""
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                print(f"[RETRY] {label} 第{i+1}/{attempts}次失败: {e}，{delay}秒后重试")
                time.sleep(delay)
    raise last_err  # type: ignore


def send_telegram(text: str):
    """发 Telegram 消息，自动重试 3 次。全失败只警告不崩。"""
    if not TG_ENABLED:
        return
    try:
        retry(
            lambda: req.post(
                f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                data={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            ),
            attempts=3, delay=3, label="TG发送",
        )
    except Exception as e:
        print(f"[TG][WARN] 发送 3 次都失败: {e}")


def fetch_tg_messages(state) -> list[str]:
    """拉取用户在 Telegram 上发来的新消息。自动重试 3 次。"""
    if not TG_ENABLED:
        return []
    try:
        offset = state.last_tg_update_id + 1
        r = retry(
            lambda: req.get(
                f"https://api.telegram.org/bot{TG_BOT_TOKEN}/getUpdates",
                params={"offset": offset, "timeout": 0, "limit": 20},
                timeout=10,
            ),
            attempts=3, delay=5, label="TG拉取",
        )
        data = r.json()
        if not data.get("ok"):
            return []
        msgs = []
        for upd in data.get("result", []):
            state.last_tg_update_id = max(state.last_tg_update_id, upd.get("update_id", 0))
            msg = upd.get("message") or upd.get("edited_message")
            if msg and msg.get("text"):
                if not msg["text"].startswith("/"):
                    msgs.append(msg["text"])
        if msgs:
            print(f"[TG] 收到 {len(msgs)} 条新消息")
        return msgs
    except Exception as e:
        print(f"[TG][WARN] 拉消息失败: {e}")
        return []

# 实盘交易
LIVE_TRADING = os.environ.get("LLM_LIVE_TRADING", "0") == "1"
if LIVE_TRADING and not os.environ.get("HL_PRIVATE_KEY"):
    print("FATAL: LLM_LIVE_TRADING=1 但未设置 HL_PRIVATE_KEY，拒绝启动")
    sys.exit(1)

SYSTEM_PROMPT = f"""你是一名专业的 Hyperliquid 永续合约加密货币短线/scalping 交易员。

账户: {COIN} | {LEVERAGE}x 杠杆 | 单笔保证金上限 ${MAX_MARGIN_USD}（仓位大小由你决定）| 最大仓位价值 ${MAX_MARGIN_USD * LEVERAGE}
手续费: {FEE_RATE * 100:.3f}%/单边（往返 {FEE_RATE * 200:.3f}%）| 滑点: {SLIPPAGE * 100:.2f}%

## 仓位规则（重要）
- 同一时间只能持有一个 {COIN} 仓位（Hyperliquid 限制）
- **如果已有同方向仓位**，调用 open_position(side=同方向) 会**加仓**（不是另开一笔）。新保证金会叠加到原仓位上，入场价变成加权平均
- **如果想反方向**，必须先 close_position 平掉当前仓位，下一轮再 open_position 反向
- **已有仓位时也可以选择 hold** 继续观望，或 close_position 止盈/止损

## 仓位大小（根据信心调整，不要每次都用满）
- 单笔保证金上限 ${MAX_MARGIN_USD}，但**你应该根据信心分档**：
  - **极高信心**：${int(MAX_MARGIN_USD*0.8)}-{int(MAX_MARGIN_USD)}（80-100%）
  - **高信心**：${int(MAX_MARGIN_USD*0.5)}-{int(MAX_MARGIN_USD*0.7)}（50-70%）
  - **中等信心**：${int(MAX_MARGIN_USD*0.3)}-{int(MAX_MARGIN_USD*0.5)}（30-50%）
  - **试探性入场**：${int(MAX_MARGIN_USD*0.15)}-{int(MAX_MARGIN_USD*0.3)}（15-30%）
- 单日连续亏损时**主动降仓**（风控纪律）：例如连亏 2 笔后下一笔只用 50% 保证金

## 下单方式（限价 IoC）
- open_position 和 close_position 用**限价单（IoC，Immediate-or-Cancel）**
- **IoC 语义**：立刻按 limit_price 或更好的价格成交，**否则整单取消**。它**不会**"挂在订单簿上等价格"
- 你传的 `limit_price` 是"**我愿意接受的最差价格**"（不是"希望成交的理想价"）
- **填法（关键，填错就成交不了）**：
  - **LONG（买入）**：填**当前价或更高**（建议 `当前价 + 5~15 点`）。价格越高越容易成交，但成交价不会比你填的更差
  - **SHORT（卖出）**：填**当前价或更低**（建议 `当前价 - 5~15 点`）。价格越低越容易成交
  - 想确保成交：填略激进（buy+15 / sell-15）；想等更好价：填略被动（buy+0 / sell-0），但要承担不成交风险
- **示例**：当前价 $62650，做空 → `limit_price: 62635`（不是 62650，更不是 62660）
- 如果不成交，说明市场快速移动了，下一轮用更激进的 limit 重试

## 等待时间 wait_seconds
- 每次操作后你必须传 wait_seconds 决定多久再回来看盘
- 范围 {MIN_WAIT}~{MAX_WAIT} 秒（小于 {MIN_WAIT} 会被钳制到 {MIN_WAIT}，大于 {MAX_WAIT} 会被钳制到 {MAX_WAIT}）
- **建议规则**：
  - 持仓 + 接近 TP/SL → 短（60-120s，盯紧）
  - 持仓 + 趋势正常 → 中（180-300s）
  - 空仓 + 等待入场信号 → 长（300-600s）
  - 刚搜完新闻 + 市场混沌 → 长（600-1200s，让尘埃落定）

## 策略
1. 判断整体趋势（上涨趋势 → 回调做多；下跌趋势 → 反弹做空）
2. 不确定时用 search_web 查新闻/市场情绪
3. 没把握就 hold。没有仓位比烂仓位好。
4. 持仓盈利时，权衡止盈落袋 vs 等到 TP。
5. **持仓中如果市场结构变化（例如趋势加速、形成更高高点），可用 `adjust_tpsl` 移动 TP/SL**

## 手续费成本意识（重要！）
- **每笔开仓 + 平仓 = 往返手续费 {FEE_RATE * 200:.3f}%**（约等于 {FEE_RATE * 200 * 1000:.2f} 个基点）
- 示例：$200 仓位 → 往返手续费 ${200 * FEE_RATE * 2:.3f}
- **盈利单**：净盈利 = 价差收益 - 手续费（吃掉一部分利润）
- **亏损单**：净亏损 = 价差亏损 + 手续费（**亏得更多**）
- 这意味着：
  - **小目标交易（TP < 30 点）几乎被手续费吞噬**，要避免
  - **频繁开平会让手续费累积**，每天 10 笔 × $0.14 = $1.4/天，相当于每天要赚 1.4% 才打平
  - 高胜率（>55%）+ 合理 TP（>50 点）才能覆盖手续费长期盈利
  - 持仓时间过短（< 5 分钟）通常是给交易所送钱

**重要：你的所有回复（content）、reason 字段、以及工具调用中的所有文本参数必须使用中文。**

每轮只调用一个工具（open_position / close_position / adjust_tpsl / hold / search_web）。"""


# ─── Tools ──────────────────────────────────────────────

def get_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "open_position",
                "description": f"开 {COIN} 仓位（限价 IoC）。如果已有同方向仓位会加仓；如果有反向仓位请先 close_position。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "side": {"type": "string", "enum": ["long", "short"]},
                        "margin_usd": {"type": "number", "description": f"保证金（美元），最大 {int(MAX_MARGIN_USD)}。已有仓位时这是加仓的额外保证金"},
                        "limit_price": {"type": "number", "description": "限价（美元）。填当前价附近；buy 略高于 mid、sell 略低于 mid 可提高成交率。偏离太远不会成交"},
                        "take_profit_points": {"type": "number", "description": "止盈点数（1 点 = $1 价格变动），基于你的分析设置"},
                        "stop_loss_points": {"type": "number", "description": "止损点数。基于你的失效判断设置，不要随便填"},
                        "wait_seconds": {"type": "number", "description": f"下单后等多久再看盘（{MIN_WAIT}-{MAX_WAIT}秒）。持仓接近 TP/SL 用短，空仓等待用长"},
                        "reason": {"type": "string", "description": "为什么这笔交易（1-2 句话，中文）"},
                    },
                    "required": ["side", "margin_usd", "limit_price", "take_profit_points", "stop_loss_points", "wait_seconds", "reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "close_position",
                "description": "以限价 IoC 平掉当前仓位。撤销已有的 TP/SL 触发单后下单。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit_price": {"type": "number", "description": "限价（美元）。想快速平仓就填略激进价（long 平仓填略低、short 平仓填略高）"},
                        "wait_seconds": {"type": "number", "description": f"平仓后等多久再看盘（{MIN_WAIT}-{MAX_WAIT}秒）"},
                        "reason": {"type": "string", "description": "平仓理由（中文）"},
                    },
                    "required": ["limit_price", "wait_seconds", "reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "hold",
                "description": "本轮不操作，继续持有当前仓位或保持空仓。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wait_seconds": {"type": "number", "description": f"等多久再看盘（{MIN_WAIT}-{MAX_WAIT}秒）。持仓接近 TP/SL 用短，空仓等待用长"},
                        "reason": {"type": "string", "description": "为什么继续观望（中文）"},
                    },
                    "required": ["wait_seconds", "reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "adjust_tpsl",
                "description": "调整当前仓位的 TP/SL 触发价（不改仓位大小）。会先撤掉旧 TP/SL 再挂新的。仅持仓时可用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "take_profit_points": {"type": "number", "description": "新的止盈点数（1 点 = $1 价格变动），基于当前入场价计算"},
                        "stop_loss_points": {"type": "number", "description": "新的止损点数，基于当前入场价计算"},
                        "wait_seconds": {"type": "number", "description": f"调整后等多久再看盘（{MIN_WAIT}-{MAX_WAIT}秒）"},
                        "reason": {"type": "string", "description": "为什么要调整 TP/SL（中文）"},
                    },
                    "required": ["take_profit_points", "stop_loss_points", "wait_seconds", "reason"],
                },
            },
        },
    ]
    if ENABLE_SEARCH:
        tools.insert(2, {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the internet for crypto news, market analysis, sentiment, or any info.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        })
    # 回复用户（Telegram 双向通信）
    tools.append({
        "type": "function",
        "function": {
            "name": "reply_to_user",
            "description": "回复用户在 Telegram 上发来的消息。可在任何轮次调用（不影响仓位操作）。如果用户问了问题、给了指令、或你需要解释当前决策，用这个工具回复。",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "回复内容（中文）。可以是分析、解释、状态汇报等"},
                    "wait_seconds": {"type": "number", "description": f"回复后等多久再看盘（{MIN_WAIT}-{MAX_WAIT}秒）"},
                },
                "required": ["message", "wait_seconds"],
            },
        },
    })
    return tools


# ─── Web search ─────────────────────────────────────────

def search_web(query: str) -> str:
    import requests as req
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0"}

    # Method 1: DuckDuckGo HTML
    try:
        r = req.post("https://html.duckduckgo.com/html/", data={"q": query}, headers=headers, timeout=10)
        snippets = []
        for block in re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', r.text, re.DOTALL | re.IGNORECASE)[:5]:
            clean = re.sub(r'<[^>]+>', '', block).strip()
            if len(clean) > 20:
                snippets.append(clean[:300])
        if snippets:
            return "\n".join(f"- {s}" for s in snippets)
    except:
        pass

    # Method 2: DuckDuckGo API
    try:
        r = req.get("https://api.duckduckgo.com/", params={"q": query, "format": "json", "no_html": 1}, headers=headers, timeout=10)
        data = r.json()
        parts = []
        if data.get("Abstract"):
            parts.append(data["Abstract"][:400])
        for t in data.get("RelatedTopics", [])[:5]:
            if isinstance(t, dict) and t.get("Text"):
                parts.append(t["Text"][:200])
        if parts:
            return "\n".join(parts)
    except:
        pass

    # Method 3: Bing (scrape)
    try:
        r = req.get(f"https://www.bing.com/search?q={query.replace(' ', '+')}+crypto", headers=headers, timeout=10)
        snippets = []
        for block in re.findall(r'<p class="b_lineclamp[^"]*"[^>]*>(.*?)</p>', r.text, re.DOTALL)[:5]:
            clean = re.sub(r'<[^>]+>', '', block).strip()
            if len(clean) > 20:
                snippets.append(clean[:300])
        if snippets:
            return "\n".join(f"- {s}" for s in snippets)
    except:
        pass

    # Method 4: Google (scrape)
    try:
        r = req.get(f"https://www.google.com/search?q={query.replace(' ', '+')}", headers=headers, timeout=10)
        snippets = []
        for block in re.findall(r'<span>(.*?)</span>', r.text)[:10]:
            clean = block.strip()
            if len(clean) > 30 and not clean.startswith("http"):
                snippets.append(clean[:200])
        if snippets:
            return "\n".join(f"- {s}" for s in snippets[:5])
    except:
        pass

    return "Search unavailable from this server. Proceed with technical analysis only."


# ─── Market data ────────────────────────────────────────

def fetch_candles(count: int, timeframe: str = CANDLE_TIMEFRAME) -> list[dict]:
    """拉取已收盘的 K 线（不含进行中的那根，避免缓存失效）。
    多取 1 根再切掉最后一根，保证拿到 count 根已收盘 K 线。"""
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        end = int(time.time() * 1000)
        tf_ms = {"1m": 60000, "5m": 300000, "15m": 900000,
                 "1h": 3600000, "4h": 14400000, "1d": 86400000}
        ms_per = tf_ms.get(timeframe, 3600000)
        # 多取 1 根，因为最后一根是进行中（会被切掉）
        start = end - int((count + 2) * ms_per * 1.5)
        raw = retry(lambda: info.candles_snapshot(COIN, timeframe, start, end),
                    attempts=3, delay=3, label=f"candles({timeframe})")
        # 切掉最后一根（进行中），保留 count 根已收盘
        closed = raw[-(count + 1):-1] if len(raw) > count else raw[:-1]
        return [{
            "t": datetime.fromtimestamp(int(c["t"]) / 1000, tz=timezone.utc).strftime("%m-%d %H:%M"),
            "_ts": int(c["t"]) / 1000,  # 原始时间戳（秒），用于去重
            "o": float(c["o"]), "h": float(c["h"]), "l": float(c["l"]),
            "c": float(c["c"]), "v": float(c["v"]),
        } for c in closed]
    except Exception as e:
        print(f"[WARN] candles({timeframe}) 拉取失败: {e}")
        return []


def fetch_multi_timeframe() -> dict[str, list[dict]]:
    """拉取多 timeframe K 线，**级联对齐**每个 timeframe 的开始/结束边界。
    每个 timeframe 只覆盖不被更细 timeframe 覆盖的时间段，且两端都对齐到稳定边界。

    缓存稳定性（两端边界都不变的时间窗口）：
      4h: [天边界, 12h边界) → 12h 稳定
      1h: [12h边界, 2h边界)  → 2h 稳定
      5m: [2h边界, 30min边界) → 30min 稳定
      1m: [30min边界, now)   → 1min 稳定 ← 唯一断点"""
    raw = {
        "4h": fetch_candles(50, "4h"),
        "1h": fetch_candles(70, "1h"),
        # 5m 需要覆盖到 aligned_5m（最远 ~10h 前），加上 aligned_1m 之后的缓冲，至少要 150 根
        "5m": fetch_candles(160, "5m"),
        "1m": fetch_candles(100, "1m"),
    }

    now = int(time.time())

    # 每层对齐到逐渐更粗的时间边界
    aligned_1m  = ((now - 5400)   // 1800)  * 1800    # 1.5h 前，对齐 30min
    aligned_5m  = ((now - 28800)  // 7200)  * 7200    # 8h 前，对齐 2h
    aligned_1h  = ((now - 172800) // 43200) * 43200   # 48h 前，对齐 12h
    aligned_4h  = ((now - 604800) // 86400) * 86400   # 7d 前，对齐 1d

    # 每个 timeframe 严格限制在 [自己的对齐起点, 下一层的对齐起点)
    raw["1m"] = [c for c in raw["1m"] if c["_ts"] >= aligned_1m]
    raw["5m"] = [c for c in raw["5m"] if aligned_5m <= c["_ts"] < aligned_1m]
    raw["1h"] = [c for c in raw["1h"] if aligned_1h <= c["_ts"] < aligned_5m]
    raw["4h"] = [c for c in raw["4h"] if aligned_4h <= c["_ts"] < aligned_1h]

    return raw


def fetch_price() -> float | None:
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        return float(retry(lambda: info.all_mids(), attempts=3, delay=3, label="fetch_price").get(COIN, 0)) or None
    except:
        return None


def compute_indicators(candles_1h: list[dict]) -> dict:
    if len(candles_1h) < 5:
        return {"price": candles_1h[-1]["c"] if candles_1h else 0}
    closes = [c["c"] for c in candles_1h]
    ind = {"price": closes[-1]}
    for p in [7, 25, 50, 99]:
        if len(closes) >= p:
            ind[f"sma{p}"] = round(sum(closes[-p:]) / p, 1)
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    if len(deltas) >= 14:
        ag = sum(max(0, d) for d in deltas[-14:]) / 14
        al = sum(max(0, -d) for d in deltas[-14:]) / 14
        ind["rsi14"] = round(100 - 100 / (1 + ag / max(al, 0.01)), 1)
    if len(closes) >= 2:
        ind["last_1h_pct"] = round((closes[-1] - closes[-2]) / closes[-2] * 100, 2)
    if len(closes) >= 6:
        ind["last_6h_pct"] = round((closes[-1] - closes[-6]) / closes[-6] * 100, 2)
    if len(closes) >= 24:
        ind["last_24h_pct"] = round((closes[-1] - closes[-24]) / closes[-24] * 100, 2)
    window = candles_1h[-24:] if len(candles_1h) >= 24 else candles_1h
    ind["high_24h"] = max(c["h"] for c in window)
    ind["low_24h"] = min(c["l"] for c in window)
    return ind


def _fmt_full(c: dict) -> str:
    chg = ((c["c"] - c["o"]) / c["o"] * 100) if c["o"] > 0 else 0
    return f"{c['t']} O{c['o']:.0f} H{c['h']:.0f} L{c['l']:.0f} C{c['c']:.0f} V{c['v']:.0f} ({chg:+.2f}%)"


def _fmt_compact(c: dict) -> str:
    return f"{c['t']} C{c['c']:.0f}"


def format_market_context(mtf: dict, ind: dict, state, price: float, user_messages: list[str] = None) -> str:
    # ─── 缓存优化后的 prompt 结构 ─────────────────────────────
    # DeepSeek API 按 prefix 匹配做自动缓存，前缀越长越省。
    # 排列原则：稳定数据在前（命缓存），动态数据在后（破坏前缀）。
    #   1. K线（4h > 1h > 5m > 1m，时间长的先列）
    #   2. 指标（基于 1h，1h 才变）
    #   3. 交易历史（仅交易时变）
    #   4. 决策日志 memory（按时间老→新，前面老条目命中缓存）
    #   5. 账户/持仓/当前价（每轮都变，放最后）
    # ────────────────────────────────────────────────────────

    # 1. K线块（最稳定，最大）
    sections = []
    for tf, label, fmt_fn, limit in [
        ("4h", "4h K线（已收盘，2-7 天前）", _fmt_full, 999),
        ("1h", "1h K线（已收盘，6h-2 天前）", _fmt_full, 999),
        ("5m", "5m K线（已收盘，1-6 小时前）", _fmt_full, 999),
        ("1m", "1m K线（已收盘，最近 1 小时）", _fmt_compact, 999),
    ]:
        candles = mtf.get(tf, [])
        if candles:
            # 按时间正序（老→新），最新的一根在末尾——即使最新根变化，前面的仍命中缓存
            lines = [fmt_fn(c) for c in candles[-limit:]]
            sections.append(f"### {label} ({len(lines)} 根)\n" + "\n".join(lines))
    candle_block = "\n\n".join(sections)

    # 2. 指标（基于 1h K线，只在 1h 跨界时变化）
    ind_lines = "\n".join(f"  {k}: {v}" for k, v in ind.items())

    # 3. 交易历史（只在有新交易时变）
    recent = state.trades[-8:]
    if recent:
        trade_lines = []
        for t in recent:
            trade_lines.append(
                f"  {t.ts} | {t.side:5s} | entry ${t.entry:.0f} → exit ${t.exit_:.0f} | "
                f"net ${t.net:+.4f} | {t.reason} | {t.dur:.0f}s"
            )
        history = "\n".join(trade_lines)
    else:
        history = "  （暂无）"

    # 4. 决策日志 memory（老→新，前面命中缓存）
    recent_log = state.decision_log[-8:]
    if recent_log:
        log_lines = []
        for d in recent_log:
            line = f"  [{d['ts']}] {d['action']:14s} @${d['price']:.0f} — {d['summary']}"
            if d.get("detail"):
                line += f"\n                  详情：{d['detail']}"
            log_lines.append(line)
        memory = "\n".join(log_lines)
    else:
        memory = "  （暂无历史决策）"

    # 5. 动态状态（每轮都变，放最后）
    if state.position:
        p = state.position
        u = (p["entry"] - price) * p["size"] if p["side"] == "short" else (price - p["entry"]) * p["size"]
        mins = (time.time() - p["open_ts"]) / 60
        entry_reason = p.get("entry_reason", "N/A")
        pos = (f"持仓: {p['side'].upper()} {p['size']:.5f} {COIN} @ ${p['entry']:.1f}\n"
               f"  未实现盈亏: ${u:+.4f} | TP {p['tp']:.0f}pts SL {p['sl']:.0f}pts | 已持仓 {mins:.1f}min\n"
               f"  开仓理由: {entry_reason}")
    else:
        pos = "无持仓"

    # 组装：稳定 → 动态
    return (
        # ─── 最稳定（命缓存）───
        f"## Market Data\n\n{candle_block}\n\n"
        f"## Indicators (基于 1h K线)\n{ind_lines}\n\n"
        # ─── 半稳定 ───
        f"## 最近交易历史 (last {len(recent)})\n{history}\n\n"
        f"## 最近决策记录（你的 memory，最近 {len(recent_log)} 条，老→新）\n"
        f"{memory}\n\n"
        # ─── 动态（破坏前缀，放最后）───
        f"## 当前账户\n"
        f"  余额: ${state.balance:.2f} | 今日盈亏: ${state.daily_pnl:.2f} | 今日交易数: {state.daily_trades}\n\n"
        f"## 当前持仓\n{pos}\n\n"
        + (
            f"## 👤 用户反馈（来自 Telegram，本轮新消息）\n"
            + "\n".join(f"  用户：{m}" for m in user_messages)
            + "\n\n**如果用户给了指令（如『平仓』『停止』『加仓』），优先考虑执行。如果有疑问，用 reply_to_user 工具回复。**\n\n"
            if user_messages else ""
        )
        + f"当前价格: ${price:.1f}"
    )


# ─── Sim engine ─────────────────────────────────────────

@dataclass
class SimTrade:
    ts: str; side: str; entry: float; exit_: float; size: float
    gross: float; fees: float; net: float; reason: str
    tp: float; sl: float; dur: float


@dataclass
class AgentState:
    balance: float = INITIAL_BALANCE
    position: dict | None = None
    trades: list[SimTrade] = field(default_factory=list)
    decision_log: list[dict] = field(default_factory=list)  # 决策日志（memory）
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_reset: str = ""
    decisions: int = 0
    start_time: float = 0.0
    running: bool = True
    last_price: float = 0.0
    last_tg_update_id: int = 0  # 已处理的最近 Telegram update_id
    did_reply_this_round: bool = False  # 本轮是否已回复用户（避免误触发"未回复缩短等待"）


# ─── Hyperliquid 实盘交易客户端 ─────────────────────────
class LiveTrader:
    """封装 Hyperliquid SDK，仅在 LIVE_TRADING=1 时使用。"""

    def __init__(self):
        from eth_account import Account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.utils import constants

        main_wallet = os.environ.get("HL_MAIN_WALLET", "").strip()
        if not main_wallet:
            print("FATAL: LIVE_TRADING=1 但未设置 HL_MAIN_WALLET（你的主钱包地址）")
            sys.exit(1)

        self.wallet = Account.from_key(os.environ["HL_PRIVATE_KEY"])
        self.agent_address = self.wallet.address
        self.main_address = main_wallet

        # SDK 初始化会立刻拉 spotMeta/marketMeta——网络抖动时会崩。
        # 加重试：最多 5 次，每次间隔 10 秒
        last_err = None
        for attempt in range(5):
            try:
                self.info = Info(constants.MAINNET_API_URL, skip_ws=True, timeout=30)
                self.exchange = Exchange(self.wallet, constants.MAINNET_API_URL,
                                         account_address=self.main_address, timeout=30)
                break
            except Exception as e:
                last_err = e
                print(f"[LIVE][WARN] SDK 初始化第 {attempt+1}/5 次失败: {e}")
                if attempt < 4:
                    print(f"[LIVE] 10 秒后重试...")
                    time.sleep(10)
        else:
            print(f"FATAL: SDK 初始化 5 次都失败，最后一次错误: {last_err}")
            sys.exit(1)

        print(f"[LIVE] 主钱包: {self.main_address}")
        print(f"[LIVE] Agent 签名钱包: {self.agent_address}")
        # 启动时设置杠杆
        for attempt in range(3):
            try:
                self.exchange.update_leverage(LEVERAGE, COIN)
                print(f"[LIVE] {COIN} 杠杆已设置为 {LEVERAGE}x")
                break
            except Exception as e:
                print(f"[LIVE][WARN] 设置杠杆第 {attempt+1}/3 次失败: {e}")
                if attempt < 2:
                    time.sleep(5)

    def get_balance(self) -> float:
        spot = retry(lambda: self.info.spot_user_state(self.main_address),
                     attempts=3, delay=3, label="get_balance")
        for b in spot.get("balances", []):
            if b.get("coin") == "USDC":
                return max(0.0, float(b.get("total", 0)) - float(b.get("hold", 0)))
        return 0.0

    def get_total_equity(self) -> float:
        us = retry(lambda: self.info.user_state(self.main_address),
                   attempts=3, delay=3, label="get_total_equity")
        return float(us.get("marginSummary", {}).get("accountValue", 0))

    def get_position(self) -> dict | None:
        us = retry(lambda: self.info.user_state(self.main_address),
                   attempts=3, delay=3, label="get_position")
        for ap in us.get("assetPositions", []):
            p = ap.get("position", {})
            if p.get("coin") == COIN and abs(float(p.get("szi", 0))) > 1e-8:
                return p
        return None

    def market_open(self, side: str, size: float):
        is_buy = side == "long"
        return retry(lambda: self.exchange.market_open(COIN, is_buy, size, None, 0.01),
                     attempts=3, delay=3, label="market_open")

    def limit_open(self, side: str, size: float, limit_px: float):
        is_buy = side == "long"
        return retry(
            lambda: self.exchange.order(
                COIN, is_buy, size, limit_px,
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=False,
            ),
            attempts=3, delay=3, label="limit_open",
        )

    def market_close(self):
        return retry(lambda: self.exchange.market_close(COIN),
                     attempts=3, delay=3, label="market_close")

    def limit_close(self, side: str, size: float, limit_px: float):
        is_buy_close = side == "short"
        return retry(
            lambda: self.exchange.order(
                COIN, is_buy_close, size, limit_px,
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=True,
            ),
            attempts=3, delay=3, label="limit_close",
        )

    def attach_tpsl(self, side: str, size: float, tp_px: float, sl_px: float):
        is_buy_close = side == "short"
        tp_type = {"trigger": {"triggerPx": tp_px, "isMarket": True, "tpsl": "tp"}}
        tp_resp = retry(
            lambda: self.exchange.order(COIN, is_buy_close, size, tp_px, tp_type, reduce_only=True),
            attempts=3, delay=3, label="attach_tp",
        )
        # 止损触发单
        sl_type = {"trigger": {"triggerPx": sl_px, "isMarket": True, "tpsl": "sl"}}
        sl_resp = retry(
            lambda: self.exchange.order(COIN, is_buy_close, size, sl_px, sl_type, reduce_only=True),
            attempts=3, delay=3, label="attach_sl",
        )
        return tp_resp, sl_resp

    def cancel_all_orders(self):
        orders = retry(lambda: self.info.open_orders(self.main_address),
                       attempts=3, delay=3, label="cancel_all_orders(open_orders)")
        coin_orders = [o for o in orders if o.get("coin") == COIN]
        if not coin_orders:
            return 0
        canceled = 0
        errors = []
        for o in coin_orders:
            try:
                oid = int(o["oid"])
                retry(lambda: self.exchange.cancel(COIN, oid),
                      attempts=3, delay=2, label=f"cancel oid={oid}")
                canceled += 1
            except Exception as e:
                errors.append(f"oid={o.get('oid')}: {e}")
        if errors:
            raise RuntimeError(f"部分撤单失败: {errors}")
        return canceled

    def get_last_close_fill(self, since_ts: float = 0, close_dir: str = "") -> dict | None:
        try:
            fills = retry(lambda: self.info.user_fills(self.main_address),
                          attempts=3, delay=3, label="get_last_close_fill")
            for f in reversed(fills or []):
                if f.get("coin") != COIN:
                    continue
                d = str(f.get("dir", ""))
                if not d.startswith("Close"):
                    continue
                if close_dir and d != close_dir:
                    continue
                if since_ts and int(f.get("time", 0)) < since_ts * 1000:
                    # 既然是倒序遍历，遇到更老的就可以停了
                    break
                return f
        except Exception as e:
            print(f"[LIVE][WARN] 查询成交记录失败: {e}")
        return None


# 实盘模式下创建单例
LIVE: LiveTrader | None = LiveTrader() if LIVE_TRADING else None


def sync_from_chain(state):
    """实盘模式下，每轮决策前从链上同步真实余额和仓位。"""
    if not LIVE:
        return
    try:
        state.balance = LIVE.get_balance()
        pos = LIVE.get_position()
        if pos:
            # 链上有仓位：以链上数据为准（覆盖本地状态）
            szi = float(pos["szi"])
            chain_side = "long" if szi > 0 else "short"
            entry = float(pos.get("entryPx", 0))
            size = abs(szi)
            if not state.position or state.position["entry"] != entry:
                # 本地无仓位或链上换了新仓：用链上数据重建本地跟踪
                if not state.position:
                    state.position = {
                        "side": chain_side, "entry": entry, "size": size,
                        "entry_fee": 0, "open_ts": time.time(),
                        "tp": 0, "sl": 0, "margin_usd": 0,
                        "entry_reason": "(synced from chain)"
                    }
                    print(f"[LIVE] 从链上同步仓位: {chain_side} {size}@${entry}")
        else:
            # 链上无仓位：清掉本地状态
            if state.position:
                # 之前有仓位，现在没了——说明 TP/SL 被交易所端触发了（或外部手动平了）
                # 撤销可能残留的另一半孤儿单
                try:
                    n = LIVE.cancel_all_orders()
                    if n > 0:
                        print(f"[LIVE] 检测到仓位已平，已撤销 {n} 个孤儿挂单")
                except Exception as e:
                    print(f"[LIVE][WARN] 撤孤儿单失败: {e}")
                old_p = state.position
                state.position = None
                # 查最近一笔 COIN 平仓成交，拿真实成交价和盈亏
                # 必须过滤时间和方向，否则会拿错（用户可能有多笔历史仓位）
                # - since_ts = 10 分钟前（TP/SL 触发后应该在很短时间内出现在 fills 里）
                # - close_dir = "Close Long"/"Close Short" 必须匹配原持仓方向
                expected_dir = "Close Long" if old_p["side"] == "long" else "Close Short"
                close_fill = LIVE.get_last_close_fill(
                    since_ts=time.time() - 600,
                    close_dir=expected_dir,
                )
                if close_fill:
                    exit_px = float(close_fill.get("px", 0))
                    closed_pnl = float(close_fill.get("closedPnl", 0))
                    fee = float(close_fill.get("fee", 0))
                    direction = close_fill.get("dir", "Close")
                    net = closed_pnl - fee
                    state.daily_pnl += net
                    state.trades.append(SimTrade(
                        now_str(), old_p["side"], old_p["entry"], exit_px,
                        old_p["size"], closed_pnl + fee, fee, net,
                        f"exchange_triggered ({direction})",
                        old_p.get("tp", 0), old_p.get("sl", 0),
                        time.time() - old_p.get("open_ts", time.time())
                    ))
                    print(f"[LIVE] 仓位被交易所触发平仓: {direction} @${exit_px:.1f} "
                          f"净盈亏=${net:+.4f}")
                    # 记进 decision_log 让 LLM 下一轮能看到
                    log_decision(state, "EXCHANGE_CLOSE",
                                 summary=f"仓位被交易所端触发平仓（{direction}）",
                                 detail=f"入场 ${old_p['entry']:.1f} → 平仓 ${exit_px:.1f}，"
                                        f"净盈亏 ${net:+.4f}（含手续费 ${fee:.4f}）。"
                                        f"原 TP={old_p.get('tp', 0):.0f}pts SL={old_p.get('sl', 0):.0f}pts")
                    emoji = "🎯" if net >= 0 else "💥"
                    send_telegram(
                        f"{emoji} *交易所自动平仓 ({direction})*\n"
                        f"入场 `${old_p['entry']:.1f}` → 平仓 `${exit_px:.1f}`\n"
                        f"净盈亏 `${net:+.4f}` (含手续费 ${fee:.4f})\n"
                        f"原 TP {old_p.get('tp', 0):.0f}pts / SL {old_p.get('sl', 0):.0f}pts"
                    )
                else:
                    # 查不到成交记录（罕见，可能是 API 延迟）
                    state.trades.append(SimTrade(
                        now_str(), old_p["side"], old_p["entry"], 0,
                        old_p["size"], 0, 0, 0, "exchange_close_unknown_price",
                        old_p.get("tp", 0), old_p.get("sl", 0), 0
                    ))
                    print(f"[LIVE] 仓位已平（成交价未知，可能是 API 延迟）")
                    log_decision(state, "EXCHANGE_CLOSE",
                                 summary="仓位被平（成交价查不到）",
                                 detail=f"原 {old_p['side']} 入场 ${old_p['entry']:.1f}，请查 UI 确认盈亏")
                    send_telegram(
                        f"❓ *仓位被平（成交价未知）*\n"
                        f"原 {old_p['side']} 入场 `${old_p['entry']:.1f}`，请到 Hyperliquid UI 查盈亏"
                    )
    except Exception as e:
        print(f"[LIVE][WARN] 从链上同步失败: {e}")


def sim_open(state, side, price, margin_usd, tp_pts, sl_pts, reason, limit_price=None):
    if state.daily_pnl <= -MAX_DAILY_LOSS:
        return f"已拦截：单日亏损达上限 ${state.daily_pnl:.2f}"
    margin_usd = max(1.0, min(margin_usd, MAX_MARGIN_USD))
    # 已有同方向仓位 → 加仓；已有反方向 → 拒绝（要求先 close）
    if state.position and state.position["side"] != side:
        return (f"已拦截：当前持有 {state.position['side']} 仓位，"
                f"不能直接开反向 {side}。请先 close_position 再开新仓。")
    adding = state.position is not None

    pv = margin_usd * LEVERAGE

    if LIVE_TRADING:
        assert LIVE is not None
        # 没传 limit_price → 用当前价兜底
        if limit_price is None or limit_price <= 0:
            limit_price = price
        size = round(pv / limit_price, 5)
        if size < 0.001:
            return f"已拦截：下单数量 {size} {COIN} 过小（最小 0.001）"
        try:
            # 1. 限价 IoC 开仓
            res = LIVE.limit_open(side, size, limit_price)
            print(f"  [LIVE-ORDER] 限价开仓 {side} 数量={size} @${limit_price:.1f} 返回={res}")
            # 2. 检查是否成交（IoC 可能未成交）
            statuses = (res.get("response", {}).get("data", {}).get("statuses", []) if isinstance(res, dict) else [])
            filled = None
            for s in statuses:
                if "filled" in s:
                    filled = s["filled"]
                    break
                if "error" in s:
                    return f"未成交（交易所拒绝）：{s['error']}"
            if not filled:
                # 根据 side 给出明确的调整建议
                if side == "long":
                    hint = f"买入需填 ≥ 当前价 ${price:.1f}（建议 ${price+10:.0f}~${price+15:.0f}）"
                else:
                    hint = f"卖出需填 ≤ 当前价 ${price:.1f}（建议 ${price-15:.0f}~${price-10:.0f}）"
                return (f"未成交：限价 ${limit_price:.1f} 无法立即匹配（当前价 ${price:.1f}）。{hint}。"
                        f"下一轮请用更激进的 limit_price 重试。")
            entry = float(filled.get("avgPx", limit_price))
            actual_fill_sz = float(filled.get("totalSz", size))
            time.sleep(1.0)
            # 3. 读回链上仓位（获取加仓后的总 size 和加权入场价）
            pos = LIVE.get_position()
            if pos:
                entry = float(pos.get("entryPx", entry))
                actual_size = abs(float(pos.get("szi", actual_fill_sz)))
            else:
                actual_size = actual_fill_sz
            fee = pv * FEE_RATE
            # 4. 加仓时先撤旧 TP/SL，然后基于新加权 entry 重挂
            if adding:
                try:
                    n = LIVE.cancel_all_orders()
                    if n > 0:
                        print(f"  [LIVE-ORDER] 加仓前撤掉 {n} 个旧 TP/SL")
                except Exception as e:
                    print(f"  [LIVE][WARN] 加仓撤旧单失败: {e}")
            # 5. 挂新的 TP/SL 触发单覆盖全部仓位
            tp_px = entry + tp_pts if side == "long" else entry - tp_pts
            sl_px = entry - sl_pts if side == "long" else entry + sl_pts
            try:
                tp_resp, sl_resp = LIVE.attach_tpsl(side, actual_size, tp_px, sl_px)
                print(f"  [LIVE-ORDER] 挂 TP=@${tp_px:.1f} SL=@${sl_p if False else sl_px:.1f} 覆盖全部仓位 size={actual_size}")
            except Exception as e:
                print(f"  [LIVE][WARN] 挂 TP/SL 触发单失败: {e}（仓位已开，但需手动盯盘）")
        except Exception as e:
            return f"错误：限价开仓失败：{e}"
    else:
        # 模拟模式：用 limit_price 作为成交价（如果传了），否则用 price ± slippage
        if limit_price and limit_price > 0:
            entry = limit_price
        else:
            entry = price * (1 + SLIPPAGE) if side == "short" else price * (1 - SLIPPAGE)
        actual_size = pv / entry
        fee = pv * FEE_RATE

    state.position = {"side": side, "entry": entry, "size": actual_size, "entry_fee": fee,
                      "open_ts": time.time() if not adding else state.position["open_ts"],
                      "tp": tp_pts, "sl": sl_pts,
                      "margin_usd": margin_usd + (state.position["margin_usd"] if adding else 0),
                      "entry_reason": reason}
    state.balance -= fee
    state.daily_trades += 1
    tp_p = entry - tp_pts if side == "short" else entry + tp_pts
    sl_p = entry + sl_pts if side == "short" else entry - sl_pts
    tag = "LIVE" if LIVE_TRADING else "SIM"
    action = "加仓" if adding else "开仓"
    msg = f"已{action} {side.upper()} @${entry:.1f} TP=${tp_p:.1f} SL=${sl_p:.1f} 手续费=${fee:.4f} 余额=${state.balance:.2f}"
    print(f"  [{tag}] {msg}")
    return msg


def sim_close(state, price, reason, llm_reason, limit_price=None):
    p = state.position
    if not p:
        return "无仓位。"

    if LIVE_TRADING:
        assert LIVE is not None
        # 没传 limit_price → 用当前价兜底（fallback 到市价）
        if limit_price is None or limit_price <= 0:
            limit_price = price
        try:
            # 1. 先撤销所有挂单（包括 TP/SL 触发单），避免平仓后留下孤儿单
            try:
                n = LIVE.cancel_all_orders()
                if n > 0:
                    print(f"  [LIVE-ORDER] 已撤销 {n} 个挂单（含 TP/SL 触发单）")
                else:
                    print(f"  [LIVE-ORDER] 无挂单需要撤销")
            except Exception as e:
                print(f"  [LIVE][WARN] 撤单失败: {e}（继续尝试平仓）")
            time.sleep(0.5)
            # 2. 限价 IoC 平仓
            res = LIVE.limit_close(p["side"], p["size"], limit_price)
            print(f"  [LIVE-ORDER] 限价平仓 {p['side']} size={p['size']} @${limit_price:.1f} 返回={res}")
            # 3. 检查是否成交
            statuses = (res.get("response", {}).get("data", {}).get("statuses", []) if isinstance(res, dict) else [])
            filled = None
            for s in statuses:
                if "filled" in s:
                    filled = s["filled"]
                    break
                if "error" in s:
                    return f"未成交（交易所拒绝）：{s['error']}"
            if not filled:
                # 平仓方向：long 仓位平仓=卖出，short 仓位平仓=买入
                close_side = "卖出" if p["side"] == "long" else "买入"
                if p["side"] == "long":
                    hint = f"平多仓={close_side}，需填 ≤ 当前价 ${price:.1f}（建议 ${price-15:.0f}~${price-10:.0f}）"
                else:
                    hint = f"平空仓={close_side}，需填 ≥ 当前价 ${price:.1f}（建议 ${price+10:.0f}~${price+15:.0f}）"
                return (f"未成交：限价 ${limit_price:.1f} 无法立即匹配（当前价 ${price:.1f}）。{hint}。"
                        f"仓位未平，下一轮请用更激进的 limit_price 重试。")
            slip = float(filled.get("avgPx", limit_price))
            actual_size = float(filled.get("totalSz", p["size"]))
            time.sleep(0.5)
            # 读链上确认已平
            pos = LIVE.get_position()
            if pos and abs(float(pos.get("szi", 0))) > 1e-8:
                print(f"  [LIVE][WARN] 平仓后链上仍有残留仓位: {pos}")
        except Exception as e:
            return f"错误：限价平仓失败：{e}"
    else:
        # 模拟模式：用 limit_price 或 price ± slippage
        if limit_price and limit_price > 0:
            slip = limit_price
        else:
            slip = price * (1 - SLIPPAGE) if p["side"] == "short" else price * (1 + SLIPPAGE)
        actual_size = p["size"]

    gross = (p["entry"] - slip) * actual_size if p["side"] == "short" else (slip - p["entry"]) * actual_size
    exit_fee = actual_size * slip * FEE_RATE
    net = gross - exit_fee
    state.balance += gross - exit_fee
    state.daily_pnl += net
    dur = time.time() - p["open_ts"]
    state.trades.append(SimTrade(now_str(), p["side"], p["entry"], slip, actual_size,
                                 gross, p["entry_fee"] + exit_fee, net, reason, p["tp"], p["sl"], dur))
    state.position = None
    tag = "LIVE" if LIVE_TRADING else "SIM"
    msg = f"已平仓 @${slip:.1f} 净盈亏=${net:+.4f} 余额=${state.balance:.2f} 持仓时长={dur:.0f}秒"
    print(f"  [{tag}] {msg} | {llm_reason}")
    save_state(state); write_report(state)
    return msg


def check_tp_sl(state, price):
    if not state.position:
        return False
    # 实盘模式下，TP/SL 由交易所端触发单处理，本地不再监控
    if LIVE_TRADING:
        return False
    p = state.position
    # tp/sl = 0 表示未设置（如从链上同步来的已有仓位），跳过自动平仓，让 LLM 决定
    if not p.get("tp") or not p.get("sl"):
        return False
    if p["side"] == "short":
        if price <= p["entry"] - p["tp"]: sim_close(state, price, "tp", "触发止盈"); return True
        if price >= p["entry"] + p["sl"]: sim_close(state, price, "sl", "触发止损"); return True
    else:
        if price >= p["entry"] + p["tp"]: sim_close(state, price, "tp", "触发止盈"); return True
        if price <= p["entry"] - p["sl"]: sim_close(state, price, "sl", "触发止损"); return True
    return False


# ─── LLM agent loop ─────────────────────────────────────

def log_decision(state, action: str, summary: str, detail: str = ""):
    """追加一条决策日志，保留最近 30 条。"""
    entry = {
        "ts": now_str(),
        "action": action,        # HOLD / OPEN / CLOSE / SEARCH
        "price": round(state.last_price, 1),
        "summary": summary[:200], # 一句话摘要（必填）
        "detail": detail[:500],   # 详情（可选，如搜索结果要点）
    }
    state.decision_log.append(entry)
    # 滚动保留最近 30 条
    if len(state.decision_log) > 30:
        state.decision_log = state.decision_log[-30:]


def adjust_tpsl(state, tp_pts, sl_pts, reason):
    """调整当前仓位的 TP/SL 触发单（实盘：撤旧 → 挂新；模拟：只更新本地状态）。"""
    p = state.position
    if not p:
        return "无仓位，无法调整 TP/SL。"

    if LIVE_TRADING:
        assert LIVE is not None
        try:
            # 1. 撤掉旧 TP/SL
            try:
                n = LIVE.cancel_all_orders()
                if n > 0:
                    print(f"  [LIVE-ORDER] 调整 TP/SL 前，撤掉 {n} 个旧触发单")
            except Exception as e:
                print(f"  [LIVE][WARN] 撤旧单失败: {e}")
            time.sleep(0.3)
            # 2. 基于当前入场价计算新的 TP/SL 价格
            entry = p["entry"]
            tp_px = entry + tp_pts if p["side"] == "long" else entry - tp_pts
            sl_px = entry - sl_pts if p["side"] == "long" else entry + sl_pts
            # 3. 挂新的（覆盖全部仓位 size）
            try:
                tp_resp, sl_resp = LIVE.attach_tpsl(p["side"], p["size"], tp_px, sl_px)
                print(f"  [LIVE-ORDER] TP/SL 已调整：TP=@${tp_px:.1f} SL=@${sl_px:.1f} size={p['size']}")
            except Exception as e:
                return f"错误：挂新 TP/SL 失败: {e}（旧单已撤，仓位暂时无 TP/SL 保护！）"
        except Exception as e:
            return f"错误：调整 TP/SL 失败: {e}"

    # 更新本地状态
    old_tp, old_sl = p["tp"], p["sl"]
    p["tp"] = tp_pts
    p["sl"] = sl_pts

    entry = p["entry"]
    tp_px = entry + tp_pts if p["side"] == "long" else entry - tp_pts
    sl_px = entry - sl_pts if p["side"] == "long" else entry + sl_pts
    tag = "LIVE" if LIVE_TRADING else "SIM"
    msg = (f"TP/SL 已调整：TP {old_tp:.0f}→{tp_pts:.0f}pts (@${tp_px:.1f})  "
           f"SL {old_sl:.0f}→{sl_pts:.0f}pts (@${sl_px:.1f})")
    print(f"  [{tag}] {msg}")
    return msg


def execute_tool(name, args, state, price):
    if name == "open_position":
        side = args["side"]
        reason = args.get("reason", "")
        margin = float(args.get("margin_usd", MAX_MARGIN_USD))
        limit_price = float(args.get("limit_price", 0))
        tp = float(args["take_profit_points"])
        sl = float(args["stop_loss_points"])
        result = sim_open(state, side, price, margin, tp, sl, reason, limit_price=limit_price)
        log_decision(state, f"OPEN {side.upper()}",
                     summary=reason,
                     detail=f"limit=${limit_price:.1f} margin=${margin:.0f} TP={tp:.0f}pts SL={sl:.0f}pts → {result[:120]}")
        # 仅在确实开/加仓成功时发通知
        if result.startswith("已开仓") or result.startswith("已加仓"):
            emoji = "🟢" if side == "long" else "🔴"
            send_telegram(
                f"{emoji} *{'加仓' if '加仓' in result else '开仓'} {side.upper()}*\n"
                f"限价 `${limit_price:.1f}` | 保证金 `${margin:.0f}` | {LEVERAGE}x\n"
                f"TP `{tp:.0f}`pts / SL `{sl:.0f}`pts\n"
                f"理由：{reason}"
            )
        else:
            send_telegram(f"⚠️ *开仓未成交*\n{result}")
        return result
    if name == "close_position":
        reason = args.get("reason", "")
        limit_price = float(args.get("limit_price", 0))
        result = sim_close(state, price, "llm_close", reason, limit_price=limit_price)
        log_decision(state, "CLOSE", summary=reason, detail=result[:200])
        if result.startswith("已平仓"):
            send_telegram(f"✖️ *平仓*\n限价 `${limit_price:.1f}`\n理由：{reason}\n`{result}`")
        else:
            send_telegram(f"⚠️ *平仓未成交*\n{result}")
        return result
    if name == "search_web":
        query = args["query"]
        print(f"  [搜索] '{query}'")
        result = search_web(query)
        print(f"  [搜索结果] {result[:200]}")
        log_decision(state, "SEARCH",
                     summary=f"搜了：{query}",
                     detail=f"结果摘要：{result[:300]}")
        return result
    if name == "adjust_tpsl":
        tp_pts = float(args["take_profit_points"])
        sl_pts = float(args["stop_loss_points"])
        reason = args.get("reason", "")
        result = adjust_tpsl(state, tp_pts, sl_pts, reason)
        log_decision(state, "ADJUST_TPSL",
                     summary=reason,
                     detail=f"TP={tp_pts:.0f}pts SL={sl_pts:.0f}pts → {result[:150]}")
        if result.startswith("TP/SL 已调整"):
            entry = state.position["entry"] if state.position else 0
            side = state.position["side"] if state.position else "?"
            tp_px = entry + tp_pts if side == "long" else entry - tp_pts
            sl_px = entry - sl_pts if side == "long" else entry + sl_pts
            send_telegram(
                f"🔧 *调整 TP/SL*\n"
                f"仓位 `{side.upper()}` 入场 `${entry:.1f}`\n"
                f"新 TP `${tp_px:.1f}` ({tp_pts:.0f}pts)\n"
                f"新 SL `${sl_px:.1f}` ({sl_pts:.0f}pts)\n"
                f"理由：{reason}"
            )
        else:
            send_telegram(f"⚠️ *调整 TP/SL 失败*\n{result}")
        return result
    if name == "hold":
        reason = args.get("reason", "")
        if state.position:
            p = state.position
            u = (p["entry"] - price) * p["size"] if p["side"] == "short" else (price - p["entry"]) * p["size"]
            log_decision(state, "HOLD", summary=reason,
                         detail=f"持仓 {p['side']} uPnL=${u:+.4f}")
            emoji = "📈" if u >= 0 else "📉"
            send_telegram(
                f"⏸️ *HOLD* (持仓中)\n"
                f"价格 `${price:.1f}` | {emoji} uPnL `${u:+.4f}`\n"
                f"理由：{reason}"
            )
            return f"已继续持有 {p['side']} 仓位。未实现盈亏=${u:+.4f}"
        log_decision(state, "HOLD", summary=reason, detail="空仓等待")
        send_telegram(f"⏸️ *HOLD* (空仓)\n价格 `${price:.1f}`\n理由：{reason}")
        return "已继续空仓观望。"
    if name == "reply_to_user":
        message = args.get("message", "")
        send_telegram(f"💬 *Bot 回复*\n{message}")
        log_decision(state, "REPLY", summary=f"我回复：{message[:500]}")
        state.did_reply_this_round = True
        return "消息已发送给用户。"
    return f"未知工具: {name}"


def run_decision(state, mtf, ind, price, user_messages=None):
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("[SKIP] 未设置 DEEPSEEK_API_KEY"); return

    client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL)
    context = format_market_context(mtf, ind, state, price, user_messages=user_messages or [])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context + "\n\nMake your trading decision. Call one tool."},
    ]

    extra_body = {}
    if ENABLE_THINKING:
        extra_body["thinking"] = {"type": "enabled"}

    for rnd in range(20):
        print(f"  [V4-Pro 第{rnd+1}轮] 思考={'开' if ENABLE_THINKING else '关'} 力度={REASONING_EFFORT}...")

        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=get_tools(),
            tool_choice="auto",
            timeout=DECISION_TIMEOUT,
            reasoning_effort=REASONING_EFFORT,
            extra_body=extra_body,
        )

        msg = resp.choices[0].message

        reasoning = getattr(msg, "reasoning_content", None) or ""
        if reasoning:
            print(f"  [思考] ({len(reasoning)} 字符，已隐藏)")

        if msg.content:
            print(f"  [V4-Pro 回复]\n{msg.content}")

        if not msg.tool_calls:
            return

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls],
        })

        made_trade_action = False
        next_wait = None  # LLM 传的 wait_seconds，由主循环使用
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except:
                args = {}
            print(f"  [调用] {name}({json.dumps(args, ensure_ascii=False)})")
            result = execute_tool(name, args, state, price)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if name in ("open_position", "close_position", "hold", "adjust_tpsl"):
                # 只有真正成功才结束本轮；未成交/报错继续让 LLM 重试
                succeeded = result.startswith("已") or result.startswith("TP/SL 已")
                if succeeded:
                    made_trade_action = True
                    raw_wait = float(args.get("wait_seconds", 180))
                    next_wait = max(MIN_WAIT, min(int(raw_wait), MAX_WAIT))
                else:
                    print(f"  [重试] {name} 未成功，继续让 LLM 决策")

        if made_trade_action:
            return next_wait

        # 没做出交易动作（例如只调了 search_web）→ 继续下一轮，让 LLM 看到搜索结果后再决策


# ─── Utils ──────────────────────────────────────────────

def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def setup_logging():
    import logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"agent_{today}.log"
    logger = logging.getLogger("llm_scalper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    class Redirect:
        def write(self, t):
            for l in t.rstrip().splitlines():
                logger.info(l)
        def flush(self): pass
    sys.stdout = Redirect()
    sys.stderr = Redirect()


def daily_reset(state):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.daily_reset != today:
        state.daily_trades = 0
        state.daily_pnl = 0.0
        state.daily_reset = today
        print(f"\n[DAILY RESET] {today} | 余额: ${state.balance:.2f}")


def save_state(state):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "balance": round(state.balance, 4),
        "daily_trades": state.daily_trades,
        "daily_pnl": round(state.daily_pnl, 4),
        "daily_reset": state.daily_reset,
        "decisions": state.decisions,
        "uptime_hours": round((time.time() - state.start_time) / 3600, 2) if state.start_time else 0,
        "total_trades": len(state.trades),
        "position": ({"side": state.position["side"], "entry": round(state.position["entry"], 2),
                      "size": state.position.get("size", 0),
                      "tp": state.position["tp"], "sl": state.position["sl"],
                      "margin_usd": state.position.get("margin_usd", 0),
                      "open_ts": state.position.get("open_ts", 0),
                      "entry_reason": state.position.get("entry_reason", "")}
                     if state.position else None),
        "trades": [{"ts": t.ts, "side": t.side, "entry": round(t.entry, 2), "exit": round(t.exit_, 2),
                    "net": round(t.net, 4), "fees": round(t.fees, 4), "reason": t.reason,
                    "tp": t.tp, "sl": t.sl, "dur_s": round(t.dur, 1)} for t in state.trades],
        "decision_log": state.decision_log[-30:],  # 持久化最近 30 条决策日志
        "last_tg_update_id": state.last_tg_update_id,  # 持久化 TG 已读位置
        "saved_at": now_str(),
    }
    STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_state(state):
    """重启时从 STATE_FILE 恢复 daily 计数、交易历史和模拟仓位。
    实盘模式下，仓位以 sync_from_chain 的链上数据为准（在 load 之后调用）。"""
    if not STATE_FILE.exists():
        return
    try:
        data = json.loads(STATE_FILE.read_text())
        state.daily_trades = data.get("daily_trades", 0)
        state.daily_pnl = data.get("daily_pnl", 0)
        state.daily_reset = data.get("daily_reset", "")
        state.decisions = data.get("decisions", 0)
        # 恢复本地仓位（实盘模式也恢复，sync_from_chain 会验证是否匹配链上并保留 tp/sl）
        if data.get("position"):
            p = data["position"]
            state.position = {
                "side": p["side"], "entry": p["entry"], "size": p.get("size", 0),
                "entry_fee": 0, "open_ts": p.get("open_ts", time.time()),
                "tp": p.get("tp", 0), "sl": p.get("sl", 0),
                "margin_usd": p.get("margin_usd", 0),
                "entry_reason": p.get("entry_reason", ""),
            }
        # 恢复交易历史
        for t in data.get("trades", [])[-200:]:
            state.trades.append(SimTrade(
                t["ts"], t["side"], t["entry"], t["exit"],
                0, 0, t["fees"], t["net"], t["reason"],
                t["tp"], t["sl"], t["dur_s"]
            ))
        # 模拟模式下恢复余额（实盘模式 sync_from_chain 覆盖）
        if not LIVE_TRADING:
            state.balance = data.get("balance", state.balance)
        # 恢复决策日志（memory）
        state.decision_log = data.get("decision_log", [])[-30:]
        state.last_tg_update_id = data.get("last_tg_update_id", 0)
        print(f"[STATE] 已载入：今日交易={state.daily_trades} 今日盈亏=${state.daily_pnl:+.2f} "
              f"历史={len(state.trades)}笔 持仓={'有' if state.position else '无'} "
              f"memory={len(state.decision_log)}条")
    except Exception as e:
        print(f"[STATE][WARN] 载入失败: {e}，从空状态开始")


def write_report(state):
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    t = state.trades
    w = [x for x in t if x.net > 0]
    total_net = sum(x.net for x in t)
    total_fees = sum(x.fees for x in t)
    up = (time.time() - state.start_time) / 3600 if state.start_time else 0
    lines = [
        "# LLM AI 交易 Agent 报告",
        "",
        f"模型: {LLM_MODEL} (thinking={'on' if ENABLE_THINKING else 'off'}, effort={REASONING_EFFORT})",
        f"运行: {up:.1f}h | 决策: {state.decisions} | 交易: {len(t)} (盈{len(w)} 亏{len(t)-len(w)})",
        (f"余额: ${state.balance:.2f} (实盘链上余额)" if LIVE_TRADING
         else f"余额: ${state.balance:.2f} (初始 ${INITIAL_BALANCE})"),
        f"净盈亏: ${total_net:+.4f} | 手续费: ${total_fees:.4f}",
        f"胜率: {len(w)/max(len(t),1)*100:.1f}%",
        "",
        "## 最近 20 笔",
        "| 时间 | 方向 | 入场 | 出场 | 净盈亏 | TP | SL | 原因 |",
        "|------|------|------|------|--------|----|----|------|",
    ]
    for x in t[-20:]:
        lines.append(f"| {x.ts} | {x.side} | {x.entry:.1f} | {x.exit_:.1f} | ${x.net:+.4f} | {x.tp:.0f} | {x.sl:.0f} | {x.reason[:40]} |")
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


# ─── Main ───────────────────────────────────────────────

def main():
    setup_logging()
    mode = "🔴 实盘（真金白银）" if LIVE_TRADING else "🟢 模拟"
    print("=" * 60)
    print(f"  LLM AI 交易 Agent — DeepSeek V4-Pro [{mode}]")
    print("=" * 60)
    print(f"  模型:     {LLM_MODEL}")
    print(f"  思考:     {'开' if ENABLE_THINKING else '关'} | 力度: {REASONING_EFFORT}")
    print(f"  币种:     {COIN} {LEVERAGE}x 单笔最大 ${MAX_MARGIN_USD}（AI 自定大小）")
    print(f"  间隔:     AI 自决（{MIN_WAIT}-{MAX_WAIT}秒/轮）| K线: {LOOKBACK_CANDLES}×{CANDLE_TIMEFRAME}")
    print(f"  工具:     open_position / close_position / search_web / hold")
    print(f"  通知:     {'✅ Telegram 已激活' if TG_ENABLED else '❌ Telegram 未配置'}")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("  ⚠️  请在 .env 中设置 DEEPSEEK_API_KEY")
    print("=" * 60)

    state = AgentState()
    state.start_time = time.time()

    # 从磁盘恢复 daily 计数和交易历史（避免重启风控失效）
    load_state(state)

    # 实盘模式：启动时立即从链上同步真实余额/仓位（在 banner 之后、循环之前）
    if LIVE_TRADING:
        assert LIVE is not None
        print(f"  [LIVE] 主钱包: {LIVE.main_address}")
        print(f"  [LIVE] Agent 签名: {LIVE.agent_address}")
        print(f"  [LIVE] 正在从链上同步...")
        sync_from_chain(state)
        # 显示完整余额信息
        try:
            total_eq = LIVE.get_total_equity()
            avail = state.balance
            print(f"  [LIVE] 账户总权益: ${total_eq:.2f}（含未实现盈亏）")
            print(f"  [LIVE] 可用保证金: ${avail:.2f}（可用来开新仓位）")
        except Exception as e:
            print(f"  [LIVE][WARN] 读余额失败: {e}")
        if state.position:
            p = state.position
            print(f"  [LIVE] 已有 {COIN} 仓位: {p['side'].upper()} {p['size']:.5f} @ ${p['entry']:.1f}")
            print(f"  ⚠️  [LIVE] 注意：这是链上已有的 {COIN} 仓位（可能你手动开的），bot 会接着管理它")
        else:
            print(f"  [LIVE] 当前无 {COIN} 仓位")
        if state.balance < MAX_MARGIN_USD:
            print(f"  ⚠️  [LIVE] 可用 ${state.balance:.2f} < 单笔上限 ${MAX_MARGIN_USD}，可能无法开仓")
        print(f"  ⚠️  [LIVE] 实盘模式已开启，AI 下单将真实成交！")
        print("=" * 60)
    else:
        print(f"  余额:     ${state.balance:.2f}（模拟）")
        print("=" * 60)

    def stop(sig, frame):
        print("\n[关闭]"); state.running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print(f"\nAgent 已启动。等待时间由 AI 决定（{MIN_WAIT}-{MAX_WAIT} 秒/轮）。\n")

    # next_wait：上一轮 LLM 决定的等待时间。初始 None → 第一次立即决策
    next_wait = None
    last_dec = 0
    poll = 0

    while state.running:
        try:
            poll += 1
            daily_reset(state)

            # 实盘模式：每轮从链上同步余额/仓位（链上为准）
            if LIVE_TRADING:
                sync_from_chain(state)

            price = fetch_price()
            if price is None:
                time.sleep(10); continue
            state.last_price = price

            if state.position:
                check_tp_sl(state, price)
                if state.position and poll % 12 == 0:
                    p = state.position
                    u = (p["entry"] - price) * p["size"] if p["side"] == "short" else (price - p["entry"]) * p["size"]
                    print(f"[监控] ${price:.1f} {p['side']} 未实现盈亏=${u:+.4f} 已持仓 {(time.time()-p['open_ts'])/60:.1f}分钟")

            # 等待 LLM 决定的 wait_seconds 过去再决策（首次 next_wait=None → 立即决策）
            wait_target = next_wait if next_wait else 0
            if time.time() - last_dec < wait_target:
                time.sleep(5); continue

            last_dec = time.time()
            state.decisions += 1
            print(f"\n{'='*50}\n[决策 #{state.decisions}] {now_str()}")

            mtf = fetch_multi_timeframe()
            candles_1h = mtf.get("1h", [])
            if len(candles_1h) < 5:
                print("[SKIP] 1h K线不足"); next_wait = 60; continue

            ind = compute_indicators(candles_1h)
            total = sum(len(v) for v in mtf.values())
            print(f"  ${price:.1f} RSI={ind.get('rsi14','?')} SMA25={ind.get('sma25','?')} | "
                  f"candles: {', '.join(f'{k}={len(v)}' for k,v in mtf.items() if v)} ({total} total)")

            # 拉取 Telegram 新消息（如果有用户反馈，会传给 LLM）
            user_messages = fetch_tg_messages(state)
            # 把用户消息也记进 memory，这样下一轮 LLM 还能看到完整对话
            for msg in user_messages:
                log_decision(state, "USER_MSG", summary=f"用户说：{msg[:200]}")

            # 重置"本轮是否回复"标记（execute_tool 里 reply_to_user 会设 True）
            state.did_reply_this_round = False

            # 运行决策；返回 LLM 决定的 wait_seconds（None 表示没动作，用兜底间隔）
            returned_wait = run_decision(state, mtf, ind, price, user_messages=user_messages)
            next_wait = returned_wait if returned_wait else 180  # 兜底 3 分钟
            # 如果有用户消息但 LLM 没回复，缩短等待（避免让用户等太久）
            if user_messages and not state.did_reply_this_round and returned_wait and returned_wait > 120:
                print(f"  [TG] 用户消息未回复，缩短等待 {returned_wait}→60s")
                next_wait = 60
            print(f"  [下次决策] {next_wait} 秒后")
            save_state(state)
            write_report(state)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback; traceback.print_exc()
            time.sleep(15)

    save_state(state)
    write_report(state)
    print(f"\nStopped. Balance: ${state.balance:.2f} Trades: {len(state.trades)}")


if __name__ == "__main__":
    main()
