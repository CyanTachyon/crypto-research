#!/usr/bin/env python3
"""诊断 prompt 缓存断点：连续跑两次 build_context，diff 看哪里变了。"""
import os, sys, time
from pathlib import Path

_env = Path("/home/cyan/default/crypto/.env")
for line in _env.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, "/home/cyan/default/crypto/scripts")
from llm_scalper import (
    fetch_price, fetch_multi_timeframe, compute_indicators,
    format_market_context, AgentState, COIN
)

state = AgentState()
state.last_price = 0

print("=== 第一次采样 ===")
p1 = fetch_price()
state.last_price = p1
mtf1 = fetch_multi_timeframe()
ind1 = compute_indicators(mtf1.get("1h", []))
ctx1 = format_market_context(mtf1, ind1, state, p1)
print(f"价格=${p1:.1f}, ctx 长度={len(ctx1)} chars")

print("\n等待 8 秒...\n")
time.sleep(8)

print("=== 第二次采样 ===")
p2 = fetch_price()
state.last_price = p2
mtf2 = fetch_multi_timeframe()
ind2 = compute_indicators(mtf2.get("1h", []))
ctx2 = format_market_context(mtf2, ind2, state, p2)
print(f"价格=${p2:.1f}, ctx 长度={len(ctx2)} chars")

# 找第一个不同的字符位置
if ctx1 == ctx2:
    print("\n✅ 两次完全相同 — 缓存应该 100% 命中")
else:
    common_prefix_len = 0
    for i, (a, b) in enumerate(zip(ctx1, ctx2)):
        if a != b:
            common_prefix_len = i
            break
    else:
        common_prefix_len = min(len(ctx1), len(ctx2))
    pct = common_prefix_len / max(len(ctx1), 1) * 100
    print(f"\n❌ 缓存在字符 {common_prefix_len} 处断开（{pct:.1f}% 命中）")
    print(f"\n=== 断点上下文（前 200 字符）===")
    start = max(0, common_prefix_len - 100)
    end = min(len(ctx1), common_prefix_len + 100)
    print(f"位置 {start}-{end}:")
    print(ctx1[start:end])
    print(f"\n=== 第一次独有的尾巴（最后 300 字符）===")
    print(ctx1[-300:])
