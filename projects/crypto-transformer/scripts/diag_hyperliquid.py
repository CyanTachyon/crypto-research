#!/usr/bin/env python3
"""诊断 Hyperliquid 账户状态：打印所有可能的余额字段。"""
import json
import os
import sys
from pathlib import Path

# 加载 .env
_env = Path(__file__).resolve().parent.parent / ".env"
env_loaded = {}
for line in _env.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        env_loaded[k.strip()] = v.strip()
        os.environ.setdefault(k.strip(), v.strip())

main_wallet = os.environ.get("HL_MAIN_WALLET", "").strip()
if not main_wallet:
    print("ERROR: HL_MAIN_WALLET 未设置")
    sys.exit(1)

from hyperliquid.info import Info
from hyperliquid.utils import constants

info = Info(constants.MAINNET_API_URL, skip_ws=True)

print(f"\n=== 查询地址: {main_wallet} ===\n")

# 1. 用户状态（仓位 + 保证金）
print("--- user_state ---")
us = info.user_state(main_wallet)
print(json.dumps(us, indent=2, ensure_ascii=False))

print("\n--- 关键字段 ---")
print(f"marginSummary:        {us.get('marginSummary')}")
print(f"crossMarginSummary:   {us.get('crossMarginSummary')}")
print(f"withdrawable:         {us.get('withdrawable')}")
print(f"assetPositions 数量:  {len(us.get('assetPositions', []))}")

# 2. 现货余额（如果存错地方会在这里）
print("\n--- spot_user_state ---")
try:
    spot = info.spot_user_state(main_wallet)
    print(json.dumps(spot, indent=2, ensure_ascii=False))
    print("\n现货余额汇总:")
    for b in spot.get("balances", []):
        if float(b.get("total", 0)) > 0:
            print(f"  {b.get('coin')}: 总={b.get('total')} 持仓={b.get('hold')} 可用={b.get('entryNtl')}")
except Exception as e:
    print(f"(无现货余额或查询失败: {e})")

# 3. 最近的入金记录
print("\n--- 最近 5 笔 fill ---")
try:
    fills = info.user_fills(main_wallet)
    for f in (fills or [])[:5]:
        print(f"  {f.get('time')} {f.get('side')} {f.get('sz')} {f.get('coin')} @{f.get('px')}")
except Exception as e:
    print(f"(查询失败: {e})")

# 4. 检查这个地址在不在 leaderboard / 有没有 clearinghouseState
print("\n--- clearinghouseState 检查 ---")
try:
    # 用 meta 直接查
    meta = info.meta()
    print(f"Perp universe 共 {len(meta.get('universe', []))} 个币")
except Exception as e:
    print(f"(查询失败: {e})")
