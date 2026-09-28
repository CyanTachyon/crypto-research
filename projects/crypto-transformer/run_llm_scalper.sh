#!/bin/bash
# LLM AI 交易 Agent 一键启动脚本（模拟交易）
#
# 用法:
#   ./run_llm_scalper.sh start       后台启动（需要先设置 API key）
#   ./run_llm_scalper.sh stop        停止
#   ./run_llm_scalper.sh status      查看状态
#   ./run_llm_scalper.sh log         查看日志
#   ./run_llm_scalper.sh report      查看交易报告
#   ./run_llm_scalper.sh run         前台运行
#   ./run_llm_scalper.sh restart     重启
#
# 配置:
#   export DEEPSEEK_API_KEY="sk-你的key"
#   # 可选:
#   export LLM_MODEL="deepseek-chat"          # 模型名
#   export LLM_DECISION_INTERVAL="60"         # 决策间隔(秒)
#   export LLM_LOOKBACK_CANDLES="48"          # 回看K线数
#   export LLM_LEVERAGE="10"                  # 杠杆
#   export LLM_MARGIN_PER_TRADE="15"          # 每笔保证金
#   export LLM_TP_POINTS="100"               # 止盈点数
#   export LLM_SL_POINTS="500"               # 止损点数

set -euo pipefail

PROJECT_DIR="/home/cyan/default/crypto"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
SCRIPT="${PROJECT_DIR}/scripts/llm_scalper.py"
PIDFILE="${PROJECT_DIR}/data/llm_scalper.pid"
LOGDIR="${PROJECT_DIR}/data/llm_logs"

mkdir -p "${LOGDIR}"

get_pid() {
    if [ -f "${PIDFILE}" ]; then
        local pid
        pid=$(cat "${PIDFILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo "${pid}"
            return 0
        fi
        rm -f "${PIDFILE}"
    fi
    return 1
}

cmd_start_bg() {
    if pid=$(get_pid); then
        echo "已在运行 (PID ${pid})"
        exit 0
    fi

    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        echo "⚠️  .env 文件不存在！请先创建。"
        exit 1
    fi

    echo "启动 LLM 交易 Agent（后台）..."
    cd "${PROJECT_DIR}"
    # 启动期间的 stderr 写到 startup_error.log，便于排查早期崩溃
    nohup "${PYTHON}" -u "${SCRIPT}" > /dev/null 2> "${PROJECT_DIR}/data/llm_logs/startup_error.log" &
    local pid=$!
    echo "${pid}" > "${PIDFILE}"
    sleep 5
    if kill -0 "${pid}" 2>/dev/null; then
        echo "启动成功 (PID ${pid})"
        echo ""
        echo "日志:   ${LOGDIR}/agent_$(date -u +%Y-%m-%d).log"
        echo "状态:   ${PROJECT_DIR}/data/llm_scalper_state.json"
        echo "报告:   ${PROJECT_DIR}/docs/llm_scalper_report.md"
        echo ""
        echo "查看日志:   $0 log"
        echo "查看状态:   $0 status"
        echo "停止:       $0 stop"
    else
        echo "启动失败！查看错误日志:"
        echo "  ${PROJECT_DIR}/data/llm_logs/startup_error.log"
        echo "  运行日志: ${LOGDIR}/agent_$(date -u +%Y-%m-%d).log"
        echo ""
        echo "=== startup_error.log 内容 ==="
        cat "${PROJECT_DIR}/data/llm_logs/startup_error.log" 2>/dev/null || echo "(空)"
        rm -f "${PIDFILE}"
        exit 1
    fi
}

cmd_stop() {
    if pid=$(get_pid); then
        echo "停止 (PID ${pid})..."
        kill "${pid}"
        sleep 2
        if kill -0 "${pid}" 2>/dev/null; then
            kill -9 "${pid}" 2>/dev/null || true
        fi
        rm -f "${PIDFILE}"
        echo "已停止"
    else
        echo "未运行"
    fi
}

cmd_status() {
    if pid=$(get_pid); then
        local uptime
        uptime=$(ps -o etime= -p "${pid}" 2>/dev/null | tr -d ' ')
        echo "运行中 (PID ${pid}, 已运行 ${uptime})"
        if [ -f "${PROJECT_DIR}/data/llm_scalper_state.json" ]; then
            echo "---"
            "${PYTHON}" -c "
import json
with open('${PROJECT_DIR}/data/llm_scalper_state.json') as f:
    d = json.load(f)
print(f\"  运行时间:   {d.get('uptime_hours', 0):.1f} 小时\")
print(f\"  当前余额:   \${d.get('balance', 0):.2f}\")
print(f\"  决策次数:   {d.get('decisions_made', 0)}\")
print(f\"  总交易数:   {d.get('total_trades', 0)}\")
print(f\"  今日交易:   {d.get('daily_trades', 0)}\")
print(f\"  今日盈亏:   \${d.get('daily_pnl', 0):.2f}\")
pos = d.get('position')
if pos:
    print(f\"  当前持仓:   {pos.get('side','?')} 入场=\${pos.get('entry',0):.1f}\")
else:
    print(f\"  当前持仓:   无\")
"
        fi
    else
        echo "未运行"
    fi
}

cmd_log() {
    local logfile
    logfile="${LOGDIR}/agent_$(date -u +%Y-%m-%d).log"
    if [ -f "${logfile}" ]; then
        echo "=== 最近 80 行日志 ==="
        tail -80 "${logfile}"
        echo ""
        echo "实时跟踪: tail -f ${logfile}"
    else
        echo "没有找到今天的日志文件"
        ls -la "${LOGDIR}" 2>/dev/null || echo "目录为空"
    fi
}

cmd_report() {
    local report="${PROJECT_DIR}/docs/llm_scalper_report.md"
    if [ -f "${report}" ]; then
        cat "${report}"
    else
        echo "报告尚未生成（需要运行并产生交易后）"
    fi
}

cmd_foreground() {
    if pid=$(get_pid); then
        echo "已在后台运行 (PID ${pid})，先停止再前台运行"
        exit 1
    fi
    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        echo "⚠️  .env 文件不存在！"
        exit 1
    fi
    cd "${PROJECT_DIR}"
    exec "${PYTHON}" -u "${SCRIPT}"
}

case "${1:-help}" in
    start|bg)   cmd_start_bg ;;
    stop)       cmd_stop ;;
    status|st)  cmd_status ;;
    log|logs)   cmd_log ;;
    report|rep) cmd_report ;;
    run|fg)     cmd_foreground ;;
    restart)
        cmd_stop
        sleep 2
        cmd_start_bg
        ;;
    *)
        echo "LLM AI 交易 Agent"
        echo ""
        echo "用法: $0 {start|stop|status|log|report|run|restart}"
        echo ""
        echo "  start    后台启动"
        echo "  stop     停止"
        echo "  status   查看运行状态和余额"
        echo "  log      查看最近日志"
        echo "  report   查看交易报告"
        echo "  run      前台运行（实时看输出）"
        echo "  restart  重启"
        echo ""
        echo "配置:"
        echo "  编辑 .env 文件设置 API key 和参数"
        echo "  vim ${PROJECT_DIR}/.env"
        exit 0
        ;;
esac
