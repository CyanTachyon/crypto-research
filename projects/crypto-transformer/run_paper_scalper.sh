#!/bin/bash
# Hyperliquid 模拟交易一键启动脚本
# 用法:
#   ./run_paper_scalper.sh          前台运行（看实时输出）
#   ./run_paper_scalper.sh start    后台运行
#   ./run_paper_scalper.sh stop     停止
#   ./run_paper_scalper.sh status   查看状态
#   ./run_paper_scalper.sh log      查看最近日志
#   ./run_paper_scalper.sh report   查看交易报告

set -euo pipefail

PROJECT_DIR="/home/cyan/default/crypto"
PYTHON="${PROJECT_DIR}/.venv/bin/python"
SCRIPT="${PROJECT_DIR}/scripts/paper_scalper.py"
PIDFILE="${PROJECT_DIR}/data/paper_scalper.pid"
LOGDIR="${PROJECT_DIR}/data/paper_logs"

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
    echo "启动模拟交易（后台）..."
    cd "${PROJECT_DIR}"
    nohup "${PYTHON}" -u "${SCRIPT}" > /dev/null 2>&1 &
    local pid=$!
    echo "${pid}" > "${PIDFILE}"
    sleep 3
    if kill -0 "${pid}" 2>/dev/null; then
        echo "启动成功 (PID ${pid})"
        echo "日志: ${LOGDIR}/bot_$(date -u +%Y-%m-%d).log"
        echo "状态: ${PROJECT_DIR}/data/paper_scalper_state.json"
        echo "报告: ${PROJECT_DIR}/docs/paper_scalper_report.md"
    else
        echo "启动失败！"
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
        if [ -f "${PROJECT_DIR}/data/paper_scalper_state.json" ]; then
            echo "---"
            "${PYTHON}" -c "
import json
with open('${PROJECT_DIR}/data/paper_scalper_state.json') as f:
    d = json.load(f)
print(f\"  运行时间:   {d.get('uptime_hours', 0):.1f} 小时\")
print(f\"  当前余额:   \${d.get('balance', 0):.2f}\")
print(f\"  总交易数:   {d.get('total_trades', 0)}\")
print(f\"  今日交易:   {d.get('daily_trades', 0)}\")
print(f\"  今日盈亏:   \${d.get('daily_pnl', 0):.2f}\")
print(f\"  当前趋势:   {d.get('trend', '?')}\")
"
        fi
    else
        echo "未运行"
    fi
}

cmd_log() {
    local logfile
    logfile="${LOGDIR}/bot_$(date -u +%Y-%m-%d).log"
    if [ -f "${logfile}" ]; then
        echo "=== 最近 50 行日志 ==="
        tail -50 "${logfile}"
        echo ""
        echo "实时跟踪: tail -f ${logfile}"
    else
        echo "没有找到今天的日志文件"
        echo "日志目录: ${LOGDIR}"
        ls -la "${LOGDIR}" 2>/dev/null || echo "目录为空"
    fi
}

cmd_report() {
    local report="${PROJECT_DIR}/docs/paper_scalper_report.md"
    if [ -f "${report}" ]; then
        cat "${report}"
    else
        echo "报告尚未生成（需要运行 30 分钟以上）"
    fi
}

cmd_foreground() {
    if pid=$(get_pid); then
        echo "已在后台运行 (PID ${pid})，先停止再前台运行"
        exit 1
    fi
    cd "${PROJECT_DIR}"
    exec "${PYTHON}" -u "${SCRIPT}"
}

case "${1:-run}" in
    start|bg)   cmd_start_bg ;;
    stop)       cmd_stop ;;
    status|st)  cmd_status ;;
    log|logs)   cmd_log ;;
    report|rep) cmd_report ;;
    run|fg|foreground) cmd_foreground ;;
    restart)
        cmd_stop
        sleep 2
        cmd_start_bg
        ;;
    *)
        echo "用法: $0 {start|stop|status|log|report|run|restart}"
        echo ""
        echo "  start    后台启动"
        echo "  stop     停止"
        echo "  status   查看运行状态和余额"
        echo "  log      查看最近日志"
        echo "  report   查看交易报告"
        echo "  run      前台运行（实时看输出）"
        echo "  restart  重启"
        exit 1
        ;;
esac
