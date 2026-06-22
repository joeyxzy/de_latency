#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# GPU Backpressure Benchmark
# 对比 max_pending_dispatches 对 TTFT/TPOT 的影响
# 用法: bash bench_gpu_backpressure.sh [--sudo]
# ============================================================

# ---- 可调参数 ----
MODEL="/home/joeyxzy/models/Qwen2.5-32B-Instruct"
VLLM_SRC="/home/joeyxzy/vllm_0.20.1/vllm"     # 修改过的 vllm 源码
CONDA_PREFIX="/home/joeyxzy/miniconda3/envs/vllm-0200"
CONDA_PYTHON="$CONDA_PREFIX/bin/python"
VLLM_CLI="$CONDA_PREFIX/bin/vllm"

HOST="0.0.0.0"
PORT=8001
SERVER_LOG="/tmp/vllm_server.log"

# 硬件: 8x3090, PP=4, TP=2
GPU_IDS="0,1,2,3,4,5,6,7"
PP_SIZE=4
TP_SIZE=2

# 实验变量
M_VALUES=(none 2 3 4)
RATES=(inf 4)
NUM_PROMPTS=50
INPUT_LEN=512
OUTPUT_LEN=256
WARMUPS=8
TIMEOUT_SEC=300

RESULT_DIR="./results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"
echo "Results -> $RESULT_DIR"

# ---- sudo? ----
USE_SUDO=""
if [[ "${1:-}" == "--sudo" ]]; then USE_SUDO="sudo -E"; shift; fi

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RESULT_DIR/run.log"; }

# ---- helpers ----

start_server() {
    local max_pd="$1"
    log ""
    log "=== Server start: max_pending=${max_pd} ==="

    local args=(
        -m vllm.entrypoints.openai.api_server
        --model "$MODEL" --host "$HOST" --port "$PORT"
        --max-model-len 20000
        --tensor-parallel-size "$TP_SIZE"
        --pipeline-parallel-size "$PP_SIZE"
        --max-num-batched-tokens 2048
    )
    [[ "$max_pd" != "none" ]] && args+=(--max-pending-dispatches "$max_pd")

    PYTHONPATH="$VLLM_SRC" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    $USE_SUDO "$CONDA_PYTHON" "${args[@]}" &>"$SERVER_LOG" &
    local pid=$!
    echo "$pid" >/tmp/vllm_pid
    log "PID=$pid (waiting for model to load, up to ${TIMEOUT_SEC}s...)"

    # 用 /v1/models 而非 /health —— 前者需要模型加载完毕
    for i in $(seq 1 $TIMEOUT_SEC); do
        if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            log "Model loaded after ${i}s"
            return 0
        fi
        sleep 1
    done
    log "TIMEOUT — server may have crashed"
    tail -30 "$SERVER_LOG"
    return 1
}

stop_server() {
    log "Stopping..."
    local pid=""
    if [[ -f /tmp/vllm_pid ]]; then
        pid=$(cat /tmp/vllm_pid)
    fi
    if [[ -n "$pid" ]]; then
        $USE_SUDO kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    # 清理残留
    $USE_SUDO pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 4
    rm -f /tmp/vllm_pid
}

run_benchmark() {
    local max_pd="$1" rate="$2" tag="${max_pd}_r${rate}"
    log ""
    log ">>> max_pending=${max_pd} rate=${rate}"

    # vllm bench serve 是 pip 安装的 console script
    "$VLLM_CLI" bench serve \
        --backend vllm --host 127.0.0.1 --port "$PORT" \
        --endpoint /v1/completions \
        --dataset-name random \
        --num-prompts "$NUM_PROMPTS" \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio 0.5 \
        --request-rate "$rate" \
        --ignore-eos \
        --num-warmups "$WARMUPS" \
        2>&1 | tee "$RESULT_DIR/${tag}.txt"
    log "Saved ${tag}.txt"
}

print_summary() {
    echo ""
    echo "============================================"
    echo "SUMMARY"
    echo "============================================"
    for f in "$RESULT_DIR"/*.txt; do
        local tag
        tag=$(printf "%-14s" "$(basename "$f" .txt)")
        local ttft tpott req_s
        ttft=$(grep "Mean TTFT" "$f" | tail -1 | awk '{print $NF}' || echo "-")
        tpott=$(grep "Mean TPOT" "$f" | tail -1 | awk '{print $NF}' || echo "-")
        req_s=$(grep "Request throughput" "$f" | tail -1 | awk '{print $NF}' || echo "-")
        printf "  %s  TTFT=%7s ms  TPOT=%7s ms  req/s=%7s\n" "$tag" "$ttft" "$tpott" "$req_s"
    done
    echo ""
    echo "Full results: $RESULT_DIR/"
}

# ============================================================
log "GPU Backpressure Experiment"
log "  PP=$PP_SIZE TP=$TP_SIZE GPUs=$GPU_IDS PYTHONPATH=$VLLM_SRC"
log "  M values: ${M_VALUES[*]}   Rates: ${RATES[*]}"
log "  Prompts=$NUM_PROMPTS (in=${INPUT_LEN}t out=${OUTPUT_LEN}t)"

for max_pd in "${M_VALUES[@]}"; do
    start_server "$max_pd" || { stop_server; exit 1; }
    for rate in "${RATES[@]}"; do run_benchmark "$max_pd" "$rate"; done
    stop_server
    sleep 5
done

print_summary

