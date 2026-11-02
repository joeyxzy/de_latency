#!/usr/bin/env bash
# Run from a terminal with sudo. No password handling or broad process cleanup.
set -euo pipefail
ROOT="$(realpath "$(dirname "$0")/..")"
SESSION="$(realpath -m "${1:?Usage: sudo bash benchmarks/ebpf_session.sh SESSION_DIR}")"
for ((i=0; i<600; i++)); do
    if [[ -s "$SESSION/target.pid" ]]; then break; fi
    sleep 1
done
read -r TARGET < "$SESSION/target.pid"
[[ "$TARGET" =~ ^[0-9]+$ ]] || exit 1
export TRACER_ZMQ_ADDR="ipc://$SESSION/events.sock"
export DE_LATENCY_SESSION_ID="$(basename "$SESSION")"
export DE_LATENCY_EBPF_READY_FILE="$SESSION/ebpf.ready"
export DE_LATENCY_STOP_FILE="$SESSION/stop"
export DE_LATENCY_EBPF_STATS_FILE="$SESSION/ebpf.stats.json"
exec "$ROOT/build/ebpf_monitor" --auto --root-pid "$TARGET" --worker-pid-file "$SESSION/targets"
