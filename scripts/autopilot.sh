#!/usr/bin/env bash
# One command to take a rented GPU box from bare to mining unattended.
#
#   export HASHBROKER_WALLET=0x...            # the mining wallet
#   export HASHBROKER_PRIVATE_KEY_FILE=/root/wallet.key   # 0600, on this host
#   bash autopilot.sh
#
# Add --dry-run to sign without broadcasting, or --no-signer to mine here and
# sign somewhere else.
set -euo pipefail

TARGET="${HASHBROKER_WORKER_DIR:-/opt/hashbroker}"
BRANCH="${HASHBROKER_BRANCH:-claude/sweet-rubin-w4jyk9}"
RAW="https://raw.githubusercontent.com/vhinafk4lee/hsdfiuhds-3/$BRANCH/scripts"
EXTRA=()
SKIP_BENCHMARK=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) EXTRA+=("--dry-run") ;;
        --no-signer) EXTRA+=("--no-signer") ;;
        --cpu) EXTRA+=("--mode" "cpu") ;;
        --skip-benchmark) SKIP_BENCHMARK=1 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

: "${HASHBROKER_WALLET:?set HASHBROKER_WALLET to the mining wallet address}"

if [ ! -d "$TARGET/repo" ]; then
    echo "== installing =="
    curl -sfSO "$RAW/bootstrap.sh"
    bash bootstrap.sh
fi

PYTHON="$TARGET/venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"
SCRIPTS="$TARGET/repo/scripts"
mkdir -p "$TARGET/logs"

if [ "$SKIP_BENCHMARK" -eq 0 ] && [ -z "${HASHBROKER_CPU_ONLY:-}" ]; then
    echo "== benchmarking =="
    "$PYTHON" "$SCRIPTS/benchmark.py" --seconds 4 | tee "$TARGET/logs/benchmark.log"
fi

if [ -n "${HASHBROKER_PRIVATE_KEY_FILE:-}" ] || [ -n "${HASHBROKER_PRIVATE_KEY:-}" ]; then
    echo "== signer preflight =="
    "$PYTHON" "$SCRIPTS/signer.py" --check
fi

echo "== starting =="
pkill -f "$SCRIPTS/run_all.py" 2>/dev/null || true
nohup "$PYTHON" "$SCRIPTS/run_all.py" \
    --wallet "$HASHBROKER_WALLET" --dir "$TARGET" "${EXTRA[@]}" \
    >>"$TARGET/logs/miner.log" 2>&1 &

sleep 5
tail -n 20 "$TARGET/logs/miner.log" || true
cat <<INFO

mining in the background as pid $!
  watch      tail -f $TARGET/logs/miner.log
  status     $PYTHON $SCRIPTS/status.py --dir $TARGET
  stop       pkill -f run_all.py
INFO
