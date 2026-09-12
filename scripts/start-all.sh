#!/usr/bin/env bash
# Start the job feed and one worker per GPU on this host.
# Reads HASHBROKER_WALLET (required) and HASHBROKER_WORKER_DIR from the environment.
set -euo pipefail

TARGET="${HASHBROKER_WORKER_DIR:-/opt/hashbroker}"
PYTHON="${HASHBROKER_PYTHON:-$TARGET/venv/bin/python}"
SCRIPTS="${HASHBROKER_SCRIPTS:-$TARGET/repo/scripts}"
WALLET="${HASHBROKER_WALLET:?set HASHBROKER_WALLET}"
LOGS="$TARGET/logs"
GPUS="${HASHBROKER_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"

mkdir -p "$LOGS"
pkill -f "$SCRIPTS/job_feed.py" || true
pkill -f "$SCRIPTS/miner.py" || true

nohup "$PYTHON" "$SCRIPTS/job_feed.py" \
    --wallet "$WALLET" --output "$TARGET/job.json" --interval 0.5 \
    >>"$LOGS/feed.log" 2>&1 &
echo "feed started"

for index in $(seq 0 $((GPUS - 1))); do
    nohup "$PYTHON" "$SCRIPTS/miner.py" \
        --wallet "$WALLET" --device "$index" \
        --job-file "$TARGET/job.json" \
        --output "$TARGET/solution-gpu$index.json" \
        --keep-mining \
        >>"$LOGS/miner-gpu$index.log" 2>&1 &
    echo "worker on GPU $index started"
done

echo "logs in $LOGS; run stop-all.sh to stop"
