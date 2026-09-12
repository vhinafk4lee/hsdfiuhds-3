#!/usr/bin/env bash
# Stop the feed and every worker on this host. The signer is left alone.
set -euo pipefail
TARGET="${HASHBROKER_WORKER_DIR:-/opt/hashbroker}"
SCRIPTS="${HASHBROKER_SCRIPTS:-$TARGET/repo/scripts}"
pkill -f "$SCRIPTS/miner.py" || true
pkill -f "$SCRIPTS/job_feed.py" || true
echo "workers stopped"
