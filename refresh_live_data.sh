#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# no-key live refresh defaults
export LIVE_INCLUDE_NOAA="${LIVE_INCLUDE_NOAA:-true}"
export ALLOW_INSECURE_SSL_FETCH="${ALLOW_INSECURE_SSL_FETCH:-false}"

python3 live_ingest.py --max-items "${1:-40}"
