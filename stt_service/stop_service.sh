#!/bin/bash
# stop_service.sh

set -euo pipefail

echo "Stopping STT Service..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

docker compose down
docker stop stt-service 2>/dev/null || true
docker rm stt-service 2>/dev/null || true
echo "✅ Service stopped"
