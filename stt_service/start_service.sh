#!/bin/bash
# start_service.sh

set -euo pipefail

echo "Starting STT Service..."

# Avoid duplicate runtime paths:
# 1) old manual container name: stt-service
# 2) compose container: stt_service-stt-service-1
docker stop stt-service 2>/dev/null || true
docker rm stt-service 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Canonical startup path: docker compose
docker compose up -d --build stt-service

echo "✅ Service started!"
echo ""
echo "Check logs: docker compose logs -f stt-service"
echo "Check health: curl http://localhost:8003/health"
