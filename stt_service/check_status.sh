#!/bin/bash
# check_status.sh

set -euo pipefail

echo "STT Service Status"
echo "================="

# Check compose service first; fallback to old manual container name.
if docker compose ps --status running stt-service 2>/dev/null | grep -q stt-service || docker ps --format '{{.Names}}' | grep -q '^stt-service$'; then
    echo "✅ Container is running"
    
    # Check health endpoint
    echo -n "Health check: "
    curl -s http://localhost:8003/health | grep -o '"status":"[^"]*"' || echo "❌ Unhealthy"
    
    # Check WebSocket connections
    echo -n "WebSocket connections: "
    curl -s http://localhost:8003/health | grep -o '"websocket_connections":[0-9]*' || echo "0"
    
    # Show container stats
    echo ""
    echo "Container stats:"
    cname="$(docker ps --format '{{.Names}}' | grep -E '^stt_service-stt-service-1$|^stt-service$' | head -n1 || true)"
    if [ -n "$cname" ]; then
      docker stats "$cname" --no-stream
    fi
    
else
    echo "❌ Container is not running"
fi

echo ""
echo "Data directory usage:"
du -sh /home/ubuntu/openbooklm-local/data/*
