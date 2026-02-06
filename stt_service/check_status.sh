#!/bin/bash
# check_status.sh

echo "STT Service Status"
echo "================="

# Check if container is running
if docker ps | grep -q stt-service; then
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
    docker stats stt-service --no-stream
    
else
    echo "❌ Container is not running"
fi

echo ""
echo "Data directory usage:"
du -sh /home/ubuntu/openbooklm-local/data/*
