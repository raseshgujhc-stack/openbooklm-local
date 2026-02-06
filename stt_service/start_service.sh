#!/bin/bash
# start_service.sh

echo "Starting STT Service..."

# Stop if already running
docker stop stt-service 2>/dev/null || true
docker rm stt-service 2>/dev/null || true

# Start new container
docker run -d \
  -p 8003:8003 \
  -v /home/ubuntu/openbooklm-local/data/stt-models:/app/models \
  -v /home/ubuntu/openbooklm-local/data/stt-uploads:/app/uploads \
  -v /home/ubuntu/openbooklm-local/data/stt-transcripts:/app/transcripts \
  -v /home/ubuntu/openbooklm-local/data/stt-logs:/app/logs \
  --env-file .env \
  --name stt-service \
  --restart unless-stopped \
  stt-service

echo "✅ Service started!"
echo ""
echo "Check logs: docker logs -f stt-service"
echo "Check health: curl http://localhost:8003/health"
