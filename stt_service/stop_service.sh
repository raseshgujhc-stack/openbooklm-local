#!/bin/bash
# stop_service.sh

echo "Stopping STT Service..."
docker stop stt-service
docker rm stt-service
echo "✅ Service stopped"
