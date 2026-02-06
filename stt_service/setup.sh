#!/bin/bash

echo "Setting up STT Service..."

# Create data directories
mkdir -p ~/openbooklm-local/data/stt-models
mkdir -p ~/openbooklm-local/data/stt-uploads
mkdir -p ~/openbooklm-local/data/stt-transcripts
mkdir -p ~/openbooklm-local/data/stt-logs

echo "Data directories created in ~/openbooklm-local/data/"

# Build and start Docker
cd ~/openbooklm-local/stt_service
echo "Building Docker image..."
docker-compose build

echo ""
echo "To start the service:"
echo "  cd ~/openbooklm-local/stt_service"
echo "  docker-compose up -d"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop the service:"
echo "  docker-compose down"
