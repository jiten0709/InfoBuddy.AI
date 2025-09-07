#!/bin/bash
# filepath: /Users/hi/jitenStuff/MyGit/InfoBuddy.AI/deploy.sh

set -e

echo "🚀 Building InfoBuddy.AI Docker Images..."

# Build with production server URL
PROD_SERVER_URL="https://your-production-server.com"

# Build server image
docker build -t your-dockerhub/infobuddy-server:latest ./server

# Build client image with production server URL
docker build \
  --build-arg NEXT_PUBLIC_SERVER_URL=$PROD_SERVER_URL \
  -t your-dockerhub/infobuddy-client:latest \
  ./client

echo "✅ Images built successfully!"

# Optionally push to registry
read -p "Push to Docker Hub? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker push your-dockerhub/infobuddy-server:latest
    docker push your-dockerhub/infobuddy-client:latest
    echo "✅ Images pushed to Docker Hub!"
fi