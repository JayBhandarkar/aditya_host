#!/bin/bash

# KVM 8 Production Deployment Script
echo "🚀 Starting KVM 8 deployment..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create environment file
cp backend/.env.example backend/.env

# Build and start services
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# Show status
echo "✅ Deployment complete!"
echo "📊 Service status:"
docker-compose ps

echo "🌐 Your application is running on:"
echo "   - HTTP: http://$(curl -s ifconfig.me)"
echo "   - Local: http://localhost"

echo "📝 To monitor logs:"
echo "   docker-compose logs -f"