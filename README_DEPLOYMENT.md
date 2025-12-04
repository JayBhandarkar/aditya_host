# KVM 8 Production Deployment Guide

## Quick Deploy
```bash
# 1. Upload project to KVM 8
scp -r aditya_host/ user@your-kvm-ip:/home/user/

# 2. SSH into KVM 8
ssh user@your-kvm-ip

# 3. Run deployment script
cd aditya_host
chmod +x deploy.sh
./deploy.sh
```

## Manual Setup
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Setup environment
cp .env.production backend/.env
# Edit backend/.env with your API keys

# Deploy
docker-compose up -d --build
```

## KVM 8 Specifications Used
- **CPU**: 8 vCPU cores (4 for ML model, 4 for web services)
- **RAM**: 16GB (8GB for backend ML model, 2GB frontend, rest for system)
- **Storage**: 320GB NVMe (fast model loading)

## Access Your Application
- **Public**: http://YOUR_KVM_IP
- **Local**: http://localhost

## Monitoring
```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Resource usage
docker stats
```