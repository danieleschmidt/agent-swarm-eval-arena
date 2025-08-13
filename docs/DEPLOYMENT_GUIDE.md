# Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Configuration](#configuration)
4. [Deployment Options](#deployment-options)
5. [Security Considerations](#security-considerations)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Scaling and Performance](#scaling-and-performance)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.5GHz
- RAM: 8GB
- Storage: 50GB SSD
- Network: 1Gbps
- OS: Ubuntu 20.04+ / CentOS 8+ / macOS 11+

**Recommended for Production:**
- CPU: 16+ cores, 3.0GHz
- RAM: 32GB+
- Storage: 200GB+ NVMe SSD
- Network: 10Gbps
- OS: Ubuntu 22.04 LTS

### Software Dependencies

```bash
# Python 3.9+
python3 --version

# Required system packages
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    redis-server \
    postgresql-14 \
    nginx \
    supervisor \
    htop \
    curl \
    git
```

### Container Requirements (Optional)

```bash
# Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## Environment Setup

### 1. Create Deployment User

```bash
# Create dedicated user for the application
sudo useradd -m -s /bin/bash swarm-arena
sudo usermod -aG sudo swarm-arena

# Switch to application user
sudo su - swarm-arena
```

### 2. Application Directory Structure

```bash
# Create directory structure
mkdir -p /opt/swarm-arena/{app,logs,data,config,scripts}
cd /opt/swarm-arena

# Clone repository
git clone https://github.com/your-org/swarm-arena.git app/
cd app/
```

### 3. Python Environment

```bash
# Create virtual environment
python3 -m venv /opt/swarm-arena/venv
source /opt/swarm-arena/venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install production dependencies
pip install gunicorn uvicorn redis psycopg2-binary
```

### 4. Database Setup

#### PostgreSQL Configuration

```bash
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE swarm_arena;
CREATE USER swarm_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE swarm_arena TO swarm_user;
ALTER USER swarm_user CREATEDB;
\q
EOF
```

#### Redis Configuration

```bash
# Configure Redis
sudo nano /etc/redis/redis.conf

# Update these settings:
# bind 127.0.0.1 ::1
# requirepass your_redis_password
# maxmemory 2gb
# maxmemory-policy allkeys-lru

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

## Configuration

### 1. Environment Variables

Create production environment file:

```bash
# /opt/swarm-arena/config/production.env
ENVIRONMENT=production
DEBUG=false

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=swarm_arena
DB_USER=swarm_user
DB_PASSWORD=your_secure_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# Security
SECRET_KEY=your_64_character_secret_key_here
JWT_SECRET=your_64_character_jwt_secret_here

# Service
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Monitoring
JAEGER_ENDPOINT=http://localhost:14268/api/traces
```

### 2. Application Configuration

```bash
# Copy production config template
cp deployment/production_config.py /opt/swarm-arena/config/

# Create configuration loader
cat > /opt/swarm-arena/config/load_config.py << 'EOF'
import os
from pathlib import Path
from deployment.production_config import get_config

# Load environment variables
env_file = Path(__file__).parent / "production.env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Get production configuration
config = get_config("production")

# Validate configuration
errors = config.validate()
if errors:
    raise ValueError(f"Configuration errors: {errors}")

print("Configuration loaded successfully")
EOF
```

### 3. Nginx Configuration

```bash
# Create Nginx configuration
sudo tee /etc/nginx/sites-available/swarm-arena << 'EOF'
upstream swarm_arena_app {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001 backup;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/swarm-arena.crt;
    ssl_certificate_key /etc/ssl/private/swarm-arena.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains";
    
    # Client max body size
    client_max_body_size 100M;
    
    # Compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    location / {
        proxy_pass http://swarm_arena_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 300s;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://swarm_arena_app/health;
        access_log off;
    }
    
    # Metrics endpoint (restrict access)
    location /metrics {
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://swarm_arena_app/metrics;
    }
    
    # Static files (if any)
    location /static/ {
        alias /opt/swarm-arena/app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/swarm-arena /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Deployment Options

### Option 1: Systemd Service

Create systemd service file:

```bash
sudo tee /etc/systemd/system/swarm-arena.service << 'EOF'
[Unit]
Description=Swarm Arena Application
After=network.target postgresql.service redis-server.service
Requires=postgresql.service redis-server.service

[Service]
Type=notify
User=swarm-arena
Group=swarm-arena
WorkingDirectory=/opt/swarm-arena/app
Environment=PATH=/opt/swarm-arena/venv/bin
EnvironmentFile=/opt/swarm-arena/config/production.env
ExecStart=/opt/swarm-arena/venv/bin/gunicorn \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --timeout 300 \
    --keep-alive 2 \
    --log-level info \
    --access-logfile /opt/swarm-arena/logs/access.log \
    --error-logfile /opt/swarm-arena/logs/error.log \
    --pid /opt/swarm-arena/app.pid \
    swarm_arena.main:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable swarm-arena
sudo systemctl start swarm-arena
```

### Option 2: Docker Deployment

#### Dockerfile

```dockerfile
# Multi-stage build
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash swarm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/swarm/.local

# Set up application
WORKDIR /app
COPY . .
RUN pip install -e .

# Change ownership
RUN chown -R swarm:swarm /app

USER swarm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "swarm_arena.main:app"]
```

#### Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  db:
    image: postgres:14
    environment:
      POSTGRES_DB: swarm_arena
      POSTGRES_USER: swarm_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - swarm_network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - swarm_network
    restart: unless-stopped

  app:
    build: .
    environment:
      - ENVIRONMENT=production
      - DB_HOST=db
      - REDIS_HOST=redis
      - DB_PASSWORD=${DB_PASSWORD}
      - REDIS_PASSWORD=${REDIS_PASSWORD}
      - SECRET_KEY=${SECRET_KEY}
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      - db
      - redis
    networks:
      - swarm_network
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 4G

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - app
    networks:
      - swarm_network
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    networks:
      - swarm_network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:

networks:
  swarm_network:
    driver: bridge
```

### Option 3: Kubernetes Deployment

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: swarm-arena
  labels:
    app: swarm-arena
spec:
  replicas: 3
  selector:
    matchLabels:
      app: swarm-arena
  template:
    metadata:
      labels:
        app: swarm-arena
    spec:
      containers:
      - name: swarm-arena
        image: swarm-arena:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DB_HOST
          value: "postgres-service"
        - name: REDIS_HOST
          value: "redis-service"
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: swarm-arena-secrets
              key: secret-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: swarm-arena-service
spec:
  selector:
    app: swarm-arena
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Security Considerations

### 1. SSL/TLS Configuration

```bash
# Generate SSL certificate (Let's Encrypt)
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com

# Or use self-signed for testing
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/swarm-arena.key \
    -out /etc/ssl/certs/swarm-arena.crt
```

### 2. Firewall Configuration

```bash
# UFW firewall rules
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# Fail2ban for additional protection
sudo apt-get install fail2ban
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
```

### 3. Secret Management

```bash
# Generate secure secrets
python3 -c "import secrets; print(secrets.token_hex(32))"  # SECRET_KEY
python3 -c "import secrets; print(secrets.token_hex(32))"  # JWT_SECRET

# Use environment-specific secret storage
# - AWS Secrets Manager
# - HashiCorp Vault
# - Kubernetes Secrets
```

### 4. Database Security

```bash
# PostgreSQL security
sudo nano /etc/postgresql/14/main/postgresql.conf
# ssl = on
# password_encryption = scram-sha-256

sudo nano /etc/postgresql/14/main/pg_hba.conf
# hostssl all all 0.0.0.0/0 scram-sha-256
```

## Monitoring and Logging

### 1. Application Logging

```bash
# Create log directories
sudo mkdir -p /var/log/swarm-arena
sudo chown swarm-arena:swarm-arena /var/log/swarm-arena

# Logrotate configuration
sudo tee /etc/logrotate.d/swarm-arena << 'EOF'
/var/log/swarm-arena/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 swarm-arena swarm-arena
    postrotate
        systemctl reload swarm-arena
    endscript
}
EOF
```

### 2. Metrics Collection

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'swarm-arena'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['localhost:9187']
```

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Swarm Arena Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### 3. Health Checks

```python
# Health check endpoint
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

@app.get("/health")
async def health_check():
    try:
        # Check database connection
        await check_database()
        
        # Check Redis connection
        await check_redis()
        
        # Check system resources
        system_status = check_system_resources()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "system": system_status
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
```

## Scaling and Performance

### 1. Horizontal Scaling

#### Load Balancer Configuration

```bash
# HAProxy configuration
sudo tee /etc/haproxy/haproxy.cfg << 'EOF'
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend swarm_arena_frontend
    bind *:80
    default_backend swarm_arena_backend

backend swarm_arena_backend
    balance roundrobin
    option httpchk GET /health
    server app1 10.0.1.10:8000 check
    server app2 10.0.1.11:8000 check
    server app3 10.0.1.12:8000 check
EOF
```

### 2. Auto-scaling

#### Docker Swarm Auto-scaling

```bash
# Initialize Docker Swarm
docker swarm init

# Deploy stack with auto-scaling
docker stack deploy -c docker-compose.prod.yml swarm-arena

# Scale service based on CPU usage
docker service update --replicas 5 swarm-arena_app
```

#### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: swarm-arena-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: swarm-arena
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 3. Performance Optimization

#### Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_agents_simulation_id ON agents(simulation_id);
CREATE INDEX CONCURRENTLY idx_messages_timestamp ON messages(timestamp);
CREATE INDEX CONCURRENTLY idx_metrics_created_at ON metrics(created_at);

-- Optimize PostgreSQL settings
-- shared_buffers = 256MB
-- effective_cache_size = 1GB
-- work_mem = 4MB
-- maintenance_work_mem = 64MB
```

#### Application Performance

```python
# Connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)

# Redis connection pooling
import redis.connection

redis_pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=20
)
```

## Troubleshooting

### Common Issues

#### 1. Application Won't Start

```bash
# Check logs
journalctl -u swarm-arena -f

# Check configuration
python3 /opt/swarm-arena/config/load_config.py

# Check dependencies
/opt/swarm-arena/venv/bin/pip check
```

#### 2. Database Connection Issues

```bash
# Test database connection
psql -h localhost -U swarm_user -d swarm_arena

# Check PostgreSQL status
sudo systemctl status postgresql

# Check network connectivity
netstat -an | grep 5432
```

#### 3. Performance Issues

```bash
# Monitor system resources
htop
iotop
nethogs

# Check application metrics
curl http://localhost:8000/metrics

# Profile application
python3 -m cProfile -o profile.stats your_script.py
```

#### 4. Memory Leaks

```bash
# Monitor memory usage
watch -n 1 'ps aux | grep swarm-arena'

# Use memory profiler
pip install memory-profiler
python3 -m memory_profiler your_script.py
```

### Log Analysis

```bash
# Application logs
tail -f /opt/swarm-arena/logs/error.log

# Nginx logs
tail -f /var/log/nginx/error.log

# System logs
journalctl -f

# Database logs
tail -f /var/log/postgresql/postgresql-14-main.log
```

## Maintenance

### 1. Regular Maintenance Tasks

#### Daily Tasks

```bash
#!/bin/bash
# daily-maintenance.sh

# Check disk space
df -h

# Check memory usage
free -h

# Check application status
systemctl status swarm-arena

# Check database status
sudo -u postgres psql -c "SELECT pg_size_pretty(pg_database_size('swarm_arena'));"

# Clean old logs
find /opt/swarm-arena/logs -name "*.log" -mtime +30 -delete
```

#### Weekly Tasks

```bash
#!/bin/bash
# weekly-maintenance.sh

# Database vacuum and analyze
sudo -u postgres psql swarm_arena -c "VACUUM ANALYZE;"

# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Check SSL certificate expiration
openssl x509 -in /etc/ssl/certs/swarm-arena.crt -text -noout | grep "Not After"

# Backup database
pg_dump -h localhost -U swarm_user swarm_arena > backup_$(date +%Y%m%d).sql
```

### 2. Backup and Recovery

#### Database Backup

```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/opt/swarm-arena/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
pg_dump -h localhost -U swarm_user swarm_arena | gzip > $BACKUP_DIR/swarm_arena_$DATE.sql.gz

# Keep only last 30 days of backups
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed: swarm_arena_$DATE.sql.gz"
```

#### Application Backup

```bash
#!/bin/bash
# backup-application.sh

BACKUP_DIR="/opt/swarm-arena/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /opt/swarm-arena/config/

# Backup application data
tar -czf $BACKUP_DIR/data_$DATE.tar.gz /opt/swarm-arena/data/

echo "Application backup completed"
```

### 3. Updates and Rollbacks

#### Application Update

```bash
#!/bin/bash
# update-application.sh

# Stop application
sudo systemctl stop swarm-arena

# Backup current version
cp -r /opt/swarm-arena/app /opt/swarm-arena/app.backup.$(date +%Y%m%d)

# Pull latest changes
cd /opt/swarm-arena/app
git pull origin main

# Update dependencies
source /opt/swarm-arena/venv/bin/activate
pip install -r requirements.txt

# Run migrations (if any)
python manage.py migrate

# Start application
sudo systemctl start swarm-arena

# Check status
sleep 10
curl -f http://localhost:8000/health

echo "Update completed successfully"
```

#### Rollback Procedure

```bash
#!/bin/bash
# rollback-application.sh

BACKUP_DATE=$1

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <backup_date>"
    echo "Available backups:"
    ls -la /opt/swarm-arena/app.backup.*
    exit 1
fi

# Stop application
sudo systemctl stop swarm-arena

# Rollback application
rm -rf /opt/swarm-arena/app
cp -r /opt/swarm-arena/app.backup.$BACKUP_DATE /opt/swarm-arena/app

# Restore database (if needed)
# gunzip -c /opt/swarm-arena/backups/swarm_arena_$BACKUP_DATE.sql.gz | psql -h localhost -U swarm_user swarm_arena

# Start application
sudo systemctl start swarm-arena

echo "Rollback to $BACKUP_DATE completed"
```

### 4. Monitoring Alerts

#### Setup Email Alerts

```bash
# Install mail utilities
sudo apt-get install mailutils

# Create alert script
cat > /opt/swarm-arena/scripts/alert.sh << 'EOF'
#!/bin/bash

SERVICE="swarm-arena"
EMAIL="admin@your-domain.com"

if ! systemctl is-active --quiet $SERVICE; then
    echo "Service $SERVICE is down!" | mail -s "ALERT: $SERVICE Down" $EMAIL
fi

# Check disk space
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    echo "Disk usage is at $DISK_USAGE%" | mail -s "ALERT: High Disk Usage" $EMAIL
fi
EOF

chmod +x /opt/swarm-arena/scripts/alert.sh

# Add to crontab
echo "*/5 * * * * /opt/swarm-arena/scripts/alert.sh" | crontab -
```

This comprehensive deployment guide provides everything needed to deploy, monitor, and maintain the Swarm Arena platform in production environments.