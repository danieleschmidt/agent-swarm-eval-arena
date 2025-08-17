
# Terragon AI Infrastructure as Code
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "terragon-swarm-arena"
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name        = "${var.cluster_name}-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  
  tags = {
    Name        = "${var.cluster_name}-igw"
    Environment = var.environment
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count = 2
  
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name        = "${var.cluster_name}-public-${count.index + 1}"
    Environment = var.environment
    Type        = "public"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count = 2
  
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name        = "${var.cluster_name}-private-${count.index + 1}"
    Environment = var.environment
    Type        = "private"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = var.cluster_name
  cluster_version = "1.27"
  
  vpc_id     = aws_vpc.main.id
  subnet_ids = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
  
  # Node groups
  eks_managed_node_groups = {
    main = {
      min_size       = 3
      max_size       = 100
      desired_size   = 3
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "main"
      }
    }
    
    compute_intensive = {
      min_size       = 1
      max_size       = 20
      desired_size   = 2
      instance_types = ["c5.9xlarge", "c5.12xlarge"]
      
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "compute-intensive"
        Workload    = "quantum-simulation"
      }
      
      taints = {
        dedicated = {
          key    = "workload"
          value  = "compute-intensive"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
  
  tags = {
    Environment = var.environment
    Project     = "terragon-swarm-arena"
  }
}

# RDS for persistent storage
resource "aws_db_instance" "main" {
  identifier     = "${var.cluster_name}-db"
  engine         = "postgresql"
  engine_version = "15.3"
  instance_class = "db.r5.large"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  
  db_name  = "swarmarena"
  username = "terragon"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.cluster_name}-final-snapshot"
  
  tags = {
    Name        = "${var.cluster_name}-db"
    Environment = var.environment
  }
}

# Redis for caching
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.cluster_name}-cache-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id         = "${var.cluster_name}-redis"
  description                  = "Redis cluster for Swarm Arena"
  
  port                = 6379
  parameter_group_name = "default.redis7"
  node_type           = "cache.r6g.large"
  num_cache_clusters  = 2
  
  subnet_group_name          = aws_elasticache_subnet_group.main.name
  security_group_ids         = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Name        = "${var.cluster_name}-redis"
    Environment = var.environment
  }
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "db_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
  sensitive   = true
}
