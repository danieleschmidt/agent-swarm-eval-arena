-- Initialize PostgreSQL database for Swarm Arena development

-- Create additional schemas
CREATE SCHEMA IF NOT EXISTS experiments;
CREATE SCHEMA IF NOT EXISTS metrics;
CREATE SCHEMA IF NOT EXISTS users;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Experiments table for storing experiment metadata
CREATE TABLE experiments.experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(255),
    tags TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Metrics table for time series data
CREATE TABLE metrics.agent_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments.experiments(id),
    agent_id VARCHAR(255) NOT NULL,
    episode INTEGER NOT NULL,
    step INTEGER NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value NUMERIC NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- System metrics table
CREATE TABLE metrics.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments.experiments(id),
    node_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    metric_value NUMERIC NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Users table for authentication (development only)
CREATE TABLE users.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'researcher',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for better performance
CREATE INDEX idx_experiments_status ON experiments.experiments(status);
CREATE INDEX idx_experiments_created_at ON experiments.experiments(created_at);
CREATE INDEX idx_experiments_tags ON experiments.experiments USING GIN(tags);

CREATE INDEX idx_agent_metrics_experiment ON metrics.agent_metrics(experiment_id);
CREATE INDEX idx_agent_metrics_agent_episode ON metrics.agent_metrics(agent_id, episode);
CREATE INDEX idx_agent_metrics_timestamp ON metrics.agent_metrics(timestamp);
CREATE INDEX idx_agent_metrics_metric_name ON metrics.agent_metrics(metric_name);

CREATE INDEX idx_system_metrics_experiment ON metrics.system_metrics(experiment_id);
CREATE INDEX idx_system_metrics_node ON metrics.system_metrics(node_id);
CREATE INDEX idx_system_metrics_timestamp ON metrics.system_metrics(timestamp);

CREATE INDEX idx_users_username ON users.users(username);
CREATE INDEX idx_users_email ON users.users(email);

-- Insert sample data for development
INSERT INTO users.users (username, email, password_hash, role) VALUES 
    ('admin', 'admin@swarm-arena.dev', crypt('admin123', gen_salt('bf')), 'admin'),
    ('researcher', 'researcher@swarm-arena.dev', crypt('research123', gen_salt('bf')), 'researcher'),
    ('student', 'student@swarm-arena.dev', crypt('student123', gen_salt('bf')), 'student');

-- Create a sample experiment
INSERT INTO experiments.experiments (name, description, config, status, created_by) VALUES 
    ('Development Test', 'Sample experiment for development testing', 
     '{"num_agents": 100, "environment": "foraging", "episodes": 10}'::jsonb, 
     'completed', 'admin');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA experiments TO swarm_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO swarm_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA users TO swarm_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA experiments TO swarm_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO swarm_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA users TO swarm_user;

-- Create views for common queries
CREATE VIEW metrics.latest_agent_metrics AS
SELECT DISTINCT ON (experiment_id, agent_id, metric_name) 
    experiment_id, agent_id, metric_name, metric_value, timestamp
FROM metrics.agent_metrics 
ORDER BY experiment_id, agent_id, metric_name, timestamp DESC;

CREATE VIEW experiments.experiment_summary AS
SELECT 
    e.id,
    e.name,
    e.status,
    e.created_at,
    e.completed_at,
    COUNT(am.id) as total_metrics,
    COUNT(DISTINCT am.agent_id) as num_agents,
    MAX(am.episode) as max_episode
FROM experiments.experiments e
LEFT JOIN metrics.agent_metrics am ON e.id = am.experiment_id
GROUP BY e.id, e.name, e.status, e.created_at, e.completed_at;