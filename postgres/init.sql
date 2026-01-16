-- ============================================================
-- Agent Logging Tables
-- ============================================================

-- LLM Audit Log Table: Track all LLM interactions for debugging and compliance
CREATE TABLE IF NOT EXISTS llm_audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    ticket_id VARCHAR(255) NOT NULL,
    model_name VARCHAR(100),
    prompt_text TEXT,
    response_text TEXT,
    decision VARCHAR(50),
    confidence VARCHAR(20),
    policy_applied VARCHAR(255),
    processing_time_ms INTEGER,
    error_message TEXT,
    metadata JSONB
);

-- Indexes for llm_audit_log
CREATE INDEX IF NOT EXISTS idx_llm_audit_ticket_id ON llm_audit_log(ticket_id);
CREATE INDEX IF NOT EXISTS idx_llm_audit_timestamp ON llm_audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_audit_decision ON llm_audit_log(decision);
CREATE INDEX IF NOT EXISTS idx_llm_audit_model ON llm_audit_log(model_name);

-- The Audit Trail Table: A detailed, step-by-step log of every action
CREATE TABLE IF NOT EXISTS agent_audit_log (
    log_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    event_type VARCHAR(50),
    event_details JSONB,
    status VARCHAR(50),
    error_message TEXT
);

-- The Metrics Table: A high-level summary of each journey run for reporting
CREATE TABLE IF NOT EXISTS agent_run_metrics (
    metric_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    journey_name VARCHAR(255),
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    duration_ms INTEGER,
    final_outcome VARCHAR(50),
    ticket_id VARCHAR(255),
    
    -- Performance metrics
    token_usage INTEGER,
    api_calls_count INTEGER,
    cache_hits INTEGER,
    cache_misses INTEGER,
    error_count INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    confidence_score VARCHAR(20),
    agent_name VARCHAR(255),
    
    -- AI/Model metrics
    model_name VARCHAR(100),
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    estimated_cost DECIMAL(10,4),
    
    -- Decision quality metrics
    decision_changed_by_human BOOLEAN DEFAULT FALSE,
    human_override_reason TEXT,
    policy_rules_applied TEXT[],
    
    -- Ticket characteristics
    ticket_priority VARCHAR(20),
    ticket_source VARCHAR(50),
    ticket_category VARCHAR(100),
    customer_sentiment VARCHAR(20),
    
    -- Security/Risk metrics
    security_scan_result VARCHAR(50),
    security_threats_detected TEXT[],
    risk_score DECIMAL(3,2),
    
    -- Response time metrics
    ticket_created_at TIMESTAMPTZ,
    first_response_time_ms INTEGER,
    resolution_time_ms INTEGER,
    sla_met BOOLEAN
);

-- The Context "Memory" Table: A persistent store of key facts about customers
CREATE TABLE IF NOT EXISTS customer_context (
    customer_id VARCHAR(255) PRIMARY KEY,
    last_interaction_date TIMESTAMPTZ,
    total_interactions INTEGER DEFAULT 1,
    total_denials INTEGER DEFAULT 0,
    custom_notes JSONB
);

-- Refund Transactions Table: Track all refund operations
CREATE TABLE IF NOT EXISTS refund_transactions (
    refund_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES agent_run_metrics(run_id) ON DELETE CASCADE,
    ticket_id VARCHAR(255) NOT NULL,
    booking_id VARCHAR(255),
    
    -- Financial details
    refund_amount DECIMAL(10, 2),
    refund_type VARCHAR(50), -- 'duplicate', 'policy', 'manual'
    refund_reason VARCHAR(100), -- 'duplicate', 'overstay_dispute', 'policy_violation', 'customer_error'
    refund_category VARCHAR(50), -- 'legitimate', 'goodwill', 'error'
    
    -- Status tracking
    refund_status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'completed', 'failed'
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Location tracking
    location_id VARCHAR(255),
    location_name VARCHAR(255),
    location_city VARCHAR(100),
    location_state VARCHAR(50),
    
    -- ParkWhiz integration
    parkwhiz_refund_id VARCHAR(255),
    parkwhiz_booking_id VARCHAR(255),
    
    -- Duplicate detection details
    duplicate_count INTEGER,
    duplicate_booking_ids TEXT[],
    time_between_duplicates_minutes INTEGER,
    
    -- Risk/Fraud flags
    dispute_flag BOOLEAN DEFAULT FALSE,
    fraud_indicators TEXT[],
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent Performance Metrics Table: Detailed performance tracking
CREATE TABLE IF NOT EXISTS agent_performance_metrics (
    metric_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES agent_run_metrics(run_id) ON DELETE CASCADE,
    agent_name VARCHAR(255),
    
    -- Timing breakdown (milliseconds)
    total_duration_ms INTEGER,
    llm_duration_ms INTEGER,
    api_duration_ms INTEGER,
    database_duration_ms INTEGER,
    freshdesk_api_duration_ms INTEGER,
    parkwhiz_api_duration_ms INTEGER,
    lakera_api_duration_ms INTEGER,
    gemini_api_duration_ms INTEGER,
    
    -- Resource usage
    token_usage INTEGER,
    api_calls_count INTEGER,
    cache_hits INTEGER,
    cache_misses INTEGER,
    api_timeout_count INTEGER,
    api_retry_count INTEGER,
    
    -- Quality metrics
    confidence_score VARCHAR(20), -- 'high', 'medium', 'low'
    error_count INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    data_quality_score DECIMAL(3,2), -- 0.0 to 1.0
    
    -- Extraction quality
    booking_extraction_method VARCHAR(50), -- 'pattern', 'llm', 'manual'
    booking_extraction_confidence VARCHAR(20),
    missing_fields TEXT[],
    
    -- Financial (if applicable)
    refund_amount DECIMAL(10, 2),
    
    -- System metrics
    queue_wait_time_ms INTEGER,
    concurrent_runs_count INTEGER,
    system_load_percent DECIMAL(5,2),
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Agent Journey Steps Table: Track individual steps within journeys
CREATE TABLE IF NOT EXISTS agent_journey_steps (
    step_id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES agent_run_metrics(run_id) ON DELETE CASCADE,
    step_name VARCHAR(100), -- 'extract_booking', 'check_security', 'detect_duplicates', 'make_decision'
    step_order INTEGER,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    duration_ms INTEGER,
    status VARCHAR(50), -- 'success', 'failed', 'skipped'
    error_message TEXT,
    step_output JSONB, -- Store step results for debugging
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Async Webhook Orchestration Tables
-- ============================================================

-- Webhook Tasks Table: Track async webhook processing tasks
CREATE TABLE IF NOT EXISTS webhook_tasks (
    task_id VARCHAR(255) PRIMARY KEY,
    hatchet_workflow_id VARCHAR(255),
    payload_hash VARCHAR(64) NOT NULL,
    webhook_payload JSONB NOT NULL,
    
    -- Task metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    -- Status tracking
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed', 'dead_letter'
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 5,
    
    -- Processing details
    worker_id VARCHAR(255),
    processing_path VARCHAR(50), -- 'fast_path', 'llm_pipeline'
    processing_time_ms INTEGER,
    
    -- Results
    decision_result JSONB,
    error_message TEXT,
    error_type VARCHAR(50), -- 'transient', 'permanent'
    
    -- Freshdesk integration
    ticket_id VARCHAR(255),
    freshdesk_updated BOOLEAN DEFAULT FALSE,
    freshdesk_update_attempts INTEGER DEFAULT 0,
    
    -- Audit trail
    trace_id VARCHAR(255),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Dead Letter Queue Table: Store unrecoverable failed tasks
CREATE TABLE IF NOT EXISTS webhook_dead_letter_queue (
    dlq_id SERIAL PRIMARY KEY,
    task_id VARCHAR(255) REFERENCES webhook_tasks(task_id),
    original_payload JSONB NOT NULL,
    failure_reason TEXT NOT NULL,
    failure_type VARCHAR(50) NOT NULL, -- 'max_retries_exceeded', 'permanent_failure'
    
    -- Failure context
    last_error_message TEXT,
    retry_history JSONB, -- Array of retry attempts with timestamps and errors
    total_retry_count INTEGER,
    
    -- Resolution tracking
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(255),
    resolution_notes TEXT,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Task Metrics Table: Performance and monitoring data
CREATE TABLE IF NOT EXISTS webhook_task_metrics (
    metric_id SERIAL PRIMARY KEY,
    task_id VARCHAR(255) REFERENCES webhook_tasks(task_id),
    
    -- Timing metrics (milliseconds)
    webhook_response_time_ms INTEGER,
    idempotency_check_time_ms INTEGER,
    task_creation_time_ms INTEGER,
    queue_wait_time_ms INTEGER,
    processing_time_ms INTEGER,
    freshdesk_update_time_ms INTEGER,
    total_time_ms INTEGER,
    
    -- Resource usage
    redis_operations_count INTEGER DEFAULT 0,
    database_operations_count INTEGER DEFAULT 0,
    api_calls_count INTEGER DEFAULT 0,
    
    -- Quality metrics
    fast_path_used BOOLEAN DEFAULT FALSE,
    llm_calls_count INTEGER DEFAULT 0,
    llm_total_time_ms INTEGER DEFAULT 0,
    
    -- Error tracking
    transient_errors_count INTEGER DEFAULT 0,
    permanent_errors_count INTEGER DEFAULT 0,
    
    -- Metadata
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

-- Idempotency Audit Table: Track idempotency key usage
CREATE TABLE IF NOT EXISTS idempotency_audit (
    audit_id SERIAL PRIMARY KEY,
    payload_hash VARCHAR(64) NOT NULL,
    task_id VARCHAR(255),
    
    -- Request details
    first_seen_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),
    duplicate_count INTEGER DEFAULT 1,
    
    -- TTL tracking
    expires_at TIMESTAMPTZ,
    expired BOOLEAN DEFAULT FALSE,
    
    -- Source tracking
    source_ip VARCHAR(45),
    user_agent TEXT,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for agent logging tables
CREATE INDEX idx_audit_log_run_id ON agent_audit_log(run_id);
CREATE INDEX idx_audit_log_timestamp ON agent_audit_log(timestamp);
CREATE INDEX idx_run_metrics_journey ON agent_run_metrics(journey_name);
CREATE INDEX idx_customer_context_last_interaction ON customer_context(last_interaction_date);

-- Indexes for refund_transactions
CREATE INDEX idx_refund_ticket ON refund_transactions(ticket_id);
CREATE INDEX idx_refund_booking ON refund_transactions(booking_id);
CREATE INDEX idx_refund_status ON refund_transactions(refund_status);
CREATE INDEX idx_refund_processed_at ON refund_transactions(processed_at);
CREATE INDEX idx_refund_location ON refund_transactions(location_id);
CREATE INDEX idx_refund_location_city ON refund_transactions(location_city);
CREATE INDEX idx_refund_reason ON refund_transactions(refund_reason);

-- Indexes for agent_performance_metrics
CREATE INDEX idx_perf_agent_name ON agent_performance_metrics(agent_name);
CREATE INDEX idx_perf_created_at ON agent_performance_metrics(created_at);
CREATE INDEX idx_perf_run_id ON agent_performance_metrics(run_id);

-- Indexes for agent_journey_steps
CREATE INDEX idx_journey_steps_run_id ON agent_journey_steps(run_id);
CREATE INDEX idx_journey_steps_name ON agent_journey_steps(step_name);
CREATE INDEX idx_journey_steps_status ON agent_journey_steps(status);
CREATE INDEX idx_journey_steps_created_at ON agent_journey_steps(created_at);

-- Indexes for async webhook orchestration tables
CREATE INDEX idx_webhook_tasks_status ON webhook_tasks(status);
CREATE INDEX idx_webhook_tasks_created_at ON webhook_tasks(created_at);
CREATE INDEX idx_webhook_tasks_payload_hash ON webhook_tasks(payload_hash);
CREATE INDEX idx_webhook_tasks_ticket_id ON webhook_tasks(ticket_id);
CREATE INDEX idx_webhook_tasks_trace_id ON webhook_tasks(trace_id);
CREATE INDEX idx_webhook_tasks_worker_id ON webhook_tasks(worker_id);

CREATE INDEX idx_dlq_task_id ON webhook_dead_letter_queue(task_id);
CREATE INDEX idx_dlq_resolved ON webhook_dead_letter_queue(resolved);
CREATE INDEX idx_dlq_created_at ON webhook_dead_letter_queue(created_at);
CREATE INDEX idx_dlq_failure_type ON webhook_dead_letter_queue(failure_type);

CREATE INDEX idx_task_metrics_task_id ON webhook_task_metrics(task_id);
CREATE INDEX idx_task_metrics_recorded_at ON webhook_task_metrics(recorded_at);

CREATE INDEX idx_idempotency_payload_hash ON idempotency_audit(payload_hash);
CREATE INDEX idx_idempotency_expires_at ON idempotency_audit(expires_at);
CREATE INDEX idx_idempotency_expired ON idempotency_audit(expired);
CREATE INDEX idx_idempotency_first_seen ON idempotency_audit(first_seen_at);


-- ============================================================
-- Dashboard Summary View
-- ============================================================

-- Dashboard Summary View: Pre-aggregated metrics for reporting
CREATE OR REPLACE VIEW dashboard_summary AS
SELECT 
    DATE(arm.start_time) as date,
    arm.agent_name,
    
    -- Volume metrics
    COUNT(*) as total_tickets,
    COUNT(CASE WHEN arm.final_outcome = 'approved' THEN 1 END) as approved_count,
    COUNT(CASE WHEN arm.final_outcome = 'denied' THEN 1 END) as denied_count,
    COUNT(CASE WHEN arm.final_outcome = 'escalated' THEN 1 END) as escalated_count,
    
    -- Performance metrics
    AVG(arm.duration_ms)/1000.0 as avg_duration_seconds,
    MIN(arm.duration_ms)/1000.0 as min_duration_seconds,
    MAX(arm.duration_ms)/1000.0 as max_duration_seconds,
    
    -- Resource usage
    AVG(arm.token_usage) as avg_token_usage,
    SUM(arm.token_usage) as total_token_usage,
    AVG(arm.api_calls_count) as avg_api_calls,
    
    -- Cache efficiency
    SUM(arm.cache_hits) as total_cache_hits,
    SUM(arm.cache_misses) as total_cache_misses,
    CASE 
        WHEN SUM(arm.cache_hits) + SUM(arm.cache_misses) > 0 
        THEN ROUND(100.0 * SUM(arm.cache_hits) / (SUM(arm.cache_hits) + SUM(arm.cache_misses)), 2)
        ELSE 0 
    END as cache_hit_rate_percent,
    
    -- Quality metrics
    SUM(arm.error_count) as total_errors,
    SUM(arm.retry_count) as total_retries,
    
    -- Refund metrics
    COUNT(rt.refund_id) as refunds_processed,
    SUM(rt.refund_amount) as total_refund_amount,
    COUNT(CASE WHEN rt.refund_status = 'completed' THEN 1 END) as successful_refunds,
    COUNT(CASE WHEN rt.refund_status = 'failed' THEN 1 END) as failed_refunds
    
FROM agent_run_metrics arm
LEFT JOIN refund_transactions rt ON arm.run_id = rt.run_id
WHERE arm.start_time > NOW() - INTERVAL '90 days'
GROUP BY DATE(arm.start_time), arm.agent_name
ORDER BY date DESC, arm.agent_name;
