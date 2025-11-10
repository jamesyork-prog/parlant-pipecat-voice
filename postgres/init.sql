-- Create products table
CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    category VARCHAR(100),
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample products
INSERT INTO products (product_name, description, price, category, stock_quantity) VALUES
('Wireless Headphones', 'Premium noise-canceling wireless headphones with 30-hour battery life', 199.99, 'Electronics', 45),
('Smart Watch', 'Fitness tracking smartwatch with heart rate monitor and GPS', 299.99, 'Electronics', 32),
('Laptop Stand', 'Ergonomic aluminum laptop stand with adjustable height', 49.99, 'Accessories', 78),
('USB-C Hub', '7-in-1 USB-C hub with HDMI, USB 3.0, and SD card reader', 39.99, 'Accessories', 120),
('Mechanical Keyboard', 'RGB backlit mechanical gaming keyboard with Cherry MX switches', 149.99, 'Electronics', 28),
('Wireless Mouse', 'Ergonomic wireless mouse with precision tracking', 59.99, 'Accessories', 95),
('Phone Case', 'Protective silicone case with raised edges for screen protection', 19.99, 'Accessories', 200),
('Portable Charger', '20000mAh portable power bank with fast charging', 44.99, 'Electronics', 65),
('Bluetooth Speaker', 'Waterproof portable speaker with 12-hour battery', 79.99, 'Electronics', 41),
('Desk Lamp', 'LED desk lamp with adjustable brightness and color temperature', 34.99, 'Accessories', 88);

-- Create orders table (for future use)
CREATE TABLE IF NOT EXISTS orders (
    order_id SERIAL PRIMARY KEY,
    order_number VARCHAR(50) UNIQUE NOT NULL,
    customer_email VARCHAR(255),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending'
);

-- Create indexes for better query performance
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_orders_number ON orders(order_number);
CREATE INDEX idx_orders_email ON orders(customer_email);

-- ============================================================
-- Agent Logging Tables
-- ============================================================

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
    ticket_id VARCHAR(255)
);

-- The Context "Memory" Table: A persistent store of key facts about customers
CREATE TABLE IF NOT EXISTS customer_context (
    customer_id VARCHAR(255) PRIMARY KEY,
    last_interaction_date TIMESTAMPTZ,
    total_interactions INTEGER DEFAULT 1,
    total_denials INTEGER DEFAULT 0,
    custom_notes JSONB
);

-- Create indexes for agent logging tables
CREATE INDEX idx_audit_log_run_id ON agent_audit_log(run_id);
CREATE INDEX idx_audit_log_timestamp ON agent_audit_log(timestamp);
CREATE INDEX idx_run_metrics_journey ON agent_run_metrics(journey_name);
CREATE INDEX idx_customer_context_last_interaction ON customer_context(last_interaction_date);
