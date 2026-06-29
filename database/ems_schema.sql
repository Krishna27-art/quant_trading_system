-- Execution Management System (EMS) Schema
-- PostgreSQL must be used for OLTP ACID compliance to manage order state transitions.

CREATE TABLE IF NOT EXISTS orders (
    order_id UUID PRIMARY KEY,
    parent_order_id UUID NULL,       -- For TWAP/VWAP child orders
    strategy_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,       -- 'BUY', 'SELL'
    order_type VARCHAR(20) NOT NULL, -- 'MARKET', 'LIMIT', 'TWAP', 'VWAP'
    total_qty INT NOT NULL,
    filled_qty INT DEFAULT 0,
    remaining_qty INT NOT NULL,
    limit_price DECIMAL(10, 2) NULL,
    avg_fill_price DECIMAL(10, 2) NULL,
    status VARCHAR(20) NOT NULL,     -- 'PENDING', 'OPEN', 'PARTIAL', 'FILLED', 'REJECTED', 'CANCELED'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    broker_id VARCHAR(50) NULL,      -- ID returned by the broker API
    broker_message TEXT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    fill_id UUID PRIMARY KEY,
    order_id UUID REFERENCES orders(order_id),
    symbol VARCHAR(20) NOT NULL,
    fill_qty INT NOT NULL,
    fill_price DECIMAL(10, 2) NOT NULL,
    commission DECIMAL(10, 2) DEFAULT 0.0,
    fill_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    broker_fill_id VARCHAR(50) NULL
);

-- Index for fast state reconciliation
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_symbol ON orders(symbol);
CREATE INDEX idx_fills_order_id ON fills(order_id);
