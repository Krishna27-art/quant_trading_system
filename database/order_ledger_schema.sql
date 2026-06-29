-- Order Ledger Schema for OMS State Reconstruction
-- 
-- This schema provides a single source of truth for order state
-- to prevent ghost orders and OMS desynchronization.
-- 
-- States: NEW, ACK, PARTIAL, FILLED, REJECTED, CANCELLED
-- 
-- Never trust broker response. Reconstruct state from:
-- - Order Book
-- - Trade Book  
-- - Position Book
-- every few seconds.

CREATE TABLE IF NOT EXISTS order_ledger (
    -- Primary keys
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_order_id VARCHAR(64) UNIQUE NOT NULL,
    broker_order_id VARCHAR(64),
    
    -- Order details
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(8) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(16) NOT NULL CHECK (order_type IN ('MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT')),
    quantity BIGINT NOT NULL CHECK (quantity > 0),
    price DECIMAL(18, 4),
    stop_price DECIMAL(18, 4),
    
    -- State management
    state VARCHAR(16) NOT NULL CHECK (state IN ('NEW', 'ACK', 'PARTIAL', 'FILLED', 'REJECTED', 'CANCELLED')),
    previous_state VARCHAR(16),
    
    -- Fill tracking
    filled_quantity BIGINT DEFAULT 0 CHECK (filled_quantity >= 0),
    avg_fill_price DECIMAL(18, 4),
    remaining_quantity BIGINT GENERATED ALWAYS AS (quantity - filled_quantity) STORED,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    rejected_at TIMESTAMP WITH TIME ZONE,
    
    -- Broker info
    broker VARCHAR(32),
    venue VARCHAR(32),
    
    -- Reconciliation
    last_reconciled_at TIMESTAMP WITH TIME ZONE,
    reconciliation_status VARCHAR(16) DEFAULT 'PENDING' CHECK (reconciliation_status IN ('PENDING', 'MATCHED', 'MISMATCH', 'RECONCILED')),
    
    -- Metadata
    strategy_id VARCHAR(64),
    portfolio_id VARCHAR(64),
    client_id VARCHAR(64),
    
    -- Audit
    created_by VARCHAR(64),
    updated_by VARCHAR(64),
    version BIGINT DEFAULT 1,
    
    -- Constraints
    CONSTRAINT check_price_positive CHECK (price IS NULL OR price > 0),
    CONSTRAINT check_stop_price_positive CHECK (stop_price IS NULL OR stop_price > 0),
    CONSTRAINT check_limit_price_required CHECK (
        (order_type = 'LIMIT' OR order_type = 'STOP_LIMIT') AND price IS NOT NULL
        OR order_type IN ('MARKET', 'STOP')
    ),
    CONSTRAINT check_stop_price_required CHECK (
        (order_type = 'STOP' OR order_type = 'STOP_LIMIT') AND stop_price IS NOT NULL
        OR order_type IN ('MARKET', 'LIMIT')
    )
);

-- Indexes for performance
CREATE INDEX idx_order_ledger_client_order_id ON order_ledger(client_order_id);
CREATE INDEX idx_order_ledger_broker_order_id ON order_ledger(broker_order_id);
CREATE INDEX idx_order_ledger_symbol ON order_ledger(symbol);
CREATE INDEX idx_order_ledger_state ON order_ledger(state);
CREATE INDEX idx_order_ledger_created_at ON order_ledger(created_at);
CREATE INDEX idx_order_ledger_updated_at ON order_ledger(updated_at);
CREATE INDEX idx_order_ledger_reconciliation_status ON order_ledger(reconciliation_status);
CREATE INDEX idx_order_ledger_strategy_id ON order_ledger(strategy_id);
CREATE INDEX idx_order_ledger_portfolio_id ON order_ledger(portfolio_id);

-- Trade Book - linked to Order Ledger
CREATE TABLE IF NOT EXISTS trade_book (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_ledger_uuid UUID NOT NULL REFERENCES order_ledger(uuid),
    trade_id VARCHAR(64) UNIQUE NOT NULL,
    
    -- Trade details
    symbol VARCHAR(32) NOT NULL,
    side VARCHAR(8) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    quantity BIGINT NOT NULL CHECK (quantity > 0),
    price DECIMAL(18, 4) NOT NULL,
    
    -- Timestamps
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    received_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Broker info
    broker VARCHAR(32),
    venue VARCHAR(32),
    broker_trade_id VARCHAR(64),
    
    -- Metadata
    commission DECIMAL(18, 4) DEFAULT 0,
    fees DECIMAL(18, 4) DEFAULT 0,
    
    -- Audit
    created_by VARCHAR(64),
    
    CONSTRAINT check_trade_price_positive CHECK (price > 0)
);

CREATE INDEX idx_trade_book_order_ledger_uuid ON trade_book(order_ledger_uuid);
CREATE INDEX idx_trade_book_trade_id ON trade_book(trade_id);
CREATE INDEX idx_trade_book_symbol ON trade_book(symbol);
CREATE INDEX idx_trade_book_executed_at ON trade_book(executed_at);

-- Position Book - reconstructed from trades
CREATE TABLE IF NOT EXISTS position_book (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(32) NOT NULL,
    portfolio_id VARCHAR(64),
    
    -- Position details
    quantity BIGINT NOT NULL,
    avg_cost DECIMAL(18, 4) NOT NULL,
    current_price DECIMAL(18, 4),
    unrealized_pnl DECIMAL(18, 4),
    realized_pnl DECIMAL(18, 4) DEFAULT 0,
    
    -- Timestamps
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Reconciliation
    last_reconciled_at TIMESTAMP WITH TIME ZONE,
    reconciliation_status VARCHAR(16) DEFAULT 'PENDING' CHECK (reconciliation_status IN ('PENDING', 'MATCHED', 'MISMATCH', 'RECONCILED')),
    
    -- Audit
    created_by VARCHAR(64),
    version BIGINT DEFAULT 1
);

CREATE UNIQUE INDEX idx_position_book_symbol_portfolio ON position_book(symbol, portfolio_id);
CREATE INDEX idx_position_book_symbol ON position_book(symbol);
CREATE INDEX idx_position_book_portfolio_id ON position_book(portfolio_id);
CREATE INDEX idx_position_book_updated_at ON position_book(updated_at);

-- Order State Transitions (audit trail)
CREATE TABLE IF NOT EXISTS order_state_transitions (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_ledger_uuid UUID NOT NULL REFERENCES order_ledger(uuid),
    
    -- Transition details
    from_state VARCHAR(16),
    to_state VARCHAR(16) NOT NULL,
    transition_reason TEXT,
    
    -- Timestamps
    transitioned_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Metadata
    triggered_by VARCHAR(64),
    external_reference VARCHAR(64),
    
    -- Audit
    created_by VARCHAR(64)
);

CREATE INDEX idx_order_state_transitions_order_ledger_uuid ON order_state_transitions(order_ledger_uuid);
CREATE INDEX idx_order_state_transitions_transitioned_at ON order_state_transitions(transitioned_at);

-- Reconciliation Log
CREATE TABLE IF NOT EXISTS reconciliation_log (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Reconciliation details
    reconciliation_type VARCHAR(32) NOT NULL CHECK (reconciliation_type IN ('ORDER', 'POSITION', 'TRADE')),
    reference_id VARCHAR(64) NOT NULL,
    
    -- Source states
    oms_state VARCHAR(16),
    broker_state VARCHAR(16),
    position_book_state VARCHAR(16),
    
    -- Reconciliation result
    status VARCHAR(16) NOT NULL CHECK (status IN ('MATCHED', 'MISMATCH', 'RECONCILED')),
    mismatch_details TEXT,
    
    -- Timestamps
    reconciled_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Audit
    reconciled_by VARCHAR(64)
);

CREATE INDEX idx_reconciliation_log_reference_id ON reconciliation_log(reference_id);
CREATE INDEX idx_reconciliation_log_reconciled_at ON reconciliation_log(reconciled_at);
CREATE INDEX idx_reconciliation_log_status ON reconciliation_log(status);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_order_ledger_updated_at BEFORE UPDATE ON order_ledger
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_position_book_updated_at BEFORE UPDATE ON position_book
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to insert order state transition
CREATE OR REPLACE FUNCTION log_order_state_transition()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.state IS DISTINCT FROM NEW.state THEN
        INSERT INTO order_state_transitions (
            order_ledger_uuid,
            from_state,
            to_state,
            transition_reason,
            triggered_by,
            created_by
        ) VALUES (
            NEW.uuid,
            OLD.state,
            NEW.state,
            'State change',
            current_user,
            current_user
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER log_order_state_transition_trigger BEFORE UPDATE ON order_ledger
    FOR EACH ROW EXECUTE FUNCTION log_order_state_transition();

-- Function to reconstruct position from trades
CREATE OR REPLACE FUNCTION reconstruct_position(p_symbol VARCHAR(32), p_portfolio_id VARCHAR(64 DEFAULT NULL)
RETURNS TABLE (
    quantity BIGINT,
    avg_cost DECIMAL(18, 4),
    unrealized_pnl DECIMAL(18, 4)
) AS $$
DECLARE
    v_quantity BIGINT;
    v_total_cost DECIMAL(18, 4);
    v_avg_cost DECIMAL(18, 4);
    v_current_price DECIMAL(18, 4);
    v_unrealized_pnl DECIMAL(18, 4);
BEGIN
    -- Calculate total quantity and cost from trades
    SELECT 
        SUM(CASE WHEN side = 'BUY' THEN quantity ELSE -quantity END),
        SUM(CASE WHEN side = 'BUY' THEN quantity * price ELSE -quantity * price END)
    INTO v_quantity, v_total_cost
    FROM trade_book tb
    JOIN order_ledger ol ON tb.order_ledger_uuid = ol.uuid
    WHERE tb.symbol = p_symbol
    AND (p_portfolio_id IS NULL OR ol.portfolio_id = p_portfolio_id);
    
    -- Calculate average cost
    IF v_quantity != 0 THEN
        v_avg_cost := v_total_cost / v_quantity;
    ELSE
        v_avg_cost := 0;
    END IF;
    
    -- Get current price (from latest trade or market data)
    SELECT price INTO v_current_price
    FROM trade_book
    WHERE symbol = p_symbol
    ORDER BY executed_at DESC
    LIMIT 1;
    
    -- Calculate unrealized PnL
    IF v_current_price IS NOT NULL AND v_quantity != 0 THEN
        v_unrealized_pnl := (v_current_price - v_avg_cost) * v_quantity;
    ELSE
        v_unrealized_pnl := 0;
    END IF;
    
    RETURN QUERY SELECT v_quantity, v_avg_cost, v_unrealized_pnl;
END;
$$ LANGUAGE plpgsql;

-- Function to reconcile order state
CREATE OR REPLACE FUNCTION reconcile_order_state(p_client_order_id VARCHAR(64))
RETURNS TABLE (
    status VARCHAR(16),
    oms_state VARCHAR(16),
    broker_state VARCHAR(16),
    mismatch_details TEXT
) AS $$
DECLARE
    v_oms_state VARCHAR(16);
    v_broker_state VARCHAR(16);
    v_status VARCHAR(16);
    v_mismatch_details TEXT;
BEGIN
    -- Get OMS state
    SELECT state INTO v_oms_state
    FROM order_ledger
    WHERE client_order_id = p_client_order_id;
    
    -- Get broker state (simulated - in production, query broker API)
    -- For now, assume broker state matches OMS state
    v_broker_state := v_oms_state;
    
    -- Compare states
    IF v_oms_state = v_broker_state THEN
        v_status := 'MATCHED';
        v_mismatch_details := NULL;
    ELSE
        v_status := 'MISMATCH';
        v_mismatch_details := format('OMS state: %s, Broker state: %s', v_oms_state, v_broker_state);
    END IF;
    
    -- Log reconciliation
    INSERT INTO reconciliation_log (
        reconciliation_type,
        reference_id,
        oms_state,
        broker_state,
        status,
        mismatch_details,
        reconciled_by
    ) VALUES (
        'ORDER',
        p_client_order_id,
        v_oms_state,
        v_broker_state,
        v_status,
        v_mismatch_details,
        current_user
    );
    
    -- Update order ledger reconciliation status
    UPDATE order_ledger
    SET 
        reconciliation_status = v_status,
        last_reconciled_at = NOW()
    WHERE client_order_id = p_client_order_id;
    
    RETURN QUERY SELECT v_status, v_oms_state, v_broker_state, v_mismatch_details;
END;
$$ LANGUAGE plpgsql;

-- View for active orders
CREATE OR REPLACE VIEW active_orders AS
SELECT 
    uuid,
    client_order_id,
    broker_order_id,
    symbol,
    side,
    order_type,
    quantity,
    price,
    stop_price,
    state,
    filled_quantity,
    remaining_quantity,
    avg_fill_price,
    created_at,
    updated_at,
    broker,
    venue,
    strategy_id,
    portfolio_id
FROM order_ledger
WHERE state IN ('NEW', 'ACK', 'PARTIAL');

-- View for order statistics
CREATE OR REPLACE VIEW order_statistics AS
SELECT 
    state,
    COUNT(*) as order_count,
    SUM(quantity) as total_quantity,
    SUM(filled_quantity) as total_filled_quantity,
    SUM(remaining_quantity) as total_remaining_quantity,
    AVG(avg_fill_price) as avg_fill_price
FROM order_ledger
GROUP BY state;

-- View for position summary
CREATE OR REPLACE VIEW position_summary AS
SELECT 
    symbol,
    portfolio_id,
    quantity,
    avg_cost,
    current_price,
    unrealized_pnl,
    realized_pnl,
    updated_at
FROM position_book
WHERE quantity != 0;
