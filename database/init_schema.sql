-- Source: database/order_ledger_schema.sql
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


-- Source: database/instrument_master_schema.sql
-- ============================================================================
-- INSTRUMENT MASTER SCHEMA - India-Specific Canonical Security Table
-- ============================================================================
-- CRITICAL: India data is messy. This schema handles:
-- - Stable internal instrument_id (never changes)
-- - Symbol changes (YES BANK type events)
-- - Mergers/delistings
-- - F&O inclusion/exclusion changes
-- - Corporate action history pointers
-- - Point-in-time resolution for backtesting
-- ============================================================================

-- ============================================================================
-- MAIN INSTRUMENT MASTER TABLE (SCD Type 2)
-- ============================================================================

CREATE TABLE IF NOT EXISTS instrument_master (
    -- Immutable identifiers
    instrument_id VARCHAR PRIMARY KEY,
    isin VARCHAR,
    
    -- Current symbol data
    nse_symbol VARCHAR NOT NULL,
    bse_symbol VARCHAR,
    
    -- Company data
    company_name VARCHAR NOT NULL,
    sector VARCHAR,
    industry VARCHAR,
    
    -- Listing data
    listing_date DATE NOT NULL,
    delisting_date DATE,
    is_delisted BOOLEAN DEFAULT FALSE,
    
    -- Corporate action tracking
    corporate_action_count INTEGER DEFAULT 0,
    last_corporate_action_date DATE,
    last_corporate_action_type VARCHAR,
    
    -- Symbol change tracking
    previous_symbols TEXT,  -- JSON array of previous symbols
    symbol_change_date DATE,
    symbol_change_reason VARCHAR,
    
    -- Merger/demerger tracking
    merged_into_instrument_id VARCHAR,
    merged_date DATE,
    demerged_from_instrument_id VARCHAR,
    demerged_date DATE,
    
    -- F&O tracking
    is_fno_eligible BOOLEAN DEFAULT FALSE,
    fno_inclusion_date DATE,
    fno_exclusion_date DATE,
    fno_lot_size INTEGER,
    
    -- Trading parameters
    face_value DOUBLE,
    tick_size DOUBLE,
    lot_size INTEGER,
    series VARCHAR,
    
    -- Metadata
    status VARCHAR DEFAULT 'active',
    effective_from DATE NOT NULL,
    effective_to DATE,
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key references
    FOREIGN KEY (merged_into_instrument_id) REFERENCES instrument_master(instrument_id),
    FOREIGN KEY (demerged_from_instrument_id) REFERENCES instrument_master(instrument_id)
);

-- Instrument Master Indexes
CREATE INDEX IF NOT EXISTS idx_instrument_master_nse_symbol ON instrument_master(nse_symbol);
CREATE INDEX IF NOT EXISTS idx_instrument_master_isin ON instrument_master(isin);
CREATE INDEX IF NOT EXISTS idx_instrument_master_status ON instrument_master(status);
CREATE INDEX IF NOT EXISTS idx_instrument_master_effective ON instrument_master(effective_from, effective_to);
CREATE INDEX IF NOT EXISTS idx_instrument_master_current ON instrument_master(is_current) WHERE is_current = TRUE;
CREATE INDEX IF NOT EXISTS idx_instrument_master_fno_eligible ON instrument_master(is_fno_eligible) WHERE is_fno_eligible = TRUE;
CREATE INDEX IF NOT EXISTS idx_instrument_master_sector ON instrument_master(sector);
CREATE INDEX IF NOT EXISTS idx_instrument_master_listing_date ON instrument_master(listing_date);

-- ============================================================================
-- SYMBOL HISTORY TABLE (SCD Type 2)
-- ============================================================================

CREATE TABLE IF NOT EXISTS instrument_symbol_history (
    history_id VARCHAR PRIMARY KEY,
    instrument_id VARCHAR NOT NULL,
    old_symbol VARCHAR NOT NULL,
    new_symbol VARCHAR NOT NULL,
    change_date DATE NOT NULL,
    change_reason VARCHAR NOT NULL,
    corporate_action_id VARCHAR,
    effective_from DATE NOT NULL,
    effective_to DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (instrument_id) REFERENCES instrument_master(instrument_id)
);

-- Symbol History Indexes
CREATE INDEX IF NOT EXISTS idx_symbol_history_instrument ON instrument_symbol_history(instrument_id);
CREATE INDEX IF NOT EXISTS idx_symbol_history_old_symbol ON instrument_symbol_history(old_symbol);
CREATE INDEX IF NOT EXISTS idx_symbol_history_new_symbol ON instrument_symbol_history(new_symbol);
CREATE INDEX IF NOT EXISTS idx_symbol_history_date ON instrument_symbol_history(change_date);
CREATE INDEX IF NOT EXISTS idx_symbol_history_effective ON instrument_symbol_history(effective_from, effective_to);
CREATE INDEX IF NOT EXISTS idx_symbol_history_ca ON instrument_symbol_history(corporate_action_id);

-- ============================================================================
-- F&O HISTORY TABLE (SCD Type 2)
-- ============================================================================

CREATE TABLE IF NOT EXISTS instrument_fno_history (
    history_id VARCHAR PRIMARY KEY,
    instrument_id VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,  -- 'inclusion' or 'exclusion'
    event_date DATE NOT NULL,
    lot_size INTEGER,
    reason VARCHAR,
    effective_from DATE NOT NULL,
    effective_to DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (instrument_id) REFERENCES instrument_master(instrument_id)
);

-- F&O History Indexes
CREATE INDEX IF NOT EXISTS idx_fno_history_instrument ON instrument_fno_history(instrument_id);
CREATE INDEX IF NOT EXISTS idx_fno_history_event_type ON instrument_fno_history(event_type);
CREATE INDEX IF NOT EXISTS idx_fno_history_date ON instrument_fno_history(event_date);
CREATE INDEX IF NOT EXISTS idx_fno_history_effective ON instrument_fno_history(effective_from, effective_to);
CREATE INDEX IF NOT EXISTS idx_fno_history_current ON instrument_fno_history(effective_to) WHERE effective_to IS NULL;

-- ============================================================================
-- MERGER/DEMERGER HISTORY TABLE (SCD Type 2)
-- ============================================================================

CREATE TABLE IF NOT EXISTS instrument_merger_history (
    history_id VARCHAR PRIMARY KEY,
    surviving_instrument_id VARCHAR NOT NULL,
    absorbed_instrument_id VARCHAR NOT NULL,
    merger_date DATE NOT NULL,
    swap_ratio DOUBLE,
    scheme TEXT,
    effective_from DATE NOT NULL,
    effective_to DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (surviving_instrument_id) REFERENCES instrument_master(instrument_id),
    FOREIGN KEY (absorbed_instrument_id) REFERENCES instrument_master(instrument_id)
);

-- Merger History Indexes
CREATE INDEX IF NOT EXISTS idx_merger_history_surviving ON instrument_merger_history(surviving_instrument_id);
CREATE INDEX IF NOT EXISTS idx_merger_history_absorbed ON instrument_merger_history(absorbed_instrument_id);
CREATE INDEX IF NOT EXISTS idx_merger_history_date ON instrument_merger_history(merger_date);
CREATE INDEX IF NOT EXISTS idx_merger_history_effective ON instrument_merger_history(effective_from, effective_to);

-- ============================================================================
-- CORPORATE ACTION POINTER TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS instrument_corporate_action_pointer (
    pointer_id VARCHAR PRIMARY KEY,
    instrument_id VARCHAR NOT NULL,
    corporate_action_id VARCHAR NOT NULL,
    action_type VARCHAR NOT NULL,
    effective_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (instrument_id) REFERENCES instrument_master(instrument_id),
    UNIQUE (instrument_id, corporate_action_id)
);

-- Corporate Action Pointer Indexes
CREATE INDEX IF NOT EXISTS idx_ca_pointer_instrument ON instrument_corporate_action_pointer(instrument_id);
CREATE INDEX IF NOT EXISTS idx_ca_pointer_ca ON instrument_corporate_action_pointer(corporate_action_id);
CREATE INDEX IF NOT EXISTS idx_ca_pointer_type ON instrument_corporate_action_pointer(action_type);
CREATE INDEX IF NOT EXISTS idx_ca_pointer_date ON instrument_corporate_action_pointer(effective_date);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Current Instrument Master View
CREATE OR REPLACE VIEW v_current_instrument_master AS
SELECT * FROM instrument_master WHERE is_current = TRUE;

-- Active Instruments View
CREATE OR REPLACE VIEW v_active_instruments AS
SELECT * FROM instrument_master 
WHERE is_current = TRUE 
  AND status = 'active' 
  AND is_delisted = FALSE;

-- F&O Eligible Instruments View
CREATE OR REPLACE VIEW v_fno_eligible_instruments AS
SELECT * FROM instrument_master 
WHERE is_current = TRUE 
  AND is_fno_eligible = TRUE 
  AND status = 'active';

-- Symbol Resolution View (for backtesting)
CREATE OR REPLACE VIEW v_symbol_resolution AS
SELECT 
    im.instrument_id,
    im.nse_symbol as current_symbol,
    im.previous_symbols,
    im.symbol_change_date,
    im.listing_date,
    im.delisting_date,
    im.status
FROM instrument_master im
WHERE im.is_current = TRUE;

-- Instruments with Symbol Changes View
CREATE OR REPLACE VIEW v_instruments_with_symbol_changes AS
SELECT 
    im.*,
    COUNT(ish.history_id) as symbol_change_count
FROM instrument_master im
LEFT JOIN instrument_symbol_history ish ON im.instrument_id = ish.instrument_id
WHERE im.is_current = TRUE
GROUP BY im.instrument_id
HAVING COUNT(ish.history_id) > 0;

-- Instruments with F&O Changes View
CREATE OR REPLACE VIEW v_instruments_with_fno_changes AS
SELECT 
    im.*,
    COUNT(ifh.history_id) as fno_change_count,
    MAX(ifh.event_date) as last_fno_change_date
FROM instrument_master im
LEFT JOIN instrument_fno_history ifh ON im.instrument_id = ifh.instrument_id
WHERE im.is_current = TRUE
GROUP BY im.instrument_id
HAVING COUNT(ifh.history_id) > 0;

-- ============================================================================
-- STORED PROCEDURES FOR COMMON OPERATIONS
-- ============================================================================

-- Get symbol as of a specific date (prevents survivorship bias)
CREATE OR REPLACE FUNCTION get_symbol_as_of(
    p_instrument_id VARCHAR,
    p_query_date DATE
) RETURNS VARCHAR AS $$
DECLARE
    v_symbol VARCHAR;
    v_change_date DATE;
    v_old_symbol VARCHAR;
BEGIN
    -- Get current symbol and most recent change date
    SELECT 
        nse_symbol, 
        symbol_change_date
    INTO v_symbol, v_change_date
    FROM instrument_master
    WHERE instrument_id = p_instrument_id
      AND is_current = TRUE;
    
    -- If there was a symbol change after the query date, find the old symbol
    IF v_change_date IS NOT NULL AND p_query_date < v_change_date THEN
        SELECT old_symbol INTO v_old_symbol
        FROM instrument_symbol_history
        WHERE instrument_id = p_instrument_id
          AND change_date <= p_query_date
          AND (effective_to IS NULL OR effective_to > p_query_date)
        ORDER BY change_date DESC
        LIMIT 1;
        
        IF v_old_symbol IS NOT NULL THEN
            RETURN v_old_symbol;
        END IF;
    END IF;
    
    RETURN v_symbol;
END;
$$ LANGUAGE plpgsql;

-- Check F&O eligibility as of a specific date
CREATE OR REPLACE FUNCTION is_fno_eligible_as_of(
    p_instrument_id VARCHAR,
    p_query_date DATE
) RETURNS BOOLEAN AS $$
DECLARE
    v_eligible BOOLEAN;
    v_inclusion_date DATE;
    v_exclusion_date DATE;
BEGIN
    -- Get current F&O status
    SELECT 
        is_fno_eligible,
        fno_inclusion_date,
        fno_exclusion_date
    INTO v_eligible, v_inclusion_date, v_exclusion_date
    FROM instrument_master
    WHERE instrument_id = p_instrument_id
      AND is_current = TRUE;
    
    -- If not currently eligible, return false
    IF NOT v_eligible THEN
        RETURN FALSE;
    END IF;
    
    -- Check if query date is before inclusion
    IF v_inclusion_date IS NOT NULL AND p_query_date < v_inclusion_date THEN
        RETURN FALSE;
    END IF;
    
    -- Check if query date is after exclusion
    IF v_exclusion_date IS NOT NULL AND p_query_date >= v_exclusion_date THEN
        RETURN FALSE;
    END IF;
    
    -- Check historical F&O history for more complex cases
    -- (e.g., instrument was included, then excluded, then included again)
    PERFORM 1 FROM instrument_fno_history
    WHERE instrument_id = p_instrument_id
      AND event_date <= p_query_date
      AND (effective_to IS NULL OR effective_to > p_query_date)
    LIMIT 1;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Record symbol change (handles YES BANK type events)
CREATE OR REPLACE FUNCTION record_symbol_change(
    p_instrument_id VARCHAR,
    p_old_symbol VARCHAR,
    p_new_symbol VARCHAR,
    p_change_date DATE,
    p_change_reason VARCHAR,
    p_corporate_action_id VARCHAR DEFAULT NULL
) RETURNS VARCHAR AS $$
DECLARE
    v_history_id VARCHAR;
    v_current_effective_to DATE;
BEGIN
    -- Generate history ID
    v_history_id := 'SYMH_' || md5(p_instrument_id || p_old_symbol || p_new_symbol || p_change_date);
    
    -- Close previous history record if exists
    UPDATE instrument_symbol_history
    SET effective_to = p_change_date
    WHERE instrument_id = p_instrument_id
      AND effective_to IS NULL;
    
    -- Insert new history record
    INSERT INTO instrument_symbol_history (
        history_id, instrument_id, old_symbol, new_symbol,
        change_date, change_reason, corporate_action_id,
        effective_from, effective_to
    ) VALUES (
        v_history_id, p_instrument_id, p_old_symbol, p_new_symbol,
        p_change_date, p_change_reason, p_corporate_action_id,
        p_change_date, NULL
    );
    
    -- Update instrument master
    UPDATE instrument_master
    SET 
        nse_symbol = p_new_symbol,
        previous_symbols = COALESCE(previous_symbols, '[]'::json) || to_json(p_old_symbol),
        symbol_change_date = p_change_date,
        symbol_change_reason = p_change_reason,
        updated_at = CURRENT_TIMESTAMP
    WHERE instrument_id = p_instrument_id;
    
    RETURN v_history_id;
END;
$$ LANGUAGE plpgsql;

-- Record F&O inclusion/exclusion
CREATE OR REPLACE FUNCTION record_fno_event(
    p_instrument_id VARCHAR,
    p_event_type VARCHAR,  -- 'inclusion' or 'exclusion'
    p_event_date DATE,
    p_lot_size INTEGER DEFAULT NULL,
    p_reason VARCHAR DEFAULT NULL
) RETURNS VARCHAR AS $$
DECLARE
    v_history_id VARCHAR;
BEGIN
    -- Generate history ID
    v_history_id := 'FNOH_' || md5(p_instrument_id || p_event_type || p_event_date);
    
    -- Close previous history record if exists
    UPDATE instrument_fno_history
    SET effective_to = p_event_date
    WHERE instrument_id = p_instrument_id
      AND effective_to IS NULL;
    
    -- Insert new history record
    INSERT INTO instrument_fno_history (
        history_id, instrument_id, event_type, event_date,
        lot_size, reason, effective_from, effective_to
    ) VALUES (
        v_history_id, p_instrument_id, p_event_type, p_event_date,
        p_lot_size, p_reason, p_event_date, NULL
    );
    
    -- Update instrument master
    IF p_event_type = 'inclusion' THEN
        UPDATE instrument_master
        SET 
            is_fno_eligible = TRUE,
            fno_inclusion_date = p_event_date,
            fno_exclusion_date = NULL,
            fno_lot_size = p_lot_size,
            updated_at = CURRENT_TIMESTAMP
        WHERE instrument_id = p_instrument_id;
    ELSIF p_event_type = 'exclusion' THEN
        UPDATE instrument_master
        SET 
            is_fno_eligible = FALSE,
            fno_exclusion_date = p_event_date,
            updated_at = CURRENT_TIMESTAMP
        WHERE instrument_id = p_instrument_id;
    END IF;
    
    RETURN v_history_id;
END;
$$ LANGUAGE plpgsql;

-- Record merger event
CREATE OR REPLACE FUNCTION record_merger(
    p_surviving_instrument_id VARCHAR,
    p_absorbed_instrument_id VARCHAR,
    p_merger_date DATE,
    p_swap_ratio DOUBLE DEFAULT NULL,
    p_scheme TEXT DEFAULT NULL
) RETURNS VARCHAR AS $$
DECLARE
    v_history_id VARCHAR;
BEGIN
    -- Generate history ID
    v_history_id := 'MERGH_' || md5(p_surviving_instrument_id || p_absorbed_instrument_id || p_merger_date);
    
    -- Insert merger history record
    INSERT INTO instrument_merger_history (
        history_id, surviving_instrument_id, absorbed_instrument_id,
        merger_date, swap_ratio, scheme, effective_from, effective_to
    ) VALUES (
        v_history_id, p_surviving_instrument_id, p_absorbed_instrument_id,
        p_merger_date, p_swap_ratio, p_scheme, p_merger_date, NULL
    );
    
    -- Update absorbed instrument
    UPDATE instrument_master
    SET 
        status = 'merged',
        merged_into_instrument_id = p_surviving_instrument_id,
        merged_date = p_merger_date,
        is_current = FALSE,
        effective_to = p_merger_date,
        updated_at = CURRENT_TIMESTAMP
    WHERE instrument_id = p_absorbed_instrument_id;
    
    -- Create new current record for absorbed instrument (historical)
    INSERT INTO instrument_master (
        instrument_id, isin, nse_symbol, company_name, sector, industry,
        listing_date, delisting_date, is_delisted, status,
        merged_into_instrument_id, merged_date,
        effective_from, effective_to, is_current
    )
    SELECT 
        instrument_id, isin, nse_symbol, company_name, sector, industry,
        listing_date, delisting_date, is_delisted, status,
        p_surviving_instrument_id, p_merger_date,
        effective_from, p_merger_date, FALSE
    FROM instrument_master
    WHERE instrument_id = p_absorbed_instrument_id
      AND is_current = TRUE;
    
    RETURN v_history_id;
END;
$$ LANGUAGE plpgsql;


-- Source: database/universe_schema.sql
-- Point-In-Time Universe Master Schema
-- Eliminates survivorship bias by tracking exact inclusion and exclusion dates

CREATE TABLE IF NOT EXISTS universe_master (
    index_name String,           -- e.g., 'NIFTY 50', 'NIFTY 500'
    symbol String,               -- e.g., 'RELIANCE'
    inclusion_date Date,         -- Date the stock was added to the index
    exclusion_date Nullable(Date), -- Date the stock was removed (NULL if currently active)
    exchange_timestamp DateTime, -- BITEMPORAL: When the exchange published the change
    local_recv_timestamp DateTime DEFAULT now(), -- BITEMPORAL: When our infra parsed it
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (index_name, symbol, inclusion_date);

-- View to get the active universe for a specific date
-- Usage: SELECT symbol FROM active_universe_view WHERE index_name = 'NIFTY 50' AND as_of_date = '2023-01-01'
-- (This view is conceptual, usually queried directly from the master table)


-- Source: database/adjusted_equity_schema.sql
-- Adjusted Equity History Schema
-- Contains corporate action adjusted OHLCV data for institutional feature stores

CREATE TABLE IF NOT EXISTS adjusted_equity_history (
    symbol String,
    date Date,
    adj_open Float32,
    adj_high Float32,
    adj_low Float32,
    adj_close Float32,
    adj_volume Int32,
    split_factor Float32 DEFAULT 1.0,
    dividend_yield Float32 DEFAULT 0.0,
    exchange_timestamp DateTime, -- BITEMPORAL: Actual time of the trade/close
    local_recv_timestamp DateTime DEFAULT now(), -- BITEMPORAL: Time our system logged it
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (symbol, date);

CREATE TABLE IF NOT EXISTS events_multiplier_table (
    symbol String,
    ex_date Date,
    multiplier Float32,
    action_type String, -- 'SPLIT', 'BONUS', 'DIVIDEND'
    last_updated DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(last_updated)
ORDER BY (symbol, ex_date, action_type);


-- Source: database/ems_schema.sql
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


-- Source: data_infra/storage/database/schema.sql
-- ============================================================================
-- INSTITUTIONAL DATABASE SCHEMA
-- ============================================================================
-- Design Philosophy: Raw → Reference → Event → Feature Tables
-- This schema follows institutional data warehouse best practices
-- ============================================================================

-- ============================================================================
-- RAW DATA PRESERVATION (Bronze Layer - Immutable Source Data)
-- ============================================================================

-- Raw Data Preservation Log
-- Tracks all raw API responses preserved in bronze layer
CREATE TABLE IF NOT EXISTS raw_data_preservation_log (
    preservation_id VARCHAR PRIMARY KEY,
    data_source VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,
    symbol VARCHAR,
    timestamp TIMESTAMP NOT NULL,
    request_params JSON,
    metadata JSON,
    data_hash VARCHAR,
    file_path VARCHAR,
    preserved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_size_bytes BIGINT
);

-- Raw Data Preservation Indexes
CREATE INDEX IF NOT EXISTS idx_raw_preservation_source ON raw_data_preservation_log(data_source);
CREATE INDEX IF NOT EXISTS idx_raw_preservation_type ON raw_data_preservation_log(data_type);
CREATE INDEX IF NOT EXISTS idx_raw_preservation_symbol ON raw_data_preservation_log(symbol);
CREATE INDEX IF NOT EXISTS idx_raw_preservation_timestamp ON raw_data_preservation_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_preservation_hash ON raw_data_preservation_log(data_hash);

-- ============================================================================
-- REFERENCE TABLES (Slowly Changing Dimension Type 2)
-- ============================================================================

-- Security Master (Single Source of Truth)
-- Contains all security reference data with effective dating
CREATE TABLE IF NOT EXISTS security_master (
    security_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    isin VARCHAR,
    company_name VARCHAR NOT NULL,
    sector VARCHAR,
    industry VARCHAR,
    listing_date DATE,
    delisting_date DATE,
    face_value DOUBLE,
    market_cap_bucket VARCHAR,
    nse_symbol VARCHAR,
    bse_symbol VARCHAR,
    lot_size INTEGER,
    series VARCHAR,
    effective_from DATE NOT NULL,
    effective_to DATE,
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading Calendar Reference
CREATE TABLE IF NOT EXISTS trading_calendar (
    date DATE PRIMARY KEY,
    is_trading_day BOOLEAN NOT NULL,
    expiry VARCHAR,
    holiday VARCHAR,
    weekday INTEGER,
    week INTEGER,
    month INTEGER,
    quarter INTEGER,
    year INTEGER
);

-- ============================================================================
-- RAW TABLES (Immutable Source Data)
-- ============================================================================

-- Raw Equity History (Bronze Layer - Immutable)
CREATE TABLE IF NOT EXISTS raw_equity_history (
    ingest_id VARCHAR,
    symbol VARCHAR NOT NULL,
    series VARCHAR,
    date DATE NOT NULL,
    prev_close DOUBLE,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    last_price DOUBLE,
    close DOUBLE,
    average_price DOUBLE,
    volume BIGINT,
    turnover DOUBLE,
    num_trades BIGINT,
    source VARCHAR DEFAULT 'NSE',
    ingest_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ingest_id, symbol, date)
);

-- Raw Options Chain (Bronze Layer - Immutable)
CREATE TABLE IF NOT EXISTS raw_options_chain (
    ingest_id VARCHAR,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR NOT NULL,
    expiry DATE NOT NULL,
    strike DOUBLE NOT NULL,
    option_type VARCHAR NOT NULL,  -- 'CE' or 'PE'
    bid DOUBLE,
    ask DOUBLE,
    ltp DOUBLE,  -- Last Traded Price
    oi BIGINT,  -- Open Interest
    oi_change BIGINT,  -- Change in Open Interest
    iv DOUBLE,  -- Implied Volatility
    delta DOUBLE,  -- Option Delta
    gamma DOUBLE,  -- Option Gamma
    theta DOUBLE,  -- Option Theta
    vega DOUBLE,  -- Option Vega
    rho DOUBLE,  -- Option Rho
    volume BIGINT,  -- Trading Volume
    source VARCHAR DEFAULT 'NSE',
    ingest_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ingest_id, symbol, timestamp, expiry, strike, option_type)
);

-- Raw FII/DII Flows (Bronze Layer - Immutable)
CREATE TABLE IF NOT EXISTS raw_fii_dii_flows (
    ingest_id VARCHAR,
    date DATE NOT NULL,
    fii_buy_value_cr DOUBLE,
    fii_sell_value_cr DOUBLE,
    fii_buy_contracts BIGINT,
    fii_sell_contracts BIGINT,
    fii_net_value_cr DOUBLE,
    fii_net_contracts BIGINT,
    source VARCHAR DEFAULT 'NSE',
    ingest_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ingest_id, date)
);

-- ============================================================================
-- EVENT TABLES (Immutable Business Events)
-- ============================================================================

-- Corporate Action Events (Immutable)
CREATE TABLE IF NOT EXISTS corporate_action_events (
    event_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    action_type VARCHAR NOT NULL,
    record_date DATE,
    ex_date DATE,
    announcement_date DATE,
    ratio DOUBLE,
    bonus_ratio VARCHAR,
    split_ratio VARCHAR,
    dividend_amount DOUBLE,
    dividend_percentage DOUBLE,
    description TEXT,
    effective_from DATE NOT NULL,
    effective_to DATE,
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Earnings Events (Immutable)
CREATE TABLE IF NOT EXISTS earnings_events (
    event_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    period_end_date DATE,
    announcement_date DATE,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    revenue DOUBLE,
    net_profit DOUBLE,
    eps DOUBLE,
    operating_margin DOUBLE,
    net_margin DOUBLE,
    yoy_revenue_growth DOUBLE,
    yoy_profit_growth DOUBLE,
    source VARCHAR DEFAULT 'NSE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index Membership History (SCD Type 2)
CREATE TABLE IF NOT EXISTS index_membership_history (
    membership_id VARCHAR PRIMARY KEY,
    index_name VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    effective_from DATE NOT NULL,
    effective_to DATE,
    is_current BOOLEAN DEFAULT TRUE,
    entry_reason VARCHAR,
    exit_reason VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Symbol Mapping History (SCD Type 2)
CREATE TABLE IF NOT EXISTS symbol_mapping_history (
    mapping_id VARCHAR PRIMARY KEY,
    old_symbol VARCHAR,
    new_symbol VARCHAR NOT NULL,
    isin VARCHAR,
    effective_from DATE NOT NULL,
    effective_to DATE,
    is_current BOOLEAN DEFAULT TRUE,
    change_reason VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- FEATURE TABLES (Computed Analytics)
-- ============================================================================

-- Equity Features (Silver/Gold Layer)
CREATE TABLE IF NOT EXISTS equity_features (
    feature_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    
    -- Price Features
    returns_1d DOUBLE,
    returns_5d DOUBLE,
    returns_20d DOUBLE,
    returns_60d DOUBLE,
    volatility_20d DOUBLE,
    volatility_60d DOUBLE,
    
    -- Technical Features
    rsi_14 DOUBLE,
    macd DOUBLE,
    macd_signal DOUBLE,
    bollinger_upper DOUBLE,
    bollinger_lower DOUBLE,
    
    -- Volume Features
    volume_ma_20d DOUBLE,
    volume_ratio_20d DOUBLE,
    
    -- Momentum Features
    momentum_20d DOUBLE,
    momentum_60d DOUBLE,
    
    -- Computed At
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (symbol, date)
);

-- Risk Metrics Features
CREATE TABLE IF NOT EXISTS risk_metrics (
    metric_id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    date DATE NOT NULL,
    
    -- Risk Metrics
    var_95 DOUBLE,
    var_99 DOUBLE,
    cvar_95 DOUBLE,
    max_drawdown DOUBLE,
    sharpe_ratio DOUBLE,
    sortino_ratio DOUBLE,
    beta DOUBLE,
    alpha DOUBLE,
    
    -- Computed At
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE (symbol, date)
);

-- ============================================================================
-- POINT-IN-TIME (PIT) TABLES - CRITICAL FOR LOOKAHEAD BIAS PREVENTION
-- ============================================================================

-- Fundamental Data PIT Table
-- All fundamental data must use PIT design to prevent look-ahead bias
CREATE TABLE IF NOT EXISTS fundamental_pit (
    id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,
    value DOUBLE NOT NULL,
    currency VARCHAR DEFAULT 'INR',
    unit VARCHAR DEFAULT 'absolute',
    announcement_date TIMESTAMP NOT NULL,
    effective_date TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    available_at TIMESTAMP NOT NULL,
    is_revised BOOLEAN DEFAULT FALSE,
    previous_value DOUBLE,
    revision_reason VARCHAR,
    
    -- Additional fields for specific data types
    promoter_name VARCHAR,
    holding_percentage DOUBLE,
    shares_held BIGINT,
    total_shares BIGINT,
    period_type VARCHAR,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    reporting_currency VARCHAR DEFAULT 'INR',
    audited BOOLEAN DEFAULT FALSE,
    period_end_date TIMESTAMP,
    revenue DOUBLE,
    net_profit DOUBLE,
    eps DOUBLE,
    operating_margin DOUBLE,
    net_margin DOUBLE,
    yoy_revenue_growth DOUBLE,
    yoy_profit_growth DOUBLE,
    consensus_eps DOUBLE,
    earnings_surprise DOUBLE,
    index_name VARCHAR,
    weight DOUBLE,
    entry_reason VARCHAR,
    exit_reason VARCHAR,
    dividend_type VARCHAR DEFAULT 'final',
    payout_ratio DOUBLE,
    face_value DOUBLE,
    action_type VARCHAR,
    ratio DOUBLE,
    bonus_ratio VARCHAR,
    split_ratio VARCHAR,
    dividend_amount DOUBLE,
    dividend_percentage DOUBLE,
    description TEXT,
    record_date TIMESTAMP,
    ex_date TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fundamental PIT Indexes
CREATE INDEX IF NOT EXISTS idx_fundamental_pit_symbol ON fundamental_pit(symbol);
CREATE INDEX IF NOT EXISTS idx_fundamental_pit_data_type ON fundamental_pit(data_type);
CREATE INDEX IF NOT EXISTS idx_fundamental_pit_available_at ON fundamental_pit(available_at);
CREATE INDEX IF NOT EXISTS idx_fundamental_pit_effective_date ON fundamental_pit(effective_date);
CREATE INDEX IF NOT EXISTS idx_fundamental_pit_symbol_available ON fundamental_pit(symbol, available_at);

-- Promoter Holding PIT Table
CREATE TABLE IF NOT EXISTS promoter_holding_pit (
    id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    promoter_name VARCHAR,
    holding_percentage DOUBLE NOT NULL,
    shares_held BIGINT,
    total_shares BIGINT,
    announcement_date TIMESTAMP NOT NULL,
    effective_date TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    available_at TIMESTAMP NOT NULL,
    is_revised BOOLEAN DEFAULT FALSE,
    previous_holding_percentage DOUBLE,
    revision_reason VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Promoter Holding PIT Indexes
CREATE INDEX IF NOT EXISTS idx_promoter_pit_symbol ON promoter_holding_pit(symbol);
CREATE INDEX IF NOT EXISTS idx_promoter_pit_available_at ON promoter_holding_pit(available_at);
CREATE INDEX IF NOT EXISTS idx_promoter_pit_symbol_available ON promoter_holding_pit(symbol, available_at);

-- Financials PIT Table
CREATE TABLE IF NOT EXISTS financials_pit (
    id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,
    value DOUBLE NOT NULL,
    period_type VARCHAR NOT NULL,
    fiscal_year INTEGER NOT NULL,
    fiscal_quarter INTEGER,
    period_end_date TIMESTAMP NOT NULL,
    announcement_date TIMESTAMP NOT NULL,
    effective_date TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    available_at TIMESTAMP NOT NULL,
    reporting_currency VARCHAR DEFAULT 'INR',
    audited BOOLEAN DEFAULT FALSE,
    is_revised BOOLEAN DEFAULT FALSE,
    previous_value DOUBLE,
    revision_reason VARCHAR,
    revenue DOUBLE,
    net_profit DOUBLE,
    eps DOUBLE,
    operating_margin DOUBLE,
    net_margin DOUBLE,
    yoy_revenue_growth DOUBLE,
    yoy_profit_growth DOUBLE,
    consensus_eps DOUBLE,
    earnings_surprise DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Financials PIT Indexes
CREATE INDEX IF NOT EXISTS idx_financials_pit_symbol ON financials_pit(symbol);
CREATE INDEX IF NOT EXISTS idx_financials_pit_period ON financials_pit(fiscal_year, fiscal_quarter);
CREATE INDEX IF NOT EXISTS idx_financials_pit_available_at ON financials_pit(available_at);
CREATE INDEX IF NOT EXISTS idx_financials_pit_symbol_available ON financials_pit(symbol, available_at);

-- Corporate Actions PIT Table (Institutional-grade)
-- Corporate actions are core infrastructure for price adjustment and backtesting
CREATE TABLE IF NOT EXISTS corporate_actions_pit (
    id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    action_type VARCHAR NOT NULL,
    announcement_date TIMESTAMP NOT NULL,
    effective_date TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    available_at TIMESTAMP NOT NULL,
    ratio DOUBLE,
    bonus_ratio VARCHAR,
    split_ratio VARCHAR,
    dividend_amount DOUBLE,
    dividend_percentage DOUBLE,
    description TEXT,
    record_date TIMESTAMP,
    ex_date TIMESTAMP,
    dividend_type VARCHAR DEFAULT 'final',
    payout_ratio DOUBLE,
    face_value DOUBLE,
    rights_ratio VARCHAR,
    rights_price DOUBLE,
    buyback_price DOUBLE,
    buyback_quantity BIGINT,
    old_symbol VARCHAR,
    new_symbol VARCHAR,
    currency VARCHAR DEFAULT 'INR',
    is_revised BOOLEAN DEFAULT FALSE,
    revision_reason VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Corporate Actions PIT Indexes
CREATE INDEX IF NOT EXISTS idx_ca_pit_symbol ON corporate_actions_pit(symbol);
CREATE INDEX IF NOT EXISTS idx_ca_pit_action_type ON corporate_actions_pit(action_type);
CREATE INDEX IF NOT EXISTS idx_ca_pit_available_at ON corporate_actions_pit(available_at);
CREATE INDEX IF NOT EXISTS idx_ca_pit_ex_date ON corporate_actions_pit(ex_date);
CREATE INDEX IF NOT EXISTS idx_ca_pit_symbol_available ON corporate_actions_pit(symbol, available_at);
CREATE INDEX IF NOT EXISTS idx_ca_pit_symbol_action ON corporate_actions_pit(symbol, action_type);

-- Index Membership PIT Table
CREATE TABLE IF NOT EXISTS index_membership_pit (
    id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    index_name VARCHAR NOT NULL,
    weight DOUBLE,
    announcement_date TIMESTAMP NOT NULL,
    effective_date TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    available_at TIMESTAMP NOT NULL,
    entry_reason VARCHAR,
    exit_reason VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index Membership PIT Indexes
CREATE INDEX IF NOT EXISTS idx_index_pit_symbol ON index_membership_pit(symbol);
CREATE INDEX IF NOT EXISTS idx_index_pit_index_name ON index_membership_pit(index_name);
CREATE INDEX IF NOT EXISTS idx_index_pit_available_at ON index_membership_pit(available_at);
CREATE INDEX IF NOT EXISTS idx_index_pit_symbol_available ON index_membership_pit(symbol, available_at);

-- ============================================================================
-- UNIVERSE HISTORY (Survivorship Bias Prevention)
-- ============================================================================

-- Universe Members Table
-- Tracks which stocks were in the universe at any point in time
CREATE TABLE IF NOT EXISTS universe_members (
    id VARCHAR PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    isin VARCHAR,
    company_name VARCHAR,
    start_date DATE NOT NULL,
    end_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    exit_reason VARCHAR,
    sector VARCHAR,
    market_cap DOUBLE,
    listing_date DATE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Universe Members Indexes
CREATE INDEX IF NOT EXISTS idx_universe_members_symbol ON universe_members(symbol);
CREATE INDEX IF NOT EXISTS idx_universe_members_active ON universe_members(is_active);
CREATE INDEX IF NOT EXISTS idx_universe_members_start_date ON universe_members(start_date);
CREATE INDEX IF NOT EXISTS idx_universe_members_end_date ON universe_members(end_date);
CREATE INDEX IF NOT EXISTS idx_universe_members_date_range ON universe_members(start_date, end_date);

-- Universe History Snapshots Table
-- Daily snapshots of universe state for backtesting
CREATE TABLE IF NOT EXISTS universe_history (
    id VARCHAR PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    total_symbols INTEGER NOT NULL,
    active_symbols INTEGER NOT NULL,
    new_entries INTEGER DEFAULT 0,
    exits INTEGER DEFAULT 0,
    symbols TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Universe History Indexes
CREATE INDEX IF NOT EXISTS idx_universe_history_date ON universe_history(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_universe_history_total_symbols ON universe_history(total_symbols);

-- Universe Definitions Table
-- Defines criteria for different universes (NIFTY50, FNO, etc.)
CREATE TABLE IF NOT EXISTS universe_definitions (
    id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,
    description TEXT,
    min_market_cap DOUBLE,
    min_liquidity DOUBLE,
    exclude_suspensions BOOLEAN DEFAULT TRUE,
    exclude_delisted BOOLEAN DEFAULT TRUE,
    include_indices TEXT,
    exclude_indices TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- MARKET CALENDAR (Institutional-grade with special sessions)
-- ============================================================================

-- Market Calendar Table
CREATE TABLE IF NOT EXISTS market_calendar (
    date DATE PRIMARY KEY,
    is_trading_day BOOLEAN NOT NULL,
    is_weekend BOOLEAN DEFAULT FALSE,
    is_holiday BOOLEAN DEFAULT FALSE,
    holiday_name VARCHAR,
    holiday_type VARCHAR,
    session_type VARCHAR DEFAULT 'NORMAL',
    session_start TIME,
    session_end TIME,
    is_expiry BOOLEAN DEFAULT FALSE,
    shifted_expiry_date DATE,
    is_revised BOOLEAN DEFAULT FALSE,
    original_holiday_date DATE,
    revision_reason VARCHAR,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market Calendar Indexes
CREATE INDEX IF NOT EXISTS idx_market_calendar_trading_day ON market_calendar(is_trading_day);
CREATE INDEX IF NOT EXISTS idx_market_calendar_holiday ON market_calendar(is_holiday);
CREATE INDEX IF NOT EXISTS idx_market_calendar_expiry ON market_calendar(is_expiry);
CREATE INDEX IF NOT EXISTS idx_market_calendar_session_type ON market_calendar(session_type);
CREATE INDEX IF NOT EXISTS idx_market_calendar_shifted_expiry ON market_calendar(shifted_expiry_date);

-- ============================================================================
-- DATA LINEAGE (Audit Trail)
-- ============================================================================

-- Data Lineage Table
CREATE TABLE IF NOT EXISTS data_lineage (
    id VARCHAR PRIMARY KEY,
    dataset VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    downloaded_at TIMESTAMP NOT NULL,
    pipeline_version VARCHAR,
    git_commit VARCHAR,
    checksum VARCHAR,
    row_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Reference Tables Indexes
CREATE INDEX IF NOT EXISTS idx_security_master_symbol ON security_master(symbol);
CREATE INDEX IF NOT EXISTS idx_security_master_isin ON security_master(isin);
CREATE INDEX IF NOT EXISTS idx_security_master_effective ON security_master(effective_from, effective_to);
CREATE INDEX IF NOT EXISTS idx_security_master_current ON security_master(is_current) WHERE is_current = TRUE;

CREATE INDEX IF NOT EXISTS idx_trading_calendar_date ON trading_calendar(date);
CREATE INDEX IF NOT EXISTS idx_trading_calendar_trading_day ON trading_calendar(is_trading_day);

-- Raw Tables Indexes
CREATE INDEX IF NOT EXISTS idx_raw_equity_history_symbol_date ON raw_equity_history(symbol, date);
CREATE INDEX IF NOT EXISTS idx_raw_equity_history_date ON raw_equity_history(date);
CREATE INDEX IF NOT EXISTS idx_raw_equity_history_ingest ON raw_equity_history(ingest_id);

CREATE INDEX IF NOT EXISTS idx_raw_options_chain_symbol_date ON raw_options_chain(symbol, date);
CREATE INDEX IF NOT EXISTS idx_raw_options_chain_expiry ON raw_options_chain(expiry);
CREATE INDEX IF NOT EXISTS idx_raw_options_chain_ingest ON raw_options_chain(ingest_id);

CREATE INDEX IF NOT EXISTS idx_raw_fii_dii_flows_date ON raw_fii_dii_flows(date);
CREATE INDEX IF NOT EXISTS idx_raw_fii_dii_flows_ingest ON raw_fii_dii_flows(ingest_id);

-- Event Tables Indexes
CREATE INDEX IF NOT EXISTS idx_corporate_action_events_symbol ON corporate_action_events(symbol);
CREATE INDEX IF NOT EXISTS idx_corporate_action_events_ex_date ON corporate_action_events(ex_date);
CREATE INDEX IF NOT EXISTS idx_corporate_action_events_effective ON corporate_action_events(effective_from, effective_to);

CREATE INDEX IF NOT EXISTS idx_earnings_events_symbol ON earnings_events(symbol);
CREATE INDEX IF NOT EXISTS idx_earnings_events_announcement ON earnings_events(announcement_date);

CREATE INDEX IF NOT EXISTS idx_index_membership_history_index ON index_membership_history(index_name);
CREATE INDEX IF NOT EXISTS idx_index_membership_history_symbol ON index_membership_history(symbol);
CREATE INDEX IF NOT EXISTS idx_index_membership_history_effective ON index_membership_history(effective_from, effective_to);

CREATE INDEX IF NOT EXISTS idx_symbol_mapping_history_symbol ON symbol_mapping_history(new_symbol);
CREATE INDEX IF NOT EXISTS idx_symbol_mapping_history_effective ON symbol_mapping_history(effective_from, effective_to);

-- Feature Tables Indexes
CREATE INDEX IF NOT EXISTS idx_equity_features_symbol_date ON equity_features(symbol, date);
CREATE INDEX IF NOT EXISTS idx_equity_features_date ON equity_features(date);

CREATE INDEX IF NOT EXISTS idx_risk_metrics_symbol_date ON risk_metrics(symbol, date);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_date ON risk_metrics(date);

-- Data Lineage Indexes
CREATE INDEX IF NOT EXISTS idx_data_lineage_target ON data_lineage(target_table);
CREATE INDEX IF NOT EXISTS idx_data_lineage_source ON data_lineage(source_system);
CREATE INDEX IF NOT EXISTS idx_data_lineage_created ON data_lineage(created_at);
CREATE INDEX IF NOT EXISTS idx_data_lineage_status ON data_lineage(status);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Current Security Master View
CREATE OR REPLACE VIEW v_current_security_master AS
SELECT * FROM security_master WHERE is_current = TRUE;

-- Active Corporate Actions View
CREATE OR REPLACE VIEW v_active_corporate_actions AS
SELECT * FROM corporate_action_events WHERE is_current = TRUE AND ex_date >= CURRENT_DATE;

-- Recent Earnings View
CREATE OR REPLACE VIEW v_recent_earnings AS
SELECT * FROM earnings_events WHERE announcement_date >= CURRENT_DATE - INTERVAL '90 DAYS';

-- Current Index Members View
CREATE OR REPLACE VIEW v_current_index_members AS
SELECT * FROM index_membership_history WHERE is_current = TRUE;

-- Data Quality Summary View
CREATE OR REPLACE VIEW v_data_quality_summary AS
SELECT 
    target_table,
    COUNT(*) as total_runs,
    AVG(data_quality_score) as avg_quality_score,
    MAX(created_at) as last_run,
    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as successful_runs,
    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failed_runs
FROM data_lineage
GROUP BY target_table;


-- Source: data_infra/storage/database/clickhouse_schema.sql
-- ============================================================================
-- CLICKHOUSE SCHEMA - Production Database
-- ============================================================================
-- High-performance, real-time analytics database
-- Architecture: NSE → Bronze Parquet → ClickHouse → Feature Store → Models
-- ============================================================================

-- Create database
CREATE DATABASE IF NOT EXISTS market;

USE market;

-- ============================================================================
-- SECURITY MASTER
-- ============================================================================

CREATE TABLE IF NOT EXISTS security_master (
    symbol String,
    isin String,
    company_name String,
    sector String,
    industry String,
    listing_date Date,
    delisting_date Date,
    face_value Float32,
    market_cap_bucket String,
    nse_symbol String,
    bse_symbol String,
    lot_size UInt32,
    index_memberships String,
    effective_from Date,
    effective_to Date,
    is_current UInt8,
    created_at DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(effective_from)
ORDER BY (symbol, effective_from)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- EQUITY HISTORY
-- ============================================================================

CREATE TABLE IF NOT EXISTS equity_history (
    symbol String,
    date Date,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume UInt64,
    vwap Float64,
    trades UInt32,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- OPTIONS CHAIN
-- ============================================================================

CREATE TABLE IF NOT EXISTS options_chain (
    timestamp DateTime,
    symbol String,
    strike Float64,
    expiry Date,
    option_type String,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume UInt64,
    oi UInt64,
    iv Float32,
    delta Float32,
    gamma Float32,
    theta Float32,
    vega Float32,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMonth(timestamp)
ORDER BY (symbol, expiry, strike, option_type, timestamp)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- CORPORATE ACTIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol String,
    event_type String,
    announcement_date Date,
    record_date Date,
    ex_date Date,
    ratio Float32,
    dividend_amount Float64,
    rights_ratio Float32,
    rights_price Float64,
    buyback_price Float64,
    buyback_quantity UInt32,
    old_symbol String,
    new_symbol String,
    currency String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(announcement_date)
ORDER BY (symbol, announcement_date, event_type)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- MARKET CALENDAR
-- ============================================================================

CREATE TABLE IF NOT EXISTS market_calendar (
    date Date,
    is_trading_day UInt8,
    is_weekend UInt8,
    is_holiday UInt8,
    holiday_name String,
    holiday_type String,
    session_type String,
    session_start Time,
    session_end Time,
    is_expiry UInt8,
    shifted_expiry_date Date,
    is_revised UInt8,
    original_holiday_date Date,
    revision_reason String,
    description String,
    created_at DateTime DEFAULT now(),
    updated_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY date
SETTINGS index_granularity = 8192;

-- ============================================================================
-- UNIVERSE MEMBERS
-- ============================================================================

CREATE TABLE IF NOT EXISTS universe_members (
    id String,
    symbol String,
    isin String,
    company_name String,
    start_date Date,
    end_date Date,
    is_active UInt8,
    exit_reason String,
    sector String,
    market_cap Float64,
    listing_date Date,
    added_at DateTime,
    updated_at DateTime
) ENGINE = MergeTree()
ORDER BY (symbol, start_date)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- UNIVERSE HISTORY
-- ============================================================================

CREATE TABLE IF NOT EXISTS universe_history (
    id String,
    snapshot_date Date,
    total_symbols UInt32,
    active_symbols UInt32,
    new_entries UInt32,
    exits UInt32,
    symbols String,
    created_at DateTime
) ENGINE = MergeTree()
ORDER BY snapshot_date
SETTINGS index_granularity = 8192;

-- ============================================================================
-- DATA LINEAGE
-- ============================================================================

CREATE TABLE IF NOT EXISTS data_lineage (
    id String,
    dataset String,
    source String,
    downloaded_at DateTime,
    pipeline_version String,
    git_commit String,
    checksum String,
    row_count UInt32,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (dataset, downloaded_at)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- POINT-IN-TIME TABLES
-- ============================================================================

-- Fundamental PIT
CREATE TABLE IF NOT EXISTS fundamental_pit (
    id String,
    symbol String,
    metric_name String,
    metric_value Float64,
    period String,
    period_end Date,
    available_at DateTime,
    valid_from DateTime,
    valid_to DateTime,
    entry_reason String,
    exit_reason String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(available_at)
ORDER BY (symbol, metric_name, available_at)
SETTINGS index_granularity = 8192;

-- Promoter Holding PIT
CREATE TABLE IF NOT EXISTS promoter_holding_pit (
    id String,
    symbol String,
    promoter_name String,
    holding_percentage Float32,
    shares_held UInt64,
    period_end Date,
    available_at DateTime,
    valid_from DateTime,
    valid_to DateTime,
    entry_reason String,
    exit_reason String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(available_at)
ORDER BY (symbol, available_at)
SETTINGS index_granularity = 8192;

-- Corporate Actions PIT
CREATE TABLE IF NOT EXISTS corporate_actions_pit (
    id String,
    symbol String,
    event_type String,
    announcement_date Date,
    record_date Date,
    ex_date Date,
    split_ratio Float32,
    bonus_ratio Float32,
    dividend_amount Float64,
    rights_ratio Float32,
    rights_price Float64,
    buyback_price Float64,
    buyback_quantity UInt32,
    old_symbol String,
    new_symbol String,
    currency String,
    is_revised UInt8,
    revision_reason String,
    available_at DateTime,
    valid_from DateTime,
    valid_to DateTime,
    entry_reason String,
    exit_reason String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(available_at)
ORDER BY (symbol, event_type, available_at)
SETTINGS index_granularity = 8192;

-- Index Membership PIT
CREATE TABLE IF NOT EXISTS index_membership_pit (
    id String,
    symbol String,
    index_name String,
    weight Float32,
    entry_date Date,
    exit_date Date,
    available_at DateTime,
    valid_from DateTime,
    valid_to DateTime,
    entry_reason String,
    exit_reason String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(available_at)
ORDER BY (symbol, index_name, available_at)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- FEATURE STORE TABLES
-- ============================================================================

-- Price Features
CREATE TABLE IF NOT EXISTS price_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Volatility Features
CREATE TABLE IF NOT EXISTS volatility_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Volume Features
CREATE TABLE IF NOT EXISTS volume_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Options Features
CREATE TABLE IF NOT EXISTS options_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Flow Features
CREATE TABLE IF NOT EXISTS flow_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Fundamental Features
CREATE TABLE IF NOT EXISTS fundamental_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;

-- Alternative Features
CREATE TABLE IF NOT EXISTS alternative_features (
    symbol String,
    date Date,
    feature_name String,
    feature_value Float64,
    lookback UInt16,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYY(date)
ORDER BY (symbol, date, feature_name)
SETTINGS index_granularity = 8192;


