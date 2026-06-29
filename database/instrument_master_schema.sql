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
