-- Fundamental Data Schemas
-- To be loaded into ClickHouse or PostgreSQL for Institutional Alpha.

CREATE TABLE IF NOT EXISTS fundamentals (
    symbol VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    filing_date DATE NOT NULL, -- Crucial for Point-In-Time (PIT) correctness
    
    -- Earnings & Profitability
    eps_ttm DECIMAL(15,4),
    revenue_ttm DECIMAL(20,4),
    net_income_ttm DECIMAL(20,4),
    gross_margin DECIMAL(10,4),
    operating_margin DECIMAL(10,4),
    roe DECIMAL(10,4), -- Return on Equity
    roa DECIMAL(10,4), -- Return on Assets
    
    -- Valuation (Calculated dynamically but cached here)
    pe_ratio DECIMAL(10,4),
    pb_ratio DECIMAL(10,4),
    ps_ratio DECIMAL(10,4),
    ev_ebitda DECIMAL(10,4),
    
    -- Balance Sheet
    total_assets DECIMAL(20,4),
    total_liabilities DECIMAL(20,4),
    total_debt DECIMAL(20,4),
    cash_and_equivalents DECIMAL(20,4),
    
    -- Cash Flow
    operating_cash_flow DECIMAL(20,4),
    free_cash_flow DECIMAL(20,4),
    
    PRIMARY KEY (symbol, filing_date)
);

-- Alternative Data: Analyst Sentiment & Estimates
CREATE TABLE IF NOT EXISTS analyst_estimates (
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    eps_estimate_current_year DECIMAL(15,4),
    eps_estimate_next_year DECIMAL(15,4),
    revenue_estimate_current_year DECIMAL(20,4),
    analyst_buy_ratings INT,
    analyst_hold_ratings INT,
    analyst_sell_ratings INT,
    price_target_mean DECIMAL(15,4),
    
    PRIMARY KEY (symbol, date)
);

-- Corporate Actions (Crucial for Return adjustments and Short Interest)
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol VARCHAR(20) NOT NULL,
    ex_date DATE NOT NULL,
    action_type VARCHAR(20) NOT NULL, -- 'DIVIDEND', 'SPLIT', 'SPINOFF', 'MERGER'
    dividend_amount DECIMAL(10,4),
    split_ratio DECIMAL(10,4),
    
    PRIMARY KEY (symbol, ex_date, action_type)
);

-- Short Interest & Borrowing Cost
CREATE TABLE IF NOT EXISTS short_interest (
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    shares_short BIGINT,
    short_ratio DECIMAL(10,4), -- Days to cover
    borrow_fee_rate DECIMAL(10,4), -- Annualized cost to borrow (%)
    
    PRIMARY KEY (symbol, date)
);
