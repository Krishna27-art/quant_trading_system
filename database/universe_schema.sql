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
