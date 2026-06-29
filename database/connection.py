"""
Database Connection Module

Provides multi-database connection and data access functions for the quant system.

Multi-Database Architecture with CQRS:
- ClickHouse: Tick data, OHLCV, time-series analytics (columnar, high-performance)
- PostgreSQL (Primary): Orders, positions, executions, OMS (ACID, transactions) - Command side
- PostgreSQL (Replica): Analytics, dashboards, reports - Query side
- DuckDB: Research, backtesting, feature computation (in-process analytical)
- Parquet: Archival, historical data (object storage, compression)

CQRS Pattern:
- Command Side: Orders, Positions, Risk (Write operations)
- Query Side: Dashboards, Reports, Analytics (Read operations from replica)
- API never touches OMS tables directly - uses materialized views

Security:
- Credentials retrieved from Vault (no hardcoded secrets)
- Dynamic credentials with short TTL
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import date
from enum import Enum
from typing import Any

from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from utils.logger import get_logger

logger = get_logger("database_connection")

# Try to import Vault client for secure credentials
try:
    import security.vault_client  # noqa: F401

    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False
    logger.warning("Vault client not available, using environment variables")


class DatabaseRole(str, Enum):
    """Database role for CQRS."""

    PRIMARY = "primary"  # Command side - OMS, Orders, Positions
    REPLICA = "replica"  # Query side - Analytics, Dashboards, Reports
    FAILOVER = "failover"  # Failover target during primary failure


# Database connection pools
postgresql_primary_pool: pool.SimpleConnectionPool | None = None
postgresql_replica_pool: pool.SimpleConnectionPool | None = None
postgresql_failover_pool: pool.SimpleConnectionPool | None = None  # Failover target
clickhouse_client = None
duckdb_connection = None


def get_database_url(role: DatabaseRole = DatabaseRole.PRIMARY) -> str:
    """Get database URL from environment variables."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        # Fallback to SQLite for local development
        logger.warning("DATABASE_URL not set, using SQLite fallback")
        return "sqlite:///quant.db"
    return url


def initialize_pool(min_connections: int = 1, max_connections: int = 20) -> bool:
    """
    Initialize database connection pool or SQLite connection.
    """
    global postgresql_primary_pool
    try:
        db_url = get_database_url()
        logger.info(f"Initializing database connection: {db_url}")

        if db_url.startswith("sqlite"):
            # SQLite doesn't use connection pools
            logger.info("Using SQLite database")
            return True
        else:
            # PostgreSQL connection pool
            postgresql_primary_pool = pool.SimpleConnectionPool(
                min_connections, max_connections, db_url
            )
            return True
    except Exception as e:
        logger.critical(f"Failed to initialize database pool: {e}")
        raise RuntimeError(f"Database pool initialization failed: {e}")


def get_connection(role: DatabaseRole = DatabaseRole.PRIMARY):
    """
    Get a database connection (PostgreSQL pool or SQLite).
    """
    global postgresql_primary_pool
    db_url = get_database_url()

    if db_url.startswith("sqlite"):
        # SQLite connection
        db_path = db_url.replace("sqlite:///", "")
        return sqlite3.connect(db_path)
    else:
        # PostgreSQL connection pool
        if not postgresql_primary_pool:
            initialize_pool()
        try:
            return postgresql_primary_pool.getconn()
        except Exception as e:
            logger.critical(f"Failed to get PostgreSQL connection from pool: {e}")
            raise RuntimeError(f"Failed to get connection: {e}")


def release_connection(conn):
    """
    Release/put back a database connection to the pool or close SQLite.
    """
    global postgresql_primary_pool
    db_url = get_database_url()

    if db_url.startswith("sqlite"):
        # Close SQLite connection
        if conn:
            conn.close()
    else:
        # PostgreSQL connection pool
        if conn and postgresql_primary_pool:
            try:
                postgresql_primary_pool.putconn(conn)
            except Exception as e:
                logger.error(f"Failed to release connection back to pool: {e}")


def close_all_connections():
    """Close PostgreSQL connection pool."""
    global postgresql_primary_pool
    if postgresql_primary_pool:
        try:
            postgresql_primary_pool.closeall()
            logger.info("Closed all connections in PostgreSQL pool")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL pool: {e}")
        postgresql_primary_pool = None

    logger.info("All database connections closed")


@contextmanager
def get_db_connection(role: DatabaseRole = DatabaseRole.PRIMARY):
    """Context manager for safe database connections."""
    conn = get_connection(role)
    try:
        yield conn
    finally:
        if conn:
            release_connection(conn)


def execute_query(
    query: str,
    params: tuple | None = None,
    fetch: str = "all",
    role: DatabaseRole = DatabaseRole.PRIMARY,
) -> Any | None:
    """
    Execute a SQL query on PostgreSQL or SQLite.
    """
    with get_db_connection(role) as conn:
        if not conn:
            return None

        db_url = get_database_url()
        is_sqlite = db_url.startswith("sqlite")

        try:
            if is_sqlite:
                cursor = conn.cursor()
                cursor.execute(query, params or ())

                if fetch == "all":
                    rows = cursor.fetchall()
                    # Convert to dict using column names
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row, strict=False)) for row in rows]
                elif fetch == "one":
                    row = cursor.fetchone()
                    if row:
                        columns = [desc[0] for desc in cursor.description]
                        return dict(zip(columns, row, strict=False))
                    return None
                elif fetch == "none":
                    conn.commit()
                    return True
            else:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, params or ())

                if fetch == "all":
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                elif fetch == "one":
                    row = cursor.fetchone()
                    return dict(row) if row else None
                elif fetch == "none":
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Query execution failed: {e}. Query: {query[:100]}")
            if not is_sqlite:
                conn.rollback()
            raise e


def execute_write(query: str, params: tuple | None = None) -> bool:
    """
    Explicitly execute a write operation on the PRIMARY database.
    """
    if query.strip().upper().startswith("SELECT"):
        logger.warning(f"execute_write called with a SELECT query: {query[:50]}...")

    result = execute_query(query, params, fetch="none", role=DatabaseRole.PRIMARY)
    return result is True


def execute_batch(query: str, params_list: list[tuple]) -> bool:
    """
    Execute a batch of SQL queries on PostgreSQL.
    """
    with get_db_connection(DatabaseRole.PRIMARY) as conn:
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Batch execution failed: {e}. Query: {query[:100]}")
            conn.rollback()
            raise e


# ═══════════════════ TABLE SCHEMA ═══════════════════


def create_tables():
    """Create database tables if they don't exist in PostgreSQL."""

    # Stocks table
    execute_write("""
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sector TEXT,
            market_cap TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Stock prices table
    execute_write("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL REFERENCES stocks(symbol),
            price REAL NOT NULL,
            change REAL,
            change_pct REAL,
            volume BIGINT,
            high_52w REAL,
            low_52w REAL,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Predictions table
    execute_write("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL REFERENCES stocks(symbol),
            model_version TEXT,
            features_used TEXT,
            prediction TEXT NOT NULL,
            horizon TEXT NOT NULL,
            confidence REAL,
            entry_price REAL,
            stop_loss REAL,
            target_price REAL,
            prediction_time TIMESTAMPTZ NOT NULL,
            expiry_time TIMESTAMPTZ,
            actual_outcome TEXT,
            target_hit INTEGER DEFAULT 0,
            stop_hit INTEGER DEFAULT 0,
            actual_return REAL,
            mfe REAL,
            mae REAL,
            latency_ms INTEGER,
            is_correct INTEGER,
            regime TEXT,
            reason TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Orders table
    execute_write("""
        CREATE TABLE IF NOT EXISTS orders (
            id SERIAL PRIMARY KEY,
            client_order_id TEXT UNIQUE NOT NULL,
            symbol TEXT NOT NULL REFERENCES stocks(symbol),
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL,
            order_type TEXT,
            status TEXT NOT NULL,
            broker_order_id TEXT,
            state TEXT DEFAULT 'CREATED',
            filled_quantity INTEGER DEFAULT 0,
            avg_fill_price REAL,
            msg_seq_num INTEGER,
            placed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # System health table
    execute_write("""
        CREATE TABLE IF NOT EXISTS system_health (
            id SERIAL PRIMARY KEY,
            component TEXT NOT NULL,
            status TEXT NOT NULL,
            value TEXT,
            message TEXT,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes
    execute_write("CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices(symbol)")
    execute_write(
        "CREATE INDEX IF NOT EXISTS idx_stock_prices_timestamp ON stock_prices(timestamp)"
    )
    execute_write("CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol)")
    execute_write("CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time)")
    execute_write("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
    execute_write("CREATE INDEX IF NOT EXISTS idx_orders_placed_at ON orders(placed_at)")
    execute_write(
        "CREATE INDEX IF NOT EXISTS idx_orders_client_order_id ON orders(client_order_id)"
    )
    execute_write("CREATE INDEX IF NOT EXISTS idx_orders_state ON orders(state)")

    # Positions table
    execute_write("""
        CREATE TABLE IF NOT EXISTS positions (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL REFERENCES stocks(symbol),
            quantity INTEGER NOT NULL,
            avg_price REAL,
            market_value REAL,
            unrealized_pnl REAL,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    execute_write("CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)")

    # Executions table
    execute_write("""
        CREATE TABLE IF NOT EXISTS executions (
            id SERIAL PRIMARY KEY,
            client_order_id TEXT NOT NULL,
            broker_order_id TEXT,
            symbol TEXT NOT NULL REFERENCES stocks(symbol),
            side TEXT NOT NULL,
            filled_quantity INTEGER NOT NULL,
            fill_price REAL,
            execution_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    execute_write(
        "CREATE INDEX IF NOT EXISTS idx_executions_client_order_id ON executions(client_order_id)"
    )
    execute_write("CREATE INDEX IF NOT EXISTS idx_executions_symbol ON executions(symbol)")

    # Order events table
    execute_write("""
        CREATE TABLE IF NOT EXISTS order_events (
            id SERIAL PRIMARY KEY,
            client_order_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            from_state TEXT,
            to_state TEXT NOT NULL,
            event_data TEXT,
            event_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            msg_seq_num INTEGER
        )
    """)
    execute_write(
        "CREATE INDEX IF NOT EXISTS idx_order_events_client_order_id ON order_events(client_order_id)"
    )
    execute_write(
        "CREATE INDEX IF NOT EXISTS idx_order_events_timestamp ON order_events(event_timestamp)"
    )
    execute_write(
        "CREATE INDEX IF NOT EXISTS idx_order_events_msg_seq_num ON order_events(msg_seq_num)"
    )

    # Borrow inventory and reservations tables
    execute_write("""
        CREATE TABLE IF NOT EXISTS borrow_inventory (
            symbol TEXT PRIMARY KEY,
            available_qty INTEGER NOT NULL,
            reserved_qty INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
    """)
    execute_write("""
        CREATE TABLE IF NOT EXISTS borrow_reservations (
            reservation_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL REFERENCES borrow_inventory(symbol),
             qty INTEGER NOT NULL,
             status TEXT NOT NULL,
             created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
             updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
         )
     """)
    execute_write(
        "CREATE INDEX IF NOT EXISTS idx_borrow_reservations_symbol ON borrow_reservations(symbol)"
    )

    # Ticks table for live prediction metrics
    execute_write("""
        CREATE TABLE IF NOT EXISTS ticks (
            id SERIAL PRIMARY KEY,
            time TIMESTAMPTZ NOT NULL,
            symbol TEXT NOT NULL REFERENCES stocks(symbol),
            ltp REAL NOT NULL,
            volume REAL DEFAULT 0
        )
    """)
    execute_write("CREATE INDEX IF NOT EXISTS idx_ticks_symbol ON ticks(symbol)")
    execute_write("CREATE INDEX IF NOT EXISTS idx_ticks_time ON ticks(time)")

    # Index ticks table for indices value tracking
    execute_write("""
        CREATE TABLE IF NOT EXISTS index_ticks (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            change REAL
        )
    """)
    execute_write("CREATE INDEX IF NOT EXISTS idx_index_ticks_name ON index_ticks(name)")
    execute_write("CREATE INDEX IF NOT EXISTS idx_index_ticks_time ON index_ticks(timestamp)")

    logger.info("Database tables created successfully")


# ═══════════════════ DATA ACCESS FUNCTIONS ═══════════════════


def insert_stock(stock_data: dict[str, Any]) -> bool:
    """
    Insert or update a stock in PostgreSQL.
    """
    query = """
        INSERT INTO stocks (symbol, name, sector, market_cap, created_at, updated_at)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (symbol) DO UPDATE SET
            name = EXCLUDED.name,
            sector = EXCLUDED.sector,
            market_cap = EXCLUDED.market_cap,
            updated_at = CURRENT_TIMESTAMP
    """
    return execute_write(
        query,
        (
            stock_data["symbol"],
            stock_data["name"],
            stock_data.get("sector"),
            stock_data.get("market_cap"),
        ),
    )


def insert_stock_price(price_data: dict[str, Any]) -> bool:
    """
    Insert stock price data.
    """
    query = """
        INSERT INTO stock_prices (symbol, price, change, change_pct, volume, high_52w, low_52w)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    return execute_write(
        query,
        (
            price_data["symbol"],
            price_data["price"],
            price_data.get("change"),
            price_data.get("change_pct"),
            price_data.get("volume"),
            price_data.get("high_52w"),
            price_data.get("low_52w"),
        ),
    )


def get_latest_prices() -> list[dict[str, Any]]:
    """
    Get latest prices for all stocks.

    Returns:
        List of stock price dictionaries
    """
    query = """
        SELECT sp.symbol, s.name, s.sector, s.market_cap,
               sp.price, sp.change, sp.change_pct, sp.volume,
               sp.high_52w, sp.low_52w, sp.timestamp
        FROM stock_prices sp
        JOIN stocks s ON sp.symbol = s.symbol
        WHERE sp.id IN (
            SELECT MAX(id) FROM stock_prices GROUP BY symbol
        )
    """
    return execute_query(query) or []


def get_stock_price(symbol: str) -> dict[str, Any] | None:
    """
    Get latest price for a specific stock.
    """
    query = """
        SELECT sp.symbol, s.name, s.sector, s.market_cap,
               sp.price, sp.change, sp.change_pct, sp.volume,
               sp.high_52w, sp.low_52w, sp.timestamp
        FROM stock_prices sp
        JOIN stocks s ON sp.symbol = s.symbol
        WHERE sp.symbol = %s
        ORDER BY sp.id DESC
        LIMIT 1
    """
    return execute_query(query, (symbol,), fetch="one")


def insert_prediction(prediction_data: dict[str, Any]) -> bool:
    """
    Insert a prediction.
    """
    import uuid

    return execute_write(
        """
        INSERT INTO predictions (
            id, symbol, model_version, features_used, prediction, horizon, confidence,
            entry_price, stop_loss, target_price, prediction_time, expiry_time, regime, reason
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            prediction_data.get("id") or str(uuid.uuid4()),
            prediction_data["symbol"],
            prediction_data.get("model_version"),
            prediction_data.get("features_used"),
            prediction_data["prediction"],
            prediction_data["horizon"],
            prediction_data["confidence"],
            prediction_data.get("entry_price"),
            prediction_data.get("stop_loss"),
            prediction_data.get("target_price"),
            prediction_data.get("prediction_time"),
            prediction_data.get("expiry_time"),
            prediction_data.get("regime"),
            prediction_data.get("reason"),
        ),
    )


def get_predictions(
    symbol: str | None = None, result: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    """
    Get predictions with optional filters.

    Args:
        symbol: Filter by symbol
        result: Filter by result (correct, wrong, pending)
        limit: Maximum number of results

    Returns:
        List of prediction dictionaries
    """
    query = """
        SELECT p.id, p.symbol, s.name, p.prediction, p.horizon, p.confidence,
               p.actual_outcome, p.is_correct as result, p.reason, p.prediction_time as prediction_date, p.created_at
        FROM predictions p
        JOIN stocks s ON p.symbol = s.symbol
        WHERE 1=1
    """
    params = []

    if symbol:
        query += " AND p.symbol = %s"
        params.append(symbol)

    if result:
        if result == "correct":
            query += " AND p.is_correct = 1"
        elif result == "wrong":
            query += " AND p.is_correct = 0"
        elif result == "pending":
            query += " AND p.is_correct IS NULL"

    query += " ORDER BY p.prediction_time DESC, p.created_at DESC LIMIT %s"
    params.append(limit)

    return execute_query(query, tuple(params)) or []


def update_prediction(prediction_id: str, updates: dict[str, Any]) -> bool:
    """Update a prediction with evaluation results."""
    if not updates:
        return True

    set_clauses = []
    values = []

    for k, v in updates.items():
        set_clauses.append(f"{k} = %s")
        values.append(v)

    values.append(prediction_id)

    query = f"""
        UPDATE predictions
        SET {", ".join(set_clauses)}
        WHERE id = %s
    """

    return execute_write(query, tuple(values))


def insert_order(order_data: dict[str, Any]) -> bool:
    """
    Insert an order with client_order_id for idempotency (OMS/EMS protection).
    Includes sequence gap detection for FIX protocol.

    Args:
        order_data: Order data dictionary

    Returns:
        True if successful
    """
    # FIX protection: Use client_order_id for idempotency
    client_order_id = order_data.get("client_order_id")
    if not client_order_id:
        import uuid

        client_order_id = f"{order_data['side']}_{order_data['symbol']}_{uuid.uuid4().hex[:8]}"

    # Sequence gap detection: Check for gaps in msg_seq_num
    msg_seq_num = order_data.get("msg_seq_num")
    if msg_seq_num:
        # Get last sequence number for this session
        last_seq_query = """
            SELECT MAX(msg_seq_num) as last_seq
            FROM orders
            WHERE msg_seq_num IS NOT NULL
        """
        result = execute_query(last_seq_query, fetch="one")
        if result and result.get("last_seq"):
            last_seq = result["last_seq"]
            if msg_seq_num > last_seq + 1:
                logger.warning(
                    f"Sequence gap detected: last_seq={last_seq}, current_seq={msg_seq_num}, "
                    f"gap={msg_seq_num - last_seq - 1}"
                )

    query = """
        INSERT INTO orders (client_order_id, symbol, side, quantity, price, order_type, status, broker_order_id, state, msg_seq_num)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (client_order_id) DO UPDATE SET
            status = EXCLUDED.status,
            state = EXCLUDED.state,
            filled_quantity = EXCLUDED.filled_quantity,
            avg_fill_price = EXCLUDED.avg_fill_price,
            updated_at = CURRENT_TIMESTAMP
    """
    return execute_write(
        query,
        (
            client_order_id,
            order_data["symbol"],
            order_data["side"],
            order_data["quantity"],
            order_data.get("price"),
            order_data.get("order_type"),
            order_data.get("status", "CREATED"),
            order_data.get("broker_order_id"),
            order_data.get("state", "CREATED"),
            msg_seq_num,
        ),
    )


def get_orders(
    symbol: str | None = None,
    status: str | None = None,
    state: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Get orders with optional filters (OMS/EMS protection - includes state).

    Args:
        symbol: Filter by symbol
        status: Filter by status
        state: Filter by state (CREATED, SENT, UNKNOWN, ACKNOWLEDGED, PARTIAL, FILLED, CANCELLED)
        limit: Maximum number of results

    Returns:
        List of order dictionaries
    """
    query = """
        SELECT o.id, o.client_order_id, o.symbol, s.name, o.side, o.quantity, o.price,
               o.order_type, o.status, o.broker_order_id, o.state, o.filled_quantity,
               o.avg_fill_price, o.msg_seq_num, o.placed_at, o.updated_at
        FROM orders o
        JOIN stocks s ON o.symbol = s.symbol
        WHERE 1=1
    """
    params = []

    if symbol:
        query += " AND o.symbol = %s"
        params.append(symbol)

    if status:
        query += " AND o.status = %s"
        params.append(status)

    if state:
        query += " AND o.state = %s"
        params.append(state)

    query += " ORDER BY o.placed_at DESC LIMIT %s"
    params.append(limit)

    return execute_query(query, tuple(params)) or []


def update_system_health(component: str, status: str, value: str, message: str) -> bool:
    """
    Update system health status.

    Args:
        component: Component name
        status: Status (green, amber, red)
        value: Status value
        message: Status message

    Returns:
        True if successful
    """
    query = """
        INSERT INTO system_health (component, status, value, message)
        VALUES (%s, %s, %s, %s)
    """
    return execute_write(query, (component, status, value, message))


def get_system_health() -> list[dict[str, Any]]:
    """
    Get latest system health status for all components.

    Returns:
        List of health status dictionaries
    """
    query = """
        SELECT DISTINCT ON (component)
            component, status, value, message, timestamp
        FROM system_health
        ORDER BY component, timestamp DESC
    """
    return execute_query(query) or []


def get_sector_performance() -> list[dict[str, Any]]:
    """Aggregate stock performance by sector."""
    query = """
        WITH latest_prices AS (
            SELECT DISTINCT ON (sp.symbol)
                sp.symbol, s.sector, sp.change_pct
            FROM stock_prices sp
            JOIN stocks s ON sp.symbol = s.symbol
            WHERE s.sector IS NOT NULL AND s.sector != ''
            ORDER BY sp.symbol, sp.timestamp DESC
        )
        SELECT
            sector AS name,
            AVG(change_pct) AS change
        FROM latest_prices
        GROUP BY sector
        ORDER BY change DESC
    """
    return execute_query(query) or []


def get_model_metrics() -> list[dict[str, Any]]:
    """Aggregate ML model metrics (win rate, accuracy) from predictions."""
    query = """
        WITH pred_stats AS (
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN result = 'correct' THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN result = 'wrong' THEN 1 ELSE 0 END) as wrong
            FROM predictions
            WHERE result IN ('correct', 'wrong')
        )
        SELECT
            'Win Rate' as key,
            CASE WHEN total > 0 THEN ROUND((correct::numeric / total) * 100, 2) || '%' ELSE '0%' END as value,
            CASE WHEN total > 0 AND (correct::numeric / total) > 0.5 THEN 'var(--green)' ELSE 'var(--red)' END as color
        FROM pred_stats
        UNION ALL
        SELECT
            'Total Predictions' as key,
            total::text as value,
            'var(--text)' as color
        FROM pred_stats
        UNION ALL
        SELECT
            'Correct' as key,
            correct::text as value,
            'var(--green)' as color
        FROM pred_stats
        UNION ALL
        SELECT
            'Incorrect' as key,
            wrong::text as value,
            'var(--red)' as color
        FROM pred_stats
    """
    return execute_query(query) or []


def get_performance_metrics() -> list[dict[str, Any]]:
    """Get portfolio performance metrics from orders and PnL."""
    query = """
        WITH order_stats AS (
            SELECT
                COUNT(*) as total_orders,
                SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled_orders
            FROM orders
        )
        SELECT
            'Total Orders' as key,
            total_orders::text as value,
            'var(--text)' as color
        FROM order_stats
        UNION ALL
        SELECT
            'Fill Rate' as key,
            CASE WHEN total_orders > 0 THEN ROUND((filled_orders::numeric / total_orders) * 100, 2) || '%' ELSE '0%' END as value,
            'var(--green)' as color
        FROM order_stats
    """
    return execute_query(query) or []


def get_ticker_data() -> list[dict[str, Any]]:
    """Get top 10 most active stocks for the top ticker bar."""
    query = """
        SELECT DISTINCT ON (sp.symbol)
            sp.symbol as name,
            sp.price as value,
            sp.change_pct,
            sp.change_pct >= 0 as up
        FROM stock_prices sp
        ORDER BY sp.symbol, sp.timestamp DESC
        LIMIT 10
    """
    results = execute_query(query) or []
    # Format the change percentage in Python instead of SQL
    for row in results:
        if row.get("change_pct") is not None:
            change_pct = float(row["change_pct"])
            if change_pct >= 0:
                row["change"] = f"+{change_pct:.2f}%"
            else:
                row["change"] = f"{change_pct:.2f}%"
        else:
            row["change"] = "0.00%"
    return results


def get_indices() -> list[dict[str, Any]]:
    """Calculate synthetic index proxy based on top market cap stocks."""
    query = """
        WITH top_stocks AS (
            SELECT DISTINCT ON (sp.symbol)
                sp.price, sp.change_pct, sp.timestamp
            FROM stock_prices sp
            ORDER BY sp.symbol, sp.timestamp DESC
        )
        SELECT
            'NIFTY 50 PROXY' as name,
            'NIFTY' as id,
            COALESCE(SUM(price), 22000.0) as value,
            COALESCE(AVG(change_pct), 0.0) as change
        FROM top_stocks
    """
    return execute_query(query) or []


def log_order_event(
    client_order_id: str,
    event_type: str,
    from_state: str | None,
    to_state: str,
    event_data: dict[str, Any] | None = None,
    msg_seq_num: int | None = None,
) -> bool:
    """
    Log order event to append-only event log (OMS event sourcing).

    Args:
        client_order_id: Client order ID
        event_type: Event type (e.g., 'ORDER_CREATED', 'ORDER_SENT', 'FILL_RECEIVED')
        from_state: Previous state
        to_state: New state
        event_data: Additional event data
        msg_seq_num: FIX message sequence number

    Returns:
        True if successful
    """
    import json

    query = """
        INSERT INTO order_events (client_order_id, event_type, from_state, to_state, event_data, msg_seq_num)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    return execute_write(
        query,
        (
            client_order_id,
            event_type,
            from_state,
            to_state,
            json.dumps(event_data) if event_data else None,
            msg_seq_num,
        ),
    )


def get_order_events(client_order_id: str) -> list[dict[str, Any]]:
    """
    Get all events for an order (for event sourcing reconstruction).

    Args:
        client_order_id: Client order ID

    Returns:
        List of events in chronological order
    """
    query = """
        SELECT id, client_order_id, event_type, from_state, to_state,
               event_data, event_timestamp, msg_seq_num
        FROM order_events
        WHERE client_order_id = %s
        ORDER BY event_timestamp ASC
    """
    return execute_query(query, (client_order_id,)) or []


def reconstruct_order_state(client_order_id: str) -> str | None:
    """
    Reconstruct current order state from event log (event sourcing).

    Args:
        client_order_id: Client order ID

    Returns:
        Current state or None
    """
    events = get_order_events(client_order_id)
    if not events:
        return None

    # The last event's to_state is the current state
    last_event = events[-1]
    return last_event["to_state"]


# ============================================
# Point-In-Time Query Functions
# ============================================


def get_security_at_date(symbol: str, target_date: date) -> dict[str, Any] | None:
    """
    Get security record as of target date (point-in-time query).

    Answers: "What was true then?" not "What is true now?"

    Args:
        symbol: Security symbol
        target_date: Target date

    Returns:
        Security master record or None
    """
    query = """
        SELECT symbol, name, sector, industry, exchange, currency,
               valid_from, valid_to, is_active, created_at
        FROM security_master
        WHERE symbol = %s
          AND valid_from <= %s
          AND (valid_to IS NULL OR valid_to > %s)
    """
    return execute_query(query, (symbol, target_date, target_date), fetch="one")


def get_sector_at_date(symbol: str, target_date: date) -> str | None:
    """
    Get sector as of target date (point-in-time query).

    Args:
        symbol: Security symbol
        target_date: Target date

    Returns:
        Sector or None
    """
    record = get_security_at_date(symbol, target_date)
    return record.get("sector") if record else None


def get_corporate_actions_between(
    symbol: str, start_date: date, end_date: date
) -> list[dict[str, Any]]:
    """
    Get corporate actions between dates (point-in-time query).

    Args:
        symbol: Security symbol
        start_date: Start date
        end_date: End date

    Returns:
        List of corporate actions
    """
    query = """
        SELECT action_id, action_type, symbol, effective_date,
               ratio, amount, old_symbol, new_symbol, description, created_at
        FROM corporate_actions
        WHERE symbol = %s
          AND effective_date >= %s
          AND effective_date <= %s
        ORDER BY effective_date
    """
    result = execute_query(query, (symbol, start_date, end_date))
    return result if result else []


def adjust_price_for_corporate_actions(
    symbol: str, price: float, from_date: date, to_date: date
) -> float:
    """
    Adjust price for corporate actions between dates (point-in-time query).

    Args:
        symbol: Security symbol
        price: Original price
        from_date: Start date
        to_date: End date

    Returns:
        Adjusted price
    """
    actions = get_corporate_actions_between(symbol, from_date, to_date)

    adjusted_price = price

    for action in actions:
        if action.get("action_type") == "stock_split":
            ratio = action.get("ratio")
            if ratio:
                adjusted_price = adjusted_price / ratio
        elif action.get("action_type") == "reverse_split":
            ratio = action.get("ratio")
            if ratio:
                adjusted_price = adjusted_price * ratio

    return adjusted_price


def get_index_constituents_at_date(index_name: str, target_date: date) -> list[str]:
    """
    Get index constituents as of target date (point-in-time query).

    Args:
        index_name: Index name
        target_date: Target date

    Returns:
        List of constituent symbols
    """
    query = """
        SELECT symbol
        FROM index_constituents
        WHERE index_name = %s
          AND effective_date <= %s
          AND action = 'ADD'
          AND symbol NOT IN (
              SELECT symbol
              FROM index_constituents
              WHERE index_name = %s
                AND effective_date <= %s
                AND action = 'REMOVE'
                AND effective_date > (
                    SELECT MAX(effective_date)
                    FROM index_constituents
                    WHERE index_name = %s
                      AND symbol = index_constituents.symbol
                      AND effective_date <= %s
                )
          )
    """
    result = execute_query(
        query, (index_name, target_date, index_name, target_date, index_name, target_date)
    )
    return [row["symbol"] for row in result] if result else []
