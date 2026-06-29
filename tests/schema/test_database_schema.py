"""
Schema Tests

Tests that verify database schema integrity and consistency.
"""

import duckdb
import pytest

from config.settings import DB_PATH


class TestDatabaseSchema:
    """Test database schema integrity."""

    @pytest.fixture
    def conn(self):
        """Create database connection."""
        conn = duckdb.connect(str(DB_PATH))
        yield conn
        conn.close()

    def test_security_master_table_exists(self, conn):
        """Test security_master table exists."""
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'security_master'
        """).fetchone()

        assert result is not None

    def test_security_master_columns(self, conn):
        """Test security_master has required columns."""
        result = conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'security_master'
            ORDER BY column_name
        """).fetchall()

        column_names = [row[0] for row in result]

        required_columns = [
            "symbol",
            "isin",
            "company_name",
            "sector",
            "industry",
            "listing_date",
            "delisting_date",
            "face_value",
            "is_current",
        ]

        for col in required_columns:
            assert col in column_names

    def test_equity_history_table_exists(self, conn):
        """Test equity_history table exists."""
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'equity_history'
        """).fetchone()

        assert result is not None

    def test_equity_history_columns(self, conn):
        """Test equity_history has required columns."""
        result = conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'equity_history'
            ORDER BY column_name
        """).fetchall()

        column_names = [row[0] for row in result]

        required_columns = ["symbol", "date", "open", "high", "low", "close", "volume"]

        for col in required_columns:
            assert col in column_names

    def test_options_chain_table_exists(self, conn):
        """Test options_chain table exists."""
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'options_chain'
        """).fetchone()

        assert result is not None

    def test_corporate_actions_pit_table_exists(self, conn):
        """Test corporate_actions_pit table exists."""
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'corporate_actions_pit'
        """).fetchone()

        assert result is not None

    def test_fundamental_pit_table_exists(self, conn):
        """Test fundamental_pit table exists."""
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'fundamental_pit'
        """).fetchone()

        assert result is not None

    def test_market_calendar_table_exists(self, conn):
        """Test market_calendar table exists."""
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'market_calendar'
        """).fetchone()

        assert result is not None

    def test_market_calendar_columns(self, conn):
        """Test market_calendar has required columns."""
        result = conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'market_calendar'
            ORDER BY column_name
        """).fetchall()

        column_names = [row[0] for row in result]

        required_columns = [
            "date",
            "is_trading_day",
            "is_weekend",
            "is_holiday",
            "session_type",
            "session_start",
            "session_end",
            "is_expiry",
            "shifted_expiry_date",
        ]

        for col in required_columns:
            assert col in column_names

    def test_universe_members_table_exists(self, conn):
        """Test universe_members table exists."""
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'universe_members'
        """).fetchone()

        assert result is not None

    def test_universe_members_columns(self, conn):
        """Test universe_members has required columns."""
        result = conn.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'universe_members'
            ORDER BY column_name
        """).fetchall()

        column_names = [row[0] for row in result]

        required_columns = ["symbol", "start_date", "end_date", "is_active", "exit_reason"]

        for col in required_columns:
            assert col in column_names

    def test_data_lineage_table_exists(self, conn):
        """Test data_lineage table exists."""
        result = conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'data_lineage'
        """).fetchone()

        assert result is not None

    def test_pit_tables_have_available_at(self, conn):
        """Test all PIT tables have available_at column."""
        pit_tables = [
            "fundamental_pit",
            "promoter_holding_pit",
            "financials_pit",
            "corporate_actions_pit",
            "index_membership_pit",
        ]

        for table in pit_tables:
            result = conn.execute(f"""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = '{table}' AND column_name = 'available_at'
            """).fetchone()

            assert result is not None, f"{table} missing available_at column"

    def test_pit_tables_have_valid_from_to(self, conn):
        """Test all PIT tables have valid_from and valid_to columns."""
        pit_tables = [
            "fundamental_pit",
            "promoter_holding_pit",
            "financials_pit",
            "corporate_actions_pit",
            "index_membership_pit",
        ]

        for table in pit_tables:
            result = conn.execute(f"""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = '{table}' AND column_name IN ('valid_from', 'valid_to')
            """).fetchall()

            column_names = [row[0] for row in result]
            assert "valid_from" in column_names, f"{table} missing valid_from column"
            assert "valid_to" in column_names, f"{table} missing valid_to column"

    def test_indexes_exist(self, conn):
        """Test critical indexes exist."""
        # Test security_master indexes
        result = conn.execute("""
            SELECT index_name FROM duckdb_indexes
            WHERE table_name = 'security_master'
        """).fetchall()

        index_names = [row[0] for row in result]
        assert len(index_names) > 0, "security_master has no indexes"
