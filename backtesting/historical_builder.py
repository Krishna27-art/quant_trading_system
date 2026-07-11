"""
Historical Data Builder

Builds comprehensive historical datasets for backtesting.
Collects:
- OHLCV data
- Options data
- Fundamentals
- News
- Macro data
- FII/DII flows
- Corporate actions

This creates the research database that powers all backtesting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from config.settings import DB_PATH, DB_LOCK
from data_platform.upstox_client import (
    get_candles,
    get_fii_activity,
    get_dii_activity,
    get_option_chain,
    get_option_expiries,
    get_company_profile,
    get_key_ratios,
    get_corporate_actions,
    get_stock_news,
    symbol_to_key,
)
from utils.logger import get_logger

logger = get_logger("historical_builder")


@dataclass
class HistoricalDataConfig:
    """Configuration for historical data building."""
    
    start_date: date
    end_date: date
    symbols: list[str]
    include_options: bool = True
    include_fundamentals: bool = True
    include_news: bool = True
    include_fii_dii: bool = True
    include_corporate_actions: bool = True


class HistoricalDataBuilder:
    """
    Builds historical datasets for backtesting.
    
    Ensures data quality and completeness before storage.
    Uses the same data sources as the live system for consistency.
    """
    
    def __init__(self, config: HistoricalDataConfig):
        """
        Initialize the historical data builder.
        
        Args:
            config: Configuration for data building
        """
        self.config = config
        self.logger = logger
        
        # Ensure output directory exists
        self.output_dir = Path("data/historical")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_dataset(self) -> dict[str, Any]:
        """
        Build the complete historical dataset.
        
        Returns:
            Summary of built data
        """
        self.logger.info(f"Building historical dataset from {self.config.start_date} to {self.config.end_date}")
        
        summary = {
            "start_date": self.config.start_date.isoformat(),
            "end_date": self.config.end_date.isoformat(),
            "symbols_count": len(self.config.symbols),
            "components": {},
        }
        
        # Build each component
        summary["components"]["price_data"] = self._build_price_data()
        
        if self.config.include_options:
            summary["components"]["options_data"] = self._build_options_data()
        
        if self.config.include_fundamentals:
            summary["components"]["fundamentals"] = self._build_fundamentals()
        
        if self.config.include_news:
            summary["components"]["news"] = self._build_news()
        
        if self.config.include_fii_dii:
            summary["components"]["fii_dii"] = self._build_fii_dii()
        
        if self.config.include_corporate_actions:
            summary["components"]["corporate_actions"] = self._build_corporate_actions()
        
        # Build master index
        self._build_master_index()
        
        self.logger.info("Historical dataset build complete")
        return summary
    
    def _build_price_data(self) -> dict[str, Any]:
        """Build OHLCV price data for all symbols."""
        self.logger.info("Building price data...")
        
        price_data = {}
        failed_symbols = []
        
        for symbol in self.config.symbols:
            try:
                candles = get_candles(
                    symbol,
                    interval="1day",
                    from_date=self.config.start_date.isoformat(),
                    to_date=self.config.end_date.isoformat()
                )
                
                if candles:
                    df = pd.DataFrame(candles)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['symbol'] = symbol
                    price_data[symbol] = df
                    self.logger.debug(f"Retrieved {len(df)} candles for {symbol}")
                else:
                    failed_symbols.append(symbol)
                    self.logger.warning(f"No price data for {symbol}")
                    
            except Exception as e:
                failed_symbols.append(symbol)
                self.logger.warning(f"Failed to get price data for {symbol}: {e}")
        
        # Save to parquet
        if price_data:
            combined_df = pd.concat(price_data.values(), ignore_index=True)
            output_path = self.output_dir / "price_data.parquet"
            combined_df.to_parquet(output_path, index=False)
            self.logger.info(f"Saved price data to {output_path}")
        
        return {
            "symbols_success": len(price_data),
            "symbols_failed": len(failed_symbols),
            "failed_symbols": failed_symbols,
            "total_records": len(combined_df) if price_data else 0,
        }
    
    def _build_options_data(self) -> dict[str, Any]:
        """Build options chain data for index and stock options."""
        self.logger.info("Building options data...")
        
        # Focus on NIFTY and BANKNIFTY for options
        index_symbols = ["NIFTY50", "BANKNIFTY"]
        
        options_data = {}
        
        for symbol in index_symbols:
            try:
                key = symbol_to_key(symbol)
                if not key:
                    continue
                
                expiries = get_option_expiries(key)
                if not expiries:
                    continue
                
                # Get data for nearest 3 expiries
                for expiry in expiries[:3]:
                    try:
                        chain = get_option_chain(key, expiry)
                        if chain:
                            df = pd.DataFrame(chain)
                            df['symbol'] = symbol
                            df['expiry'] = expiry
                            options_data[f"{symbol}_{expiry}"] = df
                            self.logger.debug(f"Retrieved options chain for {symbol} {expiry}")
                    except Exception as e:
                        self.logger.warning(f"Failed to get options chain for {symbol} {expiry}: {e}")
                        
            except Exception as e:
                self.logger.warning(f"Failed to get options data for {symbol}: {e}")
        
        # Save to parquet
        if options_data:
            combined_df = pd.concat(options_data.values(), ignore_index=True)
            output_path = self.output_dir / "options_data.parquet"
            combined_df.to_parquet(output_path, index=False)
            self.logger.info(f"Saved options data to {output_path}")
        
        return {
            "chains_retrieved": len(options_data),
            "total_records": len(combined_df) if options_data else 0,
        }
    
    def _build_fundamentals(self) -> dict[str, Any]:
        """Build fundamental data for all symbols."""
        self.logger.info("Building fundamentals...")
        
        fundamentals = {}
        failed_symbols = []
        
        for symbol in self.config.symbols:
            try:
                profile = get_company_profile(symbol)
                ratios = get_key_ratios(symbol)
                
                if profile or ratios:
                    fundamentals[symbol] = {
                        "profile": profile,
                        "key_ratios": ratios,
                    }
                    self.logger.debug(f"Retrieved fundamentals for {symbol}")
                else:
                    failed_symbols.append(symbol)
                    
            except Exception as e:
                failed_symbols.append(symbol)
                self.logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
        
        # Save to parquet
        if fundamentals:
            # Flatten for storage
            records = []
            for symbol, data in fundamentals.items():
                record = {"symbol": symbol}
                if data.get("profile"):
                    record.update(data["profile"])
                if data.get("key_ratios"):
                    record["key_ratios"] = data["key_ratios"]
                records.append(record)
            
            df = pd.DataFrame(records)
            output_path = self.output_dir / "fundamentals.parquet"
            df.to_parquet(output_path, index=False)
            self.logger.info(f"Saved fundamentals to {output_path}")
        
        return {
            "symbols_success": len(fundamentals),
            "symbols_failed": len(failed_symbols),
        }
    
    def _build_news(self) -> dict[str, Any]:
        """Build news data for all symbols."""
        self.logger.info("Building news data...")
        
        news_data = {}
        
        for symbol in self.config.symbols:
            try:
                articles = get_stock_news(symbol, limit=50)
                if articles:
                    news_data[symbol] = articles
                    self.logger.debug(f"Retrieved {len(articles)} news articles for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to get news for {symbol}: {e}")
        
        # Save to parquet
        if news_data:
            records = []
            for symbol, articles in news_data.items():
                for article in articles:
                    record = article.copy()
                    record["symbol"] = symbol
                    records.append(record)
            
            df = pd.DataFrame(records)
            output_path = self.output_dir / "news.parquet"
            df.to_parquet(output_path, index=False)
            self.logger.info(f"Saved news data to {output_path}")
        
        return {
            "symbols_with_news": len(news_data),
            "total_articles": sum(len(v) for v in news_data.values()),
        }
    
    def _build_fii_dii(self) -> dict[str, Any]:
        """Build FII/DII flow data."""
        self.logger.info("Building FII/DII data...")
        
        fii_dii_data = []
        current_date = self.config.start_date
        
        while current_date <= self.config.end_date:
            try:
                fii = get_fii_activity(current_date.isoformat())
                dii = get_dii_activity(current_date.isoformat())
                
                if fii.get("available") or dii.get("available"):
                    record = {
                        "date": current_date,
                        "fii_net_cash": fii.get("total_net_cash"),
                        "fii_net_futures": fii.get("total_net_futures"),
                        "dii_net_cash": dii.get("net_amount"),
                    }
                    fii_dii_data.append(record)
                    
            except Exception as e:
                self.logger.warning(f"Failed to get FII/DII for {current_date}: {e}")
            
            current_date += timedelta(days=1)
        
        # Save to parquet
        if fii_dii_data:
            df = pd.DataFrame(fii_dii_data)
            output_path = self.output_dir / "fii_dii.parquet"
            df.to_parquet(output_path, index=False)
            self.logger.info(f"Saved FII/DII data to {output_path}")
        
        return {
            "days_with_data": len(fii_dii_data),
        }
    
    def _build_corporate_actions(self) -> dict[str, Any]:
        """Build corporate actions data."""
        self.logger.info("Building corporate actions...")
        
        corporate_actions = {}
        
        for symbol in self.config.symbols:
            try:
                actions = get_corporate_actions(symbol)
                if actions:
                    corporate_actions[symbol] = actions
                    self.logger.debug(f"Retrieved {len(actions)} corporate actions for {symbol}")
            except Exception as e:
                self.logger.warning(f"Failed to get corporate actions for {symbol}: {e}")
        
        # Save to parquet
        if corporate_actions:
            records = []
            for symbol, actions in corporate_actions.items():
                for action in actions:
                    record = action.copy()
                    record["symbol"] = symbol
                    records.append(record)
            
            df = pd.DataFrame(records)
            output_path = self.output_dir / "corporate_actions.parquet"
            df.to_parquet(output_path, index=False)
            self.logger.info(f"Saved corporate actions to {output_path}")
        
        return {
            "symbols_with_actions": len(corporate_actions),
            "total_actions": sum(len(v) for v in corporate_actions.values()),
        }
    
    def _build_master_index(self) -> None:
        """Build a master index file for quick lookup."""
        self.logger.info("Building master index...")
        
        index_data = {
            "start_date": self.config.start_date.isoformat(),
            "end_date": self.config.end_date.isoformat(),
            "symbols": self.config.symbols,
            "components": {
                "price_data": str(self.output_dir / "price_data.parquet"),
                "options_data": str(self.output_dir / "options_data.parquet") if self.config.include_options else None,
                "fundamentals": str(self.output_dir / "fundamentals.parquet") if self.config.include_fundamentals else None,
                "news": str(self.output_dir / "news.parquet") if self.config.include_news else None,
                "fii_dii": str(self.output_dir / "fii_dii.parquet") if self.config.include_fii_dii else None,
                "corporate_actions": str(self.output_dir / "corporate_actions.parquet") if self.config.include_corporate_actions else None,
            },
            "built_at": datetime.now().isoformat(),
        }
        
        import json
        output_path = self.output_dir / "index.json"
        with open(output_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        self.logger.info(f"Saved master index to {output_path}")
    
    def validate_dataset(self) -> dict[str, Any]:
        """
        Validate the built dataset for completeness and quality.
        
        Returns:
            Validation report
        """
        self.logger.info("Validating historical dataset...")
        
        validation_report = {
            "valid": True,
            "issues": [],
            "warnings": [],
        }
        
        # Check price data
        price_path = self.output_dir / "price_data.parquet"
        if price_path.exists():
            df = pd.read_parquet(price_path)
            expected_days = (self.config.end_date - self.config.start_date).days + 1
            actual_days = df['timestamp'].dt.date.nunique()
            
            coverage = actual_days / expected_days * 100
            if coverage < 80:
                validation_report["valid"] = False
                validation_report["issues"].append(f"Price data coverage: {coverage:.1f}% (expected >80%)")
            elif coverage < 95:
                validation_report["warnings"].append(f"Price data coverage: {coverage:.1f}%")
            
            # Check for missing values
            missing_pct = df.isnull().sum() / len(df) * 100
            high_missing = missing_pct[missing_pct > 5]
            if not high_missing.empty:
                validation_report["warnings"].append(f"High missing values in columns: {high_missing.to_dict()}")
        else:
            validation_report["valid"] = False
            validation_report["issues"].append("Price data file not found")
        
        # Check other components
        components = [
            ("options_data", self.config.include_options),
            ("fundamentals", self.config.include_fundamentals),
            ("news", self.config.include_news),
            ("fii_dii", self.config.include_fii_dii),
            ("corporate_actions", self.config.include_corporate_actions),
        ]
        
        for component, required in components:
            if required:
                path = self.output_dir / f"{component}.parquet"
                if not path.exists():
                    validation_report["warnings"].append(f"{component} not found")
        
        self.logger.info(f"Validation complete: {'VALID' if validation_report['valid'] else 'INVALID'}")
        return validation_report


def build_historical_dataset(
    start_date: str | date,
    end_date: str | date,
    symbols: list[str],
    **kwargs,
) -> dict[str, Any]:
    """
    Convenience function to build historical dataset.
    
    Args:
        start_date: Start date (ISO string or date object)
        end_date: End date (ISO string or date object)
        symbols: List of symbols to include
        **kwargs: Additional configuration options
        
    Returns:
        Build summary
    """
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)
    
    config = HistoricalDataConfig(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        **kwargs
    )
    
    builder = HistoricalDataBuilder(config)
    summary = builder.build_dataset()
    validation = builder.validate_dataset()
    
    return {
        "build_summary": summary,
        "validation": validation,
    }
