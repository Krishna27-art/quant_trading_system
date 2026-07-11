"""
Signal Generator Orchestrator

Main orchestrator that coordinates signal generation, scoring, filtering, and ranking.
This is the brain of the Signal Engine.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from signal_engine.base import Signal, SignalCategory, SignalDirection, SignalSet
from signal_engine.technical import TechnicalSignalGenerator
from signal_engine.volume import VolumeSignalGenerator
from signal_engine.options import OptionsSignalGenerator
from signal_engine.fundamental import FundamentalSignalGenerator
from signal_engine.sentiment import SentimentSignalGenerator
from signal_engine.scoring import SignalScorer, ScoringConfig
from signal_engine.filtering import SignalFilter, FilterRule, MultiSignalConfirmation
from signal_engine.ranking import SignalRanker, RankingCriteria
from signal_engine.performance import SignalPerformanceTracker
from utils.logger import get_logger

logger = get_logger("signal_engine.generator")


class SignalGenerator:
    """
    Signal Generator Orchestrator.
    
    Pipeline:
    Market Data → Feature Laboratory → Signal Generator → Technical Signals
                                                      → Volume Signals
                                                      → Options Signals
                                                      → Fundamental Signals
                                                      → Sentiment Signals
                                                      → Score Every Signal
                                                      → Filter Weak Signals
                                                      → Rank Stocks
                                                      → Top Candidates
    """
    
    def __init__(
        self,
        scorer_config: Optional[ScoringConfig] = None,
        filter_rules: Optional[List[FilterRule]] = None,
        confirmation_config: Optional[MultiSignalConfirmation] = None,
        ranking_criteria: Optional[RankingCriteria] = None,
    ):
        """
        Initialize signal generator.
        
        Args:
            scorer_config: Optional scoring configuration
            filter_rules: Optional filter rules
            confirmation_config: Optional multi-signal confirmation config
            ranking_criteria: Optional ranking criteria
        """
        # Initialize signal generators
        self.technical_generator = TechnicalSignalGenerator()
        self.volume_generator = VolumeSignalGenerator()
        self.options_generator = OptionsSignalGenerator()
        self.fundamental_generator = FundamentalSignalGenerator()
        self.sentiment_generator = SentimentSignalGenerator()
        
        # Initialize engines
        self.scorer = SignalScorer(config=scorer_config)
        self.filter = SignalFilter(
            filter_rules=filter_rules,
            confirmation_config=confirmation_config,
        )
        self.ranker = SignalRanker(
            criteria=ranking_criteria,
            scorer=self.scorer,
        )
        self.performance_tracker = SignalPerformanceTracker()
        
        self._logger = get_logger("signal_engine.generator")
    
    def generate_signals(
        self,
        symbol: str,
        ohlcva_data: pd.DataFrame,
        options_data: Optional[Dict] = None,
        fundamental_data: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
    ) -> SignalSet:
        """
        Generate all signals for a single symbol.
        
        Args:
            symbol: Stock symbol
            ohlcva_data: OHLCV DataFrame
            options_data: Optional options chain data
            fundamental_data: Optional fundamental data
            sentiment_data: Optional sentiment data
            
        Returns:
            SignalSet with all generated signals
        """
        signal_set = SignalSet(
            symbol=symbol,
            timestamp=datetime.now(),
        )
        
        # Generate technical signal
        try:
            technical_signal = self.technical_generator.generate({'ohlcva': ohlcva_data})
            signal_set.add_signal(technical_signal)
            self._logger.info(f"Generated technical signal for {symbol}: score={technical_signal.score:.1f}")
        except Exception as e:
            self._logger.error(f"Failed to generate technical signal for {symbol}: {e}")
        
        # Generate volume signal
        try:
            volume_signal = self.volume_generator.generate({'ohlcva': ohlcva_data})
            signal_set.add_signal(volume_signal)
            self._logger.info(f"Generated volume signal for {symbol}: score={volume_signal.score:.1f}")
        except Exception as e:
            self._logger.error(f"Failed to generate volume signal for {symbol}: {e}")
        
        # Generate options signal
        if options_data:
            try:
                options_signal = self.options_generator.generate(options_data)
                signal_set.add_signal(options_signal)
                self._logger.info(f"Generated options signal for {symbol}: score={options_signal.score:.1f}")
            except Exception as e:
                self._logger.error(f"Failed to generate options signal for {symbol}: {e}")
        
        # Generate fundamental signal
        if fundamental_data:
            try:
                fundamental_signal = self.fundamental_generator.generate(fundamental_data)
                signal_set.add_signal(fundamental_signal)
                self._logger.info(f"Generated fundamental signal for {symbol}: score={fundamental_signal.score:.1f}")
            except Exception as e:
                self._logger.error(f"Failed to generate fundamental signal for {symbol}: {e}")
        
        # Generate sentiment signal
        if sentiment_data:
            try:
                sentiment_signal = self.sentiment_generator.generate(sentiment_data)
                signal_set.add_signal(sentiment_signal)
                self._logger.info(f"Generated sentiment signal for {symbol}: score={sentiment_signal.score:.1f}")
            except Exception as e:
                self._logger.error(f"Failed to generate sentiment signal for {symbol}: {e}")
        
        return signal_set
    
    def generate_signals_for_multiple(
        self,
        data_map: Dict[str, Dict],
    ) -> Dict[str, SignalSet]:
        """
        Generate signals for multiple symbols.
        
        Args:
            data_map: Dictionary mapping symbols to data dictionaries:
                      {
                          'SYMBOL': {
                              'ohlcva': DataFrame,
                              'options': Dict,
                              'fundamental': Dict,
                              'sentiment': Dict,
                          }
                      }
            
        Returns:
            Dictionary mapping symbols to SignalSets
        """
        signal_sets = {}
        
        for symbol, data in data_map.items():
            ohlcva_data = data.get('ohlcva')
            options_data = data.get('options')
            fundamental_data = data.get('fundamental')
            sentiment_data = data.get('sentiment')
            
            signal_set = self.generate_signals(
                symbol=symbol,
                ohlcva_data=ohlcva_data,
                options_data=options_data,
                fundamental_data=fundamental_data,
                sentiment_data=sentiment_data,
            )
            
            signal_sets[symbol] = signal_set
        
        self._logger.info(f"Generated signals for {len(signal_sets)} symbols")
        
        return signal_sets
    
    def process_signals(
        self,
        signal_sets: Dict[str, SignalSet],
        top_n: int = 10,
    ) -> Dict:
        """
        Process signals through scoring, filtering, and ranking.
        
        Args:
            signal_sets: Dictionary mapping symbols to SignalSets
            top_n: Number of top candidates to return
            
        Returns:
            Dictionary with processing results:
            - scored: Scored signal sets
            - filtered: Filtered signal sets
            - ranked: Ranked signal sets
            - top_candidates: Top N candidates
        """
        # Score signals
        scored_results = {}
        for symbol, signal_set in signal_sets.items():
            scored_results[symbol] = self.scorer.score_signal_set(signal_set)
        
        # Filter signals
        filtered_results = self.filter.filter_multiple_signal_sets(signal_sets)
        
        # Get passed symbols
        passed_symbols = self.filter.get_passed_symbols(filtered_results)
        
        # Filter to passed signal sets
        passed_signal_sets = {
            symbol: signal_sets[symbol]
            for symbol in passed_symbols
        }
        
        # Rank signals
        rankings = self.ranker.rank_signal_sets(passed_signal_sets)
        
        # Get top candidates
        top_candidates = rankings[:top_n]
        
        # Get rejection statistics
        rejection_stats = self.filter.get_rejection_stats(filtered_results)
        
        self._logger.info(
            f"Processed {len(signal_sets)} signals: "
            f"{len(passed_symbols)} passed, {len(top_candidates)} top candidates"
        )
        
        return {
            'scored': scored_results,
            'filtered': filtered_results,
            'ranked': rankings,
            'top_candidates': top_candidates,
            'passed_symbols': passed_symbols,
            'rejection_stats': rejection_stats,
        }
    
    def get_signal_dashboard(
        self,
        signal_set: SignalSet,
    ) -> Dict:
        """
        Get signal dashboard for a single signal set.
        
        Args:
            signal_set: SignalSet to visualize
            
        Returns:
            Dictionary with dashboard data
        """
        dashboard = {
            'symbol': signal_set.symbol,
            'timestamp': signal_set.timestamp.isoformat(),
            'average_score': signal_set.get_average_score(),
            'dominant_direction': signal_set.get_dominant_direction().value,
            'signals': {},
        }
        
        for category, signal in signal_set.signals.items():
            dashboard['signals'][category.value] = {
                'score': signal.score,
                'direction': signal.direction.value,
                'confidence': signal.confidence,
                'reason': signal.reason,
            }
        
        return dashboard
    
    def get_batch_dashboard(
        self,
        signal_sets: Dict[str, SignalSet],
    ) -> Dict:
        """
        Get signal dashboard for multiple signal sets.
        
        Args:
            signal_sets: Dictionary mapping symbols to SignalSets
            
        Returns:
            Dictionary with batch dashboard data
        """
        dashboards = {}
        
        for symbol, signal_set in signal_sets.items():
            dashboards[symbol] = self.get_signal_dashboard(signal_set)
        
        return dashboards
    
    def record_trade_outcome(
        self,
        symbol: str,
        signal_category: SignalCategory,
        signal_direction: str,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
    ) -> None:
        """
        Record a trade outcome for performance tracking.
        
        Args:
            symbol: Stock symbol
            signal_category: Category of the signal
            signal_direction: Direction of the signal
            entry_price: Entry price
            exit_price: Exit price
            entry_time: Entry timestamp
            exit_time: Exit timestamp
        """
        self.performance_tracker.record_trade(
            symbol=symbol,
            signal_category=signal_category,
            signal_direction=signal_direction,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
        )
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary of all signals."""
        return self.performance_tracker.get_performance_summary()
