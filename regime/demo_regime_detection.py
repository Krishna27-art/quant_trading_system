"""
Market Regime Detection Demo

Demonstrates the usage of the Market Regime Detection module.
Shows how to:
1. Detect current market regime
2. Get regime summary for dashboard
3. Filter signals based on regime
4. Query regime history
"""

from datetime import date, timedelta

from regime import get_regime_engine


def demo_regime_detection():
    """Demonstrate basic regime detection."""
    print("=" * 60)
    print("Market Regime Detection Demo")
    print("=" * 60)
    
    # Initialize the regime engine
    engine = get_regime_engine(lookback_days=200)
    
    # Detect current regime
    print("\n1. Detecting Current Market Regime...")
    classification = engine.detect_regime()
    
    print(f"\n   Regime: {classification.regime.value}")
    print(f"   Confidence: {classification.confidence}%")
    print(f"   Trend Strength: {classification.trend_strength}")
    print(f"   Volatility Level: {classification.volatility_level}")
    print(f"   Liquidity Status: {classification.liquidity_status}")
    print(f"\n   Matched Rules:")
    for rule in classification.matched_rules:
        print(f"   - {rule}")
    
    print(f"\n   Component Scores:")
    print(f"   - Trend: {classification.trend_score}")
    print(f"   - Volatility: {classification.volatility_score}")
    print(f"   - Breadth: {classification.breadth_score}")
    print(f"   - Institutional: {classification.institutional_score}")
    print(f"   - Liquidity: {classification.liquidity_score}")
    
    # Get regime summary for dashboard
    print("\n2. Getting Regime Summary for Dashboard...")
    summary = engine.get_regime_summary()
    print(f"\n   Dashboard Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Demonstrate signal filtering
    print("\n3. Demonstrating Signal Filtering by Regime...")
    
    test_signals = [
        {"type": "breakout", "symbol": "RELIANCE", "confidence": 0.85},
        {"type": "momentum", "symbol": "HDFCBANK", "confidence": 0.78},
        {"type": "mean_reversion", "symbol": "INFY", "confidence": 0.72},
        {"type": "trend_following", "symbol": "TCS", "confidence": 0.80},
    ]
    
    for signal in test_signals:
        filtered_signal = engine.filter_signal_by_regime(signal, signal["type"])
        status = "REJECTED" if filtered_signal["regime_filtered"] else "ACCEPTED"
        print(f"\n   Signal: {signal['type']} for {signal['symbol']}")
        print(f"   Original Confidence: {signal['confidence']:.2f}")
        print(f"   Adjusted Confidence: {filtered_signal['adjusted_confidence']:.2f}")
        print(f"   Status: {status}")
        print(f"   Reason: {filtered_signal['regime_explanation']}")
    
    # Get regime history
    print("\n4. Getting Regime History (Last 30 Days)...")
    history = engine.get_regime_history(
        start_date=date.today() - timedelta(days=30),
        end_date=date.today()
    )
    
    print(f"\n   Found {len(history)} regime records")
    if history:
        print(f"\n   Recent Regimes:")
        for record in history[-5:]:  # Last 5 records
            print(f"   - {record.timestamp}: {record.regime.value} ({record.confidence}%)")
    
    # Get regime distribution
    print("\n5. Getting Regime Distribution...")
    distribution = engine.history_manager.get_regime_distribution(
        start_date=date.today() - timedelta(days=30),
        end_date=date.today()
    )
    
    print(f"\n   Regime Distribution (Last 30 Days):")
    for regime, stats in distribution.items():
        print(f"   - {regime}: {stats['count']} days ({stats['percentage']}%)")
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    demo_regime_detection()
