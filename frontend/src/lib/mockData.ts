import type { EquityPoint, MarketStatus, PerformanceSummary, Position, RiskSnapshot, Signal } from '@/types'

const SYMBOLS = ['RELIANCE', 'HDFCBANK', 'INFY', 'TCS', 'ICICIBANK', 'LT', 'SBIN', 'AXISBANK']

export function mockSignals(): Signal[] {
  return SYMBOLS.slice(0, 6).map((symbol, i) => ({
    id: `sig-${i}`,
    symbol,
    side: i % 3 === 0 ? 'SELL' : 'BUY',
    strategy: ['VWAP Reversion', 'Volume Breakout', 'Cross-Sectional Momentum'][i % 3],
    winProbability: 0.52 + ((i * 7) % 20) / 100,
    entryPrice: 1000 + i * 137.5,
    stopLoss: 1000 + i * 137.5 - 18,
    target: 1000 + i * 137.5 + 42,
    generatedAt: new Date(Date.now() - i * 4 * 60_000).toISOString(),
    status: i === 0 ? 'ACTIVE' : i % 4 === 0 ? 'FILLED' : 'ACTIVE',
  }))
}

export function mockPositions(): Position[] {
  return SYMBOLS.slice(0, 4).map((symbol, i) => {
    const avg = 800 + i * 220
    const last = avg * (1 + (i % 2 === 0 ? 1 : -1) * 0.014)
    return {
      id: `pos-${i}`,
      symbol,
      side: i % 2 === 0 ? 'BUY' : 'SELL',
      quantity: 25 * (i + 1),
      avgEntryPrice: avg,
      lastPrice: Math.round(last * 100) / 100,
      unrealizedPnl: Math.round((last - avg) * 25 * (i + 1)),
      unrealizedPnlPct: Math.round(((last - avg) / avg) * 10000) / 100,
      openedAt: new Date(Date.now() - i * 3600_000).toISOString(),
    }
  })
}

export function mockRisk(): RiskSnapshot {
  return {
    killSwitchActive: false,
    dailyPnl: 18420,
    dailyPnlLimit: -50000,
    openExposure: 612000,
    maxExposure: 1500000,
    positionCount: 4,
    maxPositions: 10,
    marginUtilizationPct: 41,
    lastCheckedAt: new Date().toISOString(),
  }
}

export function mockEquityCurve(): EquityPoint[] {
  const points: EquityPoint[] = []
  let equity = 1_000_000
  let peak = equity
  for (let i = 60; i >= 0; i--) {
    equity *= 1 + (Math.sin(i / 5) * 0.004 + (((i * 37) % 11) - 5) / 900)
    peak = Math.max(peak, equity)
    points.push({
      timestamp: new Date(Date.now() - i * 86_400_000).toISOString(),
      equity: Math.round(equity),
      drawdownPct: Math.round(((peak - equity) / peak) * 10000) / 100,
    })
  }
  return points
}

export function mockPerformance(): PerformanceSummary {
  const curve = mockEquityCurve()
  return {
    totalReturn: 12.4,
    winRate: 0.56,
    sharpe: 1.31,
    maxDrawdownPct: Math.max(...curve.map((p) => p.drawdownPct)),
    totalTrades: 214,
    equityCurve: curve,
  }
}

export function mockMarketStatus(): MarketStatus {
  return {
    isOpen: true,
    session: 'REGULAR',
    nifty50Change: 0.42,
    sensexChange: 0.38,
    asOf: new Date().toISOString(),
  }
}
