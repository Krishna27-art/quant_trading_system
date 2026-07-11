export type Side = 'BUY' | 'SELL'
export type SignalStatus = 'ACTIVE' | 'FILLED' | 'EXPIRED' | 'REJECTED'
export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected'

export interface Signal {
  id: string
  symbol: string
  side: Side
  strategy: string
  winProbability: number // 0-1, thresholded against MIN_WIN_PROB
  entryPrice: number
  stopLoss: number
  target: number
  generatedAt: string // ISO timestamp
  status: SignalStatus
}

export interface Position {
  id: string
  symbol: string
  side: Side
  quantity: number
  avgEntryPrice: number
  lastPrice: number
  unrealizedPnl: number
  unrealizedPnlPct: number
  openedAt: string
}

export interface RiskSnapshot {
  killSwitchActive: boolean
  dailyPnl: number
  dailyPnlLimit: number
  openExposure: number
  maxExposure: number
  positionCount: number
  maxPositions: number
  marginUtilizationPct: number
  lastCheckedAt: string
}

export interface EquityPoint {
  timestamp: string
  equity: number
  drawdownPct: number
}

export interface PerformanceSummary {
  totalReturn: number
  winRate: number
  sharpe: number
  maxDrawdownPct: number
  totalTrades: number
  equityCurve: EquityPoint[]
}

export interface MarketStatus {
  isOpen: boolean
  session: 'PRE_OPEN' | 'REGULAR' | 'CLOSED'
  nifty50Change: number
  sensexChange: number
  asOf: string
}
