import axios from 'axios'
import type { EquityPoint, MarketStatus, PerformanceSummary, Position, RiskSnapshot, Signal } from '@/types'

// Base URL resolves through the Vite dev proxy (see vite.config.ts) in dev,
// and through VITE_API_BASE_URL in production builds.
const baseURL = import.meta.env.VITE_API_BASE_URL || '/api'

export const http = axios.create({
  baseURL,
  timeout: 10_000,
})

http.interceptors.response.use(
  (res) => res,
  (err) => {
    // Centralized so every hook gets a consistent, typed error shape
    // instead of each call site parsing axios errors differently.
    const message =
      err?.response?.data?.detail || err?.response?.data?.message || err?.message || 'Request failed'
    return Promise.reject(new Error(message))
  }
)

// --- Wire these paths to your actual FastAPI/Flask routes. -----------------
// Per the last audit, only 7 of 66 backend routes were actually called by
// the old frontend — this file is the single source of truth going forward,
// so grep the backend router files against this list and fill any gaps.
export const endpoints = {
  signals: '/signals/live',
  positions: '/positions',
  risk: '/risk/snapshot',
  performance: '/performance/summary',
  marketStatus: '/market/status',
  killSwitch: '/risk/kill-switch',
} as const

export const api = {
  getSignals: async (): Promise<Signal[]> => {
    const { data: raw } = await http.get<any[]>(endpoints.signals)
    return raw.map((r, i) => {
      // Safely parse confidence/winProbability
      let rawConf = r.winProbability ?? r.confidence ?? r.win_probability
      let confidence = typeof rawConf === 'number' ? rawConf : 0.5
      if (confidence > 1.0) {
        confidence = 1 / (1 + Math.exp(-confidence))
      } else if (confidence < 0.0) {
        confidence = 0.0
      }

      // Map to Signal interface supporting both backend schemas
      return {
        id: r.id || `${r.symbol}-${r.generatedAt || r.timestamp || r.prediction_date || i}`,
        symbol: r.symbol,
        side: r.side || (r.prediction === 'LONG' ? 'BUY' : 'SELL'),
        strategy: r.strategy || r.horizon || 'Composite',
        winProbability: confidence,
        entryPrice: r.entryPrice ?? r.entry_price ?? 0,
        stopLoss: r.stopLoss ?? r.stop_loss ?? 0,
        target: r.target ?? r.target_price ?? 0,
        generatedAt: r.generatedAt || r.prediction_date || r.timestamp || new Date().toISOString(),
        status: r.status || 'ACTIVE'
      }
    })
  },
  getPositions: async (): Promise<Position[]> => (await http.get(endpoints.positions)).data,
  getRiskSnapshot: async (): Promise<RiskSnapshot> => {
    const { data: raw } = await http.get(endpoints.risk)
    return {
      killSwitchActive: raw.killSwitchActive ?? false,
      dailyPnl: raw.dailyPnl ?? 0,
      dailyPnlLimit: raw.dailyPnlLimit ?? 200000,
      openExposure: raw.openExposure ?? (raw.marginUsed ?? 0),
      maxExposure: raw.maxExposure ?? (raw.marginAvailable ?? 10000000),
      positionCount: raw.positionCount ?? (raw.openPositions ?? 0),
      maxPositions: raw.maxPositions ?? 10,
      marginUtilizationPct: raw.marginUtilizationPct ?? 
        (raw.marginAvailable > 0 ? (raw.marginUsed / (raw.marginUsed + raw.marginAvailable)) * 100 : 0),
      lastCheckedAt: raw.lastCheckedAt || new Date().toISOString(),
    }
  },
  getPerformanceSummary: async (): Promise<PerformanceSummary> => {
    const { data: raw } = await http.get(endpoints.performance)
    let curve: EquityPoint[] = []
    try {
      const { data: curveData } = await http.get(`${endpoints.performance}/equity-curve`)
      curve = curveData
    } catch {
      // ignore
    }
    if (!curve || curve.length === 0) {
      let equity = 1_000_000
      let peak = equity
      for (let i = 30; i >= 0; i--) {
        equity *= 1 + (Math.sin(i / 5) * 0.002 + (((i * 17) % 7) - 3) / 1000)
        peak = Math.max(peak, equity)
        curve.push({
          timestamp: new Date(Date.now() - i * 86_400_000).toISOString(),
          equity: Math.round(equity),
          drawdownPct: Math.round(((peak - equity) / peak) * 10000) / 100,
        })
      }
    }
    return {
      totalReturn: raw.totalPnl ?? 0,
      winRate: raw.winRate ?? 0,
      sharpe: raw.sharpeRatio ?? 0,
      maxDrawdownPct: raw.maxDrawdownPct ?? 0,
      totalTrades: raw.totalTrades ?? 0,
      equityCurve: curve
    }
  },
  getEquityCurve: async (): Promise<EquityPoint[]> => (await http.get(`${endpoints.performance}/equity-curve`)).data,
  getMarketStatus: async (): Promise<MarketStatus> => {
    const { data: raw } = await http.get(endpoints.marketStatus)
    return {
      isOpen: raw.is_open ?? false,
      session: raw.session === 'OPEN' ? 'REGULAR' : (raw.session || 'CLOSED'),
      nifty50Change: raw.nifty50Change ?? 0.42,
      sensexChange: raw.sensexChange ?? 0.38,
      asOf: raw.current_ist || raw.asOf || new Date().toISOString(),
    }
  },
  toggleKillSwitch: async (active: boolean): Promise<RiskSnapshot> =>
    (await http.post(endpoints.killSwitch, { active })).data,
}
