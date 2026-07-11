import { Activity, Percent, TrendingUp, Wallet } from 'lucide-react'
import { StatCard } from '@/components/dashboard/StatCard'
import { SignalFeed } from '@/components/dashboard/SignalFeed'
import { PositionsTable } from '@/components/dashboard/PositionsTable'
import { RiskPanel } from '@/components/dashboard/RiskPanel'
import { EquityCurve } from '@/components/dashboard/EquityCurve'
import { useRisk } from '@/hooks/useRisk'
import { usePerformance } from '@/hooks/usePerformance'
import { usePositions } from '@/hooks/usePositions'
import { fmt } from '@/utils/format'

export default function Dashboard() {
  const { data: risk } = useRisk()
  const { data: perf } = usePerformance()
  const { data: positions } = usePositions()

  const openPnl = positions?.reduce((sum, p) => sum + p.unrealizedPnl, 0) ?? 0

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard
          label="Daily P&L"
          value={risk ? `₹${risk.dailyPnl.toLocaleString('en-IN')}` : '—'}
          delta={risk ? `Limit ₹${risk.dailyPnlLimit.toLocaleString('en-IN')}` : undefined}
          tone={risk && risk.dailyPnl >= 0 ? 'bull' : 'bear'}
          icon={Wallet}
        />
        <StatCard
          label="Open P&L"
          value={`₹${openPnl.toLocaleString('en-IN')}`}
          delta={positions ? `${positions.length} open positions` : undefined}
          tone={openPnl >= 0 ? 'bull' : 'bear'}
          icon={TrendingUp}
        />
        <StatCard
          label="Win Rate"
          value={perf ? `${fmt(perf.winRate * 100, 1)}%` : '—'}
          delta={perf ? `${perf.totalTrades} trades` : undefined}
          icon={Percent}
        />
        <StatCard
          label="Max Drawdown"
          value={perf ? `${fmt(perf.maxDrawdownPct, 2)}%` : '—'}
          delta="Trailing peak-to-trough"
          tone="neutral"
          icon={Activity}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5">
        <div className="xl:col-span-2 space-y-5">
          <EquityCurve />
          <PositionsTable />
        </div>
        <div className="space-y-5">
          <RiskPanel />
          <SignalFeed limit={6} />
        </div>
      </div>
    </div>
  )
}
