import { EquityCurve } from '@/components/dashboard/EquityCurve'
import { StatCard } from '@/components/dashboard/StatCard'
import { usePerformance } from '@/hooks/usePerformance'
import { Percent, TrendingDown, TrendingUp, Repeat } from 'lucide-react'

import { fmt } from '@/utils/format'

export default function Performance() {
  const { data: perf } = usePerformance()

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-lg font-semibold">Performance</h1>
        <p className="text-sm text-mist-500">Validated against out-of-sample walk-forward results.</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard
          label="Total Return"
          value={perf ? `${fmt(perf.totalReturn, 1)}%` : '—'}
          tone={perf && perf.totalReturn >= 0 ? 'bull' : 'bear'}
          icon={TrendingUp}
        />
        <StatCard label="Win Rate" value={perf ? `${fmt(perf.winRate * 100, 1)}%` : '—'} icon={Percent} />
        <StatCard
          label="Max Drawdown"
          value={perf ? `${fmt(perf.maxDrawdownPct, 2)}%` : '—'}
          tone="bear"
          icon={TrendingDown}
        />
        <StatCard label="Total Trades" value={perf ? String(perf.totalTrades) : '—'} icon={Repeat} />
      </div>

      <EquityCurve />
    </div>
  )
}
