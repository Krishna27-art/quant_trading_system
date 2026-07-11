import { Briefcase } from 'lucide-react'
import clsx from 'clsx'
import { Card, CardHeader } from '@/components/ui/Card'
import { SideBadge } from '@/components/ui/Badge'
import { SkeletonRows } from '@/components/ui/Skeleton'
import { EmptyState, ErrorState } from '@/components/ui/EmptyState'
import { usePositions } from '@/hooks/usePositions'
import { fmt } from '@/utils/format'

export function PositionsTable() {
  const { data: positions, isLoading, isError, error } = usePositions()

  return (
    <Card noPadding>
      <div className="p-4 pb-0">
        <CardHeader title="Open Positions" subtitle="Live mark-to-market" />
      </div>

      {isLoading && (
        <div className="p-4 pt-0">
          <SkeletonRows rows={4} />
        </div>
      )}
      {isError && (
        <div className="p-4 pt-0">
          <ErrorState message={(error as Error).message} />
        </div>
      )}

      {positions && positions.length === 0 && (
        <div className="p-4 pt-0">
          <EmptyState icon={Briefcase} title="No open positions" description="Filled orders will appear here." />
        </div>
      )}

      {positions && positions.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[11px] uppercase tracking-wide text-mist-700 border-y border-ink-600">
                <th className="px-4 py-2 font-medium">Symbol</th>
                <th className="px-4 py-2 font-medium">Side</th>
                <th className="px-4 py-2 font-medium text-right">Qty</th>
                <th className="px-4 py-2 font-medium text-right">Avg Entry</th>
                <th className="px-4 py-2 font-medium text-right">LTP</th>
                <th className="px-4 py-2 font-medium text-right">Unrealized P&amp;L</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={p.id} className="border-b border-ink-700/50 last:border-0 hover:bg-ink-700/30">
                  <td className="px-4 py-2.5 font-medium text-mist-100">{p.symbol}</td>
                  <td className="px-4 py-2.5">
                    <SideBadge side={p.side} />
                  </td>
                  <td className="px-4 py-2.5 text-right num">{p.quantity}</td>
                  <td className="px-4 py-2.5 text-right num text-mist-300">₹{fmt(p.avgEntryPrice, 2)}</td>
                  <td className="px-4 py-2.5 text-right num text-mist-300">₹{fmt(p.lastPrice, 2)}</td>
                  <td
                    className={clsx(
                      'px-4 py-2.5 text-right num font-medium',
                      p.unrealizedPnl >= 0 ? 'text-bull-400' : 'text-bear-400'
                    )}
                  >
                    {p.unrealizedPnl >= 0 ? '+' : ''}
                    ₹{p.unrealizedPnl.toLocaleString('en-IN')}
                    <span className="text-[11px] ml-1 opacity-70">
                      ({p.unrealizedPnlPct >= 0 ? '+' : ''}
                      {fmt(p.unrealizedPnlPct, 2)}%)
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}
