import { Activity } from 'lucide-react'
import { Card, CardHeader } from '@/components/ui/Card'
import { SideBadge, Badge } from '@/components/ui/Badge'
import { SkeletonRows } from '@/components/ui/Skeleton'
import { EmptyState, ErrorState } from '@/components/ui/EmptyState'
import { useSignals } from '@/hooks/useSignals'

function timeAgo(iso: string) {
  const diffMs = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diffMs / 60_000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  return `${Math.floor(mins / 60)}h ago`
}

export function SignalFeed({ limit }: { limit?: number }) {
  const { data: signals, isLoading, isError, error } = useSignals()

  return (
    <Card>
      <CardHeader title="Live Signals" subtitle="Ranked by model win probability" />

      {isLoading && <SkeletonRows rows={5} />}
      {isError && <ErrorState message={(error as Error).message} />}

      {signals && signals.length === 0 && (
        <EmptyState
          icon={Activity}
          title="No active signals"
          description="The model hasn't generated a qualifying setup this session."
        />
      )}

      {signals && signals.length > 0 && (
        <div className="divide-y divide-ink-700/60">
          {signals.slice(0, limit ?? signals.length).map((s) => (
            <div key={s.id} className="flex items-center justify-between py-2.5">
              <div className="flex items-center gap-3 min-w-0">
                <SideBadge side={s.side} />
                <div className="min-w-0">
                  <p className="text-sm font-medium text-mist-100 truncate">{s.symbol}</p>
                  <p className="text-xs text-mist-500 truncate">{s.strategy}</p>
                </div>
              </div>
              <div className="text-right shrink-0 ml-3">
                <p className="text-sm num text-mist-100">₹{typeof s.entryPrice === 'number' ? s.entryPrice.toFixed(2) : '—'}</p>
                <div className="flex items-center gap-1.5 justify-end mt-0.5">
                  <Badge tone={s.winProbability >= 0.6 ? 'saffron' : 'neutral'}>
                    {typeof s.winProbability === 'number' ? (s.winProbability * 100).toFixed(0) + '%' : '—'}
                  </Badge>
                  <span className="text-[11px] text-mist-700 num">{timeAgo(s.generatedAt)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}
