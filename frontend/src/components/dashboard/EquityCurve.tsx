import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Card, CardHeader } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { ErrorState } from '@/components/ui/EmptyState'
import { usePerformance } from '@/hooks/usePerformance'

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="panel px-3 py-2 text-xs">
      <p className="text-mist-500 num mb-1">{new Date(label).toLocaleDateString('en-IN')}</p>
      <p className="num text-mist-100 font-medium">₹{payload[0].value.toLocaleString('en-IN')}</p>
    </div>
  )
}

export function EquityCurve() {
  const { data: perf, isLoading, isError, error } = usePerformance()

  return (
    <Card>
      <CardHeader title="Equity Curve" subtitle={perf ? `${perf.totalTrades} trades · Sharpe ${perf.sharpe?.toFixed(2) ?? 'N/A'}` : undefined} />

      {isLoading && <Skeleton className="h-56 w-full" />}
      {isError && <ErrorState message={(error as Error).message} />}

      {perf && (
        <div className="h-56 -ml-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={perf.equityCurve} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#FF9F43" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#FF9F43" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis
                dataKey="timestamp"
                tickFormatter={(v) => new Date(v).toLocaleDateString('en-IN', { month: 'short', day: 'numeric' })}
                tick={{ fill: '#5B6478', fontSize: 11 }}
                axisLine={{ stroke: '#232937' }}
                tickLine={false}
                minTickGap={40}
              />
              <YAxis
                tickFormatter={(v) => `₹${(v / 100000).toFixed(1)}L`}
                tick={{ fill: '#5B6478', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
                width={56}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="equity"
                stroke="#FF9F43"
                strokeWidth={2}
                fill="url(#equityFill)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  )
}
