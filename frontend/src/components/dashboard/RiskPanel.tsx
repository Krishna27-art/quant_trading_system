import clsx from 'clsx'
import { AlertTriangle, ShieldCheck } from 'lucide-react'
import { Card, CardHeader } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Skeleton } from '@/components/ui/Skeleton'
import { ErrorState } from '@/components/ui/EmptyState'
import { useKillSwitch, useRisk } from '@/hooks/useRisk'

function Meter({ label, value, max, tone }: { label: string; value: number; max: number; tone: 'bull' | 'warn' | 'bear' }) {
  const pct = Math.min(100, Math.round((value / max) * 100))
  const barColor = tone === 'bull' ? 'bg-bull-500' : tone === 'warn' ? 'bg-warn-500' : 'bg-bear-500'
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-mist-500">{label}</span>
        <span className="num text-mist-300">{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-ink-700 overflow-hidden">
        <div className={clsx('h-full rounded-full transition-all', barColor)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

export function RiskPanel() {
  const { data: risk, isLoading, isError, error } = useRisk()
  const killSwitch = useKillSwitch()

  return (
    <Card>
      <CardHeader
        title="Risk Governance"
        subtitle="Pre-trade checks & exposure limits"
        action={
          risk && (
            <span
              className={clsx(
                'flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full border',
                risk.killSwitchActive
                  ? 'text-bear-400 border-bear-500/30 bg-bear-500/10'
                  : 'text-bull-400 border-bull-500/30 bg-bull-500/10'
              )}
            >
              {risk.killSwitchActive ? <AlertTriangle className="h-3 w-3" /> : <ShieldCheck className="h-3 w-3" />}
              {risk.killSwitchActive ? 'Halted' : 'Active'}
            </span>
          )
        }
      />

      {isLoading && (
        <div className="space-y-3">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
        </div>
      )}
      {isError && <ErrorState message={(error as Error).message} />}

      {risk && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <p className="text-xs text-mist-500">Daily P&amp;L</p>
              <p className={clsx('num font-semibold', risk.dailyPnl >= 0 ? 'text-bull-400' : 'text-bear-400')}>
                {risk.dailyPnl >= 0 ? '+' : ''}₹{risk.dailyPnl.toLocaleString('en-IN')}
              </p>
            </div>
            <div>
              <p className="text-xs text-mist-500">Positions</p>
              <p className="num font-semibold text-mist-100">
                {risk.positionCount} / {risk.maxPositions}
              </p>
            </div>
          </div>

          <Meter
            label="Open exposure"
            value={risk.openExposure}
            max={risk.maxExposure}
            tone={risk.openExposure / risk.maxExposure > 0.8 ? 'bear' : risk.openExposure / risk.maxExposure > 0.5 ? 'warn' : 'bull'}
          />
          <Meter
            label="Margin utilization"
            value={risk.marginUtilizationPct}
            max={100}
            tone={risk.marginUtilizationPct > 80 ? 'bear' : risk.marginUtilizationPct > 50 ? 'warn' : 'bull'}
          />

          <div className="pt-2 border-t border-ink-700/60 flex items-center justify-between">
            <p className="text-xs text-mist-500">
              {risk.killSwitchActive ? 'All new order placement is blocked.' : 'System is placing orders normally.'}
            </p>
            <Button
              variant={risk.killSwitchActive ? 'primary' : 'danger'}
              size="sm"
              disabled={killSwitch.isPending}
              onClick={() => killSwitch.mutate(!risk.killSwitchActive)}
            >
              {risk.killSwitchActive ? 'Resume trading' : 'Halt trading'}
            </Button>
          </div>
        </div>
      )}
    </Card>
  )
}
