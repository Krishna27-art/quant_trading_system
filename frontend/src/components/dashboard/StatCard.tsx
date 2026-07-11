import { LucideIcon } from 'lucide-react'
import clsx from 'clsx'
import { Card } from '@/components/ui/Card'

export function StatCard({
  label,
  value,
  delta,
  icon: Icon,
  tone = 'neutral',
}: {
  label: string
  value: string
  delta?: string
  icon?: LucideIcon
  tone?: 'bull' | 'bear' | 'neutral'
}) {
  return (
    <Card className="flex items-center justify-between">
      <div>
        <p className="text-xs text-mist-500 uppercase tracking-wide">{label}</p>
        <p className="text-xl font-semibold num mt-1">{value}</p>
        {delta && (
          <p
            className={clsx(
              'text-xs num mt-0.5',
              tone === 'bull' && 'text-bull-400',
              tone === 'bear' && 'text-bear-400',
              tone === 'neutral' && 'text-mist-500'
            )}
          >
            {delta}
          </p>
        )}
      </div>
      {Icon && (
        <div className="h-9 w-9 rounded-lg bg-ink-700 flex items-center justify-center shrink-0">
          <Icon className="h-4 w-4 text-saffron-400" strokeWidth={1.75} />
        </div>
      )}
    </Card>
  )
}
