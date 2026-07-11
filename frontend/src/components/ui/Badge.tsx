import { ReactNode } from 'react'
import clsx from 'clsx'

type Tone = 'bull' | 'bear' | 'neutral' | 'warn' | 'saffron'

const toneClasses: Record<Tone, string> = {
  bull: 'bg-bull-500/10 text-bull-400 border-bull-500/30',
  bear: 'bg-bear-500/10 text-bear-400 border-bear-500/30',
  neutral: 'bg-ink-600/40 text-mist-300 border-ink-500',
  warn: 'bg-warn-500/10 text-warn-500 border-warn-500/30',
  saffron: 'bg-saffron-500/10 text-saffron-400 border-saffron-500/30',
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: Tone }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-medium font-mono uppercase tracking-wide',
        toneClasses[tone]
      )}
    >
      {children}
    </span>
  )
}

export function SideBadge({ side }: { side: 'BUY' | 'SELL' }) {
  return <Badge tone={side === 'BUY' ? 'bull' : 'bear'}>{side}</Badge>
}
