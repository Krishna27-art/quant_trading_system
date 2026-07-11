import clsx from 'clsx'
import { useSignals } from '@/hooks/useSignals'

/**
 * The one signature element of the terminal: a continuously scrolling strip
 * of live model signals with color-coded win-probability confidence. It's
 * the ambient "pulse" of the ML engine — always visible, never demanding
 * attention, doubling the signal list so the CSS loop has no visible seam.
 */
export function MarketPulse() {
  const { data: signals } = useSignals()

  if (!signals || signals.length === 0) {
    return <div className="h-8 shrink-0 border-b border-ink-600 bg-ink-850" />
  }

  const loopItems = [...signals, ...signals]

  return (
    <div className="h-8 shrink-0 border-b border-ink-600 bg-ink-850 overflow-hidden relative">
      <div className="flex items-center h-full animate-ticker whitespace-nowrap w-max">
        {loopItems.map((s, i) => {
          const strong = s.winProbability >= 0.6
          return (
            <div key={`${s.id}-${i}`} className="flex items-center gap-2 px-4 text-xs border-r border-ink-700/60">
              <span className="font-semibold text-mist-300">{s.symbol}</span>
              <span className={clsx('font-mono', s.side === 'BUY' ? 'text-bull-400' : 'text-bear-400')}>
                {s.side}
              </span>
              <span
                className={clsx(
                  'font-mono px-1 rounded',
                  strong ? 'text-saffron-400' : 'text-mist-500'
                )}
              >
                {(s.winProbability * 100).toFixed(0)}% conf
              </span>
            </div>
          )
        })}
      </div>
      <div className="pointer-events-none absolute inset-y-0 left-0 w-10 bg-gradient-to-r from-ink-850 to-transparent" />
      <div className="pointer-events-none absolute inset-y-0 right-0 w-10 bg-gradient-to-l from-ink-850 to-transparent" />
    </div>
  )
}
